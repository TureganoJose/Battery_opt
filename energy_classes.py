import pandas as pd
from mip import Model, xsum, maximize, BINARY, OptimizationStatus
import matplotlib.pyplot as plt
import numpy as np

"""Class Battery
Defines the battery encapsulating the follwing methods:
add_constraints()
post_process()
write_output()
plot_outputs()

"""
class Battery:
    def __init__(self, filename, Markets: list, N: set, id: str):
        """Instantiate battery class.

        Keyword arguments:
        filename -- input file with battery parameters
        Markets -- list of markets the battery is connected to
        id -- name id of battery
        """
        self._id = id
        if not filename:
            # Load default values
            self.max_charge_rate = 2
            self.max_discharge_rate = 2
            self.max_storage_volume = 4
            self.charging_eff = 0.05
            self.discharging_eff = 0.05
            self.lifetime_cycles = 5000
            self.storage_deg = 0.001
            self.capex = 500000
            self.fix_op_cost = 5000
            print('loaded default values for battery {} \n'.format(self._id))
        else:
            parameter_table = pd.read_excel(open(filename, 'rb'),
                                               engine='openpyxl')
            self.max_charge_rate = parameter_table.values[0][1]
            self.max_discharge_rate = parameter_table.values[1][1]
            self.max_storage_volume = parameter_table.values[2][1]
            self.charging_eff = parameter_table.values[3][1]
            self.discharging_eff = parameter_table.values[4][1]
            self.lifetime_years = parameter_table.values[5][1]
            self.lifetime_cycles = parameter_table.values[6][1]
            self.storage_deg = parameter_table.values[7][1]
            self.capex = parameter_table.values[8][1]
            self.fix_op_cost = parameter_table.values[9][1]
            print('loaded parameter values for battery {} from {}\n'.format(self._id,filename))

        self.delta_time = 1e6
        if not Markets:
            # default values
            self.Markets = []
            self.delta_time = 0.5
        else:
            self.Markets = Markets
            for market in Markets:
                self.delta_time = min(self.delta_time, market.delta_t)

        self.n_markets = len(Markets)
        self.n_steps = len(N)
        self.N = N

        # Increase market granularity to match reference (finding minimum granularity)
        for i in range(self.n_markets):
            if (self.Markets[i].delta_t - self.delta_time) > 1e-6:
                size_ratio = int(self.Markets[i].delta_t / self.delta_time)
                self.Markets[i].delta_t = self.delta_time
                new_price_history = []
                for price_period in self.Markets[i].price_history:
                    temp_list = [price_period] * size_ratio
                    new_price_history.extend(temp_list)

                print("Market id {} changed granularity to {} hours \n".format(self.Markets[i]._id,
                                                                               self.Markets[i].delta_t))
                self.Markets[i].price_history = new_price_history

        # Logging placeholders for solution
        self.sol_Pb_disch = []        # Pb_disch: rate of discharge to each market [MW]
        self.sol_Pb_charg = []        # Pb_charg: rate of charge from each market [MW]
        self.sol_Pb_disch_total = []  # Pb_disch_total: total rate of discharge [MW]
        self.sol_Pb_charge_total = [] # Pb_charg_total: total rate of charge [MW]
        self.sol_SoC = []             # SoC: Battery capacity [MWh] (not really SoC per se)
        self.sol_acc_disch_Qb = []    # acc_disch_Qb: Total accumulated capacity discharged [MWh]
        self.sol_acc_charg_Qb = []    # acc_charg_Qb: Total accumulated capacity charged [MWh]
        self.sol_b_charg = []         # b_charg: Binary flag indicating battery charging [-]
        self.sol_b_disch = []         # b_disch: Binary flag indicating battery discharging [-]
        self.sol_n_cycles = []        # n_cycles: number of cycles [-]
        self.objective_revenue = 0

    def add_constraints(self, model: Model, SoC_init: int):
        """Declares all necessary variables, constraints and objective function
        for optimisation

        Note: Ideally you want to split these in different functions

        Keyword arguments:
        Model -- full optimisation model as defined in mip package
        SoC_init -- initial charge capacity of battery [MW]
        """
        # Declare variables
        Pb_charg = [[model.add_var(name='Pb_charg_' + self._id) for j in range(self.n_markets)] for i in self.N]
        Pb_disch = [[model.add_var(name='Pb_disch_' + self._id) for j in range(self.n_markets)] for i in self.N]
        Pb_charg_total = [model.add_var(name='Pb_charg_total_' + self._id) for i in self.N]
        Pb_disch_total = [model.add_var(name='Pb_disch_total_' + self._id) for i in self.N]
        SoC = [model.add_var(name='SoC_' + self._id) for i in self.N]
        acc_disch_Qb = [model.add_var(name='acc_disch_Qb_' + self._id) for i in self.N]
        acc_charg_Qb = [model.add_var(name='acc_charg_Qb_' + self._id) for i in self.N]
        b_charg = [model.add_var(name='b_charg_' + self._id, var_type=BINARY) for i in self.N]
        b_disch = [model.add_var(name='b_disch_' + self._id, var_type=BINARY) for i in self.N]
        n_cycles = [model.add_var(name='n_cycles_' + self._id, lb=0, ub=5000) for i in self.N]

        # Daily market constraint: Charge and discharge rates are the same for an entire day
        for j in range(self.n_markets):
            if self.Markets[j].b_daily_market:
                n_steps_per_day = int(24 / self.delta_time)
                n_days = int(self.n_steps / n_steps_per_day)  # round down
                for d in range(n_days):
                    for i in range((d * n_steps_per_day) + 1, ((d + 1) * n_steps_per_day)):
                        model += Pb_charg[i - 1][j] == Pb_charg[i][j]
                        model += Pb_disch[i - 1][j] == Pb_disch[i][j]

        # Charging and discharging rates limits
        # Sum of all the charges and discharges
        for i in self.N:
            model += b_charg[i] + b_disch[i] <= 1
            model += xsum(Pb_disch[i][j] for j in range(self.n_markets)) == Pb_disch_total[i]
            model += xsum(Pb_charg[i][j] for j in range(self.n_markets)) == Pb_charg_total[i]
            model += Pb_disch_total[i] <= self.max_discharge_rate * b_disch[i]
            model += Pb_charg_total[i] <= self.max_charge_rate * b_charg[i]

        # Capacity limit
        # Note that number of cycles affects maximum capacity limit (linear model)
        model += SoC[0] == SoC_init
        for i in range(1, self.n_steps):
            model += SoC[i - 1] + (Pb_charg_total[i] - Pb_disch_total[i]) * self.delta_time == SoC[i]
            model += SoC[i] <= self.max_storage_volume * (100 - self.storage_deg * n_cycles[i]) / 100

        # Number of cycles
        # Based on the accumulated charge and discharge
        # Number of cycles treated as a continuous variable
        model += acc_disch_Qb[0] == 0
        model += acc_charg_Qb[0] == 0
        for i in range(1, self.n_steps):
            model += Pb_disch_total[i] * self.delta_time + acc_disch_Qb[i - 1] == acc_disch_Qb[i]
            model += Pb_charg_total[i] * self.delta_time + acc_charg_Qb[i - 1] == acc_charg_Qb[i]
            model += 0.5 * (acc_disch_Qb[i - 1] / self.max_storage_volume)\
                     + 0.5 * (acc_charg_Qb[i - 1] / self.max_storage_volume) == n_cycles[i]

        # Objective function: Revenue + cost of efficiency loss + battery depreciation (degradation as a function of
        # number of cycles) + battery degradation cost (linear proportional to cycles)
        # + calendric degradation (proportional to number of idle hours).
        # Fixed operational cost not included
        # No calendar degradation
        model.objective = maximize(xsum(self.Markets[j].price_history[i] *
                                        (Pb_disch[i][j] - Pb_charg[i][j]
                                         -(Pb_charg[i][j] * self.charging_eff) -
                                         (Pb_disch[i][j] * self.discharging_eff)
                                         ) * self.delta_time
                                        - (self.capex/(self.n_markets*self.lifetime_years*365*24))
                                        * 0.5*(1 - (b_charg[i] + b_disch[i]))
                                        - (self.capex / (self.n_markets*self.n_steps*self.lifetime_cycles))
                                        * n_cycles[self.n_steps-1]
                                        for i in self.N for j in range(self.n_markets)))
        return model

    def post_process(self, model: Model, status):
        """Extracts values of optimum variables and store them in class

        Keyword arguments:
        Model -- full optimisation model as defined in mip package
        status -- optimisation status
        """
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            for i, v in enumerate(model.vars):
                if v.name == 'Pb_disch_' + self._id:
                    self.sol_Pb_disch.append(v.x)
                elif v.name == 'Pb_charg_' + self._id:
                    self.sol_Pb_charg.append(v.x)
                elif v.name == 'Pb_disch_total_' + self._id:
                    self.sol_Pb_disch_total.append(v.x)
                elif v.name == 'SoC_' + self._id:
                    self.sol_SoC.append(v.x)
                elif v.name == 'acc_disch_Qb_' + self._id:
                    self.sol_acc_disch_Qb.append(v.x)
                elif v.name == 'acc_charg_Qb_' + self._id:
                    self.sol_acc_charg_Qb.append(v.x)
                elif v.name == 'b_charg_' + self._id:
                    self.sol_b_charg.append(v.x)
                elif v.name == 'b_disch_' + self._id:
                    self.sol_b_disch.append(v.x)
                elif v.name == 'n_cycles_' + self._id:
                    self.sol_n_cycles.append(v.x)

        self.sol_Pb_disch = np.asarray(self.sol_Pb_disch)
        self.sol_Pb_charg = np.asarray(self.sol_Pb_charg)

        self.sol_Pb_disch = np.reshape(self.sol_Pb_disch, (self.n_steps, self.n_markets))
        self.sol_Pb_charg = np.reshape(self.sol_Pb_charg, (self.n_steps, self.n_markets))

        self.sol_b_charg = np.asarray(self.sol_b_charg)
        self.sol_b_disch = np.asarray(self.sol_b_disch)

        self.objective_revenue = model.objective_value

    def write_output(self, outputname: str):
        """Writes excel output file

        Keyword arguments:
        outputname -- name of excel output (without extension)
        """
        df = pd.DataFrame({'Pb_disch Market 1': self.sol_Pb_disch[:, 0].T,
                           'Pb_disch Market 2': self.sol_Pb_disch[:, 1].T,
                           'Pb_disch Market 3': self.sol_Pb_disch[:, 2].T,
                           'Pb_charg Market 1': self.sol_Pb_charg[:, 0].T,
                           'Pb_charg Market 2': self.sol_Pb_charg[:, 1].T,
                           'Pb_charg Market 3': self.sol_Pb_charg[:, 2].T,
                           'Capacity': np.asarray(self.sol_SoC),
                           'Number of cycles': np.asarray(self.sol_n_cycles),
                           'Revenue': self.objective_revenue})
        df.to_excel(outputname + '.xlsx', sheet_name='Sheet_name_1')

    def plot_outputs(self):
        """Returns plot figure with outputs
        """
        power_colours = ['b', 'g', 'y']
        market_colours = ['r', 'm', 'c']

        fig, axs = plt.subplots(2, 2)
        for j in range(self.n_markets):
            axs[0, 0].plot(self.sol_Pb_charg[:, j], color=power_colours[j])
        axs[0, 0].set_title('Pb_charg(KW)')
        ax00 = axs[0, 0].twinx()
        for j in range(self.n_markets):
            ax00.plot(self.Markets[j].price_history[0:self.n_steps], color=market_colours[j])
        ax00.set_ylabel('price [£/MW]', color='r')
        ax00.tick_params(axis='y', labelcolor='r')

        axs[0, 1].plot(self.sol_SoC, 'tab:orange')
        axs[0, 1].set_title('Capacity [MWh]')

        for j in range(self.n_markets):
            axs[1, 0].plot(self.sol_Pb_disch[:, j], color=power_colours[j])
        axs[1, 0].set_title('Pb_disch(KW)')
        ax10 = axs[1, 0].twinx()
        for j in range(self.n_markets):
            ax10.plot(self.Markets[j].price_history[0:self.n_steps], color=market_colours[j])
        ax10.set_ylabel('price [£/MW]', color='r')
        ax10.tick_params(axis='y', labelcolor='r')
        axs[1, 1].plot(self.sol_n_cycles, 'tab:orange')
        axs[1, 1].set_title('NCycles')
        return fig

class Market:
    def __init__(self, filename: str, excel_tab: str, time_step: float, id: str, col: int, is_daily_market: bool):
        """Instantiate Market class.

        Keyword arguments:
        filename -- input file with battery parameters
        excel_tab -- specifies which tab to read from
        time_step -- granularity of the market in hours
        id -- name id of battery
        col -- column where prices are defined
        is_daily_market -- boolean for daily market (it adds extra constraints)
        """
        self._id = id
        self.delta_t = time_step
        self.price_history = pd.read_excel(open(filename, 'rb'),
                                           engine='openpyxl',
                                           sheet_name=excel_tab,
                                           usecols=[col])
        self.price_history = self.price_history.T.values.tolist()[0]
        self.b_daily_market = is_daily_market
