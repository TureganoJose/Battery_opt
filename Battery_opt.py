
from energy_classes import Battery, Market

import pandas as pd
from mip import Model, OptimizationStatus

if __name__ == "__main__":

    # Full 2018-2020 period from exercise
    N = set(range(52608))

    # Only for debugging (10 days)
    # N = set(range(480))

    # Create model for solver
    model = Model()
    # Solver options
    model.threads = 1  # More threads needs more memory
    model.max_mip_gap_abs = 1000  # loosing up for computational reasons if needed

    # Instantiate market class: 3 different markets as defined in the exercise
    Market1 = Market('Market Data.xlsx', 'Half-hourly data', 0.5, '01', 1, is_daily_market=False)
    Market2 = Market('Market Data.xlsx', 'Half-hourly data', 0.5, '02', 2, is_daily_market=False)
    Market3 = Market('Market Data.xlsx', 'Daily data', 24, '03', 1, is_daily_market=True)
    Markets = [Market1, Market2, Market3]   # list of markets the battery is going to connect to

    # Instantiate battery class
    Aurora_Battery = Battery('Battery Parameters.xlsx', Markets, N, '01')

    # Declare variables, constraints and objective
    # YOU WANT TO LOOK INTO
    model = Aurora_Battery.add_constraints(model, 4)

    # Solve optimisation problem
    status = model.optimize()  # use max_seconds=300 if you want to find a feasable solution quickly

    # Post-process results (run before writing or plot outputs)
    Aurora_Battery.post_process(model, status)

    # Check status
    if status == OptimizationStatus.OPTIMAL:
        print('\n optimal solution cost {} found \n'.format(model.objective_value))

        # Write outputs to excel file
        Aurora_Battery.write_output('Calendric_deg')

        # Plot solution
        plot = Aurora_Battery.plot_outputs()
        plot.show()
    elif status == OptimizationStatus.FEASIBLE:
        print('\n sol.cost {} found, best possible: {} \n'.format(model.objective_value, model.objective_bound))

        # Write outputs to excel file
        Aurora_Battery.write_output('Degradation')

        # Plot solution
        plot = Aurora_Battery.plot_outputs()
        plot.show()
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))

    print("Program end")
