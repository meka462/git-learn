
import csv

from helper.location_helper import *
import streamlit as st
from helper.schema_selection import schema_selection
from ADP_files.ADP import runADP

from ADP_files.ADP2step import run_two_step_ADP
from visualize import visualization_data
from data_struct.data_struct import *
import pandas as pd
import time as t

debugging = False


def data_visualize(Dataframe):
    visualization_data(st, customers, schema, opt_route, distFromDepot)
    # st.write(Dataframe)
    st.write("Optimal delivery route is: ", str([opt_route[0]]))
    st.write("Total delivery time for all the customers: " + str(round(opt_route[1], 3)) + " minutes")
    st.write("The runtime of the algorithm is " + str(round(finish - start, 3)) + " seconds")


if __name__ == "__main__":
    customers = locations('uniform', grid_size_x, grid_size_y, count_customer)
    print(customers)
    #customers = {0: (37.59, 10.92), 1: (22.95, 14.92), 2: (15.18, 11.43), 3: (7.38, 15.33), 4: (21.25, 24.17)}
    distFromDepot = customers_distance_from_depot(customers, (depot_coordinates_x, depot_coordinates_y))

    start = t.time()
    opt_route, departure, arrival, schema, battery, recharge, flight, extrarecharge, trip = schema_selection(customers, distFromDepot)
    #opt_route, departure, arrival, schema, battery, recharge, flight, extrarecharge, trip = runADP(customers, distFromDepot)
    # opt_route, departure, arrival, schema, battery, recharge, flight, extrarecharge, trip = run_two_step_ADP(customers, distFromDepot)

    finish = t.time()

    instancelist = list(
        zip(list(i for i in customers.values())))
    instancedataframe = pd.DataFrame(instancelist, columns=['Customers'])
    instancedataframe = pd.concat([instancedataframe, pd.Series(clock_wise_timetable)], ignore_index=True, axis=1)
    instancedataframe = pd.concat([instancedataframe, pd.Series(counter_clock_wise_timetable)], ignore_index=True,
                                  axis=1)
    instancedataframe.columns = ['Customers', 'Clockwise timetable', 'Counter-Clockwise timetable']
    instancedataframe.to_csv('big_instance.csv')
    with open("big_instance.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(('Depot coordinates', 'Number of customers'))
        writer.writerow(((depot_coordinates_x, depot_coordinates_y), count_customer))

    zippedlist = list(
        zip([i for i in opt_route[0]], list(customers[i] for i in opt_route[0]),
            list(distFromDepot[i] for i in opt_route[0]),
            list(drone_direct_eligibility(distFromDepot)[i] for i in opt_route[0]),
            list(drone_vehicle_eligibility(customers)[i] for i in opt_route[0]),
            list(schema.values()),
            list(datetime.datetime.time(x).replace(microsecond=0) for x in departure.values()),
            list(datetime.datetime.time(x).replace(microsecond=0) for x in arrival.values()),
            list(flight.values()), list(trip.values()), list(battery.values()), list(recharge.values()), list(extrarecharge.values())))
    dataframe = pd.DataFrame(zippedlist,
                             columns=['Delivery order', 'Customer coordinates', 'Distance from depot',
                                      'Drone direct eligibility', 'Drone vehicle eligibility',
                                      'Selected schema', 'Drone departure time', 'Drone arrival time',
                                      'Flight time', 'Trip time', 'Current battery after recharge', 'Actual recharge time', 'Extra recharge time'])

    dataframe.to_csv('adp.csv')
    with open("adp.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(('Program runtime (seconds)', 'Total delivery time of optimal route (minutes)'))
        writer.writerow((finish - start, opt_route[1]))

    if not debugging:
        st.title("Drone parcel delivery system.")
        data_visualize(dataframe)
    print("The optimal delivery order is: ", opt_route[0])
    print("Total delivery time for all the customers: %0.3f" % opt_route[1], "minutes")
    print('The runtime of the algorithm is %0.3f' % (finish - start), "seconds")

