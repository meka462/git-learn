import random
from helper.time_table import generate_train_time_table
from data_struct.data_struct import *
import math
from math import sqrt
import numpy as np

# This function generates customer coordinates
def locations(schema, grid_size_x, grid_size_y, count_customers):
    cl = {}
    if schema == 'uniform':
        for i in range(count_customers):
            a = random.randint(700, grid_size_x * 100) / 100
            if a < 7:
                b = random.randint(700, (grid_size_y - 8) * 100) / 100
            elif a > 22:
                b = random.randint(700, grid_size_y * 100) / 100
            else:
                b = random.randint(700, grid_size_y * 100) / 100
            cl[i] = (a, b)

    return cl


# calculating the distance(dd(c))between depot to all the customers
def customers_distance_from_depot(customers_coordinates, depot_coordinates):
    Dc = []  # customer distance

    for i in range(len(customers_coordinates)):
        Dc.append(math.sqrt((customers_coordinates[i][0] - depot_coordinates[0]) ** 2 +
                            (customers_coordinates[i][1] - depot_coordinates[1]) ** 2))
    return Dc

#finding the nearest train stop(pd)from depot and their distance(d0)
def depot_to_nearest_train_stops(depot_coordinates, train_coordinates):
    dist = [math.sqrt((depot_coordinates[0] - train_stops[i][0]) ** 2 +
            (depot_coordinates[1] - train_stops[i][1]) ** 2)
        for i in range(len(train_coordinates))]
    dist_np = np.asarray(dist)
    station_index = (dist_np.argmin())  # Train stop closest to depot
    station_dist = (dist_np.min())  # Distance between depot and closest train stop
    return station_index, station_dist


# calculating the nearest point(p(c)to the customer on train route and their distance(dr(c)).
def train_path_near_to_customer(train_stops, customer_coordinates):
    distList = []  # List contains shortest distance and coordinates from given customer to two adjacent train stop for all train stops
    for stops in range(len(train_stops) - 1):
        distList.append((nearest_point_coordinate(train_stops[stops], train_stops[stops + 1], customer_coordinates),
                         stops, stops + 1))
    distList.append(
        (nearest_point_coordinate(train_stops[-1], train_stops[0], customer_coordinates), len(train_stops) - 1, 0))
    # The coordinates and distance of point on the adjacent train stops which are closest to customer
    print(distList)
    x_Cord = distList[[x[0][2] for x in distList].index(min([x[0][2] for x in distList]))][0][0]
    y_cord = distList[[x[0][2] for x in distList].index(min([x[0][2] for x in distList]))][0][1]
    dist = distList[[x[0][2] for x in distList].index(min([x[0][2] for x in distList]))][0][2]

    stop_one = distList[[x[0][2] for x in distList].index(min([x[0][2] for x in distList]))][1]
    stop_two = distList[[x[0][2] for x in distList].index(min([x[0][2] for x in distList]))][2]
    return {'pc_x': x_Cord, 'pc_y': y_cord, 'distance_to_customer': dist,
            'stop_one': stop_one, 'stop_two': stop_two}

# calculating the perpendicular coordinate p(c)between two train stops to a customer and their distance dr(c).
def nearest_point_coordinate(A, B, E):
    distA_E = stop_to_customer(A, E)

    distB_E = stop_to_customer(B, E)

    a = A[1] - B[1]
    b = B[0] - A[0]
    c = (A[0] - B[0]) * A[1] + (B[1] - A[1]) * A[0]
    temp = (-1 * (a * E[0] + b * E[1] + c) /
            (a * a + b * b))
    x4 = round(temp * a + E[0], 3)
    y4 = round(temp * b + E[1], 3)

    centerCord = (x4, y4)
    if centerCord > A and centerCord < B or centerCord < A and centerCord > B:
        distCenter_E = stop_to_customer(centerCord, E)
    else:
        distCenter_E = float("inf")

    if distA_E <= distB_E and distA_E <= distCenter_E:
        return A[0], A[1], distA_E
    elif distB_E <= distA_E and distB_E <= distCenter_E:
        return B[0], B[1], distB_E
    elif distCenter_E < distA_E and distCenter_E < distB_E:
        return centerCord[0], centerCord[1], distCenter_E


# Distance between stop and customer
def stop_to_customer(stop, customer):
    return sqrt((stop[0] - customer[0]) ** 2 + (stop[1] - customer[1]) ** 2)


# finding the nearest train stops(c)to the customer and calculating their distance ds(c
def customer_to_nearest_trainstop(customer, trains):
    dist = [math.sqrt((trains[i][0] - customer[0]) ** 2 + (trains[i][1] - customer[1]) ** 2) for i in
            range(len(trains))]
    dist_np = np.asarray(dist)
    station_index = (dist_np.argmin())
    station_dist = (dist_np.min())
    return station_index, station_dist


# calculating the minimum time to reach the coordinate p(c) from the initial train stop(pd)
def minimum_time(cord, train_stops, stopone, stoptwo):
    distlist = list()
    for stop in train_stops:
        dist = sqrt((stop[0] - cord[0]) ** 2 + (stop[1] - cord[1]) ** 2)
        distlist.append(dist)  # List of distance between coordinate
        # to all stops
    # Finding direction and distance from initial train stop to stop1 and  stop2
    first_shortest = direction(stopone, clock_wise_timetable,
                               counter_clock_wise_timetable)
    second_shortest = direction(stoptwo, clock_wise_timetable,
                                counter_clock_wise_timetable)

    # From the two stop find the one which takes shortest time to reach from initial train stop to p(c)

    timefrominitialstop, nearest_stop, Direction = min([first_shortest[0] + (distlist[first_shortest[1]] / s),
         first_shortest[1], first_shortest[2]], [second_shortest[0] + (distlist[second_shortest[1]] / s),
         second_shortest[1], second_shortest[2]])

    return timefrominitialstop, Direction


# finding the time in clockwise direction and counter-clockwise direction from the initial train stoppdto nearest train stop fromp(c).
def direction(stop, clock, anti_clock):
    if clock[stop] > anti_clock[::-1][stop]:
        return anti_clock[::-1][stop], stop, "Counter-Clockwise"
    else:
        return clock[stop], stop, "Clockwise"


# Drone direct flight time
def drone_flight_time(distFromDepot):
    dd_flight_time = {}
    for i in range(len(distFromDepot)):
        dd_flight_time[i] = ((2 * distFromDepot[i]) / v) + δ1 + δ2
    return dd_flight_time


# Drone direct schema eligibility
def drone_direct_eligibility(distFromDepot):
    dd_delivery = {}
    for i in range(len(distFromDepot)):
        if distFromDepot[i] <= (((Eo - e - e1 - e2) / (2 * p)) * v):
            dd_delivery[i] = 'drone_direct_schema'
        else:
            dd_delivery[i] = None
    return dd_delivery


#  Drone direct trip time for eligible customers
def drone_trip_time(distFromDepot):
    dd_eligibility = drone_direct_eligibility(distFromDepot)
    dft = {}
    for i in range(len(distFromDepot)):
        if dd_eligibility[i] == 'drone_direct_schema':
            dft[i] = ((2 * distFromDepot[i]) / v) + δ1 + δ2
        else:
            dft[i] = None
    return dft


#  Drone direct schema  recharge time for eligible customers
def drone_recharge_time(distFromDepot, customer_idx,
prev_customer_flight_time, current_battery):
    # Find battery consumed in previous trip
    remaining_battery = (prev_customer_flight_time * p) + e1 + e2
    # Flight time of current customer
    flight_time_curr = drone_trip_time(distFromDepot)[customer_idx]
    current_battery = current_battery - remaining_battery
    # Check if the battery is sufficient for the flight time
    if (e+e1+e2-current_battery+flight_time_curr*p)/q >0:
        # Calculate required recharge and add to our current battery
        rechargetime = (e + e1 + e2 - current_battery + flight_time_curr * p) / q
        current_battery = current_battery + rechargetime
        return rechargetime, current_battery
    else:
        return 0, current_battery


#  Drone direct schema Total delivery time
def drone_delivery_total_time(customer_idx, distFromDepot,
prev_customer_flight_time, current_battery):
    if drone_direct_eligibility(distFromDepot)[customer_idx] == 'drone_direct_schema':
        # Find time to recharge before delivery
        rechargetime, current_battery = drone_recharge_time(distFromDepot,
        customer_idx, prev_customer_flight_time, current_battery)
        # Add recharge time to delivery time to get the total time
        total_time = rechargetime + drone_trip_time(distFromDepot)[customer_idx]
        return (total_time, drone_trip_time(distFromDepot)[customer_idx],
            current_battery, 'drone_direct_schema', rechargetime, 0)
    else:
        return (None, None, current_battery, 'drone_vehicle_schema', None, None)


## Drone vehicle methods

# Drone vehicle schema eligibility
def drone_vehicle_eligibility(customers):
    dv_delivery = {}
    for idx, customer in customers.items():
        drc = train_path_near_to_customer(train_stops, customer)['distance_to_customer']
        dsc = customer_to_nearest_trainstop(customer, train_stops)[1]
        do = depot_to_nearest_train_stops((depot_coordinates_x, depot_coordinates_y), train_stops)[1]
        if drc + dsc <= (((Eo - e - e1 - e2) / p) * v) - (2 * do):
            dv_delivery[idx] = 'drone_vehicle_schema'
        else:
            dv_delivery[idx] = None
    return dv_delivery


# Drone vehicle flight time for eligible customers
def drone_vehicle_flight_time(customers):
    dv_eligibility = drone_vehicle_eligibility(customers)
    dft = {}
    for i in range(len(customers)):
        if dv_eligibility[i] == 'drone_vehicle_schema':
            drc = train_path_near_to_customer(train_stops, customers[i])['distance_to_customer']
            dsc = customer_to_nearest_trainstop(customers[i], train_stops)[1]
            do = depot_to_nearest_train_stops((depot_coordinates_x, depot_coordinates_y), train_stops)[1]
            dft[i] = ((do + drc + dsc + do) / v) + δ1 + δ2
        else:
            dft[i] = None
    return dft


# Drone vehicle recharge time for drone vehicle schema eligible customers
def drone_vehicle_recharge_time(customers, customer_idx, prev_customer_flight_time,
                                current_battery, current_time, Direction):
    # Calculate battery consumed in previous trip

    remaining_battery = prev_customer_flight_time * p
    # Flight time for the  current customer
    flight_time_curr = drone_vehicle_flight_time(customers)[customer_idx]
    current_battery = current_battery - remaining_battery

    # Check if battery is required for current customer delivery
    if (e+e1+e2-current_battery+flight_time_curr * p) / q > 0:
        primary_recharge = (e + e1 + e2 - current_battery + flight_time_curr * p) / q
        # Find the time the drone have to wait for train to arrive
        stop_reach_time = current_time + datetime.timedelta(minutes=(primary_recharge))
        # Use that time to recharge the drone at depot

        extra_recharge = extra_recharge_time(stop_reach_time, Direction)
        total_recharge = primary_recharge + (extra_recharge / q)
        rechargetime = total_recharge
        current_battery = current_battery + total_recharge
        # If the battery exccedes the capacity set it to back to maximum

        if current_battery > Eo: current_battery = Eo
        return rechargetime, current_battery, primary_recharge, extra_recharge
    else:
        # If recharge is not required still recharge drone for extra recharge
        stop_reach_time = current_time
        extra_recharge = extra_recharge_time(stop_reach_time, Direction)
        current_battery = current_battery + (extra_recharge / q)
        if current_battery > Eo: current_battery = Eo
        return extra_recharge, current_battery, 0, extra_recharge


# Calculates extra recharge time for avoiding the waiting time
def extra_recharge_time(stop_reach_time, Direction):
    # Calculate the time of arrival of drone at nearest train stop from depot
    st_index, do = depot_to_nearest_train_stops((depot_coordinates_x,
                                                 depot_coordinates_y), train_stops)
    stop_reach_time = stop_reach_time + datetime.timedelta(minutes=((do / v) + δ1))
    # Find the next earliest arrival time of train
    train_arrival_time, _ = generate_train_time_table(train_interval, stop_reach_time,
                                                      st_index, clock_wise_timetable, counter_clock_wise_timetable,
                                                      Direction)

    # Calculate the waiting time of drone for the next train
    extra_recharge = ((train_arrival_time - stop_reach_time).total_seconds()) / 60
    return extra_recharge


# Trip time for  drone vehicle schema eligible customers
def drone_vehicle_trip_time(customer_idx, customers, current_time):
    if drone_vehicle_eligibility(customers)[customer_idx] == 'drone_vehicle_schema':
        [x, y, drc, stopone, stoptwo] = list(train_path_near_to_customer(train_stops,
                                         customers[customer_idx]).values())

        pc = (x, y)
        customer_near_station_index, dsc = customer_to_nearest_trainstop(customers[customer_idx], train_stops)
        _, _, returndirection = direction(customer_near_station_index, clock_wise_timetable, counter_clock_wise_timetable)
        do = depot_to_nearest_train_stops((depot_coordinates_x,
                                           depot_coordinates_y), train_stops)[1]
        min_time, Direction = minimum_time(pc,
                                           train_stops, stopone, stoptwo)
        nearest_station_stop_time = current_time + datetime.timedelta(
            minutes=((do / v) + δ1 + (min_time) + ((drc + dsc) / v)))

        train_arrival_time_return, return_time = generate_train_time_table(train_interval,
                                  nearest_station_stop_time, customer_near_station_index,
                                  clock_wise_timetable, counter_clock_wise_timetable, returndirection)
        trip_time = (do / v) + δ1 + min_time + ((drc + dsc) / v) + return_time + (do / v) + δ2
        return trip_time, Direction


# Total time for one delivery for drone vehicle schema
def drone_vehicle_total_time(customer_idx, prev_customer_flight_time, customers, current_time, current_battery):
    if drone_vehicle_eligibility(customers)[customer_idx] == 'drone_vehicle_schema':
        # Calculate the trip time of delivery
        trip_time, Direction = drone_vehicle_trip_time(customer_idx,
        customers, current_time)
        # Calculate the total reacharge time including the extra recharge time
        rechargetime, current_battery, actual_recharge, extra_recharge = drone_vehicle_recharge_time(customers,
                        customer_idx, prev_customer_flight_time, current_battery, current_time, Direction)
        # Added trip time and recharge time to get total delivery time
        total_time = rechargetime + trip_time

        return (total_time, trip_time, current_battery, 'drone_vehicle_schema', actual_recharge, extra_recharge)
    else:
        return (None, None, current_battery, 'drone_direct_schema', None, None)
