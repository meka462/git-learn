import matplotlib.pyplot as plt
from data_struct.data_struct import grid_size_x, grid_size_y, depot_coordinates_x, depot_coordinates_y, train_path, \
    train_stops, drone_independent_cover_area
import numpy as np

# Visualize the data
def visualization_data(st, customer_coordinates, schema, opt_route, distfromdepot):
    fig = plt.figure()
    ax1 = fig.subplots()
    direct_eligible_customers = []
    vehicle_eligible_customers = []

    for i in range(len(customer_coordinates)):
        if schema[i] == 'drone_direct_schema':
            direct_eligible_customers.append(customer_coordinates[i])
        elif schema[i] == 'drone_vehicle_schema':
            vehicle_eligible_customers.append(customer_coordinates[i])

    # Draw points for Depot, Train Stoppage and Customer's.
    ax1.scatter(np.array([x for x, y in train_stops]), np.array([y for x, y in train_stops]), c='b',
                marker="s", label='Train Stoppage')
    ax1.scatter(np.array([i[0] for i in direct_eligible_customers]),
                np.array([i[1] for i in direct_eligible_customers]),
                c='#32CD32', marker="*", label='Customers(Direct)')
    ax1.scatter(np.array([i[0] for i in vehicle_eligible_customers]),
                np.array([i[1] for i in vehicle_eligible_customers]),
                c='#e35517', marker="*", label='Customers(Vehicle)')
    ax1.scatter(np.array([depot_coordinates_x]), np.array([depot_coordinates_y]), c='k', marker="o", label='Depot')

    # Draw train path
    ax1.plot([x for x, y in train_path], [y for x, y in train_path], linestyle='dashdot', label='Train Path')

    # Stop's label
    train_stop_marker = 1
    for x_cord, y_cord in train_stops:
        ax1.annotate('S-' + str(train_stop_marker), (x_cord + 1, y_cord - 0.5), size=8)
        train_stop_marker = train_stop_marker + 1

    # Customer's label
    for i in range(len(opt_route[0])):
        ax1.annotate(str(i), (customer_coordinates[opt_route[0][i]][0] + 1, customer_coordinates[opt_route[0][i]][1] - 0.5),
                     size=8)

    # Draw Drone range
    drone_range_circle = plt.Circle((depot_coordinates_x, depot_coordinates_y), drone_independent_cover_area,
                                    linestyle='dashdot', color='r', fill=False)
    ax1.add_artist(drone_range_circle)

    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('ADP V = 1')
    # plt.legend()
    # plt.xticks(range(-12, grid_size_x + 18, 12))
    # plt.yticks(range(-12, grid_size_y + 18, 12))
    plt.axis([0, grid_size_x + 10, 0, grid_size_y + 10])
    st.write(fig)
