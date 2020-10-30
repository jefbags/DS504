import numpy as np

def onehotCategorical(req, limit):
    arr = np.zeros((limit,))

    #print ("jeff here: ")
    #print (arr[req-1])

    arr[req-1] = 1
    #print (arr)
    return arr

#def switcher(count, values):

def switch_crash_related(i):
    switcher = {
        '3 Other Maintenance or Construction-Created Condition': 1,
        '5 Surface Under Water': 2,
        '7 Surface Washed Out (Caved in, Road Slippage)': 3,
        '13 Aggressive Driving/Road Rage by Non-Contact Vehicle Driver': 4,
        '14 Motor Vehicle Struck By Falling Cargo or Something That Came Loose From  or Something That Was Set in Motion By a Vehicle': 5,
        '15 Non-Occupant Struck By Falling Cargo, or Something Came Loose From or Something That Was Set In Motion By A Vehicle': 6,
        '16 Non-Occupant Struck Vehicle': 7,
        '17 Vehicle Set In Motion By Non-Driver': 8,
        '19 Recent Previous Crash Scene Nearby': 9,
        '20 Police-Pursuit-Involved': 10,
        '21 Within Designated School Zone': 11,
        '23 Indication of a Stalled/Disabled Vehicle': 12,
        '24 Unstabilized Situation Began and All Harmful Events Occurred Off of the Roadway': 13,
        '25 Toll Booth/Plaza Related': 14,
        '26 Backup Due to Prior Non-Recurring Incident': 15,
        '27 Backup Due to Prior Crash': 16,
        '28 Backup Due to Regular Congestion': 17

    }
    return switcher.get(i)

def switch_lighting(i):
    switcher = {
        '1 Daylight': 1,
        '2 Dark – Not Lighted': 2,
        '3 Dark – Lighted': 3,
        '4 Dawn': 4,
        '5 Dusk': 5,
        '6 Dark – Unknown Lighting': 6,
        '7 Other': 7
    }
    return switcher.get(i)


def switch_older_car(i):
    switcher = {
        "No": 0,
        "Yes": 1,
    }
    return switcher.get(i)

def switch_body_type(i):
    switcher = {
        '1 Convertible (Excludes Sun-Roof, T-Bar)': 1,
        '2 2-Door Sedan, Hardtop, Coupe': 2,
        '3 3-Door/2-Door Hatchback': 3,
        '4 4-Door Sedan, Hardtop': 4,
        '5 5-Door/4-Door Hatchback': 5,
        '6 Station Wagon (Excluding Van And Truck Based)': 6,
        '7 Hatchback, Number Of Doors Unknown': 7,
        '8 Sedan/Hardtop, Number of Doors Unknown': 8,
        '9 Other or Unknown Auto Type': 9,
        '17 3-Door Coupe': 17,
        '10 Auto Based Pickup (Includes El Camino, Caballero...)': 10,
        '11 Auto Based Panel (Cargo Station Wagon, Ambulance/Hearse)': 11,
        '12 Large Limousine (More Than Four Side Doors Or Stretched Chassis)': 12,
        '13 Three Wheel Auto Or Auto Derivative': 13,
        '14 Compact Utility (ANSI D-16 Util. Veh.,“Small” and “Midsize”)': 14,
        '15 Large Utility (ANSI D-16 Util. Veh.,“Full Size” and “Large”)': 15,
        '16 Utility Station Wagon': 16,
        '19 Utility Vehicle, Unknown Body Type': 18,
        '20 Minivan': 19,
        '21 Large Van – Includes Van-Based Buses': 20,
        '22 Step Van Or Walk-In Van (GVWR less than or equal to 10,000 lbs)': 21,
        '28 2Other Van Type': 22,
        '29 Unknown Van Type': 23,
        '30 Compact Pickup (S-10, LUV, Ram 50, Dakota, Sonoma...)': 24,
        '31 Standard Pickup (C10-C35, Silverado, Sierra, T100...)': 25,
        '32 Pickup With Slide-In Camper (2016-2017 Only)': 26,
        '34 Light Pickup': 27,
        '39 Unknown (Pickup Style) Light Conventional Truck': 28,
        '40 Cab Chassis Based (Incl. Rescue Veh., Dump, And Tow Truck)': 29,
        '41 Truck Based Panel': 30,
        '45 Other Light Conventional Truck Type': 32,
        '48 Unknown Light Truck Type': 33,
        '50 School Bus (Designed To Carry Students, Not Cross Country Or Transit)': 34,
        '51 Cross Country/Intercity Bus (i.e., Greyhound)': 35,
        '52 Transit Bus (City Bus)': 36,
        '55 Van-Based Bus (GVWR greater than 10,000 lbs)': 37,
        '58 Other Bus Type': 38,
        '59 Unknown Bus Type': 39,
        '60 Step Van (GVWR greater than 10,000 lbs)': 40,
        '61 Single-Unit Straight Truck or Cab-Chassis (GVWR 10,001-19,500 lbs)': 41,
        '62 Single-Unit Straight Truck or Cab-Chassis (GVWR 19,501-26,000 lbs)': 42,
        '63 Single-Unit Straight Truck or Cab-Chassis (GVWR > than 26,000 lbs)': 43,
        '64 Single Unit Straight Truck or Cab-Chassis (GVWR unknown)': 44,
        '66 Truck-Tractor (Any Number Of Trailing Units and Weight)': 46,
        '67 Medium/Heavy Pickup (GVWR > 10,000 lbs)': 47,
        '71 Unknown Single-Unit or Combo-Unit Med. Truck (GVWR 10,001-26,000 lbs)': 48,
        '72 Unknown Single-Unit or Combo-Unit Heavy Truck (GVWR > than 26,000 lbs)': 49,
        '78 Unknown Medium/Heavy Truck Type': 51,
        '42 Light Truck Based Motor Home (Chassis Mounted)': 31,
        '65 Medium/Heavy Truck-Based Motor Home': 45,
        '73 Camper or Motor Home, Unknown Truck Type': 50,
        '80 Two Wheel Motorcycle (excluding motor scooters)': 52,
        '81 Moped (Motorized Bicycle)': 53,
        '82 Three-wheel Motorcycle (2 Rear Wheels)': 54,
        '83 Off-Road Motorcycle': 55,
        '84 Motor Scooter': 56,
        '85 Unenclosed 3 Wheel Motorcycle/Unenclosed Autocycle (1 Rear Wheel)': 57,
        '86 Enclosed 3 Wheel Motorcycle/Enclosed Autocycle (1 Rear Wheel)': 58,
        '87 Unknown 3 Wheel Motorcycle Type': 59,
        '88 Other Motored Cycle Type (Mini-bikes, Pocket Motorcycles)': 60,
        '89 Unknown Motored Cycle Type': 61,
        '90 ATV (All-Terrain Vehicle) / ATC (All-Terrain Cycle)': 62,
        '91 Snowmobile': 63,
        '92 Farm Equipment Other Than Trucks': 64,
        '93 Construction Equipment Other Than Trucks (Includes Graders)': 65,
        '94 Low Speed Vehicle (LSV)/Neighborhood Electric Vehicle (NEV)': 66,
        '95 Golf Cart': 67,
        '96 Recreational Off-Highway Vehicle (ROV)': 68,
        '97 Other Vehicle Type (Includes Go-Cart, Fork-Lift, Street Sweeper)': 69
    }
    return switcher.get(i)