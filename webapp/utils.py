import numpy as np

def onehotCategorical(req, limit):
    arr = np.zeros((limit,))

    #print ("jeff here: ")
    #print (arr[req-1])

    arr[req-1] = 1
    #print (arr)
    return arr

#def switcher(count, values):


def switch_month(month):
    switcher = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }
    return switcher.get(month)


def switch_week(i):
        switcher = {
            'Sun': 1,
            'Mon': 2,
            'Tue': 3,
            'Wed': 4,
            'Thur': 5,
            'Fri': 6,
            'Sat': 7
        }
        return switcher.get(i)


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