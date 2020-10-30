import joblib
from utils import *
from flask import Flask, render_template, request
import sklearn

# Init Flask
application = Flask(__name__)
application.secret_key = 'very-secret-key'

# GET '/'
@application.route('/')
@application.route('/index')
def index():
    # Render
    return render_template('index.html', title='CarSafetyApp')

# model = joblib.load('rm.pkl')

@application.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        # get request values

        OLD_CAR = request.form['old_car']
        OLD_CAR = switch_older_car(OLD_CAR)

        # one-hot encode categorical variables

        CF = request.form['cf']
        CF = switch_crash_related(CF)
        # print(CF)
        CF_3, CF_5, CF_7, CF_13, CF_14, CF_15, CF_16, CF_17, CF_19, CF_20, CF_21, CF_23, CF_24, CF_25, CF_26, CF_27, CF_28 = onehotCategorical(
            CF, 17)

        LGTCON_IM = request.form['lighting']
        LGTCON_IM = switch_lighting(LGTCON_IM)
        LGTCON_IM_1, LGTCON_IM_2, LGTCON_IM_3, LGTCON_IM_4, LGTCON_IM_5, LGTCON_IM_6, LGTCON_IM_7 = onehotCategorical(
            LGTCON_IM, 7)

        TYPE_INT_LIST = ['1 Not an Intersection',
                         '2 Four-Way Intersection',
                         '3 T-Intersection',
                         '4 Y-Intersection',
                         '5 Traffic Circle',
                         '6 Roundabout',
                         '7 Five-Point, or More',
                         '10 L-Intersection']
        TYPE_INT = TYPE_INT_LIST.index(request.form['type_int'])
        list_len = len(TYPE_INT_LIST)
        TYP_INT_1, TYP_INT_2, TYP_INT_3, TYP_INT_4, TYP_INT_5, TYP_INT_6, TYP_INT_7, TYP_INT_10 = onehotCategorical(
            TYPE_INT, list_len)

        REL_ROAD_LIST = ['1 On Roadway',
                         '2 On Shoulder',
                         '3 On Median',
                         '4 On Roadside',
                         '5 Outside Trafficway',
                         '6 Off Roadway – Location Unknown',
                         '7 In Parking Lane/Zone',
                         '8 Gore',
                         '10 Separator',
                         '11 Continuous Left Turn Lane']

        REL_ROAD = REL_ROAD_LIST.index(request.form['rel_road'])
        list_len = len(REL_ROAD_LIST)
        REL_ROAD_1, REL_ROAD_2, REL_ROAD_3, REL_ROAD_4, REL_ROAD_5, REL_ROAD_6, REL_ROAD_7, REL_ROAD_8, REL_ROAD_10, REL_ROAD_11 = onehotCategorical(
            REL_ROAD, list_len)

        WRK_ZONE_LIST = ['0 None',
                         '1 Construction',
                         '2 Maintenance',
                         '3 Utility']
        WRK_ZONE = WRK_ZONE_LIST.index(request.form['wrk_zone'])
        list_len = len(WRK_ZONE_LIST)
        WRK_ZONE_0, WRK_ZONE_1, WRK_ZONE_2, WRK_ZONE_3 = onehotCategorical(WRK_ZONE, list_len)

        WEATHR_IM_LIST = ['1 Clear',
                          '2 Rain',
                          '3 Sleet or Hail',
                          '4 Snow',
                          '5 Fog, Smog, Smoke',
                          '6 Severe Crosswinds',
                          '7 Blowing Sand, Soil, Dirt',
                          '8 Other',
                          '10 Cloudy',
                          '11 Blowing Snow',
                          '12 Freezing Rain or Drizzle']
        WEATHR_IM = WEATHR_IM_LIST.index(request.form['weather_im'])
        list_len = len(WEATHR_IM_LIST)
        WEATHR_IM_1, WEATHR_IM_2, WEATHR_IM_3, WEATHR_IM_4, WEATHR_IM_5, WEATHR_IM_6, WEATHR_IM_7, WEATHR_IM_8, WEATHR_IM_10, WEATHR_IM_11, WEATHR_IM_12 = onehotCategorical(
            WEATHR_IM, list_len)

        ALCHL_IM_LIST = ['1 Alcohol Involved', '2 No Alcohol Involved']
        ALCHL_IM = ALCHL_IM_LIST.index(request.form['alchl_im'])
        list_len = len(ALCHL_IM_LIST)
        ALCHL_IM_1, ALCHL_IM_2 = onehotCategorical(ALCHL_IM, list_len)

        URBANICITY_LIST = ['1 Urban', '2 Rural']
        URBANICITY = URBANICITY_LIST.index(request.form['urbancity'])
        list_len = len(URBANICITY_LIST)
        URBANICITY_1, URBANICITY_2 = onehotCategorical(URBANICITY, list_len)

        SPEED_LIST = ['Less than 30 MPH',
                      'Between 30 and 65 MPH',
                      'More than 65 MPH']
        SPEED = SPEED_LIST.index(request.form['speed'])
        list_len = len(SPEED_LIST)
        SPD_L30MPH, SPD_30_65MPH, SPD_G65MPH = onehotCategorical(SPEED, list_len)

        BDTYPE_IM = request.form['bdtyp_im']
        BDTYPE_IM = switch_body_type(BDTYPE_IM)
        BDYTYP_IM_1, BDYTYP_IM_2, BDYTYP_IM_3, BDYTYP_IM_4, BDYTYP_IM_5, BDYTYP_IM_6, BDYTYP_IM_7, BDYTYP_IM_8, BDYTYP_IM_9, BDYTYP_IM_10, BDYTYP_IM_11, BDYTYP_IM_12, BDYTYP_IM_13, BDYTYP_IM_14, BDYTYP_IM_15, BDYTYP_IM_16, BDYTYP_IM_17, BDYTYP_IM_19, BDYTYP_IM_20, BDYTYP_IM_21, BDYTYP_IM_22, BDYTYP_IM_28, BDYTYP_IM_29, BDYTYP_IM_30, BDYTYP_IM_31, BDYTYP_IM_32, BDYTYP_IM_34, BDYTYP_IM_39, BDYTYP_IM_40, BDYTYP_IM_41, BDYTYP_IM_42, BDYTYP_IM_45, BDYTYP_IM_48, BDYTYP_IM_50, BDYTYP_IM_51, BDYTYP_IM_52, BDYTYP_IM_55, BDYTYP_IM_58, BDYTYP_IM_59, BDYTYP_IM_60, BDYTYP_IM_61, BDYTYP_IM_62, BDYTYP_IM_63, BDYTYP_IM_64, BDYTYP_IM_65, BDYTYP_IM_66, BDYTYP_IM_67, BDYTYP_IM_71, BDYTYP_IM_72, BDYTYP_IM_73, BDYTYP_IM_78, BDYTYP_IM_80, BDYTYP_IM_81, BDYTYP_IM_82, BDYTYP_IM_83, BDYTYP_IM_84, BDYTYP_IM_85, BDYTYP_IM_86, BDYTYP_IM_87, BDYTYP_IM_88, BDYTYP_IM_89, BDYTYP_IM_90, BDYTYP_IM_91, BDYTYP_IM_92, BDYTYP_IM_93, BDYTYP_IM_94, BDYTYP_IM_95, BDYTYP_IM_96, BDYTYP_IM_97 = onehotCategorical(
            BDTYPE_IM, 69)

        SPEEDREL_LIST = ['0 No',
                         '2 Yes, Racing',
                         '3 Yes, Exceeded Speed Limit',
                         '4 Yes, Too Fast for Conditions',
                         '5 Yes, Specifics Unknown']
        SPEEDREL = SPEEDREL_LIST.index(request.form['speedrel'])
        list_len = len(SPEEDREL_LIST)
        SPEEDREL_0, SPEEDREL_2, SPEEDREL_3, SPEEDREL_4, SPEEDREL_5 = onehotCategorical(SPEEDREL, list_len)

        VALIGN_LIST = ['0 Non-Road or Driveway',
                       '1 Straight',
                       '2 Curve Right',
                       '3 Curve Left',
                       '4 Curve – Unknown Dir.']
        VALIGN = VALIGN_LIST.index(request.form['valign'])
        list_len = len(VALIGN_LIST)
        VALIGN_0, VALIGN_1, VALIGN_2, VALIGN_3, VALIGN_4 = onehotCategorical(VALIGN, list_len)

        VPROFILE_LIST = ['0 Non-Road or Driveway',
                         '1 Level',
                         '2 Grade, Unknown Slope',
                         '3 Hillcrest',
                         '4 Sag (Bottom)',
                         '5 Uphill',
                         '6 Downhill']
        VPROFILE = VPROFILE_LIST.index(request.form['vprofile'])
        VPROFILE_0, VPROFILE_1, VPROFILE_2, VPROFILE_3, VPROFILE_4, VPROFILE_5, VPROFILE_6 = onehotCategorical(VPROFILE,
                                                                                                               len(
                                                                                                                   VPROFILE_LIST))

        VSURCOND_LIST = ['0 Non-Road or Driveway',
                         '1 Dry',
                         '2 Wet',
                         '3 Snow',
                         '4 Ice/Frost',
                         '5 Sand',
                         '6 Water',
                         '7 Oil',
                         '8 Other',
                         '10 Slush',
                         '11 Mud, Dirt, Gravel']
        VSURCOND = VSURCOND_LIST.index(request.form['vsurcond'])
        VSURCOND_0, VSURCOND_1, VSURCOND_2, VSURCOND_3, VSURCOND_4, VSURCOND_5, VSURCOND_6, VSURCOND_7, VSURCOND_8, VSURCOND_10, VSURCOND_11 = onehotCategorical(
            VSURCOND, len(VSURCOND_LIST))

        PCRASH1_IM_LIST = ['0 No/Unknown if Driver Present',
                           '1 Going Straight',
                           '2 Decelerating in Road',
                           '3 Accelerating in Road',
                           '4 Starting in Road',
                           '5 Stopped in Roadway',
                           '6 Passing or Overtaking Another Veh.',
                           '7 Disabled or Parked in Travel Lane',
                           '8 Leaving a Parking Position',
                           '9 Entering a Parking Position',
                           '10 Turning Right',
                           '11 Turning Left',
                           '12 Making a U-turn',
                           '13 Backing Up (Other than to Park)',
                           '14 Negotiating a Curve',
                           '15 Changing Lanes',
                           '16 Merging',
                           '17 Successful Corrective Action',
                           '98 Other']
        PCRASH1_IM = PCRASH1_IM_LIST.index(request.form['pcrash1_im'])
        PCRASH1_IM_0, PCRASH1_IM_1, PCRASH1_IM_2, PCRASH1_IM_3, PCRASH1_IM_4, PCRASH1_IM_5, PCRASH1_IM_6, PCRASH1_IM_7, PCRASH1_IM_8, PCRASH1_IM_9, PCRASH1_IM_10, PCRASH1_IM_11, PCRASH1_IM_12, PCRASH1_IM_13, PCRASH1_IM_14, PCRASH1_IM_15, PCRASH1_IM_16, PCRASH1_IM_17, PCRASH1_IM_98 = onehotCategorical(
            PCRASH1_IM, len(PCRASH1_IM_LIST))

        DR_SF_LIST = ['6 Careless Driving',
                      '8 Road Rage/Aggressive Driving',
                      '9 Emergency Services Personnel',
                      '10 Looked But Did Not See',
                      '16 Police or Law Enforcement Officer',
                      '18 Traveling on Prohibited Trafficways',
                      '20 Leaving Vehicle Unattended with Engine Running; Leaving Vehicle Unattended in Roadway',
                      '21 Overloading or Improper Loading of Vehicle with Passenger or Cargo',
                      '22 Towing or Pushing Vehicle Improperly',
                      '23 Failing to Dim Lights or to Have Lights on When Required',
                      '24 Operating Without Required Equipment',
                      '32 Opening Vehicle Closure into Moving Traffic or Vehicle is in Motion or Operating at Erratic or Suddenly Changing Speeds',
                      '36 Operating Vehicle in an Erratic, Reckless, Careless or Negligent Manner',
                      '37 Police Pursuing this Driver or Police Officer in Pursuit',
                      '50 Driving Wrong Way on One-Way Trafficway',
                      '51 Driving on Wrong Side of Two-Way Trafficway',
                      '54 Stopping in Roadway (Vehicle Not Abandoned)',
                      '55 Improper Management of Vehicle Controls',
                      '56 Object Interference with Vehicle Controls',
                      '57 Driving with Tire-Related Problems',
                      '58 Over Correcting',
                      '59 Getting Off/Out of a Vehicle',
                      '60 Alcohol and/or Drug Test Refused',
                      '91 Non-Traffic Violation Charged (Manslaughter, Homicide or Other)']
        DR_SF = DR_SF_LIST.index(request.form['dr_sf'])
        DR_SF_6, DR_SF_8, DR_SF_9, DR_SF_10, DR_SF_16, DR_SF_18, DR_SF_20, DR_SF_21, DR_SF_22, DR_SF_23, DR_SF_24, DR_SF_32, DR_SF_36, DR_SF_37, DR_SF_50, DR_SF_51, DR_SF_54, DR_SF_55, DR_SF_56, DR_SF_57, DR_SF_58, DR_SF_59, DR_SF_60, DR_SF_91 = onehotCategorical(
            DR_SF, len(DR_SF_LIST))

        IMPAIRED_LIST = ['0 None/Apparently Normal',
                         '1 Ill, Blackout',
                         '2 Asleep or Fatigued',
                         '3 Walking with a Cane or Crutches, etc.',
                         '4 Paraplegic or in Wheelchair',
                         '5 Impaired Due to Previous Injury',
                         '6 Deaf',
                         '7 Blind',
                         '8 Emotional (Depressed, Angry, Disturbed, etc.)',
                         '9 DUI of Alcohol, Drugs or Meds',
                         '10 Physical Impairment – No Details',
                         '95 No/Unknown if Driver Present',
                         '96 Other Physical Impairment']
        IMPAIRED = IMPAIRED_LIST.index(request.form['impaired'])
        IMPAIRED_NONE, IMPAIRED_BLACKOUT, IMPAIRED_ASLEEP, IMPAIRED_CANE, IMPAIRED_PARAPALEGIC, IMPAIRED_PREINJ, IMPAIRED_DEAF, IMPAIRED_BLIND, IMPAIRED_EMOTIONAL, IMPAIRED_DUI, IMPAIRED_PHY_UNK, IMPAIRED_NO_DRIVER, IMPAIRED_OTHER = onehotCategorical(
            IMPAIRED, len(IMPAIRED_LIST))

        REST_USE_LIST = ['0 Not Applicable',
                         '1 Shoulder Belt Only Used',
                         '2 Lap Belt Only Used',
                         '3 Lap and Shoulder Belt Used',
                         '4 Child Restraint Type Unknown',
                         '5 DOT-Compliant Motorcycle Helmet',
                         '7 None Used',
                         '8 Restraint Used – Type Unknown',
                         '10 Child Restraint System – Forward Facing',
                         '11 Child Restraint System – Rear Facing',
                         '12 1Booster Seat',
                         '16 Helmet, Other than DOT-Compliant Motorcycle Helmet',
                         '17 No Helmet',
                         '19 Helmet, Unknown if DOT-Compliant',
                         '20 None Used / Not Applicable',
                         '29 Unknown if Helmet Worn',
                         '96 Not a Motor Vehicle Occupant',
                         '97 Other']
        REST_USE = REST_USE_LIST.index(request.form['rest_use'])
        REST_USE_0, REST_USE_1, REST_USE_2, REST_USE_3, REST_USE_4, REST_USE_5, REST_USE_7, REST_USE_8, REST_USE_10, REST_USE_11, REST_USE_12, REST_USE_16, REST_USE_17, REST_USE_19, REST_USE_20, REST_USE_29, REST_USE_96, REST_USE_97 = onehotCategorical(
            REST_USE, len(REST_USE_LIST))

        REST_MIS_LIST = ['No', 'Yes']
        REST_MIS = REST_MIS_LIST.index(request.form['rest_mis'])
        REST_MIS_0, REST_MIS_1 = onehotCategorical(REST_MIS, len(REST_MIS_LIST))

        DRUGS_LIST = ['No (Drugs Not Involved)',
                      'Yes (Drugs Involved)']
        DRUGS = DRUGS_LIST.index(request.form['drugs'])
        DRUGS_0, DRUGS_1 = onehotCategorical(DRUGS, len(DRUGS_LIST))

        MFACTOR_LIST = ['0 None',
                        '1 Tires',
                        '2 Brake System',
                        '3 Steering System-Tie Rod, Kingpin, Ball Joint, etc.',
                        '4 Suspension-Springs, Shock Absorbers, Struts, etc.',
                        '5 Power Train-Universal Joint, Drive Shaft, Transmission, etc.',
                        '6 Exhaust System',
                        '7 Headlights',
                        '8 Signal Lights',
                        '9 Other Lights',
                        '10 Wipers',
                        '11 Wheels',
                        '12 Mirrors',
                        '13 Windows/Windshield',
                        '14 Body, Doors',
                        '15 Truck Coupling/Trailer Hitch/Safety Chains',
                        '16 Safety Systems',
                        '17 Vehicle Contributing Factors-No Details',
                        '97 Other']
        MFACTOR = MFACTOR_LIST.index(request.form['mfactor'])
        MFACTOR_0, MFACTOR_1, MFACTOR_2, MFACTOR_3, MFACTOR_4, MFACTOR_5, MFACTOR_6, MFACTOR_7, MFACTOR_8, MFACTOR_9, MFACTOR_10, MFACTOR_11, MFACTOR_12, MFACTOR_13, MFACTOR_14, MFACTOR_15, MFACTOR_16, MFACTOR_17, MFACTOR_97 = onehotCategorical(
            MFACTOR, len(MFACTOR_LIST))

        MDRDSTRD_LIST = ['0 Not Distracted',
                         '1 Looked But Did Not See',
                         '3 By Other Occupants',
                         '4 By a Moving Object In Vehicle',
                         '5 While Talking Or Listening To Cellular Phone',
                         '6 While Manipulating Cellular Phone',
                         '7 While Adjusting Audio Or Climate Controls',
                         '9 While Using Other Component/Controls Integral To Vehicle',
                         '10 While Using Or Reaching For Device/Object Brought into Vehicle',
                         '12 Distracted By Outside Person, Object Or Event',
                         '13 Eating Or Drinking',
                         '14 Smoking Related',
                         '15 Other Cellular Phone Related',
                         '16 No Driver Present/Unknown if Driver Present',
                         '17 Distraction/Inattention',
                         '18 Distraction/Careless',
                         '19 Careless/Inattentive',
                         '92 Distraction (Distracted), Details Unknown',
                         '93 Inattention (Inattentive), Details Unknown',
                         '96 Not Reported',
                         '97 Lost In Thought/Day Dreaming',
                         '98 Other Distraction']
        MDRDSTRD = MDRDSTRD_LIST.index(request.form['mdrdstrd'])
        MDRDSTRD_0, MDRDSTRD_1, MDRDSTRD_3, MDRDSTRD_4, MDRDSTRD_5, MDRDSTRD_6, MDRDSTRD_7, MDRDSTRD_9, MDRDSTRD_10, MDRDSTRD_12, MDRDSTRD_13, MDRDSTRD_14, MDRDSTRD_15, MDRDSTRD_16, MDRDSTRD_17, MDRDSTRD_18, MDRDSTRD_19, MDRDSTRD_92, MDRDSTRD_93, MDRDSTRD_96, MDRDSTRD_97, MDRDSTRD_98 = onehotCategorical(MDRDSTRD, len(MDRDSTRD_LIST))

        # manually specify competition distance
        comp_dist = 5458.1

        # build 1 observation for prediction
        entered_li = [CF_3, CF_5, CF_7, CF_13, CF_14, CF_15, CF_16, CF_17, CF_19, CF_20, CF_21, CF_23,
                      CF_24, CF_25, CF_26, CF_27, CF_28, LGTCON_IM_1, LGTCON_IM_2, LGTCON_IM_3, LGTCON_IM_4,
                      LGTCON_IM_5, LGTCON_IM_6, LGTCON_IM_7,
                      TYP_INT_1, TYP_INT_2, TYP_INT_3, TYP_INT_4, TYP_INT_5, TYP_INT_6, TYP_INT_7, TYP_INT_10,
                      REL_ROAD_1, REL_ROAD_2, REL_ROAD_3, REL_ROAD_4, REL_ROAD_5, REL_ROAD_6, REL_ROAD_7, REL_ROAD_8,
                      REL_ROAD_10, REL_ROAD_11, WRK_ZONE_0, WRK_ZONE_1, WRK_ZONE_2, WRK_ZONE_3, WEATHR_IM_1,
                      WEATHR_IM_2, WEATHR_IM_3, WEATHR_IM_4, WEATHR_IM_5, WEATHR_IM_6, WEATHR_IM_7, WEATHR_IM_8,
                      WEATHR_IM_10, WEATHR_IM_11, WEATHR_IM_12, ALCHL_IM_1, ALCHL_IM_2, URBANICITY_1, URBANICITY_2, OLD_CAR, SPD_L30MPH, SPD_30_65MPH, SPD_G65MPH, BDYTYP_IM_1,
                      BDYTYP_IM_2, BDYTYP_IM_3, BDYTYP_IM_4, BDYTYP_IM_5, BDYTYP_IM_6, BDYTYP_IM_7, BDYTYP_IM_8,
                      BDYTYP_IM_9, BDYTYP_IM_10, BDYTYP_IM_11, BDYTYP_IM_12, BDYTYP_IM_13, BDYTYP_IM_14, BDYTYP_IM_15,
                      BDYTYP_IM_16, BDYTYP_IM_17, BDYTYP_IM_19, BDYTYP_IM_20, BDYTYP_IM_21, BDYTYP_IM_22, BDYTYP_IM_28,
                      BDYTYP_IM_29, BDYTYP_IM_30, BDYTYP_IM_31, BDYTYP_IM_32, BDYTYP_IM_34, BDYTYP_IM_39, BDYTYP_IM_40,
                      BDYTYP_IM_41, BDYTYP_IM_42, BDYTYP_IM_45, BDYTYP_IM_48, BDYTYP_IM_50, BDYTYP_IM_51, BDYTYP_IM_52,
                      BDYTYP_IM_55, BDYTYP_IM_58, BDYTYP_IM_59, BDYTYP_IM_60, BDYTYP_IM_61, BDYTYP_IM_62, BDYTYP_IM_63,
                      BDYTYP_IM_64, BDYTYP_IM_65, BDYTYP_IM_66, BDYTYP_IM_67, BDYTYP_IM_71, BDYTYP_IM_72, BDYTYP_IM_73,
                      BDYTYP_IM_78, BDYTYP_IM_80, BDYTYP_IM_81, BDYTYP_IM_82, BDYTYP_IM_83, BDYTYP_IM_84, BDYTYP_IM_85,
                      BDYTYP_IM_86, BDYTYP_IM_87, BDYTYP_IM_88, BDYTYP_IM_89, BDYTYP_IM_90, BDYTYP_IM_91, BDYTYP_IM_92,
                      BDYTYP_IM_93, BDYTYP_IM_94, BDYTYP_IM_95, BDYTYP_IM_96, BDYTYP_IM_97, SPEEDREL_0, SPEEDREL_2,
                      SPEEDREL_3, SPEEDREL_4, SPEEDREL_5, VALIGN_0, VALIGN_1, VALIGN_2, VALIGN_3, VALIGN_4, VPROFILE_0, VPROFILE_1,
                      VPROFILE_2, VPROFILE_3, VPROFILE_4, VPROFILE_5, VPROFILE_6, VSURCOND_0, VSURCOND_1, VSURCOND_2,
                      VSURCOND_3, VSURCOND_4, VSURCOND_5, VSURCOND_6, VSURCOND_7, VSURCOND_8, VSURCOND_10, VSURCOND_11,
                      PCRASH1_IM_0, PCRASH1_IM_1, PCRASH1_IM_2, PCRASH1_IM_3, PCRASH1_IM_4, PCRASH1_IM_5, PCRASH1_IM_6,
                      PCRASH1_IM_7, PCRASH1_IM_8, PCRASH1_IM_9, PCRASH1_IM_10, PCRASH1_IM_11, PCRASH1_IM_12,
                      PCRASH1_IM_13, PCRASH1_IM_14, PCRASH1_IM_15, PCRASH1_IM_16, PCRASH1_IM_17, PCRASH1_IM_98, DR_SF_6,
                      DR_SF_8, DR_SF_9, DR_SF_10, DR_SF_16, DR_SF_18, DR_SF_20, DR_SF_21, DR_SF_22, DR_SF_23, DR_SF_24,
                      DR_SF_32, DR_SF_36, DR_SF_37, DR_SF_50, DR_SF_51, DR_SF_54, DR_SF_55, DR_SF_56, DR_SF_57,
                      DR_SF_58, DR_SF_59, DR_SF_60, DR_SF_91, IMPAIRED_NONE, IMPAIRED_BLACKOUT, IMPAIRED_ASLEEP,
                      IMPAIRED_CANE, IMPAIRED_PARAPALEGIC, IMPAIRED_PREINJ, IMPAIRED_DEAF, IMPAIRED_BLIND,
                      IMPAIRED_EMOTIONAL, IMPAIRED_DUI, IMPAIRED_PHY_UNK, IMPAIRED_NO_DRIVER, IMPAIRED_OTHER, REST_USE_0, REST_USE_1, REST_USE_2, REST_USE_3,
                      REST_USE_4, REST_USE_5, REST_USE_7, REST_USE_8, REST_USE_10, REST_USE_11, REST_USE_12,
                      REST_USE_16, REST_USE_17, REST_USE_19, REST_USE_20, REST_USE_29, REST_USE_96, REST_USE_97,
                      REST_MIS_0, REST_MIS_1, DRUGS_0, DRUGS_1, MFACTOR_0, MFACTOR_1, MFACTOR_2, MFACTOR_3, MFACTOR_4,
                      MFACTOR_5, MFACTOR_6, MFACTOR_7, MFACTOR_8, MFACTOR_9, MFACTOR_10, MFACTOR_11, MFACTOR_12,
                      MFACTOR_13, MFACTOR_14, MFACTOR_15, MFACTOR_16, MFACTOR_17, MFACTOR_97, MDRDSTRD_0, MDRDSTRD_1,
                      MDRDSTRD_3, MDRDSTRD_4, MDRDSTRD_5, MDRDSTRD_6, MDRDSTRD_7, MDRDSTRD_9, MDRDSTRD_10, MDRDSTRD_12,
                      MDRDSTRD_13, MDRDSTRD_14, MDRDSTRD_15, MDRDSTRD_16, MDRDSTRD_17, MDRDSTRD_18, MDRDSTRD_19,
                      MDRDSTRD_92, MDRDSTRD_93, MDRDSTRD_96, MDRDSTRD_97, MDRDSTRD_98]

        # entered_li = [StoreType_a, StoreType_b, StoreType_c, StoreType_d, Assortment_a, Assortment_b, Assortment_c, StateHoliday_0, StateHoliday_a, StateHoliday_b, StateHoliday_c, comp_dist, Promo2, DayOfWeek,Month,SchoolHoliday]
        print(entered_li)

        # make prediction

        # CORRECT
        model = joblib.load('gb.pkl')
        prediction = model.predict(np.array(entered_li).reshape(1, -1))
        #label = str(np.squeeze(prediction.round(2)))

        if prediction == 0:
            #print("ok")
            label = "NO: A severe accident is unlikely based on the factors presented."
        else:
            label = "YES: A severe accident is likely  based on the factors presented."


        return render_template('index.html', label=label)


if __name__ == "__main__":
    application.run(load_dotenv=True, use_reloader=True)

    model = joblib.load('gb.pkl')
# run the app.


# Ref
# https://www.likeanswer.com/question/330646
