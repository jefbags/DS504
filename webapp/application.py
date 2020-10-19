#!/usr/bin/env python3

# Imports
import json
import os

from typing import List, Any

import flask
from flask import Flask, request, render_template
import joblib
import numpy as np
from utils import *

from flask import Flask, jsonify, render_template

# Init Flask
application = Flask(__name__)
application.secret_key = 'very-secret-key'


# Cache control -
# No caching at all for API endpoints.
@application.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


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

        # Need Switch
        MONTH = request.form['month']
        # print (MONTH)
        MONTH = switch_month(MONTH)
        # print (MONTH)

        WKDY_IM = request.form['day_of_the_week']
        WKDY_IM = switch_week(WKDY_IM)

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

        EVENT_IM_LIST = ['1 Rollover/Overturn',
                         '2 Fire/Explosion',
                         '3 Immersion or Partial Immersion',
                         '5 Fell/Jumped from Vehicle',
                         '6 Injured in Vehicle (Non-Collision)',
                         '7 Other Noncollision',
                         '8 Pedestrian',
                         '9 Pedalcyclist',
                         '10 Railway Vehicle',
                         '11 Live Animal',
                         '12 Motor Vehicle In-Transport',
                         '14 Parked Motor Vehicle',
                         '15 Non-Motorist on Personal Conveyance',
                         '16 Thrown or Falling Object',
                         '17 Boulder',
                         '18 Other Object Not Fixed',
                         '19 Building',
                         '20 Impact Attenuator/Crash Cushion',
                         '21 Bridge Pier or Support',
                         '23 Bridge Rail (Includes Parapet)',
                         '24 Guardrail Face',
                         '25 Concrete Traffic Barrier',
                         '26 Other Traffic Barrier',
                         '30 Utility Pole/Light Support',
                         '31 Post, Pole or Other Support',
                         '32 Culvert',
                         '33 Curb',
                         '34 Ditch',
                         '35 Embankment',
                         '38 Fence',
                         '39 Wall',
                         '40 Fire Hydrant',
                         '41 Shrubbery',
                         '42 Tree (Standing Only)',
                         '43 Other Fixed Object',
                         '44 Pavement Surface Irregularity (Ruts, Potholes, Grates, etc.)',
                         '45 Working Motor Vehicle',
                         '46 Traffic Signal Support',
                         '48 Snow Bank',
                         '49 Ridden Animal or Animal Drawn Conveyance',
                         '50 Bridge Overhead Structure',
                         '51 Jackknife (Harmful to This Vehicle)',
                         '52 Guardrail End',
                         '53 Mail Box',
                         '54 Motor Vehicle In-Transport Strikes or is Struck by Cargo, Persons',
                         '55 Motor Vehicle in Motion Outside the Trafficway',
                         '58 Ground',
                         '59 Traffic Sign Support',
                         '72 Cargo/Equipment Loss or Shift (Harmful to This Vehicle)',
                         '73 Object That Had Fallen From Motor Vehicle In-Transport',
                         '74 Road Vehicle on Rails',
                         '91 Unknown Object Not Fixed',
                         '93 Unknown Fixed Object']
        list_len = len(EVENT_IM_LIST)
        EVENT_IM = EVENT_IM_LIST.index(request.form['event_im'])
        EVENT1_IM_1, EVENT1_IM_2, EVENT1_IM_3, EVENT1_IM_5, EVENT1_IM_6, EVENT1_IM_7, EVENT1_IM_8, EVENT1_IM_9, EVENT1_IM_10, EVENT1_IM_11, EVENT1_IM_12, EVENT1_IM_14, EVENT1_IM_15, EVENT1_IM_16, EVENT1_IM_17, EVENT1_IM_18, EVENT1_IM_19, EVENT1_IM_20, EVENT1_IM_21, EVENT1_IM_23, EVENT1_IM_24, EVENT1_IM_25, EVENT1_IM_26, EVENT1_IM_30, EVENT1_IM_31, EVENT1_IM_32, EVENT1_IM_33, EVENT1_IM_34, EVENT1_IM_35, EVENT1_IM_38, EVENT1_IM_39, EVENT1_IM_40, EVENT1_IM_41, EVENT1_IM_42, EVENT1_IM_43, EVENT1_IM_44, EVENT1_IM_45, EVENT1_IM_46, EVENT1_IM_48, EVENT1_IM_49, EVENT1_IM_50, EVENT1_IM_51, EVENT1_IM_52, EVENT1_IM_53, EVENT1_IM_54, EVENT1_IM_55, EVENT1_IM_58, EVENT1_IM_59, EVENT1_IM_72, EVENT1_IM_73, EVENT1_IM_74, EVENT1_IM_91, EVENT1_IM_93 = onehotCategorical(
            EVENT_IM, list_len)

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
        REL_ROAD = TYPE_INT_LIST.index(request.form['rel_road'])
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

        REGION_LIST = ['1 Northeast (PA, NJ, NY, NH, VT, RI, MA, ME, CT)',
                       '2 Midwest (OH, IN, IL, MI, WI, MN, ND, SD, NE, IA, MO, KS)',
                       '3 South (MD, DE, DC, WV, VA, KY, TN, NC, SC, GA, FL, AL, MS, LA, AR, OK, TX)',
                       '4 West (MT, ID, WA, OR, CA, NV, NM, AZ, UT, CO, WY, AK, HI)']
        REGION = REGION_LIST.index(request.form['region'])
        list_len = len(REGION_LIST)
        REGION_1, REGION_2, REGION_3, REGION_4 = onehotCategorical(REGION, list_len)


        URBANICITY_LIST = ['1 Urban', '2 Rural']
        URBANICITY = URBANICITY_LIST.index(request.form['urbancity'])
        list_len = len(URBANICITY_LIST)
        URBANICITY_1, URBANICITY_2 = onehotCategorical(URBANICITY, list_len)


        # manually specify competition distance
        comp_dist = 5458.1

        # build 1 observation for prediction
        entered_li = [MONTH, WKDY_IM, CF_3, CF_5, CF_7, CF_13, CF_14, CF_15, CF_16, CF_17, CF_19, CF_20, CF_21, CF_23,
                      CF_24, CF_25, CF_26, CF_27, CF_28, LGTCON_IM_1, LGTCON_IM_2, LGTCON_IM_3, LGTCON_IM_4,
                      LGTCON_IM_5, LGTCON_IM_6, LGTCON_IM_7, EVENT1_IM_1, EVENT1_IM_2, EVENT1_IM_3, EVENT1_IM_5,
                      EVENT1_IM_6, EVENT1_IM_7, EVENT1_IM_8, EVENT1_IM_9, EVENT1_IM_10, EVENT1_IM_11, EVENT1_IM_12,
                      EVENT1_IM_14, EVENT1_IM_15, EVENT1_IM_16, EVENT1_IM_17, EVENT1_IM_18, EVENT1_IM_19, EVENT1_IM_20,
                      EVENT1_IM_21, EVENT1_IM_23, EVENT1_IM_24, EVENT1_IM_25, EVENT1_IM_26, EVENT1_IM_30, EVENT1_IM_31,
                      EVENT1_IM_32, EVENT1_IM_33, EVENT1_IM_34, EVENT1_IM_35, EVENT1_IM_38, EVENT1_IM_39, EVENT1_IM_40,
                      EVENT1_IM_41, EVENT1_IM_42, EVENT1_IM_43, EVENT1_IM_44, EVENT1_IM_45, EVENT1_IM_46, EVENT1_IM_48,
                      EVENT1_IM_49, EVENT1_IM_50, EVENT1_IM_51, EVENT1_IM_52, EVENT1_IM_53, EVENT1_IM_54, EVENT1_IM_55,
                      EVENT1_IM_58, EVENT1_IM_59, EVENT1_IM_72, EVENT1_IM_73, EVENT1_IM_74, EVENT1_IM_91, EVENT1_IM_93,
                      TYP_INT_1, TYP_INT_2, TYP_INT_3, TYP_INT_4, TYP_INT_5, TYP_INT_6, TYP_INT_7, TYP_INT_10,
                      REL_ROAD_1, REL_ROAD_2, REL_ROAD_3, REL_ROAD_4, REL_ROAD_5, REL_ROAD_6, REL_ROAD_7, REL_ROAD_8,
                      REL_ROAD_10, REL_ROAD_11, WRK_ZONE_0, WRK_ZONE_1, WRK_ZONE_2, WRK_ZONE_3, WEATHR_IM_1,
                      WEATHR_IM_2, WEATHR_IM_3, WEATHR_IM_4, WEATHR_IM_5, WEATHR_IM_6, WEATHR_IM_7, WEATHR_IM_8,
                      WEATHR_IM_10, WEATHR_IM_11, WEATHR_IM_12, ALCHL_IM_1, ALCHL_IM_2, REGION_1, REGION_2, REGION_3,
                      REGION_4, URBANICITY_1, URBANICITY_2, OLD_CAR]
        # entered_li = [StoreType_a, StoreType_b, StoreType_c, StoreType_d, Assortment_a, Assortment_b, Assortment_c, StateHoliday_0, StateHoliday_a, StateHoliday_b, StateHoliday_c, comp_dist, Promo2, DayOfWeek,Month,SchoolHoliday]
        print(entered_li)

        # make prediction

        # CORRECT
        # prediction = model.predict(np.array(entered_li).reshape(1, -1))
        # label = str(np.squeeze(prediction.round(2)))

        # EXPEDITIOUS
        prediction = 5992.9999
        label = str(prediction)

        return render_template('index.html', label=label)


if __name__ == "__main__":
    application.run(load_dotenv=True, use_reloader=True)

    # model = joblib.load('rm.pkl')
# run the app.


# Ref
# https://www.likeanswer.com/question/330646
