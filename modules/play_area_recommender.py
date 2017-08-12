import sys, os
directory = os.path.dirname(__file__)
sys.path.append(directory + '/uzuuzuindex_nn')

from final_regression_nn import uzu_uzu_index_predictor as uui_nn


"""
latest weather data (ja) = http://www.data.jma.go.jp/obd/stats/data/mdrr/docs/csv_dl_readme.html
historical weather data (ja) = http://www.data.jma.go.jp/gmd/risk/obsdl/index.php
resas api = https://opendata.resas-portal.go.jp/docs/api/v1/index.html

"""

def generate_places():
    """
    because all the places are on a positive x y axis, distance is calculable as
    (x1 - x) * (y1 - y) = euclidean distance where x and y is the child, and x1 and y1 is the place
    :return:
    """
    places = {}
    places['park_b'] = [0.8,0.7]
    places['park_a'] = [0.6,1.0]
    places['leisure_facility'] = [0.7,0.3]
    places['museum'] = [0.3,0.3]
    places['library'] = [0.2,0.2]
    places['zoo'] = [0.4,0.7]

    return places

def play_recommendation(uui, outdoor_preference, weather_forecast):
    # initial hard coded places
    # todo: train places on the uzu uzu index graph by feedback loop from child
    # e.g. 'did you like it?' 'no' 'why?' 'weather was bad' > detect keyword weather, update preference

    places = generate_places()
    # key[0] = uui
    # kay[1] = outdoor pref
    # larger than uui
    # smaller than outdoor pref
    #print(places)

    # reduce outdoor preference by half in the event of rain (weather < 0.5)
    if weather_forecast > 0.5:
        outdoor_preference = outdoor_preference / 2.0

    recommendations = []
    place_vectors = []
    distances = []
    child_vector = [uui, outdoor_preference]
    for key, value in places.items():
        if value[0] >= uui and value[1] <= outdoor_preference:
                recommendations.append(key)
                place_vectors.append(value)

    for p_vector in place_vectors:
        print(p_vector)
        print(child_vector)
        calc1 = abs(p_vector[0] - child_vector[0])
        calc2 = abs(p_vector[1] - child_vector[1])
        if calc1 != 0.0 and calc2 != 0.0:
            distances.append(calc1*calc2)
        else:
            distances.append(calc1+calc2)
    #print('I recommend: ', recommendations)

    # sort recommendations by the child's uzu_uzu index as well as outdppr preference
    distances, recommendations = (list(x) for x in zip(*sorted(zip(distances, recommendations), key=lambda pair: pair[0])))

    return recommendations

def help():
    # references
    col_labels = ['day of the week',
                  'cloud_cover',
                  'dew_point',
                  'rain_mm',
                  'rh',
                  'snow_cover',
                  'sunlight_hrs',
                  'temp_celsius',
                  'visibility',
                  'weather_rating',
                  'forest area coverage',
                  'LowPressure']

    print('input list must be in the following format: ')
    print(col_labels)


if __name__ == '__main__':
    print('running sample to show how it works')

    # print reference dataset headers
    help()
    # MAIN
    # sample child
    test_input = ['Wed', 5, 20.8, 0.3, 56, 0, 1.96, 30.6, 10, 2, 0.087407407, 0.355]
    outdoor_pref = 0.7
    forecast_weather = test_input[3]
    # currently weather is rain_mm so it is not normalized to 0-1. rain of more than 0.5mm will screw up the recommendation


    uui = uui_nn(new_input=test_input, load_saved_model=True)
    uui = uui[0]

    # output
    responses = play_recommendation(uui, outdoor_pref, forecast_weather)

    print('uui: ', uui)
    print('recommendations, sorted by distance :', responses)