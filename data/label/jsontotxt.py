import os
import json

data_dir = 'prejson/'
all_json = os.listdir(data_dir)

with open("trainq.txt", "w") as f:
    for j_name in all_json:
        f.write(j_name + '\n')
        j = open(data_dir + j_name, encoding='utf-8')
        info = json.load(j)
        print(len(info['shapes']))
        num = len(info['shapes'])
        length = int(num/6)
        for i in range(length):
            x1 = info['shapes'][5+i*6]['points'][0][0]
            y1 = info['shapes'][5+i*6]['points'][0][1]
            x2 = info['shapes'][5+i*6]['points'][1][0]
            y2 = info['shapes'][5+i*6]['points'][1][1]
            w = str(round(x2 - x1, 2))
            h = str(round(y2 - y1, 2))
            x1 = str(round(x1, 2))
            y1 = str(round(y1, 2))
            d1x = str(round(info['shapes'][0+i*6]['points'][0][0], 2))
            d1y = str(round(info['shapes'][0+i*6]['points'][0][1], 2))
            d2x = str(round(info['shapes'][1+i*6]['points'][0][0], 2))
            d2y = str(round(info['shapes'][1+i*6]['points'][0][1], 2))
            d3x = str(round(info['shapes'][2+i*6]['points'][0][0], 2))
            d3y = str(round(info['shapes'][2+i*6]['points'][0][1], 2))
            d4x = str(round(info['shapes'][3+i*6]['points'][0][0], 2))
            d4y = str(round(info['shapes'][3+i*6]['points'][0][1], 2))
            d5x = str(round(info['shapes'][4+i*6]['points'][0][0], 2))
            d5y = str(round(info['shapes'][4+i*6]['points'][0][1], 2))
            label = x1 + ' ' + y1 + ' ' + w + ' ' + h + ' ' + d1x + ' ' + d1y + ' ' + '0.0' + ' ' + d2x + ' ' + d2y + ' ' + '0.0' + ' ' + d3x + ' ' + d3y + ' ' + '0.0' + ' ' + d4x + ' ' + d4y + ' ' + '0.0' + ' ' + d5x + ' ' + d5y + ' ' + '0.0' + ' ' + '1'

            f.write(label + '\n')

        # x1 = info['shapes'][5]['points'][0][0]
        # y1 = info['shapes'][5]['points'][0][1]
        # x2 = info['shapes'][5]['points'][1][0]
        # y2 = info['shapes'][5]['points'][1][1]
        # w = str(round(x2 - x1, 2))
        # h = str(round(y2 - y1, 2))
        # x1 = str(round(x1, 2))
        # y1 = str(round(y1, 2))
        # d1x = str(round(info['shapes'][0]['points'][0][0], 2))
        # d1y = str(round(info['shapes'][0]['points'][0][1], 2))
        # d2x = str(round(info['shapes'][1]['points'][0][0], 2))
        # d2y = str(round(info['shapes'][1]['points'][0][1], 2))
        # d3x = str(round(info['shapes'][2]['points'][0][0], 2))
        # d3y = str(round(info['shapes'][2]['points'][0][1], 2))
        # d4x = str(round(info['shapes'][3]['points'][0][0], 2))
        # d4y = str(round(info['shapes'][3]['points'][0][1], 2))
        # d5x = str(round(info['shapes'][4]['points'][0][0], 2))
        # d5y = str(round(info['shapes'][4]['points'][0][1], 2))
        # label = x1 + ' ' + y1 + ' ' + w + ' ' + h + ' ' + d1x + ' ' + d1y + ' ' + '0.0' + ' ' + d2x + ' ' + d2y + ' ' + '0.0' + ' ' + d3x + ' ' + d3y + ' ' + '0.0' + ' ' + d4x + ' ' + d4y + ' ' + '0.0' + ' ' + d5x + ' ' + d5y + ' ' + '0.0' + ' ' + '1'
        #
        # f.write(label + '\n')




