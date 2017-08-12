print('please wait while python loads all modules...')
from http.server import BaseHTTPRequestHandler
import json, sys
import urllib.parse as urlparse
sys.path.append('./modules')
# the above trick only works for the current file. for module files, use
# os.path.dirname(__file__) and append that directory path to the folder that you're pointing to
import play_area_recommender

def parse_queries(query_string):
    inputs = [x for x in query_string.split('&')]
    print('number of inputs found:', len(inputs))
    checker = 0
    for data in inputs:
        print('checking: ', data)
        if 'uui' in data:
            temp = data.split('=')[-1].split(',')
            if len(temp) == 12:
                uui_result = play_area_recommender.uui_nn(new_input=temp, load_saved_model=True)[0]
                forecast_weather = temp[3]
                checker += 2

            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(bytes('number of features must be 12.', 'utf-8'))

        if 'outdoor' in data:
            outdoor_pref = data.split('=')[-1]
            checker += 1


    if checker == 3:
        print(uui_result, outdoor_pref, forecast_weather)
        response = play_area_recommender.play_recommendation(float(uui_result),
                                                             float(outdoor_pref),
                                                             float(forecast_weather))
        print('response from recommender', response)
        return response
    else:
        return None

class GetHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_path = urlparse.urlparse(self.path)

        # for debugging, just throw the word 'debug' somewhere in the url
        if 'debug' in self.path:
            message = '\n'.join([
                'CLIENT VALUES:',
                'client_address=%s (%s)' % (self.client_address,
                    self.address_string()),
                'command=%s' % self.command,
                'path=%s' % self.path,
                'real path=%s' % parsed_path.path,
                'query=%s' % parsed_path.query,
                'request_version=%s' % self.request_version,
                '',
                'SERVER VALUES:',
                'server_version=%s' % self.server_version,
                'sys_version=%s' % self.sys_version,
                'protocol_version=%s' % self.protocol_version,
                '',
                ])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(message, 'utf-8'))

        recommendations = parse_queries(parsed_path.query)

        if recommendations is not None:
            json_out = json.dumps({'recommendations': recommendations})
            self.send_response(200)
            self.end_headers()
            #self.wfile.write(bytes(','.join(recommendations), 'utf-8'))
            self.wfile.write(bytes(json_out, 'utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes('an error has occured, recommendations is not passing anything', 'utf-8'))

        return

    # no sorry no post request implemented yet
    def do_POST(self):
        content_len = int(self.headers.getheader('content-length'))
        post_body = self.rfile.read(content_len)
        self.send_response(200)
        self.end_headers()

        data = json.loads(post_body)

        self.wfile.write(data['foo'])
        return

if __name__ == '__main__':
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 8080), GetHandler)
    print('Starting server at http://localhost:8080')
    server.serve_forever()