from flask import Flask, request

app = Flask(__name__)

@app.route('/input_image', methods=['POST'])
def receive_upload():
    file = request.files['file']
    string_param = request.form['userid']
    
    save_path = f'./input_image/{string_param}.{file.filename.split(".")[-1]}'
    file.save(save_path)
    
    return 'Success', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20)
