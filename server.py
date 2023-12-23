from flask import Flask, jsonify, request, render_template;
from flask_cors import CORS, cross_origin
from transformers import pipeline


app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['GET', 'POST'])
def home():
    return "NLP Tasks"


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

question_answering_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


@app.route('/summary_generate/<data>', methods=['GET', 'POST'])
@cross_origin()
def summary_generate(data):
    output = summarizer(data, max_length=200, min_length=30, do_sample=False);
    print(output);
    return(jsonify(output))


@app.route('/summary_generate_csv/<values>', methods=['GET', 'POST'])
@cross_origin()
def summary_generate_csv(values):
    output = summarizer(values, max_length=200, min_length=30, do_sample=False);
    print(output);
    return(jsonify(output))


@app.route('/answer_the_question/', methods=['GET', 'POST'])
@cross_origin()
def answer_the_question():
    question = request.args.get('question')
    context = request.args.get('context')
    print(question)

    # contextArr = "";

    # for text in values:
    #     contextArr = contextArr + text;

    answer = question_answering_pipeline(question=question, context=context)
    print(answer)
    return jsonify(answer)




if __name__ == "__main__":
    app.run(debug=True)
