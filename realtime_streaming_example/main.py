from flask import Flask, Response, request
import struct, time
from orpheus_tts import OrpheusModel
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# engine = OrpheusModel(model_name="canopylabs/3b-es_it-pretrain-research_release")
engine = OrpheusModel(model_name="canopylabs/3b-es_it-ft-research_release")

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt', 'Hola, parece que olvidaste incluir un texto')
    voice = request.args.get('voice', 'maria')
    temperature = float(request.args.get('temperature', 0.6))
    top_p = float(request.args.get('top_p', 0.9))
    
    def generate_audio_stream():
        yield create_wav_header()

        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice=voice,
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=temperature,
            top_p=top_p
        )
        for chunk in syn_tokens:
            yield chunk

    return Response(generate_audio_stream(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
