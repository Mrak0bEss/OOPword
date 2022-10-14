from model import Model, DecoderType
from main import *


decoder_mapping = {'bestpath': DecoderType.BestPath,
                   'beamsearch': DecoderType.BeamSearch,
                   'wordbeamsearch': DecoderType.WordBeamSearch}
decoder_type = decoder_mapping['bestpath']

model = Model(char_list_from_file(), decoder_type, must_restore=True, dump='store_true')


def infer_lol_my(img):
    print(img)
    return infer(model, img)
    
import gradio as gr

print('taktaktak')

int1 = gr.Interface(fn=infer_lol_my, inputs=gr.Image(image_mode='L', source='canvas', shape=(256, 256), invert_colors=False), outputs="label")
int2 = gr.Interface(fn=infer_lol_my, inputs=gr.Image(image_mode='L', source='upload', tool="editor", type="numpy", invert_colors=False), outputs="label")
gr.TabbedInterface([int1, int2], ["draw", "upload"]).launch(server_name="0.0.0.0", server_port=8888)
