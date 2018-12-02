import vggish_slim
import vggish_params
import vggish_input

def EmbeddingsFromVGGish(vgg, x, sr):
    '''Run the VGGish model, starting with a sound (x) at sample rate
    (sr). Return a dictionary of embeddings from the different layers
    of the model.'''
    # Produce a batch of log mel spectrogram examples.
    input_batch = vggish_input.waveform_to_examples(x, sr)
    # print('Log Mel Spectrogram example: ', input_batch[0])

    layer_names = vgg['layers'].keys()
    tensors = [vgg['layers'][k] for k in layer_names]

    results = sess.run(tensors,
                       feed_dict={vgg['features']: input_batch})

    resdict = {}
    for i, k in enumerate(layer_names):
        resdict[k] = results[i]

    return resdict