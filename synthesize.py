# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data
import tensorflow as tf
from train import Graph
from utils import spectrogram2wav
from scipy.io.wavfile import write
import os
import numpy as np



frases=["A inauguração da vila é quarta ou quinta-feira",
        "Vote se você tiver o título de eleitor",
        "Hoje é fundamental encontrar a razão da existência humana",
        "A temperatura é mais amena à noite",
        "Em muitas cidades a população está diminuindo.",
        "Nunca se deve ficar em cima do morro",
        "Para as pessoas estranhas o panorama é desolador",
        "É bom te ver colhendo flores menino",
        "Eu finjo me banhar num lago ao amanhecer",
        "Sua sensibilidade mostrará o caminho",
        "A Amazônia é a reserva ecológica do globo",
        "O ministério mudou demais com a eleição",
        "Novas metas surgem na informática",
        "O capital de uma empresa depende de sua produção",
        "Se não fosse ela tudo teria sido melhor",
        "A principal personagem no filme é uma gueixa",
        "Espere seu amigo em casa",
        "A juventude tinha que revolucionar a escola",
        " A cantora terá quatro meses para ensaiar seu canto",
        "Esse tema foi falado no congresso."]

from matplotlib import pylab as plt

def plot_alignment_with_text(alignment,text, info=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation=None)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel,fontsize = 'x-large')
    plt.ylabel('Encoder timestep',fontsize = 'x-large')
    #plt.yticks(range(len(text)), list(text.upper()),fontsize ='xx-large' )
    ax.set_yticks(range(len(text)))
    ax.set_yticklabels(list(text.upper()),fontsize ='medium')
    plt.tight_layout()
    return fig



def synthesize():
    if not os.path.exists(hp.sampledir): os.mkdir(hp.sampledir)

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    # Load data
    texts = load_data(mode="synthesize")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

        # Feed Forward
        ## mel
        y_hat = np.zeros((texts.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
        for j in tqdm.tqdm(range(200)):
            _y_hat,aligs = sess.run([g.y_hat,g.alignments], {g.x: texts, g.y: y_hat})
            y_hat[:, j, :] = _y_hat[:, j, :]

        #plot aligments
        for i,al in enumerate(aligs):
            fig = plot_alignment_with_text(al,frases[i])
            fig.savefig(os.path.join(hp.sampledir,'align_'+str(i+1)+'_val.png'))
        ## mag
        mags = sess.run(g.z_hat, {g.y_hat: y_hat})
        for i, mag in enumerate(mags):
            print("File {}.wav is being generated ...".format(i+1))
            audio = spectrogram2wav(mag)
            write(os.path.join(hp.sampledir, '{}.wav'.format(i+1)), hp.sr, audio)
        
if __name__ == '__main__':
    synthesize()
    print("Done")

