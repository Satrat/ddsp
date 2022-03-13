import gin
import librosa
import numpy as np
import tensorflow as tf
import pretty_midi

PHONEMES = ['sil', 'b','d','f','g','h','j','k','l','m','n','p','r','s','t','v','w','z','zh','ch','sh','th','dh','ng','y','ae','ei','e','ii','i','ai','a','ou','u','ao','uu','oi','au','eo','er','oo']
PHONEME2ID={}
for i,p in enumerate(PHONEMES):
  PHONEME2ID[p] = i

def load_midi(filename):
    midi_data = pretty_midi.PrettyMIDI(filename)
    if len(midi_data.instruments) > 1:
        raise AssertionError(f"More than 1 track detected in {filename}")

    return midi_data.instruments[0].notes

def load_text(filename):
    text = open(filename).read()
    text = ' '.join(text.split())

    graph = text.split(' ')
    if graph[-1] == '':
        graph = graph[:-1]

    return graph

@gin.register
def annotate_f0_and_phoneme(audio, sample_rate, midi_filename, phoneme_filename, frame_rate):
  """Parse Fundamental frequency (f0) annotations from midi_filename once per frame_rate,
     Also parse phoneme annotations from phoneme_filename once per frame_rate

  Args:
    audio: Numpy ndarray of single audio (16kHz) example. Shape [audio_length,].
    sample_rate: Sample rate in Hz.
    midi_filename: path to MIDI file containing pitch annotations for audio.
    phoneme_filename: text file of syllables corresponding to notes in midi_filename
    frame_rate: Rate of f0 frames in Hz.

  Returns:
    f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
    phoneme_frames: Index of phonemes. Shape [n_frames,].
  """

  audio_len_sec = audio.shape[-1] / float(sample_rate)
  num_frames = int(audio_len_sec * frame_rate)

  f0_hz = np.zeros(num_frames, dtype=np.float32)
  phoneme_frames = np.zeros(num_frames, dtype=np.int64)
  midi_data = load_midi(midi_filename)
  phoneme_data = load_text(phoneme_filename)
  for i,m in enumerate(midi_data):
    start_frame = int(m.start * frame_rate)
    end_frame = int(m.send * frame_rate)
    f0_hz[start_frame:end_frame] = librosa.midi_to_hz(m.pitch)

    phonemes = phoneme_data[i].split("_")
    num_phonemes = len(phonemes)
    frames_per_phoneme = int((end_frame - start_frame) / num_phonemes)
    phoneme_frames[start_frame:end_frame] = PHONEME2ID[phonemes[-1]] #default to last phoneme to deal with rounding
    for p in range(num_phonemes):
      a = start_frame + (i * frames_per_phoneme)
      b = a + frames_per_phoneme
      phoneme_frames[a:b] = PHONEME2ID[phonemes[p]]
  
  return f0_hz, phoneme_frames