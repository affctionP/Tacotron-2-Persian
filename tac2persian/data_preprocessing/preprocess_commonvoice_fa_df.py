
import argparse
import os
import shutil
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tac2persian.utils.generic import load_config
from tac2persian.utils.g2p.g2p import Grapheme2Phoneme
from tac2persian.utils.audio import log_melspectrogram, trim_silence
import pandas as pd

def normalize_text(text):
    if text[-1] not in [".", "!", "?"]:
        text = text + "."
    return text


def compute_features(source_audio_path, 
                     file_name, 
                     text, 
                     speaker_name, 
                     out_melspecs_path, 
                     g2p,
                     itr,
                     count_files):
    print(f"Processing file {itr}/{count_files}")
    try:
        text = normalize_text(text)

        out_mel_path = os.path.join(out_melspecs_path, file_name + ".npy")
 
        phoneme = g2p.text_to_phone(text, language="fa")

        phoneme_idx = g2p.phone_to_sequence(phoneme)
        phoneme_idx = ','.join(map(str, phoneme_idx))
        audio, _ = librosa.core.load(source_audio_path+'.wav', sr=config["mel_params"]["sample_rate"])
        audio = trim_silence(audio, config["ref_level_db"])
        melspec = log_melspectrogram(audio, **config["mel_params"])
        np.save(out_mel_path, melspec)
        meta_line = f"{file_name}|{speaker_name}|{text}|{phoneme}|{melspec.shape[1]}|{phoneme_idx}"
        # print(f"line - {meta_line}")
        return meta_line

    except Exception as e:
        import traceback
        print(f"Error in processing {file_name} error {e}")
        traceback.print_exc()
        return None

def preprocess(dataset_path, output_path, config, num_workers,meta_path):
    r"""Preprocesses audio files in the dataset."""
    
    # Load G2P module
    g2p = Grapheme2Phoneme()

    executor = ProcessPoolExecutor(max_workers=num_workers)
    
    # Create metafile and copy files
    metafile = []


    df=pd.read_csv(meta_path)
    count_files=len(df)
    speaker_name = "speaker_fa_atefeh"
    out_melspecs_path = os.path.join(output_path, "melspecs")
    os.makedirs(out_melspecs_path, exist_ok=True)
    for index, row in df.iterrows():
            file_name=str(row['filename'])
            text=row['text']
            file_name_ = speaker_name + "_" + file_name
            source_audio_path = os.path.join(dataset_path, "wavs/wavs/", file_name)
            meta_line = executor.submit(partial(compute_features, 
                                                source_audio_path, 
                                                file_name_, 
                                                text, 
                                                speaker_name, 
                                                out_melspecs_path, 
                                                g2p,
                                                index,
                                                count_files))


            metafile.append(meta_line)

    metafile = [metaline.result() for metaline in metafile if metaline is not None]
    print(metafile)
    
    # Write metafile
    with open(os.path.join(output_path, "metadata.txt"), "w") as final_meta:
        for l in metafile:
            if l != None :
                final_meta.write(l + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=5)
    args = parser.parse_args()
    # target_speakers = ["b8e506d7d9b5f2a9cfb4e08a088819f7f54bfd8c3e0ab86d80f07b48fb50981da2264ed13ff1a77fefbaccc6a014a530bace3d0ac32e2c993ebf0ed0dd1712a8"]

    config = load_config(os.path.join(args.config_path, "config.yml"))
    preprocess(args.dataset_path, args.output_path, config, args.num_workers)