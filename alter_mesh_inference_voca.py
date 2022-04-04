'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''


import os
import cv2
import scipy
import tempfile
import numpy as np
import tensorflow as tf
from subprocess import call
from scipy.io import wavfile

from psbody.mesh import Mesh
from utils.audio_handler import  AudioHandler
from utils.rendering import render_mesh_helper
import argparse
from tqdm import tqdm
from util_flame_render import Facerender
import pickle

def process_audio(ds_path, audio, sample_rate):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1

    tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    audio_handler = AudioHandler(config)
    return audio_handler.process(tmp_audio)['subj']['seq']['audio']

def output_sequence_meshes(sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    mesh_out_path = os.path.join(out_path, 'meshes')
    if not os.path.exists(mesh_out_path):
        os.makedirs(mesh_out_path)

    if os.path.exists(uv_template_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
    else:
        vt, ft = None, None

    num_frames = sequence_vertices.shape[0]
    for i_frame in tqdm(range(num_frames), desc="Writing meshes"):
        out_fname = os.path.join(mesh_out_path, '%05d.obj' % i_frame)
        out_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            out_mesh.vt, out_mesh.ft = vt, ft
        if os.path.exists(texture_img_fname):
            out_mesh.set_texture_image(texture_img_fname)
        out_mesh.write_obj(out_fname)

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, 
                        uv_template_fname='', texture_img_fname='', 
                        out_file_name="video"):

    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (512, 512), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (512, 512), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    for i_frame in tqdm(range(num_frames), desc="rendering frames"):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
            
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)

        writer.write(img)
    writer.release()

    video_fname = os.path.join(out_path, out_file_name+'.mp4')
    cmd = ('ffmpeg' + ' -y -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)
    
    # remove_command = ('rm %s'%tmp_video_file)
    print("seq completed")
    print()

class Alter_shape_identity():

    def __init__(self, all_user_templates,
                 default_template_fname) -> None:

        # Load previously saved meta graph in the default graph
        self.face_render = Facerender()
        self.templates_data = pickle.load(open(all_user_templates, 'rb'), encoding='latin1')
        self.template_conditional_subjects = list(self.templates_data.keys())

        #
        self.template = Mesh(filename=default_template_fname)


    def inference(self,
                    subj_name,
                    audio_fname,
                    meshes_for_sequence,
                    out_path,
                    out_file_name,
                    uv_template_fname='',
                    texture_img_fname=''):
        """
        Function is used to remove the default template from the result meshes without shape params
        and add the used specfic template on all the meshes in the seqUence and render them with condition

        :param audio_fname:
        :param template_fname:
        :param out_path:
        :param out_file_name:
        :param uv_template_fname:
        :param texture_img_fname:
        :return:
        """

        org_predicted_vertices = np.stack(meshes_for_sequence, axis=0)
        offsets = org_predicted_vertices - np.expand_dims(self.template.v, axis=0)

        current_template = np.expand_dims(self.templates_data[subj_name], axis=0)
        # current_template = np.expand_dims(self.template.v, axis=0)

        current_template = np.repeat(current_template, org_predicted_vertices.shape[0], axis=0)
        predictd_vertices_w_template = current_template + offsets

        # offsets = org_predicted_vertices - self.template.v
        # predictd_vertices_w_template = self.templates_data[subj_name] + offsets

        # remove the old template
        out_mesh_path = os.path.join(out_path, "meshes_w_identity")
        os.makedirs(out_mesh_path, exist_ok=True)

        out_vid_path = os.path.join(out_path, "videos_w_identity")
        os.makedirs(out_vid_path, exist_ok=True)


        # output_sequence_meshes(predictd_vertices_w_template, self.template, out_mesh_path)
        self.render_sequence_meshes(audio_fname, predictd_vertices_w_template, self.template, out_vid_path, uv_template_fname, texture_img_fname, out_file_name)

    def render_sequence_meshes(self, audio_fname, sequence_vertices, template, out_path, 
                        uv_template_fname='', texture_img_fname='', 
                        out_file_name="video"):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
        if int(cv2.__version__[0]) < 3:
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (512, 512), True)
        else:
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (512, 512), True)

        if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
            uv_template = Mesh(filename=uv_template_fname)
            vt, ft = uv_template.vt, uv_template.ft
            tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
        else:
            vt, ft = None, None
            tex_img = None

        num_frames = sequence_vertices.shape[0]
        center = np.mean(sequence_vertices[0], axis=0)
        for i_frame in tqdm(range(num_frames), desc="rendering frames"):
            render_mesh = Mesh(sequence_vertices[i_frame], template.f)
            if vt is not None and ft is not None:
                render_mesh.vt, render_mesh.ft = vt, ft
                
            # img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
            self.face_render.add_face(render_mesh.v, render_mesh.f)
            img = self.face_render.render()

            writer.write(img)
        writer.release()

        video_fname = os.path.join(out_path, out_file_name+'.mp4')
        cmd = ('ffmpeg' + ' -y -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            audio_fname, tmp_video_file.name, video_fname)).split()
        call(cmd)
        
        print("seq completed")
        print()

def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice operated character animation')
    parser.add_argument('--model_path', default='./pretrained_model', help='Path to store')

    parser.add_argument('--template_fname', default='./template/FLAME_sample.ply', help='Path of "zero pose" template mesh in" FLAME topology to be animated')
    parser.add_argument('--all_user_templates_path', default='./projects/dataset/voca/templates.pkl', help='Path of "zero pose" template mesh in" FLAME topology to be animated')

    parser.add_argument('--uv_template_fname', default='', help='Path of a FLAME template with UV coordinates')
    parser.add_argument('--texture_img_fname', default='', help='Path of the texture image')

    args = parser.parse_args()
    model_path = args.model_path

    uv_template_fname = args.uv_template_fname
    texture_img_fname = args.texture_img_fname

    print("\n\n\n")
    print("model_path", model_path)
    print("all configs", args)
    print("\n\n\n")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load all the files in the audio
    audio_root_path = os.path.join(os.getenv("HOME"), "projects/dataset/voca/audio/")

    template_fname = args.template_fname
    all_user_templates_path = os.path.join(os.getenv("HOME"), args.all_user_templates_path)
    shape_alter = Alter_shape_identity(all_user_templates_path,
                                       template_fname)

    input_meshes_paths = os.path.join(model_path, "meshes")

    a = os.listdir(input_meshes_paths)

    for seq_dir in os.listdir(input_meshes_paths):

        print("Running on sequenence", seq_dir)

        # load the all the files in the seq dir
        all_files = os.listdir(os.path.join(input_meshes_paths, seq_dir, "meshes"))
        all_files = sorted(all_files)
        # load all the obj vertices in the seq dir
        all_vertices = [Mesh(filename=os.path.join(input_meshes_paths, seq_dir, "meshes", file)).v for file in all_files]

        b = seq_dir.split("_")
        subj = "_".join(b[:4])
        sen = "_".join(b[4:5])
        cond = int(b[-1])
        current_condition = shape_alter.template_conditional_subjects[cond]

        # create the referencing files
        out_file_name = subj + "_" + sen + "_" + current_condition
        audio_fname = os.path.join(audio_root_path, subj, sen+".wav")

        #
        shape_alter.inference( subj,
                    audio_fname,
                    all_vertices,
                    model_path,
                    out_file_name)