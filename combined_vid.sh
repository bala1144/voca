#!/bin/sh

single_quote="'"
double_quote="\""

#  Voca combine video val

out_folder=voca_diff_method_comp
rm -r  $out_folder
mkdir $out_folder
echo "creating folder ${out_folder}"

m1=pretrained_model
m2=training/run_unprocessed_audio/gstep_52280.model
m3=training/run_unprocessed_audio/gstep_130700.model
m4=training/run1/gstep_52280.model
m5=training/run1/gstep_130700.model

text="drawtext=fontfile=/usr/share/fonts/fonts-go/Go-Regular.ttf:text=${single_quote}Pretrained-model${single_quote}:fontcolor=red:fontsize=40:boxborderw=5:x=30:y=60"
text2="drawtext=fontfile=/usr/share/fonts/fonts-go/Go-Regular.ttf:text=${single_quote}Ds-model-best${single_quote}:fontcolor=red:fontsize=40:boxborderw=5:x=830:y=60"
text3="drawtext=fontfile=/usr/share/fonts/fonts-go/Go-Regular.ttf:text=${single_quote}Ds-model-last${single_quote}:fontcolor=red:fontsize=40:boxborderw=5:x=1630:y=60"
text4="drawtext=fontfile=/usr/share/fonts/fonts-go/Go-Regular.ttf:text=${single_quote}Preprocessed-Ds-model-best${single_quote}:fontcolor=red:fontsize=40:boxborderw=5:x=2430:y=60"
text5="drawtext=fontfile=/usr/share/fonts/fonts-go/Go-Regular.ttf:text=${single_quote}Preprocessed-Ds-model-last${single_quote}:fontcolor=red:fontsize=40:boxborderw=5:x=3230:y=60"

vid_dir=videos
vid_files=$(ls $m1/$vid_dir | cut -d"." -f1)
echo "all videos files" $vid_files

for vid in $vid_files; do
    echo $vid
    ls $m1/$vid_dir/$vid.mp4
    ls $m2/$vid_dir/$vid.mp4
    ls $m5/$vid_dir/$vid.mp4

    
    # # combine videos
    ffmpeg -y -hwaccel cuda -i $m1/$vid_dir/$vid.mp4 -i $m2/$vid_dir/$vid.mp4 -i $m3/$vid_dir/$vid.mp4 -i $m4/$vid_dir/$vid.mp4  -i $m5/$vid_dir/$vid.mp4 -filter_complex hstack=inputs=5 $out_folder/${vid}_combine.mp4
    # add text and create a new vide
    ffmpeg -y -hwaccel cuda -i $out_folder/${vid}_combine.mp4 -vf $text,$text2,$text3,$text4,$text5 -codec:a copy $out_folder/${vid}_combine_wtxt.mp4  
    # rm the old video
    rm $out_folder/${vid}_combine.mp4

    echo "\n"
done