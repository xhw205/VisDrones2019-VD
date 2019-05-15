#!/usr/bin/env bash
hdfs dfs -get $PAI_DATA_DIR
tar -I pigz -xf VID-train.tar
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/cc/
pip3 install tqdm
cd core/external
make
cd ..
cd models/py_utils/_cpools
python3 setup.py install --user
cd ../../../../
python3 train.py CornerNet_Saccade
tar -cf ./cc.tar ./cache/nnet/CornerNet_Saccade/
hdfs dfs -put -f cc.tar $PAI_DEFAULT_FS_URI/data/models/$PAI_USER_NAME/cc/