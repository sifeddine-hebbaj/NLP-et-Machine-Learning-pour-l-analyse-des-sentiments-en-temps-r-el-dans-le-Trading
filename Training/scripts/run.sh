#!/bin/bash
set -e
cd /app
echo "<<<<<<<<<<<<<<<<<<<TRAIN>>>>>>>>>>>>>>>>>>>>"
python train.py
echo "<<<<<<<<<<<<<<<<<<<TEST>>>>>>>>>>>>>>>>>>>>"
python test.py
