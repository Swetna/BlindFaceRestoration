# ML-Project-Multi-Prior Learning via Neural Architecture Search for Blind Face Restoration
## requirement
GPU required
***
## run
To generate the high quality images from low quality images, run (modify checkpoints and data path in ./config_utils/test_args.py, the option for pretrained model(checkpoints) saved in the folders adam, sgd, no_heat, no_dict, no_parse fold respectively, data path already saved in args as default)
```python
python demo.py
```
To search the architecture for neural network, run (modify data path in ./config_utils/search_args.py, you can use the other dataset based on your local dictionary, also you can use the same path saved in test_args.py, that is the same dataset we used to train our model in experiment)
```python
python search.py
```
To train the model, you just need to run(also modify path as above in ./config_utils/train_args.py ):
```python
python train.py
```
***
## experiment
for experiment 1, modify args for optimizer in ./config_utils/train_args.py "solver", "adam" for Adam optimizer, "sgd" for sigmond optimizer
***
for experiment 2, 
you should choose different prior features that you want to train the model on (parse_feature, dict_feature, heat_feature) in ./models/build_model.py line 175, note: just choose any two out of three
***
To evaluate the result, you should run evaluation.py, and modify the path of output folder and ground trurh folder
(our output for five models are saved in the folders result_adam, result_sgd, result_heat, esult_parse, result_dict respectively)
