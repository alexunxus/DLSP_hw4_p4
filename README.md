# DLSP_hw4_p4
DLSP homework 4 problem 4

# Usage
Basically you only have to run the shell script named by your last name on the corresponding machine and the data log will be collected in `logs/`.

```
$ chmod +x run_$YOURNAME$.sh

$ ./run_$YOURNAME$.sh
# training log is recorded in ./logs/

$ git add --all

$ git push -u origin master
```

# Collaboration
| ResNet | 18   | 20   | 32   | 44   | 56   |
|--------|------|------|------|------|------|
| K80    | Alex | Alex | Alex | Alex | Alex |
| P100   | Kuo  | Kuo  | Kuo  | Lin  | Lin  |
| V100   | Josh | Josh | Josh | Wu   | Wu   |

Basically Kuo and Lin have to use P100 to train the model; Josh and Wu have to use V100. I suggest using VertexAI workbench provided by GCP. The google cloud setup is provided in `Google Cloud Setup.pdf`

# Keep your machine running even if your computer is disconnected

The VertexAI benchmark already provides `screen`. You can first create a screen by 
```
$ screen -S NAME
```
. Afterward, running the bashscript and click `Command+A+D` simultaneously to exit this background program. If you want to step back to the background program, you can simply type
```
$ screen -r NAME
```
. And you can check out if your background program run smoothly or not.

# Monitor the training progress

You can simply check `logs/history___.json` to see how many epoch is finished and the corresponding training loss and accuracy. If you want to add validation set to the training pipeline, you can simply add a validation data loader of CIFAR10 and pass it with argument `valid_loader=valid_loader` to the `trainer` class.

# Throughput(images/sec)
| ResNet | 18       | 20       | 32       | 44        | 56       |
|--------|------    |------    |------    |------     |------    |
| K80    | 2051.968 | 2243.456 | 1430.912 | 1030.912  | 815.104  |
| P100   | 6251.520 | 6480.512 | 4490.240 | 3759.488  | 2571.648 |
| V100   | 9613.824 | 9321.600 | 6704.128 | 4259.712  | 4136.832 |
