import subprocess
import os

for i in range(11, 2000):
    cmd=str.format('python main.py --mode=train --reloadid=%d'%(i))
    ret=os.system(cmd)
    if ret != 0:
        break


