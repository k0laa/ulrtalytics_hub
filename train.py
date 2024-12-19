from ultralytics import YOLO, checks, hub

checks()

with open('keys.txt') as f:
    key = f.readline().strip()
    link = f.readline().strip()
    f.close()

hub.login(key)

model = YOLO(link)
results = model.train()