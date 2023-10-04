import json

data_file="./data/data_fold/data_0/dailydialog_train.json"
f = open(data_file)
data = json.load(f)
f.close()

for doc_id,content in data.items():
    content=content[0]
    print(doc_id,content)
    for turn_data in content:
        # print(turn_data)
        if 'expanded emotion cause evidence' in turn_data:
            print("****************")
            print(turn_data)
        # break
    print("\n")
        