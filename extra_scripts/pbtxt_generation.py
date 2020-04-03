# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:52:54 2020

@author: Tim
"""

from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format


def convert_classes(classes, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text


if __name__ == '__main__':
    txt = convert_classes(['Red ','Yellow ','Green '])
    print(txt)
    with open('label_map.pbtxt', 'w') as f:
        f.write(txt)
