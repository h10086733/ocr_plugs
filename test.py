import pandas as pd
from paddleocr import PaddleOCR


class AnalysisOcr:
    def __init__(self):

        pass

    def load_img(self, img_path):

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        data = []
        ocr = PaddleOCR(
            use_angle_cls=True,  # 启用方向分类器
            lang='ch',  # 使用简体中文模型
            det_algorithm='DB',  # 文本检测算法选择DB
            rec_algorithm='SVTR_LCNet',  # 文本识别算法选择SVTR_LCNet
            use_space_char=True,
            layout=True,  # 启用布局分析
            table=True,  # 启用表格识别
            det_db_box_thresh=0.6,
            det_db_thresh=0.4,
        )

        result = ocr.ocr(img_path,cls=True)
        print(result)
        for idx in range(len(result)):
            res = result[idx]
        for line in res:
            index = line[0]
            row = {"index_1": index[0][0], "index_2": index[0][1], "index_3": index[1][0], "index_4": index[1][1],
                   "index_5": index[2][0], "index_6": index[2][1], "index_7": index[3][0], "index_8": index[3][1],
                   "key": line[1][0].replace("：", ":").replace("（", "(").replace("）", ")"), "accuracy": line[1][1]}
            data.append(row)
            print(row)
        data = pd.DataFrame(data)
        # TODO 合并指定字段
        data = data.sort_values(by=['index_1', 'index_2'], ascending=[True, True])
        # 行高容错设置
        data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2.5
        data["y_offset_up"] = data["index_2"] + data["row_height"]
        data["y_offset_low"] = data["index_2"] - data["row_height"]
        data["x_offset_up"] = data["index_3"] * 1.01
        data["x_offset_low"] = data["index_7"] * 0.99
        data["filed_length"] = data["index_3"] - data["index_7"]
        data["filed_height"] = data["index_6"] - data["index_4"]
        self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
        self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2

        self.data = data

    def analysis_index(self, key, direction, end_key=None, block=-1):
        """
        解析key 方向的匹配
        :param key:
        :param direction: below / right like
        :return:
        """
        append_block_filter = ""
        if block == 1:
            append_block_filter = f"and  index_2 >= {self.middle_y} and index_3 <= {self.middle_x}"
        if block == 2:
            append_block_filter = f"and index_2 >= {self.middle_y} and index_3 >= {self.middle_x}"
        if block == 3:
            append_block_filter = f"and index_2 <= {self.middle_y} and  index_3 <= {self.middle_x}"
        if block == 4:
            append_block_filter = f"and index_2 <= {self.middle_y} and index_3 >= {self.middle_x}"

        start_in_words = self.data.query(f'key=="{key}"')
        if direction == "like":
            expr = 'key.str.contains(@key, case=False, na=False)'
            expr += append_block_filter
            curr_key = self.data.query(expr, engine='python')
            print(curr_key["key"].tolist())
            return curr_key["key"].tolist()
        if end_key is not None:
            end_in_words = self.data.query('key.str.contains(@end_key, case=False, na=False)', engine='python')
            end_row = end_in_words.iloc[0]
        if not start_in_words.empty:
            first_row = start_in_words.iloc[0]
            if end_key is not None and direction == "right":
                query_str = f'{first_row["y_offset_low"]} < index_2 < {first_row["y_offset_up"]} and index_1 != {first_row["index_1"]} and index_8 <= {end_row["index_8"]}'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)

            elif end_key is None and direction == "right":
                query_str = f'{first_row["y_offset_low"]} < index_2 < {first_row["y_offset_up"]} and index_1 != {first_row["index_1"]} and {first_row["index_3"]}<= index_8 <= {first_row["index_3"] + first_row["filed_length"]}'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)

            elif end_key is None and direction == "below":
                # 左对齐 或右对齐
                filed_height = first_row["filed_height"]
                query_str = f' ({first_row["x_offset_low"]} < index_3 < {first_row["x_offset_up"]} and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height} and index_2 != {first_row["index_2"]}) or ({first_row["x_offset_low"]} < index_7 < {first_row["x_offset_up"]} and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height} and index_2 != {first_row["index_2"]})'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)

            elif end_key is not None and direction == "below":
                # 左对齐 或右对齐
                filed_height = end_row["filed_height"]
                query_str = f' ({first_row["x_offset_low"]} < index_3 < {first_row["x_offset_up"]} and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height} and index_2 != {first_row["index_2"]}) or ({first_row["x_offset_low"]} < index_7 < {first_row["x_offset_up"]} and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height} and index_2 != {first_row["index_2"]})'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)
            print(filter_values_words_value["key"].tolist())
            return filter_values_words_value["key"].tolist()
        else:
            return []


if __name__ == '__main__':
    ao = AnalysisOcr()
    ao.load_img("imgs/1.jpg")
    ao.analysis_index(key="发票号码:", direction="like")
    ao.analysis_index(key="开票日期:", direction="like")
    ao.analysis_index(key="项目名称", direction="below")
    ao.analysis_index(key="规格型号", direction="below")
    ao.analysis_index(key="单位", direction="below")
    ao.analysis_index(key="数量", direction="below")
    ao.analysis_index(key="金额", direction="below")
    ao.analysis_index(key="税额", direction="below")
    ao.analysis_index(key="价税合计(大写)", direction="right", end_key="小写")
    ao.analysis_index(key="名称:", direction="like", block=1)
    ao.analysis_index(key="名称:", direction="like", block=2)
    ao.analysis_index(key="开票人:", direction="like")
