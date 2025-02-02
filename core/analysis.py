import copy
import time
import re
import pandas as pd
from paddleocr import PaddleOCR
import cv2
import numpy as np

class Analysis:
    def determine_ocr_type(self, data):
        # 初始化ocr_type为"unknow"，假设没有匹配项
        ocr_type = "unknow"
        key = "".join(data["key"])
        print(key)
        # 判断是否为航空发票
        if "空运输电子客票行程单" in key:
            ocr_type = "plane_invoice"
        # 判断是否为铁路发票
        elif any(keyword in key for keyword in ["高铁票", "中国铁路","车站"]):
            ocr_type = "train_invoice"
        # 判断是否为支付类发票（银联、支付宝、微信）
        elif any(keyword in key for keyword in ["银联商", "支付宝", "微信"]):
            ocr_type = "pay_invoice"
        elif "增值税专用发票" in key:
            ocr_type = "vat_special_invoice"
        # 判断是否为增值税普通发票或增值税专用发票
        elif "增值税普通发票" in key or re.search(r'服务名称', key):
            ocr_type = "vat_invoice"
        # 判断是否为普通发票
        elif key in("\n普通发票") or all(keyword in key for keyword in  ["普通发票", "电子发票"]) or "发票普发票" in key:
            ocr_type = "ordinary_invoice"
        # 判断是否为发票联且包含增值税
        elif "机打号码" in key or "机器编号" in key:
            ocr_type = "smart_invoice"
        elif any(keyword in key for keyword in ["账单详情", "全部账单"]):
            ocr_type = "detail_invoice"

        return ocr_type


    def __init__(self, img_stream):

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        data = []
        ocr = PaddleOCR(
            use_angle_cls=True,  # 启用方向分类器
            lang='ch',  # 使用简体中文模型
            det_algorithm='DB',  # 文本检测算法选择DB
            rec_algorithm='CRNN',  # 文本识别算法选择SVTR_LCNet
            use_space_char=True,
            layout=True,  # 启用布局分析
            table=True,  # 启用表格识别
            det_db_box_thresh=0.2,
            det_db_thresh=0.1,
            rec_model_dir='./models/ch_PP-OCRv4_rec_infer',
        )
        result = ocr.ocr(img_stream, cls=True)
        print(result)
        for idx in range(len(result)):
            res = result[idx]
        for line in res:
            index = line[0]
            row = {"index_1": index[0][0], "index_2": index[0][1], "index_3": index[1][0], "index_4": index[1][1],
                   "index_5": index[2][0], "index_6": index[2][1], "index_7": index[3][0], "index_8": index[3][1],
                   "key": line[1][0].replace("：", ":").replace("_", "").replace("（", "(").replace("）", ")").replace(" ",
                                                                                                                    ""),
                   "accuracy": line[1][1]}
            data.append(row)
        data = pd.DataFrame(data)
        self.data_old=data
        data = data.sort_values(by=['index_1', 'index_2'], ascending=[True, True])

        ocr_type=self.determine_ocr_type(data)

        # 如果最大行高 > 最大 列高，需反转
        data["pre_filed_height"] = data["index_6"] - data["index_4"]
        max_pre_filed_height = data['pre_filed_height'].max()

        data["pre_filed_length"] = data["index_3"] - data["index_7"]
        max_pre_filed_length = data['pre_filed_length'].max()

        if max_pre_filed_height > max_pre_filed_length:
            data["index_1"], data["index_2"], data["index_7"], data["index_8"], data["index_5"], data["index_6"], data[
                "index_3"], data["index_4"] = data["index_2"], data["index_1"], data["index_4"], data["index_3"], data[
                "index_6"], data["index_5"], data["index_8"], data["index_7"]

        data = data.sort_values(by=['index_1', 'index_2'], ascending=[True, True])
        if ocr_type == 'vat_invoice':
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]

            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 0
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 0
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 0
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * 0
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2
        elif ocr_type == 'vat_special_invoice':
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]

            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 0
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 0
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 0
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * 0
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2
        elif ocr_type == 'ordinary_invoice':
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2.5
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]

            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 0
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 0
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 0
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * 0
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2
        elif ocr_type == 'smart_invoice':
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2.5
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]

            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 1
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 0
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 2
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * -2
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2
        if ocr_type == 'plane_invoice':
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]
            # 横向合并
            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 1
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 1
            # 纵向
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 0
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * 0
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2
        if ocr_type == 'detail_invoice':
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]
            # 横向合并
            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 0.5
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 0.5
            # 纵向
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 2
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * 2
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2
        else:
            # 行高容错设置
            data["row_height"] = abs(data["index_4"] - data["index_6"]) / 2.5
            data["filed_height"] = data["index_6"] - data["index_4"]
            data["filed_length"] = data["index_3"] - data["index_7"]

            data["y_offset_up"] = data["index_6"] + data["filed_height"] * 0
            data["y_offset_low"] = data["index_2"] - data["filed_height"] * 0
            data["x_offset_up"] = data["index_3"] + data["filed_height"] * 2
            data["x_offset_low"] = data["index_7"] - data["filed_height"] * 2
            self.middle_y = (max(data["index_2"]) - min(data["index_2"])) / 2
            self.middle_x = (max(data["index_3"]) - min(data["index_3"])) / 2

        print(data)
        self.data = data
        self.data_copy = copy.copy(data)
        self.ocr_type=ocr_type


    def data_handle(self):
        keys = self.data["key"].tolist()
        ocr_type=self.ocr_type
        analysis_result = getattr(self, ocr_type + "_analysis")()
        return ocr_type,analysis_result

    def merge_raw_data(self, fileds: dict):
        """
        合并原始数据
        :return:
        """

        def merge(filtered_df, join_flag=True):
            # filtered_df = filtered_df.sort_values(by=['index_2'], ascending=[True])
            if filtered_df.shape[0] >= 2:
                indexes_list = filtered_df.index.tolist()
                # 合并操作

                first_key = str(filtered_df["key"].iloc[0])
                if ":" in first_key and not first_key.endswith(":"):
                    return

                first_key = first_key.replace(":", "")
                remaining_keys = filtered_df["key"].iloc[1:]
                if join_flag:
                    new_key = first_key + ":" + "".join(remaining_keys.tolist())
                else:
                    new_key = "".join(filtered_df["key"].tolist())
                self.data.loc[indexes_list[0], 'key'] = new_key
                self.data.loc[indexes_list[0], 'index_3'] = filtered_df.loc[indexes_list[-1], 'index_3']
                self.data.loc[indexes_list[0], 'index_2'] = filtered_df.loc[indexes_list[-1], 'index_2']
                self.data.loc[indexes_list[0], 'index_7'] = filtered_df.loc[indexes_list[-1], 'index_7']

                self.data.drop(indexes_list[1:], inplace=True)

        for filed in fileds.keys():
            extension_multiplier_x = fileds.get(filed)
            if not extension_multiplier_x:
                if ":" in filed:
                    filed = filed.split(":")[0]
                start_text, end_text = filed[0], filed[-1]
                filtered_df = self.data[
                    (self.data['key'].str.startswith(start_text) | self.data['key'].str.endswith(end_text)) &
                    (self.data['y_offset_up'] >= self.data['index_2']) &
                    (self.data['index_2'] >= self.data['y_offset_low'])
                    ]
                merge(filtered_df, False)
            else:
                curr_df = self.data[self.data['key'].str.startswith(filed)]
                if curr_df.empty:
                    continue
                # x偏移设置
                shape = curr_df.shape[0]
                for i in range(shape):
                    curr_index_3 = curr_df.iloc[i]['index_3']
                    curr_filed_length = curr_df.iloc[i]['filed_length'] * extension_multiplier_x
                    curr_y_offset_up = curr_df.iloc[i]['y_offset_up']
                    y_offset_low = curr_df.iloc[i]['y_offset_low']

                    # y：index_6 比index_2大 ; x：3比7大
                    merge_x_offset = curr_index_3 + curr_filed_length
                    filtered_df = self.data[
                        (self.data['index_3'] >= curr_index_3) &
                        (merge_x_offset >= self.data['index_7']) &
                        (curr_y_offset_up >= self.data['index_2']) &
                        (y_offset_low < self.data['index_6'])
                        ]
                    merge(filtered_df)

    def vat_special_invoice_analysis(self):
        # 需要合并的字段
        fileds = {"价税合计(大写)": 5}
        self.merge_raw_data(fileds)
        print(self.data)

        名称 = self.analysis_index(key="称:", direction="like")
        if len(名称) == 2:
            购买方名称 = self.extract_name_after_cheng(r'称:([\u4e00-\u9fa5]+)',名称[0])
            销售方名称 = self.extract_name_after_cheng(r'称:([\u4e00-\u9fa5]+)',名称[1])
        else:
            销售方名称 = self.analysis_index(key="称:", direction="like", block=1)
            if len(销售方名称) == 1:
                销售方名称 = [self.extract_name_after_cheng(r'称:([\u4e00-\u9fa5]+)',销售方名称[0])]
            购买方名称 = self.analysis_index(key="称:", direction="like", block=3)
            if len(购买方名称) == 1:
                购买方名称 = [self.extract_name_after_cheng(r'称:([\u4e00-\u9fa5]+)',购买方名称[0])]

        价税合计 = self.analysis_index(key="价税合计", direction="like")
        价税合计小写, 价税合计大写 = [], []
        if len(价税合计) != 0 and ":" in 价税合计[0] and not 价税合计[0].endswith(":"):
            价税合计 = 价税合计[0].split(":")[1].replace("小写", "").replace("(", "").replace(")", "")
            pattern = r'[1234567890¥.]*'
            matches = re.findall(pattern, 价税合计)
            价税合计小写 = [match for match in matches if match]
            if len(价税合计小写) > 0:
                # 价税合计小写.append(价税合计小写[0])
                价税合计大写.append(价税合计.replace(价税合计小写[0], ""))
                # # 取不到再取一次
            else:
                key = r'([壹贰叁肆伍陆柒捌玖拾佰仟万亿零]+(?:零)?)*[圆园元](?:[零壹贰叁肆伍陆柒捌玖拾]+角)?(?:[零壹贰叁肆伍陆捌玖拾]+分)?(?:整)?'
                expr = f'key.str.contains(@key, case=False, na=False)'
                curr_key = self.data_copy.query(expr, engine='python')
                if not curr_key.empty:
                    价税合计大写 = curr_key["key"].tolist()
                    # TODO 大写转小写
                    # 价税合计小写
                if len(价税合计小写) == 0:
                    价税合计小写=self.analysis_index(key=r"^\(小写\)[￥¥]?([0-9,]+(\.\d{1,2})?)", direction="like")
                    if len(价税合计小写) > 0:
                        价税合计小写 = 价税合计小写[-1:]
        if  len(价税合计大写)==0:
            价税合计大写=self.analysis_index(
            key=r'([壹贰叁肆伍陆柒捌玖拾佰仟零]+(?:零)?)*[圆园元](?:[零壹贰叁肆伍陆柒捌玖拾]+角)?(?:[零壹贰叁肆伍陆捌玖拾]+分)?(?:整)?',
            direction="like")
        开票日期 = self.analysis_index(key=r'(\d{4}[年]\d{2}[月]\d{2})', direction="like")
        发票号码 = self.analysis_index(key="发票号码", direction="like")
        开票人 = self.analysis_index(key="开票人:", direction="like")
        税率 = self.analysis_index(key="税率", direction="below")
        收款人 = self.analysis_index(key="收款人:", direction="like", block=1)
        金额 = self.analysis_index(key="金额", end_key="￥",direction="below")
        税额 = self.analysis_index(key="税额", end_key="￥",direction="below")
        项目名称 = self.analysis_index(key="项目名称", direction="below")
        if 项目名称:
            项目名称 = 项目名称[0]

        print(self.data)
        data = {"开票日期": 开票日期, "发票号码": 发票号码, "购买方名称": 购买方名称, "销售方名称": 销售方名称,"项目名称":项目名称,"金额":金额,"税额":税额,
                "价税合计大写": 价税合计大写, "价税合计小写": 价税合计小写, "开票人": 开票人}
        print(data)
        return data
    def extract_name_after_cheng(self,regex,text):
        # 查找 "称:" 后面的中文内容
        match = re.search(regex, text)
        if match:
            return match.group(1)
        return text
    def vat_invoice_analysis(self):
        # 需要合并的字段
        fileds = {"价税合计(大写)": 6, "开票人:": 4, "称:": 3,"金":4}
        self.merge_raw_data(fileds)
        print(self.data)

        名称 = self.analysis_index(key=r"称[:·]", direction="like")
        print("名称",名称)
        if len(名称) == 2:
            购买方名称 = self.extract_name_after_cheng(r'称[:·]([\u4e00-\u9fa5]+)',名称[0])
            销售方名称 = self.extract_name_after_cheng(r'称[:·]([\u4e00-\u9fa5]+)',名称[1])
        else:
            销售方名称 = self.analysis_index(key="称:", direction="like", block=1)
            if len(销售方名称) == 1:
                销售方名称 = self.extract_name_after_cheng(r'称[:·]([\u4e00-\u9fa5]+)',销售方名称[0])
            购买方名称 = self.analysis_index(key="称:", direction="like", block=3)
            if len(购买方名称) == 1:
                购买方名称 = self.extract_name_after_cheng(r'称[:·]([\u4e00-\u9fa5]+)',购买方名称[0])

        价税合计 = self.analysis_index(key="价税合计", direction="like")
        价税合计小写, 价税合计大写 = [], []
        if len(价税合计) != 0 and ":" in 价税合计[0] and not 价税合计[0].endswith(":"):
            价税合计 = 价税合计[0].split(":")[1].replace("小写", "").replace("(", "").replace(")", "")
            pattern = r'[1234567890¥.]*'
            matches = re.findall(pattern, 价税合计)
            价税合计小写 = [match for match in matches if match]
            if len(价税合计小写) > 0:
                # 价税合计小写.append(价税合计小写[0])
                价税合计大写.append(价税合计.replace(价税合计小写[0], ""))
                # # 取不到再取一次
            else:
                key = r'([壹贰叁肆伍陆柒捌玖拾佰仟万亿零]+(?:零)?)*[圆园元](?:[零壹贰叁肆伍陆柒捌玖拾]+角)?(?:[零壹贰叁肆伍陆捌玖拾]+分)?(?:整)?'
                expr = f'key.str.contains(@key, case=False, na=False)'
                curr_key = self.data_copy.query(expr, engine='python')
                if not curr_key.empty:
                    价税合计大写 = curr_key["key"].tolist()
                    # TODO 大写转小写
                    # 价税合计小写
                if len(价税合计小写) == 0:
                    价税合计小写=self.analysis_index(key=r"^\(小写\)[￥¥]?([0-9,]+(\.\d{1,2})?)", direction="like")
                    if len(价税合计小写) > 0:
                        价税合计小写 = 价税合计小写[-1:]
        if  len(价税合计大写)==0:
            价税合计大写=self.analysis_index(
            key=r'([壹贰叁肆伍陆柒捌玖拾佰仟零]+(?:零)?)*[圆园元](?:[零壹贰叁肆伍陆柒捌玖拾]+角)?(?:[零壹贰叁肆伍陆捌玖拾]+分)?(?:整)?',
            direction="like")
        else:
            价税合计大写 = re.sub(r'(整|分).*', r'\1', 价税合计大写[0])

        if len(价税合计小写) == 0:
            价税合计小写=self.analysis_index(key=r"^\(小写\)[￥¥]?([0-9,]+(\.\d{1,2})?)", direction="like")

        开票日期 = self.analysis_index(key=r'(\d{4}[年]\d{2}[月]\d{2})', direction="like")

        发票号码 = self.analysis_index(key="发票号码", direction="like")
        if len(发票号码)==0:
            发票号码 = self.analysis_index(key=r"^No?\d+", direction="like")
        开票人 = self.analysis_index(key="开票人:", direction="like")
        税率 = self.analysis_index(key="税率", direction="below")
        收款人 = self.analysis_index(key="收款人:", direction="like", block=1)
        金额 = self.analysis_index(key="金额", end_key="￥",direction="below")
        if len(金额)==0:
            金额 = self.analysis_index(key="金:额", end_key="￥",direction="below")
        税额 = self.analysis_index(key="税额", end_key="￥",direction="below")
        if len(税额)==0:
            税额 = self.analysis_index(key="税率:税额", end_key="￥",direction="below")
        项目名称 = self.analysis_index(key="货物或应税劳务、服务名称", direction="below")
        if 项目名称:
            项目名称 = 项目名称[0]

        print(self.data)
        data = {"开票日期": 开票日期, "发票号码": 发票号码, "购买方名称": 购买方名称, "销售方名称": 销售方名称,"项目名称":项目名称,"金额":金额,"税额":税额,
                "价税合计大写": 价税合计大写, "价税合计小写": 价税合计小写, "开票人": 开票人}
        print(data)
        return data

    def train_invoice_analysis(self):

        姓名 = self.analysis_index(key=r'\d{6,10}\*{4,8}[\dXx]+', direction="like")  #
        时间 = self.analysis_index(key=r'\d{4}年\d{2}月\d{2}日\d{2}:\d{2}', direction="like")  #
        stand = self.analysis_index(key='站', direction="like")
        print(stand)
        stand = [item for item in stand if '车站' not in item]
        出发地, 到达地 = [], []
        stand = [item for item in stand if item.endswith('站')]
        if len(stand) == 2:
            出发地, 到达地 = [stand[0]], [stand[1]]
        价格 = self.analysis_index(key='￥', direction="like")  #

        data = {"姓名": 姓名, "时间": 时间, "出发地": 出发地, "到达地": 到达地, "价格": 价格}
        print(self.data)
        return data

    def ordinary_invoice_analysis(self):

        # 需要合并的字段
        fileds = {"数量": None, "合计": None}
        self.merge_raw_data(fileds)
        名称 = self.analysis_index(key=r"名?称:", direction="like")
        if 名称 and len(名称) > 0:
            购买方名称 = 名称[0].split(":")[1]
        else:
            购买方名称 = "未知购买方名称"  # 或记录日志

        if 名称 and len(名称) > 1:
            销售方名称 = 名称[1].split(":")[1]
        else:
            销售方名称 = "未知销售方名称"  # 或记录日志
        价税合计 = self.analysis_index(
            key=r'([壹贰叁肆伍陆柒捌玖拾佰仟零]+(?:零)?)*[圆园元](?:[零壹贰叁肆伍陆柒捌玖拾]+角)?(?:[零壹贰叁肆伍陆捌玖拾]+分)?(?:整)?',
            direction="like")
        价税合计大写 = "未知"
        if 价税合计:
            价税合计大写 = 价税合计[0]
        价税合计小写 = self.analysis_index(key="(小写)", direction="like")
        开票日期 = self.analysis_index(key="开票日期:", direction="like")
        发票号码 = self.analysis_index(key="发票号码:", direction="like")
        开票人 = self.analysis_index(key="开票人:", direction="like")

        项目名称 = self.analysis_index(key="项目名称", direction="below")
        if 项目名称:
            项目名称 = 项目名称[0]
        规格型号 = self.analysis_index(key="规格型号", direction="below")
        单位 = self.analysis_index(key="单位", direction="below")
        数量 = self.analysis_index(key="数量", direction="below")
        金额 = self.analysis_index(key="金额", direction="below")
        税额 = self.analysis_index(key="税额", direction="below")
        金额合计 = self.analysis_index(key="金额", end_key="￥", direction="below")
        税额合计 = self.analysis_index(key="税额", end_key="￥", direction="below")
        if 金额合计:
            金额合计 = 金额合计[0]

        # "规格型号": 规格型号, "单位": 单位, "数量": 数量, "金额": 金额, "税额": 税额,
        data = {"开票日期": 开票日期, "发票号码": 发票号码, "购买方名称": 购买方名称, "销售方名称": 销售方名称,
                "价税合计大写": 价税合计大写, "价税合计小写": 价税合计小写, "项目名称": 项目名称,
                "金额合计": 金额合计,
                "税额合计": 税额合计, "开票人": 开票人}
        print(data)
        return data

    def pay_invoice_analysis(self):
        银行卡号 = self.analysis_index(key=r"(\d{5,6}[\*]+)([\d]{4})", direction="like")
        金额 = self.analysis_index(key=r"(金额:)?RMB\:?\d+(\.\d+)?[\d)]$", direction="like")
        商户名称 = self.analysis_index(key="商户名称", direction="like")
        if len(商户名称) == 0 or 商户名称[0].endswith(":"):
            商户名称 = self.analysis_index(start_key="商户名称", row_index=1)
        时间 = self.analysis_index(key=r'(\d{4}[/\.-]\d{2}[/\.-]\d{2})\s*?(\d{2}:\d{2}:\d{2})', direction="like")
        银行名称 = ""

        if 银行卡号:
            银行名称 = self.get_bank_name(银行卡号[0])
        data = {"银行名称": 银行名称, "银行卡号": 银行卡号, "金额": 金额, "时间": 时间, "商户名称": 商户名称}
        print(data)
        return data

    def smart_invoice_analysis(self):
        fileds = {"开票日期": 3,"购买方名称": 3}
        self.merge_raw_data(fileds)
        print(self.data)

        开票日期 = self.analysis_index(key=r'(2\d{3}-?\d{2}-?\d{2})$', direction="likeOld")        
        if len(开票日期)>0:
            开票日期 = 开票日期[0].replace("开票日期","")
        发票号码 = self.analysis_index(key='发票号码', direction="like")
        if len(发票号码)==0:
            发票号码 = self.analysis_index(key='发票号码', direction="right")

        发票代码 = self.analysis_index(key='发票代码', direction="like")
        if len(发票代码)==0:
            发票代码 = self.analysis_index(key='发票代码', direction="right")

        金额 = self.analysis_index(key="金额", direction="below")
        if len(金额)==0 or not re.match(r'^\d+(\.\d+)?$', 金额[0]):
            金额 = self.analysis_index(key="合计金额(小写)", direction="right")
        if len(金额)==0 or not re.match(r'^\d+(\.\d+)?$', 金额[0]):
            金额 = self.analysis_index(key="合计金额", direction="likeOld")
        if len(金额)>0:
            金额=金额[0]
        项目 = self.analysis_index(key="项目", direction="below",below_height=8)
        if len(项目)>0:
            项目 = [item for item in 项目 if "税总函" not in item]

        购买方名称 = self.analysis_index(key=r'购买方名称:?', direction="like")
        if len(购买方名称)>0:
            购买方名称 = 购买方名称[0].replace("购买方名称","")
        销售方名称 = self.analysis_index(key=r'销售方名称:?', direction="like")
        if len(销售方名称) == 0 or 销售方名称[0] == "销售方名称":
            销售方名称 = self.analysis_index(start_key=r'^销售方名称', row_index=1)
            if "纳税人" in 销售方名称:
                销售方名称 = self.analysis_index(start_key=r'^销售方名称', row_index=-1)

        if len(销售方名称)>0:
            销售方名称 = 销售方名称[0].replace("销售方名称","")
        data = {"购买方名称": 购买方名称, "销售方名称": 销售方名称, "发票号码": 发票号码,"发票代码":发票代码, "开票日期": 开票日期,
                "金额": 金额, "项目": 项目}
        print(data)
        return data

    # def smart_vat_invoice_analysis(self):
    #     fileds = {"合计金额(小写)": 3}
    #     self.merge_raw_data(fileds)

    #     开票日期 = self.analysis_index(key=r'^2\d{3}-?\d{2}-?\d{2}$', direction="right")
    #     发票号码 = self.analysis_index(key='发票号码', direction="like")
    #     发票代码 = self.analysis_index(key='发票代码', direction="like")
    #     金额 = self.analysis_index(key="合计金额(小写)", direction="like")
    #     项目 = self.analysis_index(key="项目", direction="below")
    #     销售方名称 = self.analysis_index(key=r'销售方名称:?', direction="like")

    #     购买方名称 = self.analysis_index(key=r'购买方名称:?', direction="like")
    #     if 购买方名称 is None or 购买方名称[0] == "购买方名称":
    #         购买方名称 = self.analysis_index(start_key=r'^购买方名称', row_index=1)

    #     data = {"购买方名称": 购买方名称, "销售方名称": 销售方名称, "发票号码": 发票号码,"发票代码":发票代码, "开票日期": 开票日期,
    #             "金额": 金额, "项目": 项目}
    #     print(data)
    #     return data

    def plane_invoice_analysis(self):
        旅客姓名 = self.analysis_index(key=r'旅客姓名NAMEOFPASSENGER', direction="below")
        if len(旅客姓名)==0:
            旅客姓名 = self.analysis_index(key='旅客姓名', direction="below")
        身份证号码 = self.analysis_index(key='有效身份证件号码ID.NO', direction="below")
        if len(身份证号码)==0:
            身份证号码 = self.analysis_index(key='有效身份证件号码', direction="below")
        航班号 = self.analysis_index(key='FLIGHT', direction="below")
        if len(航班号)==0:
            航班号 = self.analysis_index(key='航班号', direction="below")
        出发地 = self.analysis_index(key='自FROM', direction="right")
        if len(出发地)==0:
            出发地 = self.analysis_index(key=r'^自:.+', direction="like")
        到达地 = self.analysis_index(key='至TO', direction="right")
        if len(到达地)==0:
            到达地 = self.analysis_index(key=r'^至:.+', direction="like")
        日期时间 = self.analysis_index(key=r'(2\d{3}-?\d{2}-?\d{2})\d{2}:\d{2}$', direction="like")
        if len(日期时间)==0:
            日期 = self.analysis_index(key='日期', direction="below")
            时间 = self.analysis_index(key='时间', direction="below")
            日期时间= 日期 + 时间
        价格 = self.analysis_index(key='合计TOTAL', direction="below")
        if len(价格)==0:
            价格 = self.analysis_index(key='合计', direction="below")

        data = {"旅客姓名":旅客姓名,"身份证号码":身份证号码,"航班号":航班号,"出发地":出发地,"到达地":到达地,"日期时间":日期时间,"价格":价格}
        print(data)
        return data

    def detail_invoice_analysis(self):

        #时间 = self.analysis_index(key=r'\d{4}[/\.-年]\d{1,2}[/\.-月]\d{1,2}日?\s*?(\d{2}:\d{2}:\d{2})', direction="like")
        时间 = self.analysis_index(key='支付时间', direction="right")
        商品说明 = self.analysis_index(key='商品说明', direction="right")
        付款方式 = self.analysis_index(key='付款方', direction="right")
        if len(付款方式)==0:
            付款方式 = self.analysis_index(key='付款方式', direction="right")
            if len(付款方式)==0:
                付款方式 = self.analysis_index(key='支付方式', direction="right")
        收款方 = self.analysis_index(key='收款方全称', direction="right")
        if len(收款方)==0:
            收款方 = self.analysis_index(key=r'收单机构', direction="right")

        金额 = self.analysis_index(key=r'^-.*\.', direction="likeOld")
        if 金额:
            金额=金额[0]
        #商户 = self.analysis_index(start_key=r'^-.*\.', row_index=-1)
        商户 = self.analysis_index(start_key=金额, row_index=-1)

        data = {"商户": 商户, "金额": 金额, "收款方": 收款方, "付款方式": 付款方式, "时间": 时间, "商品说明": 商品说明}
        print(data)
        return data

    def unknow_analysis(self):
        data={}
        return data


    # 银行卡号BIN段示例（这里只列出了一些常见的银行）
    BANK_BIN_DICT = {
        "中国工商银行": ["6222", "6226", "6227", "6228", "6212", "6216"],
        "中国建设银行": ["6227", "6228", "6258"],
        "中国农业银行": ["6225", "6223", "6251"],
        "中国银行": ["6232", "6213", "6222", "6217"],
        "交通银行": ["6222", "6215", "6225"],
        "招商银行": ["6225", "6219", "6212", "6210"],
        "兴业银行": ["6229", "6230"],
        "中信银行": ["6223", "6225", "6210"],
        "浦发银行": ["6213", "6221"],
        "广发银行": ["6212", "6227"],
        "中国光大银行": ["6282", "6220"],
        "中国民生银行": ["6226", "6225"],
        "平安银行": ["6226", "6210"],
        "华夏银行": ["6222", "6219"],
        "北京银行": ["6229", "6212"],
        "上海银行": ["6213", "6227"],
        "渤海银行": ["6229", "6225"],
        "南京银行": ["6225", "6222"],
        "宁波银行": ["6222", "6228"],
        "浙商银行": ["6226", "6213"],
        "富滇银行": ["6222", "6223"],
        "VISA": ["4"],  # VISA卡的BIN是以4开头
        "MasterCard": ["5"],  # MasterCard卡的BIN是以5开头
        "美国运通": ["34", "37"],  # AMEX卡的BIN是以34或37开头
        "发现卡": ["6011"],  # Discover卡的BIN是以6011开头
        "JCB": ["3568", "3528", "3529"],  # JCB卡的BIN是以3528-3589开头
        "Diners Club": ["36", "38"],  # Diners Club卡的BIN是以36、38开头
        "银联卡": ["62"],  # 银联卡的BIN是以62开头
        "韩亚银行": ["6213"],
        "招商永隆银行": ["6219"],
        "安盛银行": ["6215"],
        "光大银行": ["6282"],
        "交通银行": ["6222", "6215", "6225"],
        "中信银行": ["6223", "6225", "6210"],
        "邮储银行": ["6221"],
        "恒丰银行": ["6228"],
        "民生银行": ["6225", "6226"],
        "重庆银行": ["6227"],
        "长沙银行": ["6228"],
        "青岛银行": ["6229"],
        "厦门银行": ["6230"],
        "杭州银行": ["6232"],
        "广西银行": ["6223", "6228"],
        "江苏银行": ["6229", "6218"],
        "四川银行": ["6222", "6233"],
        "广州市商业银行": ["6227"],
        "深圳发展银行": ["6212"],
        "中原银行": ["6227", "6229"]
    }

    def get_bank_name(self, card_number):
        # 获取银行卡号的前六位作为BIN号段
        cleaned_card_number = re.sub(r'\D', '', card_number)
        bin_prefix = cleaned_card_number[:6]

        # 查找卡号对应的银行
        for bank_name, bin_list in self.BANK_BIN_DICT.items():
            for bin in bin_list:
                if bin_prefix.startswith(bin):
                    return bank_name

        return "无法识别该银行卡"

    def clean_data(self, key, data_list):
        # 统一处理：删除完全等于 key 的项，或者去除 key 后的内容
        cleaned_data = []
        for item in data_list:
            if item.startswith(key):
                # 去除 key 后剩余的部分
                cleaned_item = re.sub(key, '', item)
                # 只保留非空字符串
                if cleaned_item.strip():
                    cleaned_data.append(re.sub(r'\(.*?\)', '', cleaned_item.strip()))
            else:
                cleaned_data.append(item)
        return cleaned_data

    def analysis_index(self, direction=None, key=None, end_key=None, block=-1, below_height=2, row_index=0,
                       start_key=None):

        """
        解析key 方向的匹配
        :param key:
        :param direction: below / right like
        :return:
        """

        append_block_filter = ""
        if block == 1:
            append_block_filter = f"and  index_2 >= {self.middle_y} and index_7 <= {self.middle_x}"
        if block == 2:
            append_block_filter = f"and index_2 >= {self.middle_y} and index_7 >= {self.middle_x}"
        if block == 3:
            append_block_filter = f"and index_2 <= {self.middle_y} and  index_7 <= {self.middle_x}"
        if block == 4:
            append_block_filter = f"and index_2 <= {self.middle_y} and index_7 >= {self.middle_x}"

        start_in_words = self.data.query(f'key=="{key}"')
        if start_key:
            start_in_words = self.data.query('key.str.contains("^" + @start_key, case=False, na=False)',
                                             engine='python')
            key = start_key
        filter_values_words_value = None
        # 按行范围查找，并返回偏移行
        if not start_in_words.empty and row_index != 0:
            first_row = start_in_words.iloc[0]
            curr_index = start_in_words.index[0]
            query_str = f' {first_row["x_offset_low"]} < index_3 and {first_row["x_offset_up"]} > index_7'
            filter_values_words_value = self.data.query(query_str).sort_values(by='index_2', ascending=True)
            if not filter_values_words_value.empty:
                curr_loc = filter_values_words_value.index.get_loc(curr_index)
                next_row = filter_values_words_value.iloc[curr_loc + row_index]
                return next_row["key"] if next_row is not None else None
        if direction == "likeOld":
            expr = 'key.str.contains(@key, case=False, na=False)'
            expr += append_block_filter
            curr_key = self.data_old.query(expr, engine='python')
            if not curr_key.empty:
                return self.clean_data(key,curr_key["key"].tolist())
        if direction == "like":
            expr = 'key.str.contains(@key, case=False, na=False)'
            expr += append_block_filter
            curr_key = self.data.query(expr, engine='python')
            if not curr_key.empty:
                return self.clean_data(key,curr_key["key"].tolist())
        if end_key is not None:
            end_key = end_key.strip()
            end_in_words = self.data[self.data['key'].str.startswith(end_key, na=False)]
            if end_in_words.empty:
                print(f"Error: No data found for key: {key} and end_key: {end_key}")
                return None  # 或者你可以返回一个默认值
            end_row = end_in_words.iloc[0]
        if not start_in_words.empty:
            first_row = start_in_words.iloc[0]
            if end_key is not None and direction == "right":
                # query_str = f'{first_row["y_offset_low"]} <= index_2 <= {first_row["y_offset_up"]} and index_1 != {first_row["index_1"]} and index_7 <= {end_row["index_7"]}'
                query_str = f'{first_row["y_offset_low"]} <= index_2 <= {first_row["y_offset_up"]} and index_1 != {first_row["index_1"]} and index_7 <= {end_row["index_3"]}'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)

            elif end_key is None and direction == "right":
                query_str = f'{first_row["y_offset_low"]} < index_2 < {first_row["y_offset_up"]} and index_1 != {first_row["index_1"]} and {first_row["index_3"]}<= index_7 <= {first_row["index_3"] + first_row["filed_length"]}'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)

            elif end_key is None and direction == "below":
                # 左对齐 或右对齐
                filed_height = first_row["filed_height"] * below_height
                # query_str = f' x_offset_low <= {first_row["index_3"]} and x_offset_up >= {first_row["index_7"]} and  {first_row["index_2"]}+{filed_height} >= index_2  and {first_row["index_2"]}<= index_2'
                query_str = f' {first_row["x_offset_low"]} <= index_3 and {first_row["x_offset_up"]} >= index_7 and  {first_row["index_2"]}+{filed_height} >= index_2  and {first_row["index_2"]}<= index_6'
                # query_str = f' ({first_row["x_offset_low"]} < index_3 < {first_row["x_offset_up"]} and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height * 2} and index_2 != {first_row["index_2"]}) or ({first_row["x_offset_low"]} < index_7 < {first_row["x_offset_up"]} and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height * 2} and index_2 != {first_row["index_2"]})'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)

            elif end_key is not None and direction == "below":
                # 左对齐 或右对齐
                query_str = f' {first_row["x_offset_low"]} <= index_3 and {first_row["x_offset_up"]} >= index_7 and  {end_row["y_offset_up"]} >= index_2  and {end_row["y_offset_low"]} <= index_6'
                query_str += append_block_filter
                filter_values_words_value = self.data.query(query_str)
            if filter_values_words_value is None:
                return ""
            return self.clean_data(key, filter_values_words_value["key"].tolist())
        else:
            return []


if __name__ == '__main__':
    ao = Analysis("vat_invoice", "../uploadfile/img_13.png")
    ao.data_handle("vat_invoice")
    # ao.analysis_index(key="价税合计(大写)", direction="right", end_key="小写")
    # ao.analysis_index(key="项目名称", direction="below")
    # ao.analysis_index(key="规格型号", direction="below")
    # ao.analysis_index(key="单位", direction="below")
    # ao.analysis_index(key="数量", direction="below")
    # ao.analysis_index(key="金额", direction="below")
    # ao.analysis_index(key="税额", direction="below")
    # ao.analysis_index(key="名称:", direction="like", block=1)
    # ao.analysis_index(key="名称:", direction="like", block=2)
    # index_1  index_2  index_3  index_4  index_5  index_6  index_7  index_8  \
# 26    472.0    143.0    499.0    143.0    499.0    154.0    472.0    154.0    税率
# 33    483.0    156.0    500.0    156.0    500.0    169.0    483.0    169.0    3%
# 60    147.0   1105.0   1054.0   1110.0   1054.0   1146.0    147.0   1142.0


# query_str = f' ({first_row["x_offset_low"]} < index_3 < {first_row["x_offset_up"]}
# and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height}
# and index_2 != {first_row["index_2"]}) or
# ({first_row["x_offset_low"]} < index_7 < {first_row["x_offset_up"]}
# and {first_row["index_2"]} <= index_2 <= {first_row["index_2"]}+ {filed_height}
# and index_2 != {first_row["index_2"]})'

#      index_1  index_2  index_3  index_4  index_5  index_6  index_7  index_8     filed_length  y_offset_up  y_offset_low  x_offset_up  x_offset_low
# 12   2637.0    964.0   2854.0    964.0   2854.0   1043.0   2637.0   1043.0        217.0       1043.0         964.0       3012.0        2479.0
# 13   2923.0    964.0   3277.0    964.0   3277.0   1039.0   2923.0   1039.0        354.0       1039.0         964.0       3427.0        2773.0  税率/征收率
# 19   2607.0   1039.0   2850.0   1039.0   2850.0   1113.0   2607.0   1113.0        243.0       1113.0        1039.0       2998.0        2459.0
#  {first_row["x_offset_low"]} <= index_3 and {first_row["x_offset_up"]} >= index_7 and  {first_row["index_2"]}+{filed_height} >= index_2  and {first_row["index_2"]}<= index_2'
