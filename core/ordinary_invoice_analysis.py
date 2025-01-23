from core.common_analysis import commonAnalysis


class OrdinaryInvoiceAnalysis(commonAnalysis):

    def __init__(self, img_stream, ):
        super().__init__(img_stream)
        self.field = []

    def analysis(self):
        价税合计 = self.analysis_index(key="价税合计(大写)", direction="right", end_key="小写")
        项目名称 = self.analysis_index(key="项目名称", direction="below")
        规格型号 = self.analysis_index(key="规格型号", direction="below")
        单位 = self.analysis_index(key="单位", direction="below")
        数量 = self.analysis_index(key="数量", direction="below")
        金额 = self.analysis_index(key="金额", direction="below")
        税额 = self.analysis_index(key="税额", direction="below")
        开票日期=self.analysis_index(key="开票日期:", direction="like")
        发票号码=self.analysis_index(key="发票号码:", direction="like")
        开票人=self.analysis_index(key="开票人:", direction="like")
        购买方名称 = self.analysis_index(key="名称:", direction="like", block=1)
        if 购买方名称:
            购买方名称 = 购买方名称[0].split(":")[1]
        销售方名称 = self.analysis_index(key="名称:", direction="like", block=2)
        if 销售方名称:
            销售方名称 = 销售方名称[0].split(":")[1]

        data = {"开票日期":开票日期,"发票号码":发票号码,"价税合计": 价税合计, "项目名称": 项目名称, "规格型号": 规格型号, "单位": 单位, "数量": 数量,
                "金额": 金额, "税额": 税额, "购买方名称": 购买方名称, "销售方名称": 销售方名称,"开票人":开票人}
        return data


if __name__ == '__main__':
    oia = OrdinaryInvoiceAnalysis("../imgs/1.jpg")
    oia.analysis()
