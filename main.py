import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import uvicorn
from PIL import Image
import numpy as np
from core.analysis import Analysis

app = FastAPI()

# 提供静态文件服务，指向 `static` 文件夹
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/ocr")
async def ocr_analysis(file: UploadFile = File(...)):
    # 读取上传的文件内容
    contents = await file.read()

    # 使用 BytesIO 将文件内容转换为字节流对象
    img_byte_stream = BytesIO(contents)
    # 将字节流转换为图片格式（PIL）
    try:
        # img = Image.open(img_byte_stream)
        # img = np.array(img)  # 转换为 np.ndarray 类型
        # img = Image.open(img_byte_stream)
        # img = np.array(img)  # 转换为 np.ndarray 类型

        img = Image.open(img_byte_stream)

        # 将 PIL 图像转换为 numpy 数组
        img = np.array(img)

        # 1. 自动方向矫正（解决手机拍摄倒置问题）
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 注意：PIL 默认是 RGB 格式
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        # 将图像转回 PIL 对象（如果需要继续处理为 PIL 图像）
        img = Image.fromarray(img)
        img = np.array(img)
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

    result = Analysis(img)
    ocr_type, analysis_result = result.data_handle()
    
    # 将ocr_type转为中文
    ocr_type_map = {
        "ordinary_invoice": "普通发票",
        "vat_invoice": "增值税发票",
        "vat_special_invoice": "增值税电子专票",
        "pay_invoice": "微信或者支付宝支付",
        "detail_invoice": "账单详情",
        "train_invoice": "高铁票",
        "plane_invoice": "机票",
        "smart_invoice": "小发票联",
        "smart_vat_invoice": "小发票联增税",
        "unknow": "未知"
    }

    # 获取中文名称
    ocr_type_chinese = ocr_type_map.get(ocr_type, "未知")

    return {"ocr_type": ocr_type_chinese, "analysis_result": analysis_result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
