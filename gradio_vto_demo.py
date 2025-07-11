import gradio as gr
import boto3
import base64
import json
import io
from PIL import Image
import numpy as np
import os
import datetime
import logging
from pathlib import Path

class VirtualTryOnDemo:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.setup_logging()
    def setup_logging(self):
        """设置日志系统和目录结构"""
        # 创建主日志目录
        self.log_dir = Path("gradio_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置全局日志记录器
        log_file = self.log_dir / f"gradio_vto_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Gradio VTO Demo 日志系统初始化完成 ===")
    
    def generate_session_id(self):
        """生成唯一的会话ID"""
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    
    def create_session_folder(self, session_id):
        """为每个会话创建独立的文件夹"""
        session_folder = self.log_dir / session_id
        session_folder.mkdir(exist_ok=True)
        return session_folder
    
    def save_image_to_session(self, image, session_folder, filename):
        """保存图片到会话文件夹"""
        if image is None:
            return None
            
        try:
            # 转换并保存图片
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            file_path = session_folder / filename
            image.save(file_path, 'PNG')
            
            self.logger.info(f"图片已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存图片失败 ({filename}): {str(e)}")
            return None
    
    def save_payload_to_session(self, session_folder, request_body, response_body=None):
        """保存请求和响应payload到会话文件夹"""
        try:
            # 创建payload数据
            payload_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "request": self.sanitize_request_for_json(request_body),
                "response": self.sanitize_response_for_json(response_body) if response_body else None
            }
            
            # 保存到JSON文件
            payload_file = session_folder / "payload.json"
            with open(payload_file, 'w', encoding='utf-8') as f:
                json.dump(payload_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Payload已保存: {payload_file}")
            return str(payload_file)
            
        except Exception as e:
            self.logger.error(f"保存payload失败: {str(e)}")
            return None
    
    def sanitize_request_for_json(self, request_body):
        """清理请求体中的base64数据以便JSON保存"""
        sanitized = json.loads(json.dumps(request_body))
        
        if 'virtualTryOnParams' in sanitized:
            vto_params = sanitized['virtualTryOnParams']
            
            # 替换base64图片数据为元信息
            if 'sourceImage' in vto_params:
                data_len = len(vto_params['sourceImage'])
                vto_params['sourceImage'] = {
                    "_type": "base64_image_data",
                    "_length": data_len,
                    "_description": "Source image base64 data"
                }
            
            if 'referenceImage' in vto_params:
                data_len = len(vto_params['referenceImage'])
                vto_params['referenceImage'] = {
                    "_type": "base64_image_data", 
                    "_length": data_len,
                    "_description": "Reference image base64 data"
                }
            
            if 'imageBasedMask' in vto_params and 'maskImage' in vto_params['imageBasedMask']:
                data_len = len(vto_params['imageBasedMask']['maskImage'])
                vto_params['imageBasedMask']['maskImage'] = {
                    "_type": "base64_image_data",
                    "_length": data_len,
                    "_description": "Mask image base64 data"
                }
        
        return sanitized
    
    def sanitize_response_for_json(self, response_body):
        """清理响应体中的base64数据以便JSON保存"""
        if not response_body:
            return None
            
        sanitized = json.loads(json.dumps(response_body))
        
        # 处理结果图片
        if 'images' in sanitized:
            for i, image_data in enumerate(sanitized['images']):
                sanitized['images'][i] = {
                    "_type": "base64_image_data",
                    "_length": len(image_data),
                    "_description": f"Result image {i+1} base64 data"
                }
        
        # 处理mask图片
        if 'maskImage' in sanitized:
            data_len = len(sanitized['maskImage'])
            sanitized['maskImage'] = {
                "_type": "base64_image_data",
                "_length": data_len,
                "_description": "Output mask base64 data"
            }
        
        return sanitized
    
    def log_request_details(self, session_id, session_folder, **kwargs):
        """记录请求详细信息"""
        self.logger.info(f"=== 新请求开始 [Session: {session_id}] ===")
        self.logger.info(f"会话文件夹: {session_folder}")
        
        # 记录所有参数
        params_log = []
        for key, value in kwargs.items():
            if key not in ['source_image', 'reference_image', 'mask_image']:  # 图片单独处理
                params_log.append(f"{key}: {value}")
        
        self.logger.info(f"请求参数: {' | '.join(params_log)}")
    
    def decode_and_save_response_images(self, session_id, session_folder, response_body):
        """解码并保存响应中的图片到会话文件夹"""
        result_images = []
        mask_image = None
        saved_files = []
        
        try:
            # 处理结果图片
            if 'images' in response_body and response_body['images']:
                for i, image_b64 in enumerate(response_body['images']):
                    image_data = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # 保存图片
                    filename = f"result_{i+1}.png" if len(response_body['images']) > 1 else "result.png"
                    saved_path = self.save_image_to_session(image, session_folder, filename)
                    
                    result_images.append(image)
                    saved_files.append(saved_path)
                    self.logger.info(f"结果图片 {i+1} 已解码并保存")
            
            # 处理Mask图片
            if 'maskImage' in response_body:
                mask_data = base64.b64decode(response_body['maskImage'])
                mask_image = Image.open(io.BytesIO(mask_data))
                
                # 保存Mask图片
                saved_path = self.save_image_to_session(mask_image, session_folder, "mask_output.png")
                saved_files.append(saved_path)
                self.logger.info(f"Mask图片已解码并保存")
            
            return result_images, mask_image, saved_files
            
        except Exception as e:
            self.logger.error(f"解码响应图片失败: {str(e)}")
            return [], None, []
        
    def invert_mask(self, mask_image):
        """对mask图像进行反色处理"""
        if mask_image is None:
            return None
        
        try:
            # 转换为灰度图
            if mask_image.mode != 'L':
                mask_gray = mask_image.convert('L')
            else:
                mask_gray = mask_image.copy()
            
            # 反色处理：255 - 原值
            import numpy as np
            mask_array = np.array(mask_gray)
            inverted_array = 255 - mask_array
            
            # 转回PIL图像
            from PIL import Image
            inverted_mask = Image.fromarray(inverted_array, mode='L')
            
            self.logger.info("Mask图像已进行反色处理")
            return inverted_mask
            
        except Exception as e:
            self.logger.error(f"Mask反色处理失败: {str(e)}")
            return mask_image  # 返回原图像
    
    def extract_source_and_mask(self, source_input):
        """从 ImageEditor 输入中提取人物图片和mask，并对mask进行反色处理"""
        if source_input is None:
            return None, None
        
        # 如果是 ImageEditor 的输出（字典格式）
        if isinstance(source_input, dict):
            background = source_input.get('background')  # 原始人物图片
            composite = source_input.get('composite')    # 合成后的图片（包含涂抹）
            
            if background is None:
                return None, None
                
            # 如果有涂抹层，提取mask
            if composite is not None and composite != background:
                # 这里可以通过对比 background 和 composite 来提取 mask
                # 简化处理：如果有 layers，使用第一层作为 mask
                layers = source_input.get('layers', [])
                if layers:
                    raw_mask = layers[0]
                    # 对mask进行反色处理
                    inverted_mask = self.invert_mask(raw_mask)
                    return background, inverted_mask
            
            return background, None
        
        # 如果是直接的 PIL 图像
        elif hasattr(source_input, 'mode'):
            return source_input, None
        
        return None, None
        """处理 ImageEditor 输入，提取 mask 图像"""
        if mask_input is None:
            return None
        
        # 如果是 ImageEditor 的输出（字典格式）
        if isinstance(mask_input, dict):
            # 尝试获取 composite 图像（合成后的图像）
            if 'composite' in mask_input and mask_input['composite'] is not None:
                return mask_input['composite']
            # 如果没有 composite，尝试获取 background
            elif 'background' in mask_input and mask_input['background'] is not None:
                return mask_input['background']
            # 如果有 layers，合并所有层
            elif 'layers' in mask_input and mask_input['layers']:
                # 这里可以实现更复杂的层合并逻辑
                # 暂时返回第一层
                if len(mask_input['layers']) > 0:
                    return mask_input['layers'][0]
            return None
        
        # 如果是直接的 PIL 图像
        elif hasattr(mask_input, 'mode'):  # PIL Image
            return mask_input
        
    def process_mask_input(self, mask_input):
        """处理传统mask输入，提取mask图像并进行反色处理"""
        if mask_input is None:
            return None
        
        # 如果是 ImageEditor 的输出（字典格式）
        if isinstance(mask_input, dict):
            # 尝试获取 composite 图像（合成后的图像）
            if 'composite' in mask_input and mask_input['composite'] is not None:
                raw_mask = mask_input['composite']
            # 如果没有 composite，尝试获取 background
            elif 'background' in mask_input and mask_input['background'] is not None:
                raw_mask = mask_input['background']
            # 如果有 layers，合并所有层
            elif 'layers' in mask_input and mask_input['layers']:
                # 这里可以实现更复杂的层合并逻辑
                # 暂时返回第一层
                if len(mask_input['layers']) > 0:
                    raw_mask = mask_input['layers'][0]
                else:
                    return None
            else:
                return None
            
            # 对mask进行反色处理
            return self.invert_mask(raw_mask)
        
        # 如果是直接的 PIL 图像
        elif hasattr(mask_input, 'mode'):  # PIL Image
            # 对mask进行反色处理
            return self.invert_mask(mask_input)
        
        return None
    
    def encode_image_to_base64(self, image):
        """将PIL图像转换为base64编码"""
        if image is None:
            return None
        
        # 转换为RGB格式（如果不是的话）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 保存到字节流
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # 编码为base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_base64
    
    def virtual_try_on(self, 
                      # 基础图像
                      source_image, reference_image, mask_image,
                      # Mask类型和参数
                      mask_type, garment_class, mask_shape_garment, mask_shape_prompt,
                      mask_prompt, long_sleeve_style, tucking_style, outer_layer_style,
                      # Mask排除选项
                      preserve_body_pose, preserve_hands, preserve_face,
                      # 合并样式
                      merge_style,
                      # 图像生成配置
                      number_of_images, quality, cfg_scale, seed):
        
        # 强制开启returnMask
        return_mask = True
        
        # 生成会话ID和创建会话文件夹
        session_id = self.generate_session_id()
        session_folder = self.create_session_folder(session_id)
        
        try:
            # 从 ImageEditor 中提取人物图片和涂抹的mask
            extracted_source, extracted_mask = self.extract_source_and_mask(source_image)
            
            if extracted_source is None:
                error_msg = "错误: 请上传人物图片"
                self.logger.error(f"[{session_id}] {error_msg}")
                return None, None, error_msg
            # 记录请求开始
            self.log_request_details(
                session_id=session_id,
                session_folder=session_folder,
                mask_type=mask_type,
                garment_class=garment_class,
                mask_shape_garment=mask_shape_garment,
                mask_shape_prompt=mask_shape_prompt,
                mask_prompt=mask_prompt,
                long_sleeve_style=long_sleeve_style,
                tucking_style=tucking_style,
                outer_layer_style=outer_layer_style,
                preserve_body_pose=preserve_body_pose,
                preserve_hands=preserve_hands,
                preserve_face=preserve_face,
                merge_style=merge_style,
                return_mask=return_mask,
                number_of_images=number_of_images,
                quality=quality,
                cfg_scale=cfg_scale,
                seed=seed
            )
            
            # 保存输入图片到会话文件夹
            input_files = []
            source_path = self.save_image_to_session(extracted_source, session_folder, "source.png")
            if source_path:
                input_files.append(source_path)
                
            reference_path = self.save_image_to_session(reference_image, session_folder, "reference.png")
            if reference_path:
                input_files.append(reference_path)
                
            # 处理 mask（优先使用涂抹的mask，其次使用上传的mask）
            final_mask = extracted_mask if extracted_mask is not None else self.process_mask_input(mask_image)
            mask_input_path = None
            if final_mask is not None:
                mask_input_path = self.save_image_to_session(final_mask, session_folder, "mask_input.png")
                if mask_input_path:
                    input_files.append(mask_input_path)
            
            # 编码图像
            source_b64 = self.encode_image_to_base64(extracted_source)
            reference_b64 = self.encode_image_to_base64(reference_image)
            
            if not source_b64 or not reference_b64:
                error_msg = "错误: 请上传源图像和参考图像"
                self.logger.error(f"[{session_id}] {error_msg}")
                return None, error_msg
            
            # 构建请求参数
            request_body = {
                "taskType": "VIRTUAL_TRY_ON",
                "virtualTryOnParams": {
                    "sourceImage": source_b64,
                    "referenceImage": reference_b64,
                    "maskType": mask_type,
                    "mergeStyle": merge_style,
                    "returnMask": return_mask
                },
                "imageGenerationConfig": {
                    "numberOfImages": number_of_images,
                    "quality": quality,
                    "cfgScale": cfg_scale
                }
            }
            
            # 添加seed（如果指定）
            if seed >= 0:
                request_body["imageGenerationConfig"]["seed"] = seed
            
            # 根据mask类型添加相应参数
            if mask_type == "IMAGE" and final_mask is not None:
                mask_b64 = self.encode_image_to_base64(final_mask)
                request_body["virtualTryOnParams"]["imageBasedMask"] = {
                    "maskImage": mask_b64
                }
            
            elif mask_type == "GARMENT":
                garment_mask = {
                    "maskShape": mask_shape_garment,
                    "garmentClass": garment_class
                }
                
                # 添加服装样式参数（如果适用）
                styling = {}
                if long_sleeve_style != "DEFAULT":
                    styling["longSleeveStyle"] = long_sleeve_style
                if tucking_style != "DEFAULT":
                    styling["tuckingStyle"] = tucking_style
                if outer_layer_style != "DEFAULT":
                    styling["outerLayerStyle"] = outer_layer_style
                
                if styling:
                    garment_mask["garmentStyling"] = styling
                
                request_body["virtualTryOnParams"]["garmentBasedMask"] = garment_mask
            
            elif mask_type == "PROMPT":
                request_body["virtualTryOnParams"]["promptBasedMask"] = {
                    "maskShape": mask_shape_prompt,
                    "maskPrompt": mask_prompt
                }
            
            # 添加mask排除选项
            exclusions = {}
            if preserve_body_pose != "DEFAULT":
                exclusions["preserveBodyPose"] = preserve_body_pose
            if preserve_hands != "DEFAULT":
                exclusions["preserveHands"] = preserve_hands
            if preserve_face != "DEFAULT":
                exclusions["preserveFace"] = preserve_face
            
            if exclusions:
                request_body["virtualTryOnParams"]["maskExclusions"] = exclusions
            
            # 调用API
            self.logger.info(f"[{session_id}] 开始调用 Nova Canvas API...")
            response = self.bedrock.invoke_model(
                modelId='amazon.nova-canvas-v1:0',
                body=json.dumps(request_body),
                contentType='application/json'
            )
            
            # 解析响应
            response_body = json.loads(response['body'].read())
            
            # 解码并保存响应图片
            result_images, mask_output, output_files = self.decode_and_save_response_images(session_id, session_folder, response_body)
            
            # 保存payload到会话文件夹
            payload_file = self.save_payload_to_session(session_folder, request_body, response_body)
            
            if result_images:
                # 使用第一张结果图片作为主要输出
                result_image = result_images[0]
                
                # 构建详细信息，包含文件路径
                info_parts = [
                    f"**🎉 API调用成功! [Session: {session_id}]**",
                    "",
                    f"**📁 会话文件夹:** `{session_folder}`",
                    "",
                    "**📁 保存的文件:**"
                ]
                
                # 添加输入文件信息
                if source_path:
                    info_parts.append(f"- 源图像: `source.png`")
                if reference_path:
                    info_parts.append(f"- 参考图像: `reference.png`")
                if mask_input_path:
                    info_parts.append(f"- 输入Mask: `mask_input.png`")
                
                # 添加输出文件信息
                for i, _ in enumerate(result_images):
                    filename = f"result_{i+1}.png" if len(result_images) > 1 else "result.png"
                    info_parts.append(f"- 结果图片: `{filename}`")
                
                if mask_output:
                    info_parts.append(f"- 输出Mask: `mask_output.png`")
                
                if payload_file:
                    info_parts.append(f"- 请求数据: `payload.json`")
                
                info_parts.extend([
                    "",
                    "**⚙️ 请求参数:**",
                    f"- Mask类型: {mask_type}",
                    f"- 合并样式: {merge_style}",
                    f"- 图像数量: {number_of_images}",
                    f"- 质量: {quality}",
                    f"- CFG Scale: {cfg_scale}",
                    f"- Seed: {seed if seed >= 0 else '随机'}",
                    f"- 返回Mask: {'是' if return_mask else '否'}"
                ])
                
                if mask_type == "GARMENT":
                    info_parts.extend([
                        "",
                        "**👔 服装参数:**",
                        f"- 服装类别: {garment_class}",
                        f"- Mask形状: {mask_shape_garment}",
                        f"- 长袖样式: {long_sleeve_style}",
                        f"- 塞入样式: {tucking_style}",
                        f"- 外层样式: {outer_layer_style}"
                    ])
                elif mask_type == "PROMPT":
                    info_parts.extend([
                        "",
                        "**💬 Prompt参数:**",
                        f"- Mask形状: {mask_shape_prompt}",
                        f"- Mask提示词: {mask_prompt}"
                    ])
                
                info_parts.extend([
                    "",
                    "**🛡️ 保留选项:**",
                    f"- 保留身体姿势: {preserve_body_pose}",
                    f"- 保留手部: {preserve_hands}",
                    f"- 保留面部: {preserve_face}"
                ])
                
                info = "\n".join(info_parts)
                
                self.logger.info(f"[{session_id}] 请求处理完成，成功生成 {len(result_images)} 张图片")
                # 返回结果图片、mask图片和详细信息
                return result_image, mask_output, info
            else:
                error_msg = f"错误: API响应中没有图像数据\n响应: {response_body}"
                self.logger.error(f"[{session_id}] {error_msg}")
                # 仍然保存payload以便调试
                self.save_payload_to_session(session_folder, request_body, response_body)
                return None, None, error_msg
                
        except Exception as e:
            error_msg = f"错误: {str(e)}"
            self.logger.error(f"[{session_id}] API调用异常: {error_msg}")
            # 保存失败的请求信息以便调试
            try:
                self.save_payload_to_session(session_folder, request_body if 'request_body' in locals() else None, None)
            except:
                pass
            return None, None, error_msg

def create_demo():
    demo_instance = VirtualTryOnDemo()
    
    with gr.Blocks(title="Nova Canvas 虚拟换装 Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Nova Canvas 虚拟换装 Demo")
        gr.Markdown("这个demo展示了Amazon Nova Canvas虚拟换装API的所有可调节参数，并自动记录所有请求日志")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📸 输入图像")
                
                # 使用 ImageEditor 支持在人物图片上直接涂抹 mask
                source_image = gr.ImageEditor(
                    label="sourceImage (人物图片) - 可直接涂抹mask区域",
                    type="pil",
                    height=350,
                    brush=gr.Brush(
                        default_size=15,
                        color_mode="fixed",
                        default_color="white",
                        colors=["white", "black"]
                    ),
                    eraser=gr.Eraser(default_size=15),
                    sources=["upload", "webcam", "clipboard"],
                    transforms=["crop"],
                    crop_size=None
                )
                
                reference_image = gr.Image(
                    label="referenceImage (服装/物品图片)",
                    type="pil",
                    height=300
                )
                
                # 保留传统的 mask 上传选项（可选）
                mask_image = gr.Image(
                    label="maskImage (可选) - 传统mask上传",
                    type="pil",
                    height=200
                )
                
                gr.Markdown("""
                **🖌️ 人物图片编辑说明:**
                - **上传人物图片**: 点击上传或拖拽图片
                - **涂抹mask区域**: 用白色笔刷涂抹要替换的部分
                - **橡皮擦**: 擦除错误的涂抹
                - **只有maskType设定为IMAGE时才生效**
                """)
                
                gr.Markdown("## ⚙️ Mask配置")
                
                mask_type = gr.Dropdown(
                    choices=["GARMENT", "PROMPT", "IMAGE"],
                    value="GARMENT",
                    label="maskType"
                )
                
                # 服装类别选项
                garment_class = gr.Dropdown(
                    choices=[
                        "UPPER_BODY", "LOWER_BODY", "FULL_BODY", "FOOTWEAR",
                        "LONG_SLEEVE_SHIRT", "SHORT_SLEEVE_SHIRT", "NO_SLEEVE_SHIRT", "OTHER_UPPER_BODY",
                        "LONG_PANTS", "SHORT_PANTS", "OTHER_LOWER_BODY",
                        "LONG_DRESS", "SHORT_DRESS", "FULL_BODY_OUTFIT", "OTHER_FULL_BODY",
                        "SHOES", "BOOTS", "OTHER_FOOTWEAR"
                    ],
                    value="UPPER_BODY",
                    label="garmentClass (GARMENT模式)"
                )
                
                with gr.Row():
                    mask_shape_garment = gr.Dropdown(
                        choices=["DEFAULT", "CONTOUR", "BOUNDING_BOX"],
                        value="DEFAULT",
                        label="maskShape (GARMENT)"
                    )
                    
                    mask_shape_prompt = gr.Dropdown(
                        choices=["DEFAULT", "CONTOUR", "BOUNDING_BOX"],
                        value="DEFAULT",
                        label="maskShape (PROMPT)"
                    )
                
                mask_prompt = gr.Textbox(
                    label="maskPrompt (PROMPT模式)",
                    placeholder="例如: hat, sunglasses, necklace",
                    value=""
                )
                
                gr.Markdown("## 👔 garmentStyling")
                
                with gr.Row():
                    long_sleeve_style = gr.Dropdown(
                        choices=["DEFAULT", "SLEEVE_DOWN", "SLEEVE_UP"],
                        value="DEFAULT",
                        label="longSleeveStyle"
                    )
                    
                    tucking_style = gr.Dropdown(
                        choices=["DEFAULT", "UNTUCKED", "TUCKED"],
                        value="DEFAULT",
                        label="tuckingStyle"
                    )
                    
                    outer_layer_style = gr.Dropdown(
                        choices=["DEFAULT", "CLOSED", "OPEN"],
                        value="DEFAULT",
                        label="outerLayerStyle"
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("## 🛡️ maskExclusions")
                
                with gr.Row():
                    preserve_body_pose = gr.Dropdown(
                        choices=["DEFAULT", "ON", "OFF"],
                        value="DEFAULT",
                        label="preserveBodyPose"
                    )
                    
                    preserve_hands = gr.Dropdown(
                        choices=["DEFAULT", "ON", "OFF"],
                        value="DEFAULT",
                        label="preserveHands"
                    )
                    
                    preserve_face = gr.Dropdown(
                        choices=["DEFAULT", "ON", "OFF"],
                        value="DEFAULT",
                        label="preserveFace"
                    )
                
                gr.Markdown("## 🎨 imageGenerationConfig")
                
                merge_style = gr.Dropdown(
                    choices=["BALANCED", "SEAMLESS", "DETAILED"],
                    value="BALANCED",
                    label="mergeStyle"
                )
                
                # 强制开启returnMask，不显示给用户
                # return_mask = gr.Checkbox(
                #     label="returnMask",
                #     value=True
                # )
                
                gr.Markdown("**returnMask: 已强制开启** ✅")
                
                with gr.Row():
                    number_of_images = gr.Slider(
                        minimum=1,
                        maximum=1,
                        value=1,
                        step=1,
                        label="numberOfImages"
                    )
                    
                    quality = gr.Dropdown(
                        choices=["standard", "premium"],
                        value="standard",
                        label="quality"
                    )
                
                with gr.Row():
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=6.5,
                        step=0.1,
                        label="cfgScale"
                    )
                    
                    seed = gr.Number(
                        label="seed (负数为随机)",
                        value=-1,
                        precision=0
                    )
                
                gr.Markdown("## 🚀 生成")
                
                generate_btn = gr.Button(
                    "生成虚拟换装",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("## 📊 结果")
                
                result_image = gr.Image(
                    label="换装结果",
                    format="png",
                    height=400
                )
                
                # 添加mask预览
                mask_image_output = gr.Image(
                    label="Mask输出预览",
                    format="png",
                    height=300
                )
                
                result_info = gr.Markdown(
                    label="详细信息"
                )
        
        if 0:
            # 添加日志信息显示区域
            with gr.Row():
                gr.Markdown("## 📋 日志信息")
            
            with gr.Row():
                gr.Markdown(f"""
                **日志保存位置:** `gradio_logs/`
                
                **目录结构:**
                ```
                gradio_logs/
                ├── gradio_vto_YYYYMMDD.log          # 全局日志文件
                ├── 20250709_070123_456/             # 会话文件夹 (session_id)
                │   ├── source.png                   # 用户上传的人物图片
                │   ├── reference.png                # 用户上传的参考图片
                │   ├── mask_input.png               # 用户上传的Mask图片 (可选)
                │   ├── result.png                   # API返回的结果图片
                │   ├── mask_output.png              # API返回的Mask图片 (可选)
                │   └── payload.json                 # 请求和响应数据
                └── 20250709_070456_789/             # 另一个会话文件夹
                    ├── source.png
                    ├── reference.png
                    ├── result.png
                    └── payload.json
                ```
                
                **会话ID格式:** `YYYYMMDD_HHMMSS_mmm` (年月日_时分秒_毫秒)
                
                每次请求都会创建独立的会话文件夹，包含完整的输入输出数据，方便追踪和调试。
                """)
        
        
        # 绑定生成按钮
        generate_btn.click(
            fn=demo_instance.virtual_try_on,
            inputs=[
                source_image, reference_image, mask_image,
                mask_type, garment_class, mask_shape_garment, mask_shape_prompt,
                mask_prompt, long_sleeve_style, tucking_style, outer_layer_style,
                preserve_body_pose, preserve_hands, preserve_face,
                merge_style,
                number_of_images, quality, cfg_scale, seed
            ],
            outputs=[result_image, mask_image_output, result_info],
            concurrency_limit=3,
        )
        
        # 添加示例
        gr.Markdown("## 📝 API参数说明")
        gr.Markdown("""
        ### maskType 三种模式:
        
        **1. GARMENT模式** (推荐用于服装换装)
        - 选择具体的 garmentClass
        - 系统自动识别服装区域
        - 可调节 garmentStyling 参数
        
        **2. PROMPT模式** (推荐用于配饰)
        - 使用 maskPrompt 文字描述要替换的物品
        - 适合帽子、眼镜、项链等配饰
        - 例如: "hat", "sunglasses", "necklace"
        
        **3. IMAGE模式** (高级用户)
        - 上传自定义 maskImage 或使用笔刷工具绘制
        - 精确控制替换区域
        - **笔刷编辑功能**:
          - 白色区域 = 要替换的部分
          - 黑色区域 = 要保留的部分
          - 支持上传图片后再用笔刷编辑
          - 可调节笔刷大小和橡皮擦
        
        ### 主要参数:
        - **mergeStyle**: BALANCED(平衡), SEAMLESS(无缝), DETAILED(细节)
        - **cfgScale**: 控制生成图像与提示的匹配度 (1.0-10.0)
        - **maskExclusions**: 控制是否保留原图的特定部分
        - **garmentStyling**: 服装样式控制 (longSleeveStyle, tuckingStyle, outerLayerStyle)
        - **imageGenerationConfig**: 图像生成配置 (numberOfImages, quality, cfgScale, seed)
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8003,
        # root_path="/nova_canvas_vto",
        show_api=False,
        share=False,
        debug=False,
    )
