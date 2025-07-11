# Nova Canvas 虚拟换装 Demo

这是一个基于 Amazon Nova Canvas API 的虚拟换装演示应用，提供了完整的 Web UI 界面来体验虚拟试衣功能。

## 🚀 快速开始

### 启动应用

```bash
uv run ./gradio_vto_demo.py
```

应用将在 `http://localhost:8003` 启动。

## ✨ 主要功能

### 🎨 虚拟换装
- **人物图片上传**: 支持上传、摄像头拍摄、剪贴板粘贴
- **服装参考图片**: 上传要试穿的服装图片
- **实时预览**: 生成换装效果图和 Mask 预览
- **多种 Mask 模式**: 支持服装自动识别、文字描述、手动绘制

### 🖌️ 智能 Mask 编辑
- **笔刷工具**: 直接在人物图片上涂抹要替换的区域
- **橡皮擦**: 修正错误的涂抹区域
- **可调节笔刷**: 支持不同大小的笔刷和橡皮擦
- **自动反色处理**: 自动处理 Mask 图像以符合 API 要求

### 📊 完整参数控制
- **Mask 类型**: GARMENT（服装）、PROMPT（文字描述）、IMAGE（图像）
- **服装样式**: 长袖样式、塞入样式、外层样式
- **保留选项**: 身体姿势、手部、面部保留控制
- **生成配置**: 图像质量、CFG Scale、随机种子等

### 📁 自动日志记录
- **会话管理**: 每次请求创建独立的会话文件夹
- **完整记录**: 保存输入图片、输出结果、API 请求数据
- **调试友好**: 便于问题追踪和参数调优

## 🛠️ 技术特性

### API 集成
- 基于 Amazon Bedrock Nova Canvas v1.0
- 支持所有官方 API 参数
- 自动错误处理和重试机制

### 用户界面
- 基于 Gradio 构建的现代化 Web UI
- 响应式设计，支持移动端访问
- 实时参数调节和预览

### 数据管理
- 自动创建日志目录结构
- 会话 ID 格式：`YYYYMMDD_HHMMSS_mmm`
- JSON 格式保存请求和响应数据

## 📋 使用说明

### 1. 基础换装流程

1. **上传人物图片**: 在 "sourceImage" 区域上传或拍摄人物照片
2. **上传服装图片**: 在 "referenceImage" 区域上传要试穿的服装
3. **选择 Mask 模式**: 
   - **GARMENT**: 自动识别服装区域（推荐）
   - **PROMPT**: 使用文字描述要替换的物品
   - **IMAGE**: 手动绘制或上传 Mask 图片
4. **调整参数**: 根据需要调整生成参数
5. **点击生成**: 等待 API 处理并查看结果

### 2. Mask 模式详解

#### GARMENT 模式（推荐用于服装）
- 选择具体的服装类别（上身、下身、全身、鞋类等）
- 系统自动识别对应的服装区域
- 可调节服装样式参数（长袖、塞入、外层等）

#### PROMPT 模式（推荐用于配饰）
- 使用文字描述要替换的物品
- 适合帽子、眼镜、项链等配饰
- 示例：`hat`, `sunglasses`, `necklace`

#### IMAGE 模式（高级用户）
- **笔刷绘制**: 直接在人物图片上用白色笔刷涂抹要替换的区域
- **传统上传**: 上传黑白 Mask 图片（白色=替换区域，黑色=保留区域）
- **精确控制**: 最精确的区域控制方式

### 3. 高级参数

#### 合并样式 (mergeStyle)
- **BALANCED**: 平衡模式，兼顾自然度和细节
- **SEAMLESS**: 无缝模式，注重边缘融合
- **DETAILED**: 细节模式，保留更多纹理细节

#### 保留选项 (maskExclusions)
- **preserveBodyPose**: 保留身体姿势
- **preserveHands**: 保留手部细节
- **preserveFace**: 保留面部特征

#### 生成配置
- **quality**: standard（标准）/ premium（高级）
- **cfgScale**: 控制生成图像与提示的匹配度（1.0-10.0）
- **seed**: 随机种子，负数表示随机生成

## 📁 日志结构

```
gradio_logs/
├── gradio_vto_YYYYMMDD.log          # 全局日志文件
├── 20250711_030123_456/             # 会话文件夹
│   ├── source.png                   # 输入的人物图片
│   ├── reference.png                # 输入的参考图片
│   ├── mask_input.png               # 输入的 Mask 图片（可选）
│   ├── result.png                   # API 返回的结果图片
│   ├── mask_output.png              # API 返回的 Mask 图片
│   └── payload.json                 # 完整的请求和响应数据
└── 20250711_030456_789/             # 另一个会话文件夹
    └── ...
```

## 🔧 环境要求

### Python 依赖
- gradio
- boto3
- Pillow (PIL)
- numpy

### AWS 配置
- 需要配置 AWS 凭证（通过 AWS CLI 或环境变量）
- 需要 Amazon Bedrock 访问权限
- 模型 ID: `amazon.nova-canvas-v1:0`
- 区域: `us-east-1`

## 🎯 使用技巧

### 获得最佳效果
1. **人物图片**: 使用高清、光线良好的正面或侧面照片
2. **服装图片**: 选择清晰的服装产品图，最好是平铺或模特展示
3. **Mask 绘制**: 涂抹时要覆盖完整的替换区域，避免遗漏边缘
4. **参数调节**: 
   - 对于复杂服装，使用 DETAILED 合并样式
   - 对于简单替换，使用 SEAMLESS 模式
   - 调整 CFG Scale 来平衡真实度和创意度

### 常见问题解决
- **生成效果不理想**: 尝试调整 cfgScale 值或更换 mergeStyle
- **边缘不自然**: 使用 SEAMLESS 模式或手动绘制更精确的 Mask
- **保留原有特征**: 启用相应的 maskExclusions 选项

## 📝 更新日志

- 支持 ImageEditor 组件的笔刷绘制功能
- 自动 Mask 反色处理
- 完整的会话日志记录系统
- 支持所有 Nova Canvas API 参数
- 优化的用户界面和交互体验

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 📄 许可证

本项目遵循相应的开源许可证。
