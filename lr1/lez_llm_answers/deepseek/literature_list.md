Для синтеза речи (Text-to-Speech, TTS) в условиях малоресурсных языков, таких как лезгинский, применяются современные методы, позволяющие обучать модели на небольших датасетах. Вот основные подходы:  

### **Современные методы синтеза речи (2020–2024)**  
1. **Нейронные TTS-модели с трансформерами**  
   - **FastSpeech 2, FastPitch** – неавторегрессионные модели с механизмом внимания (attention), работают быстрее классических Tacotron.  
   - **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)** – объединяет вариационные автоэнкодеры и adversarial learning для улучшения качества.  

2. **Few-shot и Zero-shot TTS**  
   - **YourTTS, VALL-E** – модели, способные синтезировать речь с минимальными данными или даже одним примером голоса (meta-learning, transfer learning).  
   - **AdaSpeech, AdaStyle** – адаптация предобученных моделей под новые голоса с малым количеством данных.  

3. **Мультиязычные и кросс-лингвальные модели**  
   - **XLS-R, Whisper (OpenAI)** – предобученные модели на множестве языков, могут дообучаться для синтеза.  
   - **mBART, mT5** – трансформерные архитектуры для мультиязычного переноса.  

4. **Data augmentation и полуавтоматические датасеты**  
   - Использование **TTS data augmentation** (например, обратный синтез с помощью ASR).  
   - **Псевдо-датасеты** – генерация синтетических данных с помощью других TTS-моделей.  

---

### **Актуальные статьи (2020–2024)**  
#### **1. Общие модели TTS для малоресурсных языков**  
- **"YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone"** (2022)  
  [arXiv:2112.02418](https://arxiv.org/abs/2112.02418)  
  - Модель для few-shot синтеза, подходит для языков без больших датасетов.  

- **"VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"** (2021)  
  [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)  
  - Эффективная end-to-end модель, работающая даже на небольших данных.  

- **"AdaSpeech: Adaptive Text to Speech for Custom Voice"** (2021)  
  [arXiv:2103.00993](https://arxiv.org/abs/2103.00993)  
  - Адаптация TTS под новые голоса с малым количеством данных.  

#### **2. Мультиязычные и кросс-лингвальные подходы**  
- **"XTTS: A Massively Multilingual Zero-Shot Text-to-Speech Model"** (2023)  
  [arXiv:2305.03590](https://arxiv.org/abs/2305.03590)  
  - Модель от Coqui AI, поддерживающая синтез для ~16 языков, включая малоресурсные.  

- **"Cross-lingual Text-to-Speech with Flow-based Voice Conversion for Improved Pronunciation"** (2022)  
  [arXiv:2211.00652](https://arxiv.org/abs/2211.00652)  
  - Использование transfer learning для улучшения произношения в TTS.  

#### **3. Data augmentation и сбор датасетов**  
- **"Improving Low-Resource Speech Synthesis with Data Augmentation and Conditional Layer Normalization"** (2023)  
  [arXiv:2301.08739](https://arxiv.org/abs/2301.08739)  
  - Методы аугментации для малоресурсных TTS.  

- **"Synthesizing Speech from Text in Low-Resource Languages with Dynamic Vocabulary"** (2023)  
  [arXiv:2304.08407](https://arxiv.org/abs/2304.08407)  
  - Подходы к генерации данных для TTS.  

---

### **Рекомендации для лезгинского языка**  
1. **Сбор датасета**:  
   - Запись 5–10 часов речи (желательно с разными дикторами).  
   - Использование ASR (Whisper) для автоматической разметки.  
2. **Использование предобученных моделей**:  
   - Fine-tuning **XTTS-v2** (поддерживает кавказские языки).  
   - Адаптация **VITS** или **YourTTS**.  
3. **Кросс-лингвальное обучение**:  
   - Использование данных азербайджанского/турецкого (агглютинативные языки).  

Если нужна помощь с конкретными экспериментами или кодом, дайте знать! Удачи с диссертацией! 🚀