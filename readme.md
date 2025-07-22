## 1. Crea un entorno virtual (opcional pero recomendado):
```
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

## 2. Crea la estructura del proyecto
```
langchain_project/
│
├── venv/                 # (si usas entorno virtual)
├── main.py               # Script principal
├── requirements.txt      # Lista de dependencias
└── .env                  # Variables de entorno (opcional, para API keys)
```

## 3. Instala LangChain y dependencias básicas
```
requirements.txt
pip install -r requirements.txt

langchain
openai
python-dotenv
```

## 4. Configura tus claves de API en .env
```
OPENAI_API_KEY=sk-xxxxxxx
```

## 5. Ejecutar proyecto python
```
python app.py
```

## 5. Ejecutar CLI Langgraph
```
https://langchain-ai.github.io/langgraph/concepts/application_structure/
http://127.0.0.1:2024
```