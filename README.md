# Construye una aplicación de recomendación de productos basada en un artículo del cliente

Bienvenido al taller para desplegar tu propia aplicación de recomendaciones, usando DataStax AstraDB y AWS Bedrock, con **Anthropic Claude 3 Sonnet** y **Amazon Titan G1** (Multimodal embeddings).

Se aprovecha el uso de [DataStax RAGStack](https://docs.datastax.com/en/ragstack/docs/index.html), que es una colección del mejor software open-source para facilitar la implementación. Además de la aplicación como tal, usaremos [LangSmith](https://www.langchain.com/langsmith), en este ejercicio no se invocan los wrappers de Langchain, si no que se demuestra el uso de RunTree API de LangSmith para los casos de modelos que no estén ya incluidos en el SDK.

Qué queremos aprender:
- Cómo aprovechar las capacidades de interpretación semántica de los modelos de GenAI (modelos LLM) disponibles, para generar los elementos necesarios para realizar una búsqueda de similitud semántica.
- Aprovechar las capacidades de búsqueda vectorial de [AstraDB](https://astra.datastax.com) para obtener resultados relevantes de acuerdo al objeto buscado. 
- Como usar [Streamlit](https://streamlit.io) para desplegar facilmente tu app. 

## Prerequisitos
Se asume que ya se tiene acceso a: 
1. [Una cuenta Github](https://github.com)

Adicionalmente se necesitan accesos a los siguientes sistemas de forma gratuita:
1. [DataStax Astra DB](https://astra.datastax.com) (se puede crear una cuenta a través de Github)
2. [LangSmith](https://www.langchain.com/langsmith) (se puede crear una cuenta a través de Github)
3. [Streamlit](https://streamlit.io) si deseas desplegar tu aplicación (se puede crear una cuenta a través de Github)

Servicios con contratación:
1. [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html): el uso de estos modelos requiere una cuenta con AWS, con cargo a una tarjeta de crédito, así como solicitar la habilitación en la cuenta de AWS del modelo Anthropic Claude 3 Sonnet. Sin embargo, es posible reproducir este workshop con otros modelos de interpretación semántica de imágenes y generación de embeddings multimodales de otros proveedores, por ejemplo, OpenAI.
2. [AWS S3](https://aws.amazon.com/es/s3/): El servicio de almacenamiento s3 será será usado para contener las imágenes de catálogo. 

## Configuración del entorno
Para el funcionamiento de la aplicación es necesario lo siguiente:
1. Instalación de [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) y su configuración. La configuración inicial puede hacerse en la terminal con ```aws configure```, donde deberán suministrarse los datos de credenciales de la cuenta de AWS así como la región preferida.
2. Se recomienda crear un entorno virtual de Python (virtual environment) para mantener todos los paquetes dentro de ese entorno virtual. 
3. Instalación de las librerías adicionales de Python, para esto hay que ejecutar el paquete ```pip install -r requirements.txt``` - este archivo se encuentra en el repositorio.
4. Creación de un archivo ```secrets.toml``` dentro de la carpeta .streamlit, este archivo requiere los datos de configuración y conexión con AstraDB. El repositorio contiene un archivo ```secrets_example.txt``` como muestra.

## Preparación de los datos
Para poder hacer la búsqueda vectorial necesitamos tener un catálogo de productos, en este caso en particular usamos un [dataset público de relojes de pulsera](https://www.kaggle.com/datasets/mathewkouch/a-dataset-of-watches). El dataset contiene las imágenes, marca, nombre del producto y precio. Para cargar el dataset en AstraDB se incluye en el repositorio un Jupyter lab ```/JupyterLab/SeedCollections.ipyn```, el cual requiere que se suministren las credenciales de AWS y AstraDB. Las imágenes se almecenaron directamente en s3.

## Uso de LangSmith
En el repositorio se incluyeron dos scripts de Python con la aplicación, uno hace uso del decorador @traceable para hacer el registro del log en LangSmith. El segundo hace uso del API runtree para hacer los registros en el log, este modo requiere la creación de un pipeline principal o "padre", y la creación de un proceso "hijo" el cual se suscribe al pipeline pricipal, para finalmente mandar los datos a LangSmith.

LangSmith requiere que sus datos de autenticación y configuración existan en variables de entorno, las cuales se establecen en los scripts bash del repositorio. Al ejecutar alguno de esos scripts se establece el API key de LangSmith así como la identificación del proyecto, para después ejecutar el programa en Python. Estos scripts deben editarse con los datos correctos para que las llamadas a los LLM se registren de forma correcta. 

También es necesario [crear la configuración de los modelos](https://docs.smith.langchain.com/how_to_guides/tracing/calculate_token_based_costs) con sus datos de costo correctos dentro del panel de administración de LangSmith.
En el script de Python pueden verificarse cuales son, los modelos que se ocupan y su configuración son las siguientes:

### Codificación para LangSmith
Archivo ```traceable-decorator-app.py```
En esta versión de utiliza el decorador ```@traceable``` antes de los métodos que devuelven el objeto para el log, automaticamente el SDK de LangSmith manda los datos, usando el API key que se haya proporcionado en las variables de entorno, dentro del script bash. 

En este caso cada llamada al LLM se registra como un registro independiente en el panel de LangSmith.

Archivo ```runtree-logging-app.py```
En esta versión se utiliza la clase runtree para crear un objeto "Runtree" ```ls_pipeline```, al cual se suscriben los procesos hijos, en este caso uno para la llamada a Claude3 y otro para la llamada a Titan G1. Se ocupan los métodos ```end``` y ```post``` para el registro y envío de los eventos a LangSmith. 

En el panel de LangSmith las dos llamadas aparecerán como parte de un mismo registro.

## Listos para comenzar a usar la aplicación
Para ejecutar la aplicación lo más sencillo es ejecutar alguno de los dos scripts bash, ```app-decorator.sh``` o ```app-runtree.sh```. 

Si deseas ejecutar la aplicación sin usar LangSmith puede ejecutar el programa en la terminal:

```streamlit run traceable-decorator-app.py``` o ```streamlit run `runtree-logging-app.py```
