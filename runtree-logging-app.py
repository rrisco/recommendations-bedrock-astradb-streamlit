import streamlit as st
import pandas as pd
import io
import boto3
import base64
import json
import time
import pandas as pd
from PIL import Image
from datetime import datetime
from datetime import timezone 
from astrapy.db import AstraDB, AstraDBCollection
from langchain_astradb import AstraDBVectorStore
from langchain_core.messages import HumanMessage
from langsmith.run_trees import RunTree

# It must be called at the start 
st.set_page_config(page_title="Búsqueda de accesorios", layout="wide")

# Max size (width or height) we allowed for the image
IMAGE_MAX_SIZE = 800

###################################
# Cache the connection to AstraDB #
###################################
@st.cache_resource(show_spinner='Conectando con Astra')
def astra_connection():
    astra = AstraDB(
        token=st.secrets['ASTRA_DB_TOKEN'],
        api_endpoint=st.secrets['ASTRA_DB_ENDPOINT']
    )
    return astra
astra_conn = astra_connection()

@st.cache_resource()
def catalog_collection():
    catalog = AstraDBCollection(
        collection_name="watches_catalog", 
        astra_db=astra_conn
    )
    return catalog

#######################
# Set AWS credentials #
#######################

# required aws_cli
@st.cache_resource()
def s3_client():
    return boto3.client('s3')

@st.cache_resource()
def bedrock_runtime():
    return boto3.client('bedrock-runtime')

#############################
# Set Bedrock configuration #
#############################

ls_provider= "AWS"
embedding_modelId = "amazon.titan-embed-image-v1"
embedding_output_length = 1024
modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

##########
# Prompt #
##########
category = 'wrist watches'
prompt = """
    Identify the following product in the image provided. Your answer should be in Spanish.
    Product Category: {product_category}
    
    Return an enhanced description of the product based on the image for better search results.
    Do not include any specific details that can not be confirmed from the image such as the quality of materials, other color options, or exact measurements.
    """

#######################
# Application methods #
#######################

# LangSmith runtree pipeline
def ls_pipeline():
    if 'ls_pipeline' not in st.session_state or st.session_state.ls_pipeline is None:
        st.session_state.ls_pipeline = RunTree(
            name="Image Pipeline", 
            run_type="llm",
            inputs={"question": prompt},
            model=modelId,
            provider=ls_provider
        )
    return st.session_state.ls_pipeline

def get_claude_messages_content(input_image): 
    messages_content = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": input_image,
                    }
                },
                {
                    "type": "text",
                    "text": prompt.format(product_category=category)
                }
            ]
        }
    ]
    return messages_content

# Creation of the Langchain log message
def log_image_model(messages: list, input_tokens: int, output_tokens: int, result: str):
    return {
        "choices": [
            {
                "role" : "assistant",
                "message" : result 
            }
        ],
        "created": time.time(),
        "model": modelId,
        "ls_provider": ls_provider, 
        "ls_model_name": modelId,
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }

def log_embedding_model(messages: list, result: str):
    return {
        "choices": [
            {
                "role" : "Embedding model",
                "message" : result 
            }
        ],
        "created": time.time(),
        "model": embedding_modelId,
        "ls_provider": ls_provider, 
        "ls_model_name": embedding_modelId,
        "usage_metadata": {
            "input_tokens": st.session_state.prompt_tokens,
            "output_tokens": st.session_state.embedding_tokens,
            "total_tokens": st.session_state.embedding_tokens + st.session_state.prompt_tokens,
        },
    }

# Methods to call the LLM
def get_claude_description(image):
    contentType = 'application/json'
    accept = 'application/json'

    messages = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", 
        "max_tokens": 1024,
        "temperature" : 0,
        "messages": get_claude_messages_content(image)
    })
    
    output = bedrock_runtime().invoke_model(
        modelId=modelId,
        contentType=contentType,
        accept=accept,
        body=messages
    )
    final_response = json.loads(output.get('body').read())

    st.session_state.prompt_tokens = final_response['usage']['input_tokens']
    st.session_state.image_tokens = final_response['usage']['output_tokens']
    st.session_state.claude_tokens = st.session_state.prompt_tokens + st.session_state.image_tokens

    return final_response['content'][0]['text']

def generate_titan_embedding(image, text):
    contentType = 'application/json'
    accept = 'application/json'

    embedding_messages = json.dumps({
        "inputText": text,
        "inputImage": image,
        "embeddingConfig": {
            "outputEmbeddingLength": embedding_output_length
        }
    })

    titan_response = bedrock_runtime().invoke_model(
        modelId=embedding_modelId,
        contentType=contentType,
        accept=accept,
        body=embedding_messages
    )

    final_response = json.loads(titan_response.get('body').read())

    st.session_state.embedding_tokens = final_response['inputTextTokenCount']

    return final_response['embedding']

# Vector seach in Astra
def vector_search(query_vector):
    documents = catalog_collection().vector_find(
            query_vector,
            limit=5,
            fields=["_id", "file_path", "brand", "product_name", "$vector"],  # remember the dollar sign (reserved name)
            include_similarity=True,
        )
    similar_items = pd.DataFrame(documents)
    return similar_items

# Image Handling methods
def get_full_image_path(id_name):
    path = f"https://aws-speed-date.s3.amazonaws.com/{id_name}"
    return path

def resize_image(img_data):
    image = Image.open(img_data)
    width, height = image.size
    if width > IMAGE_MAX_SIZE or height > IMAGE_MAX_SIZE:
        ratio = min(IMAGE_MAX_SIZE/width, IMAGE_MAX_SIZE/height)
        newsize = int(width*ratio), int(height*ratio)
        image = image.resize(newsize)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr 

def image_to_base64():
    raw_image = st.session_state.uploaded_image
    return base64.b64encode(raw_image.getvalue()).decode("utf-8")

# UI components callbacks
def on_image_selected():
    st.session_state.stage = 0

def on_get_description():
    image = image_to_base64()
    st.session_state.description = get_claude_description(image)

    inputs = [
        HumanMessage(
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt.format(product_category=category)
                },
            ]
        )
    ]
    st.session_state.child_llm_run = ls_pipeline().create_child(
        name="Claude3 Call",
        run_type="llm",
        inputs={"messages": inputs},
        tags=["Claude3"],
        extra={ "metadata" : 
            { "ls_model_name": modelId,
                "ls_provider": ls_provider }}
    )
    st.session_state.child_llm_run.end(outputs=log_image_model(inputs,
                    st.session_state.prompt_tokens,
                    st.session_state.image_tokens, 
                    st.session_state.claude_tokens))
    st.session_state.child_llm_run.post()
    st.session_state.stage = 2

def on_get_embedding():
    image = image_to_base64()
    st.session_state.embedding = generate_titan_embedding(image, st.session_state.description)
    
    inputs = [
        HumanMessage(
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    },
                },
                {
                    "type": "text",
                    "text": st.session_state.description
                },
            ]
        )
    ]

    st.session_state.child_embedding_run = ls_pipeline().create_child(
        name="Tital Call",
        run_type="llm",
        inputs={"messages": inputs},
        tags=["embedding"],
        extra={ "metadata" : 
            { "ls_model_name": embedding_modelId,
                "ls_provider": ls_provider }}
    )
    st.session_state.child_embedding_run.end(outputs=log_embedding_model(inputs,  
                        st.session_state.embedding))
    st.session_state.child_embedding_run.post()
    st.session_state.stage = 3

def on_vector_search():
    results = vector_search(st.session_state.embedding)
    st.session_state.similar_vectors = results
    st.session_state.stage = 4

# Session initialization
if 'prompt_tokens' not in st.session_state:
    st.session_state.prompt_tokens = 0
if 'image_tokens' not in st.session_state:
    st.session_state.image_tokens = 0
if 'image_out_tokens' not in st.session_state:
    st.session_state.image_out_tokens = 0
if 'claude_tokens' not in st.session_state:
    st.session_state.claude_tokens = 0
if 'claude_out_tokens' not in st.session_state:
    st.session_state.claude_out_tokens = 0
if 'embedding_tokens' not in st.session_state:
    st.session_state.embedding_tokens = 0
if 'stage' not in st.session_state:
    st.session_state.stage = 0
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'dynamic_message' not in st.session_state:
    st.session_state.dynamic_message = ""
if 'description' not in st.session_state:
    st.session_state.description = None
if 'embedding' not in st.session_state:
    st.session_state.embedding = None
if 'similar_vectors' not in st.session_state:
    st.session_state.similar_vectors = None
if 'child_llm_run' not in st.session_state:
    st.session_state.child_llm_run = None
if 'child_embedding_run' not in st.session_state:
    st.session_state.child_embedding_run = None

# Stages
# 0 - wainting for image
# 1 - Image loaded
# 2 - Got description
# 3 - Got embedding
# 4 - Vector search

st.title('Recomendaciones de accesorios')
st.markdown("""La Inteligencia Artificial Generativa se considera como el motor de la siguiente revolución industrial.  
¡Te ayudaré a encontrar rejojes similares al que tienes!""")

uploaded_image = st.sidebar.file_uploader(label="Imagen para buscar, jpg o png", key="image_file", on_change=on_image_selected)
if st.session_state.image_file is not None:
    resized = resize_image(uploaded_image)
    st.session_state.uploaded_image = resized

if st.session_state.uploaded_image:
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.markdown("""Esta es la imagen que buscaré:""")
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, output_format='auto')
        if st.session_state.stage == 0:
            st.session_state.stage = 1

if st.session_state.stage == 1:
    st.session_state.dynamic_message = """Ahora pidamos a "Claude 3 Sonnet" en Bedrock la descripción de nuestro artículo."""
    st.markdown(st.session_state.dynamic_message)
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.button(label="Obtener descripción", on_click=on_get_description)

if st.session_state.stage > 1:  
    st.write(st.session_state.description)

if st.session_state.stage == 2:
    st.markdown("""El siguiente paso es crear un embedding multimodal usando la imagen y la descripción obtenida, "usando Bedrock Tital v1""")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.button(label="Obtener embedding multimodal", on_click=on_get_embedding)

if st.session_state.stage == 3:
    st.markdown("""Este es el embedding obtenido del model "Titan v1". Este embedding nos permite hacer una búsqueda de similitud en una base vectorial, para encontrar los productos más relevantes.""")
    df_to_show = pd.DataFrame(st.session_state.embedding).transpose()
    st.dataframe(df_to_show)
    st.markdown("¡Ahora búsquemos los productos para recomendar a nuestro cliente!")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.button(label="Ejecutar búsqueda vectorial", on_click=on_vector_search)

    final_response = {
        "response": [
            {
                "message": {
                    "role" : "assistant",
                    "content" : st.session_state.description
                }
            }
        ]
    }
    ls_pipeline().end(outputs={"answer": final_response["response"][0]["message"]["content"]})
    ls_pipeline().post()
    # Avoid sending more events to the same log entry if a new image is searched
    st.session_state.ls_pipeline = None

if st.session_state.stage == 4:
    left_co, second_co, cent_co, fourth_co, last_co = st.columns(5)
    cell_height = "100px"
    with left_co:
        st.write(f"Marca: {st.session_state.similar_vectors.loc[0, 'brand']}")
        st.markdown(f"""<p style="height: {cell_height}; overflow-y: auto">Producto: {st.session_state.similar_vectors.loc[0, 'product_name']}</p>""",
    unsafe_allow_html=True,)
        st.image(get_full_image_path(st.session_state.similar_vectors.loc[0, 'file_path']), output_format='auto')
    with second_co:
        st.write(f"Marca: {st.session_state.similar_vectors.loc[1, 'brand']}")
        st.markdown(f"""<p style="height: {cell_height}; overflow-y: auto">Producto: {st.session_state.similar_vectors.loc[1, 'product_name']}</p>""",
    unsafe_allow_html=True,)
        st.image(get_full_image_path(st.session_state.similar_vectors.loc[2, 'file_path']), output_format='auto')
    with cent_co:
        st.write(f"Marca: {st.session_state.similar_vectors.loc[2, 'brand']}")
        st.markdown(f"""<p style="height: {cell_height}; overflow-y: auto">Producto: {st.session_state.similar_vectors.loc[2, 'product_name']}</p>""",
    unsafe_allow_html=True,)
        st.image(get_full_image_path(st.session_state.similar_vectors.loc[2, 'file_path']), output_format='auto')
    with fourth_co:
        st.write(f"Marca: {st.session_state.similar_vectors.loc[3, 'brand']}")
        st.markdown(f"""<p style="height: {cell_height}; overflow-y: auto">Producto: {st.session_state.similar_vectors.loc[3, 'product_name']}</p>""",
    unsafe_allow_html=True,)
        st.image(get_full_image_path(st.session_state.similar_vectors.loc[3, 'file_path']), output_format='auto')
    with last_co:
        st.write(f"Marca: {st.session_state.similar_vectors.loc[4, 'brand']}")
        st.markdown(f"""<p style="height: {cell_height}; overflow-y: auto">Producto: {st.session_state.similar_vectors.loc[4, 'product_name']}</p>""",
    unsafe_allow_html=True,)
        st.image(get_full_image_path(st.session_state.similar_vectors.loc[4, 'file_path']), output_format='auto')

