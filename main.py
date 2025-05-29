# IMPORTACIÓN DE LIBRERÍAS NECESARIAS

# Estas son las herramientas que necesitamos para que nuestro programa funcione

import PyPDF2          # Librería para leer y extraer texto de archivos PDF
import json            # Librería para trabajar con datos en formato JSON (como un diccionario organizado)
import os              # Librería para interactuar con el sistema operativo (archivos, carpetas, variables)
from openai import OpenAI     # Librería para conectarnos con la inteligencia artificial de OpenAI
from dotenv import load_dotenv # Librería para cargar variables secretas desde un archivo .env
import re              # Librería para trabajar con expresiones regulares (patrones de texto)


# CONFIGURACIÓN INICIAL DEL PROGRAMA


# Cargar variables de entorno desde archivo .env
# Esto nos permite mantener nuestra clave API secreta y segura
# En lugar de escribir la clave directamente en el código, la guardamos en un archivo .env
load_dotenv()

# Inicializar cliente de OpenAI con la API key
# Esto crea una "conexión" con el servicio de inteligencia artificial
# La clave API es como una contraseña que nos permite usar el servicio
# base_url especifica que usaremos OpenRouter en lugar del API directo de OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")


# FUNCIÓN 1: EXTRAER TEXTO DE UN PDF

def extraer_texto_pdf(ruta_pdf):
    """
    Esta función toma un archivo PDF y extrae todo su texto para poder analizarlo.
    Modificada para devolver SOLO la primera página (fácil de cambiar para procesar más páginas después).
    """
    texto_completo = ""
    try:
        with open(ruta_pdf, 'rb') as archivo:
            lector_pdf = PyPDF2.PdfReader(archivo)
            # SOLO PRIMERA PÁGINA (fácil de cambiar para más páginas)
            if len(lector_pdf.pages) > 0:
                pagina = lector_pdf.pages[0]
                texto_pagina = pagina.extract_text()
                texto_completo += f"\n--- PÁGINA 1 ---\n"
                texto_completo += texto_pagina
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return None
    return texto_completo


# FUNCIÓN 2: ANALIZAR DOCUMENTO CON IA

def analizar_estructura_documento(texto_documento):
    """
    Usa OpenAI para analizar SOLO la primera hoja del RUT y extraer la información en la estructura JSON especificada.
    """
    print(f"[DEPURACIÓN] Longitud del texto enviado a la IA: {len(texto_documento)} caracteres")
    if len(texto_documento) > 6000:
        print("[ADVERTENCIA] El texto extraído es muy grande. Podrías estar superando el límite de tokens del modelo.")
    modelo_usado = "deepseek/deepseek-r1:free"
    print(f"[DEPURACIÓN] Modelo usado: {modelo_usado}, max_tokens=1500, temperature=0")

    # PROMPT ESTRUCTURADO PARA LA IA
    prompt = f"""
Eres un experto en extracción de datos de documentos oficiales. Analiza letra por letra el siguiente texto extraído de la PRIMERA HOJA de un RUT colombiano. Extrae la información en la siguiente estructura JSON, siguiendo el ORDEN y las CATEGORÍAS especificadas. Si alguna categoría no tiene información, pon el valor "" (cadena vacía). No inventes datos. No incluyas categorías que no estén en la lista. El JSON debe tener exactamente esta estructura y nombres de clave:

{{
  "informacion_inicial": {{
    "concepto": "",
    "numero de formulario": "",
    "numero de identificacion tributaria": "",
    "direccion seccional": "",
    "buzon de elecctronico": ""
  }},
  "identificacion": {{
    "tipo de contribuyente": "",
    "tipo de documento": "",
    "numero de identificacion": "",
    "fecha de expedicion": "",
    "lugar de expedicion": {{
      "pais": "",
      "departamento": "",
      "ciudad/municipio": ""
    }},
    "primer apellido": "",
    "segundo apellido": "",
    "primer nombre": "",
    "otros nombres": "",
    "razon social": "",
    "nombre comercial": ""
  }},
  "ubicacion": {{
    "pais": "",
    "departamento": "",
    "ciudad/municipio": "",
    "direccion principal": "",
    "correo electronico": "",
    "codigo postal": "",
    "telefono 1": "",
    "telefono 2": ""
  }},
  "clasificacion": {{
    "actividad economica": {{
      "actividad principal": {{"codigo": "", "fecha de inicio de actividad": ""}},
      "actividad secundaria": {{"codigo": "", "fecha de inicio de actividad": ""}},
      "otras actividades": [{{"codigo": "", "fecha de inicio de actividad": ""}}]
    }},
    "ocupacion": {{
      "codigo": "",
      "numero de establecimientos": ""
    }}
  }},
  "responsabilidades_calidades_atributos": {{
    "codigos de responsabilidades": [],
    "nombre de los codigos": {{}}
  }},
  "obligados_aduaneros": {{
    "codigos aduaneros": []
  }},
  "exportadores": {{
    "forma": "",
    "tipo": "",
    "modo": "",
    "cpc": ""
  }}
}}

EXTRAE SOLO LA INFORMACIÓN DE LA PRIMERA HOJA. Si alguna categoría no aparece, pon "" o [] según corresponda. No agregues texto adicional, solo el JSON. Si no puedes extraer nada, responde un JSON con todos los campos vacíos.

TEXTO DEL DOCUMENTO:
{texto_documento}
"""

    try:
        respuesta = client.chat.completions.create(
            model=modelo_usado,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en extracción de datos de documentos. Analiza cuidadosamente la estructura y extrae solo la información solicitada en formato JSON válido."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2500
        )
        contenido_respuesta = respuesta.choices[0].message.content.strip()
        try:
            with open("respuesta_ia_raw.txt", "w", encoding="utf-8") as f:
                f.write(contenido_respuesta)
            print("[DEPURACIÓN] Respuesta cruda de la IA guardada en 'respuesta_ia_raw.txt'")
        except Exception as e:
            print(f"[DEPURACIÓN] Error al guardar la respuesta cruda: {e}")
        try:
            datos_extraidos = json.loads(contenido_respuesta)
            return datos_extraidos
        except json.JSONDecodeError:
            print("Error: La respuesta no es un JSON válido")
            print("Respuesta recibida:", contenido_respuesta)
            return None
    except Exception as e:
        print(f"Error al procesar con OpenAI: {e}")
        return None


# FUNCIÓN 3: FILTRAR Y VALIDAR DATOS

def filtrar_rectangulos_validos(datos_extraidos):
    """
    Esta función actúa como un "filtro de calidad" para los datos que extrajo la IA.
    
    ¿Por qué necesitamos esto?
    A veces la IA puede incluir rectángulos que no tienen información útil,
    o puede confundirse con el formato. Esta función verifica que los datos
    sean realmente útiles antes de mostrarlos al usuario.
    
    ¿Qué verifica?
    1. Que cada rectángulo tenga tanto categoría como información
    2. Que la información sea sustancial (más de 5 caracteres)
    3. Que la categoría y la información no sean iguales
    4. Que los textos tengan sentido y no estén vacíos
    
    Parámetros:
        datos_extraidos (dict): Los datos que nos devolvió la IA
    
    Retorna:
        dict: Los mismos datos pero filtrados y validados
        None si no hay datos válidos
    """
    # Si no hay datos para filtrar, retornar None
    if not datos_extraidos:
        return None
    
    # Lista para guardar solo los rectángulos que sí tienen información válida
    rectangulos_filtrados = []
    
    # REVISAR CADA RECTÁNGULO UNO POR UNO
    # Obtenemos la lista de rectángulos, o una lista vacía si no existe
    for rectangulo in datos_extraidos.get("rectangulos_con_informacion", []):
        
        # Extraer y limpiar la categoría (quitar espacios extra)
        categoria = rectangulo.get("categoria", "").strip()
        
        # Extraer y limpiar la información (quitar espacios extra)
        informacion = rectangulo.get("informacion", "").strip()
        
        # APLICAR FILTROS DE CALIDAD
        # Verificar que:
        # - La categoría tenga más de 2 caracteres (no solo "a" o "ab")
        # - La información tenga más de 5 caracteres (contenido sustancial)
        # - La categoría y la información no sean iguales (evitar duplicados)
        if (len(categoria) > 2 and len(informacion) > 5 and 
            categoria.lower() != informacion.lower()):
            
            # Si pasa todos los filtros, agregar a la lista de rectángulos válidos
            rectangulos_filtrados.append(rectangulo)
    
    # ACTUALIZAR LOS DATOS CON LOS RECTÁNGULOS FILTRADOS
    # Reemplazar la lista original con la lista filtrada
    datos_extraidos["rectangulos_con_informacion"] = rectangulos_filtrados
    
    # Retornar los datos actualizados
    return datos_extraidos


# FUNCIÓN PRINCIPAL: COORDINA TODO EL PROCESO

def extraer_datos_documento_pdf(ruta_pdf):
    """
    Esta es la función "maestra" que coordina todo el proceso de extracción.
    Es como un director de orquesta que hace que todos los músicos toquen en orden.
    
    ¿Qué hace paso a paso?
    1. Extrae el texto del PDF
    2. Envía el texto a la IA para análisis
    3. Filtra y valida los resultados
    4. Muestra un resumen de lo que encontró
    
    Esta función usa todas las otras funciones en el orden correcto.
    
    Parámetros:
        ruta_pdf (str): La ubicación del archivo PDF en tu computadora
    
    Retorna:
        dict: Todos los datos extraídos y organizados del documento
        None si hubo algún error en el proceso
    """
    print("🔍 Iniciando extracción de datos del PDF...")
    
    # PASO 1: EXTRAER TEXTO DEL PDF
    # Usar nuestra función para convertir el PDF en texto
    print("📄 Extrayendo texto del PDF...")
    texto_documento = extraer_texto_pdf(ruta_pdf)
    
    # Verificar si la extracción funcionó
    if not texto_documento:
        print(" Error: No se pudo extraer texto del PDF")
        return None
    
    # Mostrar cuánto texto extrajimos (para que el usuario sepa que funcionó)
    print(f" Texto extraído exitosamente ({len(texto_documento)} caracteres)")
    
    # PASO 2: ANALIZAR ESTRUCTURA CON IA
    # Enviar el texto a la inteligencia artificial para que lo analice
    print(" Analizando estructura del documento con IA...")
    datos_extraidos = analizar_estructura_documento(texto_documento)
    
    # Verificar si el análisis funcionó
    if not datos_extraidos:
        print(" Error: No se pudo analizar el documento")
        return None
    
    # PASO 3: FILTRAR Y VALIDAR DATOS
    # Limpiar y verificar que los datos sean de buena calidad
    print(" Filtrando rectángulos válidos...")
    datos_finales = filtrar_rectangulos_validos(datos_extraidos)
    
    # MOSTRAR RESUMEN DE RESULTADOS
    # Informar al usuario qué encontramos
    if datos_finales:
        # Contar cuántos rectángulos válidos encontramos
        num_rectangulos = len(datos_finales.get("rectangulos_con_informacion", []))
        # Contar cuánta información externa encontramos
        num_info_externa = len(datos_finales.get("informacion_externa", []))
        
        print(f"Extracción completada:")
        print(f"   Rectángulos con información: {num_rectangulos}")
        print(f"   Información externa: {num_info_externa}")
    
    # Retornar todos los datos procesados
    return datos_finales


# FUNCIÓN PARA GUARDAR RESULTADOS

def guardar_datos_extraidos(datos, nombre_archivo="datos_extraidos.json"):
    """
    Esta función guarda todos los datos extraídos en un archivo JSON en tu computadora.
    
    ¿Por qué JSON?
    JSON es un formato de archivo que mantiene la estructura de los datos organizados,
    como si fuera una caja con compartimentos etiquetados. Es fácil de leer
    tanto para humanos como para otros programas.
    
    ¿Qué hace?
    1. Toma todos los datos extraídos
    2. Los convierte a formato JSON
    3. Los guarda en un archivo en tu disco duro
    4. Te dice dónde guardó el archivo
    
    Parámetros:
        datos (dict): Todos los datos extraídos del documento
        nombre_archivo (str): Cómo quieres que se llame el archivo (opcional)
    """
    try:
        # Abrir/crear un archivo para escribir
        # 'w' = write (escribir), 'utf-8' = codificación para caracteres especiales
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            
            # Convertir los datos a formato JSON y escribirlos en el archivo
            # indent=4: hace que el JSON sea fácil de leer (con espacios y líneas)
            # ensure_ascii=False: permite caracteres especiales como acentos
            json.dump(datos, archivo, indent=4, ensure_ascii=False)
        print(f" Datos guardados en: {nombre_archivo}")
    except Exception as e:
        # Si hay algún problema al guardar (disco lleno, permisos, etc.)
        print(f" Error al guardar datos: {e}")


def mostrar_resultados(datos):
    """
    Esta función toma todos los datos extraídos y los muestra en la pantalla
    de una manera organizada y fácil de leer para el usuario.
    
    Es como crear un reporte bien formateado de todo lo que encontramos.
    
    ¿Qué muestra?
    1. Todos los rectángulos con su categoría e información
    2. Información que estaba fuera de rectángulos
    3. Datos adicionales como nombres, direcciones, teléfonos, etc.
    4. Todo organizado con títulos, numeración y separadores visuales
    
    Parámetros:
        datos (dict): Los datos extraídos del documento
    """
    # Si no hay datos para mostrar, informar y salir
    if not datos:
        print(" No hay datos para mostrar")
        return
    
    # CREAR ENCABEZADO PRINCIPAL
    # Mostrar un título grande y llamativo
    print("\n" + "="*60)
    print("           RESULTADOS DE EXTRACCIÓN DE DATOS")
    print("="*60)
    
    # SECCIÓN 1: MOSTRAR RECTÁNGULOS CON INFORMACIÓN
    # Obtener la lista de rectángulos, o lista vacía si no existe
    rectangulos = datos.get("rectangulos_con_informacion", [])
    
    if rectangulos:
        # Mostrar título de sección con contador
        print(f"\nRECTÁNGULOS CON INFORMACIÓN ({len(rectangulos)}):")
        print("-" * 40)
        
        # Mostrar cada rectángulo numerado
        for i, rect in enumerate(rectangulos, 1):
            print(f"\n{i}.  CATEGORÍA: {rect.get('categoria', 'N/A')}")
            print(f"    INFORMACIÓN: {rect.get('informacion', 'N/A')}")
    
    # SECCIÓN 2: MOSTRAR INFORMACIÓN EXTERNA
    # Información que estaba fuera de los rectángulos
    info_externa = datos.get("informacion_externa", [])
    
    if info_externa:
        # Mostrar título de sección con contador
        print(f"\nINFORMACIÓN EXTERNA ({len(info_externa)}):")
        print("-" * 40)
        
        # Mostrar cada elemento de información externa numerado
        for i, info in enumerate(info_externa, 1):
            print(f"{i}. {info}")
    
    # SECCIÓN 3: MOSTRAR DATOS ADICIONALES
    # Datos específicos como nombres, direcciones, etc.
    datos_adicionales = datos.get("datos_adicionales", {})
    
    if datos_adicionales:
        print(f"\nDATOS ADICIONALES:")
        print("-" * 40)
        
        # Recorrer cada categoría de datos adicionales
        for categoria, items in datos_adicionales.items():
            
            # Solo mostrar categorías que tengan elementos
            if items:
                # Convertir nombre de categoría a formato legible
                # 'nombres_mencionados' -> 'NOMBRES MENCIONADOS'
                titulo_categoria = categoria.upper().replace('_', ' ')
                print(f"\n {titulo_categoria}: {len(items)}")
                
                # Mostrar cada elemento con viñeta
                for item in items:
                    print(f"   • {item}")


# SECCIÓN PRINCIPAL DEL PROGRAMA

# Esta es la parte que se ejecuta cuando corres el programa
if __name__ == "__main__":
    
    # CONFIGURACIÓN: UBICACIÓN DEL ARCHIVO PDF
    # ¡IMPORTANTE! Cambia esta ruta por la ubicación de tu archivo PDF
    ruta_archivo_pdf = "documents/RUT.pdf"
    
    # VERIFICAR QUE EL ARCHIVO EXISTE
    # Antes de intentar procesarlo, verificamos que el archivo esté donde dijimos
    if not os.path.exists(ruta_archivo_pdf):
        # Si el archivo no existe, mostrar mensajes de ayuda
        print(f"Error: El archivo {ruta_archivo_pdf} no existe")
        print("Asegúrate de:")
        print("   1. Colocar tu archivo PDF en la carpeta 'documents'")
        print("   2. O cambiar la variable 'ruta_archivo_pdf' con la ruta correcta")
        print("   3. Verificar que el nombre del archivo sea exacto (incluyendo mayúsculas/minúsculas)")
        
    else:
        # SI EL ARCHIVO EXISTE, EJECUTAR TODO EL PROCESO
        print(" ¡Archivo encontrado! Iniciando procesamiento...")
        
        # Llamar a la función principal que coordina todo
        resultados = extraer_datos_documento_pdf(ruta_archivo_pdf)
        
        # Si obtuvimos resultados exitosos
        if resultados:
            print("\n¡Procesamiento completado exitosamente!")
            
            # Mostrar los resultados en pantalla de manera organizada
            mostrar_resultados(resultados)
            
            # Guardar los resultados en un archivo JSON
            guardar_datos_extraidos(resultados)
            
            print("\n¡Proceso terminado! Revisa el archivo 'datos_extraidos.json' para ver todos los datos.")
            
        else:
            # Si no se pudieron extraer datos
            print("  No se pudieron extraer datos del documento")
            print("  Posibles causas:")
            print("   - El PDF está corrupto o protegido")
            print("   - El PDF es una imagen escaneada (necesita OCR)")
            print("   - Problemas de conexión con la IA")
            print("   - El documento no tiene el formato esperado")









