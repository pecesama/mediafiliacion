# Media Filiación SNSP v2.0

Aplicación Streamlit para dos flujos:

1. **Foto → ficha:** fotografía frontal obligatoria + fotografía de perfil opcional/recomendada.
2. **Ficha → imagen:** generación directa de un retrato hablado sintético con `gemini-3.1-flash-image` (Nano Banana 2).

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración rápida con API key

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edita `.streamlit/secrets.toml`:

```toml
GOOGLE_GENAI_BACKEND = "developer_api"
GEMINI_API_KEY = "TU_API_KEY"
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
```

## Configuración con Gemini Enterprise Agent Platform

```toml
GOOGLE_GENAI_BACKEND = "agent_platform"
GOOGLE_CLOUD_PROJECT = "tu-proyecto-gcp"
GOOGLE_CLOUD_LOCATION = "global"
GOOGLE_GENAI_USE_ENTERPRISE = true
GEMINI_TEXT_MODEL = "gemini-3.5-flash"
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
```

Autenticación recomendada:

```bash
gcloud auth application-default login
```

El proyecto debe tener facturación y Agent Platform API habilitada.

## Ejecutar

```bash
streamlit run app.py
```

## Cambios principales

- Se agrega el flujo operativo de **dos imágenes**: frente + perfil.
- Se migra de `google-generativeai` a `google-genai`.
- Se agrega generación directa de imágenes con `gemini-3.1-flash-image`.
- Se elimina la conversión de tonos de piel a etnia, herencia o nacionalidad.
- Se muestran y descargan tanto el prompt final como el PNG generado.
- Se agregan avisos de privacidad, revisión humana y no inferencia de identidad.
