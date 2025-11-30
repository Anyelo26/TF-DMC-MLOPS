# Ejecución de Tests Funcionales del Modelo de Crédito

### Paso 0: Ingrese al Escritorio remoto

### Paso 1: Fork del Repositorio Original

En el navegador, inicie sesión en Github. Luego, vaya al enlace del proyecto original (https://github.com/lcajachahua/model-credit) y dé click al botón "Fork". Esto copiará todo el proyecto en su usuario de Github.


### Paso 2: Levantar el contenedor de Python

```
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```


### Paso 3: Configurar git

Abra una Terminal en JupyterLab e ingrese los siguientes comandos

```
git config --global user.name "<USER>"
git config --global user.email <CORREO>
```


### Paso 4: Clonar el Proyecto desde su propio Github

```
git clone https://github.com/<USER>/model-credit.git
```


### Paso 5: Instalar los pre-requisitos

```
cd model-credit/

pip install -r requirements.txt
```


### Paso 6: Ejecutar las pruebas en el entorno

```
cd src

python make_dataset.py

python train.py

python evaluate.py

python predict.py

cd ..
```


### Paso 7: Guardar los cambios en el Repo

```
git add .

git commit -m "Pruebas Finalizadas"

git push

```

Ingrese su usuario y Personal Access Token de Github. Puede revisar que los cambios se hayan guardado en el repositorio. Luego, puede finalizar JupyterLab ("File" => "Shut Down").