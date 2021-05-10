# Para correr el siguiente script es necesario instalar twint, esto se puede lograr corriendo desde la terminal
# las siguientes intrucciones
# git clone --depth=1 https://github.com/twintproject/twint.git
# cd twint
# pip3 install . -r requirements.txt

# una vez instalado twint podemos ejecutar el siguiente script desde la terminal mediante:
# python twiter_scraper.py


import twint

#configuracion de los datos que seran extraidos.

c = twint.Config()

#Nombre del usuario del cual se extraera la informacion
c.Username = "trafico889"
#Oculta los resultados al correr desde la terminal
c.Hide_output = True
#c.Limit = 10
#opcion para generar un archivo json o un csv
#c.Store_csv = True
#c.Output ="cdmx.csv"
c.Store_json=True
#nombre del archivo de salida
c.Output ="datasets/accidentes_cdmx.json"

#palabra clave de busqueda
c.Search = "choque"
#fechas de inicio y final
c.Since = "2010-01-01 21:30:00"
c.Until= "2019-12-30 22:30:00"
#ejecuta la busqueda y extraccion
twint.run.Search(c)


