# Proyecto - Accidentes vehiculares en la CDMX

# Equipo 6


* Angel Uriel Meléndez Rivera: amelendezr1100@alumno.ipn.mx
* Adalberto Benitez Zapata: adalb518@gmail.com
* Sergio Maldonado Rodriguez: sermalrod@outlook.com

# Contenido del Repositorio
* JNotebook: contiene los notebook del proyecto de este modulo
* Presentacion: contiene la presentacion del proyecto
* Reporte version git: contiene el reporte del proyecto en version git
* Video: contiene el video del proyecto


# Identificación del Problema

Sin duda alguna los automóviles han sido y son parte esencial de la sociedad, su movilidad y su transporte, desafortunadamente los siniestros causados por ellos son parte inherente de su uso y el impacto de ellos repercute de manera directa o indirecta en la vida de los involucrados, pero también de las personas que los rodean.

De acuerdo con un reporte de la Organización Mundial de la Salud en 2017, los accidentes viales son causantes de más de 1.3 de millones de muertes al año, y un estimado de alrededor de 50 millones de personas con traumatismos no mortales siendo así la quinta causa de los fallecimientos mundiales anuales.

Aunado a lo anterior, la OMS ahonda en su reporte, indicando que el 93% de los accidentes de tránsito suceden en los países de ingresos bajos y medianos. De este 93%, 20% acaece en la región de las Américas, en donde las estadísticas reportan más de 155’000 muertes por año por accidentes automovilísticos.

Por su parte México y de acuerdo con los datos del Instituto Nacional de Salud Pública (INSP), el país se encuentra en el séptimo lugar a nivel mundial en el rubro de siniestros automovilísticos mortales, y ocupa el tercer lugar en la región de Latinoamérica en este mismo rubro

Además de las repercusiones sociales y humanas, los accidentes viales, representan en números un estimado de más de 220 mil millones de pesos en costos, cifra la cual supera el 2% del PIB del país. Eventualmente los costos no son todos absorbidos por las instituciones del gobierno, y es el propio usuario en quién recae la mayor cantidad de responsabilidad económica asociada al siniestro. De nuevo, este último factor es un atenuante para la sociedad puesto que, continuando con los estudios presentados por el INSP, en promedio el gasto por accidente alcanza un costo de alrededor de 25´000 pesos, y si a esto se le suma que derivado de los siniestros automovilísticos se suman alrededor de 50´000 personas al grupo de personas con discapacidad motora, de los cuáles 70% no vuelve a conseguir empleo. Los costos en seguridad social, reparación de daños, entre otros regresan para ser solventados por las instituciones de gobierno.

Bajo este tenor, y con el fin de atenuar la situación descrita anteriormente, resulta útil identificar cuáles son los factores que tienen mayor influencia en los choques automovilísticos, cuáles son los perfiles más propensos a generar un choque, y simultáneamente diseñar estrategias que permitan atacar las causas raíz de los siniestros automovilísticos en México.

# Planteamiento de preguntas clave
1. ¿Qué delegación/municipio registró más accidentes?
2. ¿Cuál es el promedio de accidentes por delegación del año 2010 a 2019?
3. ¿Cuáles son las tres delegaciones con más accidentes y como es su tendencia?
4. ¿En que año se han registrado más accidentes en la CDMX y cuál es su tendencia?
5. ¿Quiénes son mas propensos a sufrir accidentes, los hombres o las mujeres y como es la tendencia?
6. ¿Cuáles son los meses con mayores accidentes a los largo de los años?
7. ¿Cuáles son los días de la semana con mayores accidentes a los largo de los años?
8. ¿Cuál es la edad más propensa a sufrir accidentes?
9. ¿Qué porcentaje de datos de edades de 99 y 100 años hay en los datos?
10. ¿Cuál es la tendencia de las edades más propensas a sufrir accidentes del 2010 al 2019?
11. ¿Qué vehiculo es el que causa más accidentes?
12. ¿Cuál es la tendencia de los 6 tipos de vehiculos que causan más accidentes?

# Análisis de datos
El análisis de datos incluye la obtención, limpieza y transformación de los datos. Adicionalmente se incluye el análisis exploratorio y las conclusiones.
* [Jupyter Notebook](https://github.com/Bedu-Equipo13/Proyecto_Python_Accidentes/blob/main/Jupyter_notebook/Proyecto_Python.ipynb)

# Conclusiones
Los análisis inferenciales deberán ir enfocados en evaluar 3 aspectos específicos.

El primer rubro será el análisis del impacto de los distintos factores o variables sobre los choques, para esto será necesario llevar a cabo análisis de regresión multivariable en los cuáles se puedan evaluar los coeficientes, así como las interacciones de estos factores. Dentro de las herramientas estadísticas que podremos utilizar para este análisis destacan ANOVA, K - Means, Random Forests, entre otros.

El segundo rubro será ahondar en el análisis comparativo con otros archivos de datos como lo es el parque vehicular, la cantidad de hombres y mujeres registrados, persona registradas por delegación, personas registradas por grupos de edad, entre otros. Los cuáles serán eficacez para eliminar ciertos tipos de sesgo que por ahora se tienen y que impiden un análisis más profundo de los factores determinates para la generación de un choque.

Como último rubro, será importante seguir enfocando hacía la predicción de choques bajo diferentes tipos de escenarios, para eto será útil efectuar métodos como las series de tiempo. asimismo y complementadno lo anterior estas predicciones podrán ayudar a también generar posibles análisis económetricos de la situación.

Finalmente, englobando todo este futuro desarrollo, estipulado sobre las base del análisis exploratorio que se genero, la propuesta final será diseñar estrategias que permitan mitigar los factores (causas raíz) que propician mayor número de choques. A manera de ejemplo, durante este primer análisis encontramos que a medida que incrementaba la hora del día, aumentaba el número de choques, y que el Viernes era el día con mayor número de choques registrados. Aunado a esto encontramos, qué alcaldías, erán las que tenían un reporte de mayor número de accidentes, por lo que, considerando estos tres factores, podriamos diseñar una medida de contigencía de alchólimetros mejor ubicados y distribuidos sobre las delegaciones con más concentración de choques y en horarios que quizas antes no se tenían contemplados, con esto probablemente reducir choques y consecuentemente reducir gasto público e indivdual. Esto claramente fue un somero esbozo del tipo de soluciones que podrían proponerse, siendo conscientes de que ese tipo de decisiones deberá contemplar también otros factores como movilidad en la zona, presupuesto de tránsito y vialidad, entre otros que los expertos en el tema considerarán, sin embargo, este ejercicio tenía como mera finalidad evidenciar el potencial de los resultados que pudiera llegar a tener este análisis y la utilidad sobre la problemática planteada al inicio del proyecto.

# Fuente de los Datos 
* [Diccionario de datos](https://www.inegi.org.mx/rnm/index.php/catalog/506/):
* [Pagina de los datos](https://www.inegi.org.mx/programas/accidentes/):
