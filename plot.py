import numpy as np
import matplotlib.pyplot as plt

v1 = 176.30496215820312
v2 = 241.98590087890625

val = np.array([269.0768916134045, 274.76682939754437, 276.1996430468367, 266.06077214288405, 263.91825458751845, 242.1956177355635])
val2 = np.array([257.4942161002926, 279.7237568653652, 273.7715487034112, 260.8980058000399, 252.91300952064626, 265.03167969209625])
nbp = [i for i in range(3, 9)]
s = [val2[i]/val[i] for i in range(6)]
plt.title("evolution du Speedup en fonction du nombre de processus")
plt.ylabel("speedup")
plt.xlabel("nombre de processus")
plt.plot(nbp, s)
plt.savefig("./ressources/speedup_final.png")