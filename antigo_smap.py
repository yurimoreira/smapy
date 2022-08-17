import numpy as np
import csv
from pyswarm import pso
import matplotlib.pyplot as plt
from matplotlib.mlab import stineman_interp

# import matplotlib.pyplot as plt
# Coeficientes de peso para calculo do erro
coefq = [0.5, 0.535714286, 0.571428571, 0.607142857, 0.642857143, 0.678571429, 0.714285714, 0.75, 0.785714286,
         0.821428571, 0.857142857, 0.892857143, 0.928571429, 0.964285714, 1, 1.035714286, 1.071428571, 1.107142857,
         1.142857143, 1.178571429, 1.214285714, 1.25, 1.285714286, 1.321428571, 1.357142857, 1.392857143, 1.428571429,
         1.464285714, 1.5]

# Dados de Salto Caxias
with open('parametros.txt', newline='') as parfile:
    parametros = csv.reader(parfile, dialect=csv.excel_tab)
    cont = 0
    par = []
    p = {}
    for row in parametros:
        par.append(row)
        (k, v) = par[cont]
        p[k] = float(v)
        cont = cont + 1
    print(p)
parfile.close
locals().update(p)

c_obs = [0.12, 2.55, 0.09, 22.90, 1.13, 1.60, 0.44, 0.26, 3.16, 1.68, 0.47, 0.22, 0.17, 10.16, 0.89, 7.09, 1.04, 3.67,
         11.46, 13.55, 28.13, 21.14, 0.26, 0.15, 0.29, 0.19, 12.14, 8.08, 0.14, 0.13, 1.20, 2.27]

c_prev = [3.78, 3.42, 1.14, 3.62, 25.28, 0.11, 0.02, 9.23, 6.45, 6.16, 6.16, 0.03, 0.02, 0.01, 1.73, 4.688, 0.018,
          0.051, 1.158, 0, 0.006, 0, 0, 0.003, 0, 0, 0.01, 0.04, 0, 0, 9.428, 0, 0, 0, 0.051, 0.629]

qobs_aux = [2403.32, 2963.8, 2602.96, 2878.46, 2995.45, 2725.62, 2274.89, 1863.36, 1829.89, 1566.66, 1393.03, 1235.14,
            1237.45, 1268.27, 1196.82, 1167.16, 1101.01, 1015.95, 913.33, 883.42, 817.96, 753.99, 714.92, 675.02,
            655.48, 621.96, 600.07, 572.51, 543.56]


def chuva():
    lobs = len(c_obs)
    lprev = len(c_prev)
    cp_aux = []
    i = 3
    while i < len(c_obs) - 1:
        v0 = c_obs[(i - 3)]
        v1 = c_obs[(i - 2)]
        v2 = c_obs[(i - 1)]
        v3 = c_obs[i]
        v4 = c_obs[(i + 1)]
        y = (float(v0) * t3 + float(v1) * t2 + float(v2) * t1 + float(v3) * t0 + float(v4) * t00)
        cp_aux.append(y)
        i = i + 1
    v0 = c_obs[(i - 3)]
    v1 = c_obs[(i - 2)]
    v2 = c_obs[(i - 1)]
    v3 = c_obs[i]
    y = float(v0) * t3 + float(v1) * t2 + float(v2) * t1 + float(v3) * t0
    cp_aux.append(y)
    return np.array(cp_aux)


cp = chuva()

tuin = 40.
ebin = 1007.
rsupi = 26
rsoloi = (tuin / 100) * sat
rsubi = (1 / (1 - ((0.5) ** (1 / kkt)))) * (ebin / Ad) * 86.4

aii = np.array([ai] * len(cp))
qobs = np.array(qobs_aux)
coefq = np.array(coefq)
# es=[0,8.89181147,3.536165245,0,0,0,0,0,0,0,0,0.027285509,0,0.002017212,0,0.001673814,0.231472281,0.543781371,1.345991638,1.284552638,0.031070005,0,0,0,0.071185939,0.087924344,0,0,0]
# er=[1.430113614,1.6,1.6,1.304945761,1.225481116,1.151202544,1.373538483,1.336133867,1.118893592,1.209923279,1.14705826,1.6,1.739282568,1.6,1.749273439,1.6,1.6,1.6,1.6,1.6,1.6,1.217910144,1.345469493,1.408694788,1.6,1.6,1.684612128,1.442760154,1.547303277]
# rec=[0.261075559,0.254570072,0.738411332,1.089546915,1.033095473,0.976205444,0.919874717,0.879775733,0.84044843,0.793963798,0.754921689,0.716203982,0.739946063,0.724580984,0.72443939,0.709923645,0.70944751,0.795077288,0.929992447,1.144692571,1.34540427,1.346684982,1.258626008,1.191146011,1.134357982,1.164486041,1.19934438,1.141305467,1.069967023]
ep = [2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000,
      2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.00000,
      2.00000, 2.00000, 2.00000, 2.00000, 2.00000, 2.35484, 2.35484, 2.35484]


def vaz():
    rsolo = np.empty(len(cp) + 1)
    rsolo[0] = (tuin / 100) * sat
    rsub = np.empty(len(cp) + 1)
    rsub[0] = rsubi
    rsup = np.empty(len(cp) + 1)
    rsup[0] = rsupi
    tu = np.empty(len(cp))
    es = np.empty(len(cp))
    rec = np.empty(len(cp))
    er = np.empty(len(cp))
    eb = np.empty(len(cp))
    ed = np.empty(len(cp))
    qcalc = np.empty(len(cp))
    boo = cp >= aii
    i = 0
    while i < len(cp):
        tu[i] = rsolo[i] / sat
        es[i] = (((cp[i] - ai) ** 2) / (cp[i] - ai + sat - rsolo[i])) * boo[i]
        rec[i] = crec * 0.01 * tu[i] * (rsolo[i] - (capc * 0.01 * sat))

        if (cp[i] - es[i]) > ep[i]:
            er[i] = ep[i] * ecof
        else:
            er[i] = cp[i] - es[i] + ((ep[i] * ecof) - (cp[i] - es[i])) * tu[i]

        eb[i] = rsub[i] * (1 - (0.5 ** (1 / kkt)))
        ed[i] = (rsup[i] + es[i]) * (1 - (pow(0.5, (1 / k2t))))
        qcalc[i] = (ed[i] + eb[i]) * (Ad / 86.4)

        i = i + 1

        rsolo[i] = rsolo[i - 1] + cp[i - 1] - es[i - 1] - er[i - 1] - rec[i - 1]
        rsup[i] = rsup[i - 1] + es[i - 1] - ed[i - 1]
        rsub[i] = rsub[i - 1] - eb[i - 1] + rec[i - 1]
    return qcalc[:]


print(qobs)
print(vaz())


def nash(qobs, qcalc):
    mqobs = np.mean(qobs)
    vn = np.sum((qobs - qcalc) ** 2 * coefq[:len(qcalc)])
    vt = np.sum((qobs - mqobs) ** 2 * coefq[:len(qcalc)])
    return 1 - (vn / vt)


def mape(qobs, qcalc):
    vt = np.mean(((np.absolute(qcalc - qobs)) / qobs) * coefq[:len(qcalc)])
    return 1 - vt


def objective(qcalc, qobs):
    return (nash(qobs, qcalc) + mape(qobs, qcalc))


print(objective(vaz(), qobs))

# Geracao de grafico
x = np.arange(0., 29., 1.)
xi = np.linspace(x[0], x[-1], 100000)
yp = None
qcalci = stineman_interp(xi, x, vaz(), yp)
qobsi = stineman_interp(xi, x, qobs, yp)
# ebi=stineman_interp(xi, x, eb, yp)

# plt.plot(x, yi, 'r', x, qobs, 'b')
fig, ax = plt.subplots()
ax.plot(xi, qcalci, 'r', xi, qobsi, 'b')
ax.grid()
ylimit = np.amax(np.append(qcalci, qobsi))
plt.ylim(0, ylimit + 50)
fig.text(0.90, 0.85, 'LAMMOC', fontsize=50, color='gray', ha='right', va='top', alpha=0.5)
plt.show()

# xopt, fopt = pso(objective(vaz(),qobs), lb, ub, f_ieqcons=con)
'''opt=np.ones(len(cp))
cp=cp*opt
o=[0,0]
n=500
maior=0
i=0
while o[0]<2 or i==len(cp):
    opt[i]=0.5
    for j in range(n):
        cp=cp*opt
        o[0]=objective(vaz(),qobs)
        if o[0]>o[1]:
            o[1]=o[0]
            maior=opt[i]
        opt[i]=opt[i]+(2/n)
    opt[i]=maior
    i=i+1'''