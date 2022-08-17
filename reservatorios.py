import numpy as np


Rsolo(t) = Mín {Rsolo(t-1) + P(t) – Es(t) – Er(t) – Rec(t) ; Str}
Rsub(t) = Rsub(t-1) + Rec(t) – Eb(t)
Rsup(t) = Rsup(t-1) + Es(t) – Marg(t) – Ed(t) – Ed3(t) + Máx {0 ; [(Rsolo(t-1) + P(t) – Es(t) – Er(t) – Rec(t)) – Str]}
Rsup2(t) = Rsup2(t-1) + Marg(t) – Ed2(t) – Emarg(t)

def rsolo(rsolo, prec, es, er, rec, str):
    return min(rsolo + prec - es - er - rec, str)

def rsub(rsub, rec, eb):
    return rsub + rec + eb

def rsup(rsup, es, marg, ed, ed3, rsolo, str):
    return rsup + es + marg - ed - ed3 + rsolo-str

def rsup2(rsup2, marg, ed2, emarg):
    return rsup2 + marg - ed2 - emarg


