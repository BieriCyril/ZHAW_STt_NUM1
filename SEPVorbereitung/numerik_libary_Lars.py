"""
numerik.py  –  Numerik-Library für ET/ST
=========================================
Funktionen 1:1 wie in den Praktika P1–P14.
Verwendung:  import numerik as n   ODER  from numerik import linsolve, RK4, ...
"""

import numpy as np


# ════════════════════════════════════════════════════════════════
#  KAPITEL 0 – NUMERISCHE ABLEITUNG  (P1)
# ════════════════════════════════════════════════════════════════

def vorwaerts(f, x0, h):
    """
    Vorwärtsdifferenzenquotient  (P1)
    f'(x0) ≈ (f(x0+h) - f(x0)) / h     Fehlerordnung O(h)

    Beispiel (wie in P1):
        f  = np.cos
        x0 = 1
        h  = np.logspace(0, -10, 500)
        errD1 = np.abs(vorwaerts(f, x0, h) - (-np.sin(x0)))
    """
    return (f(x0 + h) - f(x0)) / h


def zentral(f, x0, h):
    """
    Zentraler Differenzenquotient  (P1)
    f'(x0) ≈ (f(x0+h) - f(x0-h)) / (2h)     Fehlerordnung O(h²)

    Beispiel (wie in P1):
        errD2 = np.abs(zentral(f, x0, h) - (-np.sin(x0)))
    """
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


# ════════════════════════════════════════════════════════════════
#  KAPITEL 1a – LR-ZERLEGUNG  (P2)
# ════════════════════════════════════════════════════════════════

def pivot(A, I, k):
    """
    Bestimmt die Pivotzeile (relatives Spaltenmaximum) und tauscht Zeilen.  (P2)
    Verändert A und I direkt (in-place).
    """
    temp = np.abs(A[k:, k])
    p = k + np.argmax(temp)
    if p != k:
        A[[k, p], :] = A[[p, k], :]
        I[[k, p]]    = I[[p, k]]


def LR(A_in):
    """
    LR-Zerlegung mit Spaltenpivotisierung (relatives Spaltenmaximum).  (P2)

    Rückgabe: L, R, I
        L  – untere Dreiecksmatrix (Diagonale = 1)
        R  – obere Dreiecksmatrix
        I  – Permutationsvektor  →  A[I] = L @ R

    Beispiel (wie in P2):
        L, R, I = LR(A)
        print(np.linalg.norm(L @ R - A[I]))   # ≈ 0
    """
    A = A_in.copy().astype(float)
    n = A.shape[0]
    I = np.arange(n)
    assert A.shape[0] == A.shape[1], "A muss quadratisch sein"

    for k in range(n - 1):
        pivot(A, I, k)
        for j in range(k + 1, n):
            A[j, k]     = A[j, k] / A[k, k]
            A[j, k+1:]  = A[j, k+1:] - A[j, k] * A[k, k+1:]

    L = np.tril(A, -1) + np.eye(n)
    R = np.triu(A)
    return L, R, I


def fbsub(L, R, b):
    """
    Vorwärts- und Rückwärtseinsetzen für L @ R @ x = b.  (P2, P4)

    Funktioniert für LR- und Cholesky-Zerlegung (ℓᵢᵢ ≠ 1 wird korrekt behandelt).

    Beispiel (wie in P2):
        L, R, I = LR(A)
        x = fbsub(L, R, b[I])
        print(np.linalg.norm(A @ x - b))   # ≈ 0
    """
    n = b.shape[0]

    # Vorwärtseinsetzen: L y = b
    y = np.zeros_like(b, dtype=float)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]

    # Rückwärtseinsetzen: R x = y
    x = np.zeros_like(b, dtype=float)
    x[-1] = y[-1] / R[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.sum(R[i, i+1:] * x[i+1:])) / R[i, i]

    return x


def linsolve(A, b):
    """
    Löst A @ x = b via LR-Zerlegung mit Pivotisierung.  (P2)

    Beispiel (wie in P2):
        x = linsolve(A, b)
        print(np.linalg.norm(A @ x - b))   # ≈ 0
    """
    L, R, I = LR(A)
    return fbsub(L, R, b[I])


# ════════════════════════════════════════════════════════════════
#  KAPITEL 1b – TRIDIAGONALE LGS  (P3)
# ════════════════════════════════════════════════════════════════
#
#  Kompaktes Speicherformat (3 × n Matrix M):
#    M[0, :] = untere Nebendiagonale  (M[0, 0] unbenutzt)
#    M[1, :] = Hauptdiagonale
#    M[2, :] = obere Nebendiagonale   (M[2, -1] unbenutzt)

def ind(i, j):
    """
    Index-Umrechnung: volle Matrixposition (i,j) → kompaktes Format (Zeile, Spalte).  (P3)

    Beispiel (wie in P3):
        M[ind(i, j)] = wert
    """
    if i == j + 1: return 0, i
    if i == j:     return 1, i
    if i == j - 1: return 2, i
    assert False, f"Kein Tridiagonal-Index: ({i},{j})"


def LR_tri(m):
    """
    LR-Zerlegung einer Tridiagonalmatrix im kompakten 3×n Format.  (P3)

    Eingabe/Ausgabe im kompakten Format:
        m[0, :] = untere Nebendiagonale  (m[0,0] unbenutzt)
        m[1, :] = Hauptdiagonale
        m[2, :] = obere Nebendiagonale   (m[2,-1] unbenutzt)

    Beispiel (wie in P3):
        m = np.zeros((3, n))
        m[0, 1:] = -1;  m[1] = 2;  m[2, :-1] = -1
        LR = LR_tri(m)
        assert np.linalg.norm(L @ R - A) < 1e-10
    """
    n = m.shape[1]
    LR = np.zeros_like(m, dtype=float)
    r, c = ind(0, 0)
    LR[r, c] = m[r, c]
    for j in range(1, n):
        LR[ind(j, j-1)] = m[ind(j, j-1)] / LR[ind(j-1, j-1)]
        LR[ind(j-1, j)] = m[ind(j-1, j)]
        LR[ind(j, j)]   = m[ind(j, j)] - LR[ind(j, j-1)] * LR[ind(j-1, j)]
    return LR


def fbSub_tri(LR, b):
    """
    Vorwärts- und Rückwärtseinsetzen für Tridiagonalmatrizen (kompaktes Format).  (P3)

    Beispiel (wie in P3):
        LR = LR_tri(m)
        x  = fbSub_tri(LR, b)
        assert np.linalg.norm(x1 - x2) < 1e-10
    """
    n = b.shape[0]
    y = np.zeros_like(b, dtype=float)
    x = np.zeros_like(b, dtype=float)

    # Vorwärtseinsetzen (Diagonale von L ist 1)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - LR[ind(i, i-1)] * y[i-1]

    # Rückwärtseinsetzen
    x[n-1] = y[n-1] / LR[ind(n-1, n-1)]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - LR[ind(i, i+1)] * x[i+1]) / LR[ind(i, i)]

    return x


def linsolve_tri(m, b):
    """
    Löst ein tridiagonales LGS (kompaktes 3×n Format).  (P3)

    Beispiel (wie in P3):
        x = linsolve_tri(m, b)
    """
    return fbSub_tri(LR_tri(m), b)


def thomas(lower, main, upper, r):
    """
    Thomas-Algorithmus: löst tridiagonales LGS direkt aus lower/main/upper/r.  (P3, P13, P14)

    Parameter:
        lower – untere Nebendiagonale, shape (n-1,)
        main  – Hauptdiagonale,        shape (n,)
        upper – obere Nebendiagonale,  shape (n-1,)
        r     – rechte Seite,          shape (n,)

    Beispiel (wie in P14):
        x = thomas(lower, main, upper, r)
    """
    n = len(main)
    m   = main.copy().astype(float)
    u   = upper.copy().astype(float)
    rhs = r.copy().astype(float)

    for i in range(1, n):
        faktor  = lower[i-1] / m[i-1]
        m[i]   -= faktor * u[i-1]
        rhs[i] -= faktor * rhs[i-1]

    x = np.zeros(n)
    x[-1] = rhs[-1] / m[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - u[i] * x[i+1]) / m[i]
    return x


# ════════════════════════════════════════════════════════════════
#  KAPITEL 1c – CHOLESKY-ZERLEGUNG  (P4, P5, Plattenkondensator)
# ════════════════════════════════════════════════════════════════

def cholesky(A_in):
    """
    Cholesky-Zerlegung Variante 1:  A = L Lᵀ  (P4, P5)

    A muss symmetrisch und positiv definit sein.

    Rückgabe: L (untere Dreiecksmatrix)
    Raises ValueError wenn A nicht positiv definit.

    Beispiel (wie in P4):
        L = cholesky(A)
        print(np.linalg.norm(L @ L.T - A))   # ≈ 0
    """
    A = A_in.copy().astype(float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for k in range(n):
        if A[k, k] < 0:
            raise ValueError("Matrix ist nicht positiv definit")
        L[k, k] = np.sqrt(A[k, k])
        l = A[k+1:, k] / L[k, k]
        L[k+1:, k] = l
        A[k+1:, k+1:] -= np.outer(l, l)
    return L


def cholesky2(A_in):
    """
    Cholesky-Zerlegung Variante 2:  A = L̃ D L̃ᵀ  (P4, Plattenkondensator)

    L̃ hat Einsen auf der Hauptdiagonalen, D ist eine Diagonalmatrix.

    Rückgabe: L (Diag. = 1), D (Diagonalmatrix)
    Raises ValueError wenn A nicht positiv definit.

    Beispiel (wie in P4):
        L, D = cholesky2(A)
        print(np.linalg.norm(L @ D @ L.T - A))   # ≈ 0
    """
    A = A_in.copy().astype(float)
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros_like(A)
    for k in range(n):
        tmp = sum(L[k, j]**2 * D[j, j] for j in range(k))
        D[k, k] = A[k, k] - tmp
        if D[k, k] <= 0:
            raise ValueError("Matrix ist nicht positiv definit")
        for i in range(k + 1, n):
            tmp = sum(L[i, j] * D[j, j] * L[k, j] for j in range(k))
            L[i, k] = (A[i, k] - tmp) / D[k, k]
    return L, D


def linsolve_chol(A, b):
    """
    Löst A @ x = b via Cholesky Var. 1 (A = LLᵀ).  (P4)
    A muss symmetrisch und positiv definit sein.

    Beispiel (wie in P4):
        x = linsolve_chol(A, b)
        print("Ohne Diagonalmatrix:", np.linalg.norm(A @ x - b))
    """
    assert np.linalg.norm(A - A.T) < 1e-12, "A muss symmetrisch sein"
    L = cholesky(A)
    return fbsub(L, L.T, b)


def linsolve_chol2(A, b):
    """
    Löst A @ x = b via Cholesky Var. 2 (A = LDLᵀ).  (P4)
    A muss symmetrisch und positiv definit sein.

    Beispiel (wie in P4):
        x = linsolve_chol2(A, b)
        print("Mit Diagonalmatrix:", np.linalg.norm(A @ x - b))
    """
    assert np.linalg.norm(A - A.T) < 1e-12, "A muss symmetrisch sein"
    L, D = cholesky2(A)
    return fbsub(L @ D, L.T, b)


# Hilfsfunktionen aus dem Plattenkondensator-Praktikum
def fsub(L, b):
    """Vorwärtseinsetzen L y = b  (Plattenkondensator)."""
    n = b.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def dsolve(D, v):
    """Löst D x = v für Diagonalmatrix D  (Plattenkondensator)."""
    return v / np.diag(D)

def bsub(R, y):
    """Rückwärtseinsetzen R x = y  (Plattenkondensator)."""
    n = y.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    return x

def linsolve_chol2_voll(A, b):
    """
    Löst A @ x = b via Cholesky Var. 2 mit separatem fsub/dsolve/bsub.
    Exakt wie im Plattenkondensator-Praktikum:
        L, D = cholesky2(A)
        v    = fsub(L, b)       # L v = b
        y    = dsolve(D, v)     # D y = v
        x    = bsub(L.T, y)    # Lᵀ x = y
    """
    L, D = cholesky2(A)
    v = fsub(L, b)
    y = dsolve(D, v)
    return bsub(L.T, y)


# ════════════════════════════════════════════════════════════════
#  KAPITEL 3 – AUSGLEICHSRECHNUNG  (P5, P6, P7, P8)
# ════════════════════════════════════════════════════════════════

def solve_normalengleichung_cholesky(A, y):
    """
    Löst lineares Ausgleichsproblem via Normalgleichungen + Cholesky.  (P5)
    AᵀA x = Aᵀy

    Beispiel (wie in P5):
        A = build_matrix(t, n=5)
        x = solve_normalengleichung_cholesky(A, y)
    """
    AtA = A.T @ A
    Aty = A.T @ y
    L = cholesky(AtA)
    # Vorwärts/Rückwärts wie in P5
    n = len(Aty)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (Aty[i] - np.dot(L[i, :i], z[:i])) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - np.dot(L.T[i, i+1:], x[i+1:])) / L.T[i, i]
    return x


def householder(a):
    """
    Householder-Spiegelungsmatrix H zu Vektor a: H @ a = ‖a‖ e₁.  (P6)

    Beispiel (wie in P6):
        H = householder(a)
        assert np.linalg.norm(H @ a [1:]) < 1e-10
        assert np.linalg.norm(H.T @ H - np.eye(n)) < 1e-10
    """
    a = np.asarray(a, dtype=float)
    e1 = np.zeros_like(a); e1[0] = 1.0
    n_vec = a + np.linalg.norm(a) * e1
    n_vec = n_vec / np.linalg.norm(n_vec)
    return np.eye(len(a)) - 2 * np.outer(n_vec, n_vec)


def QR(A):
    """
    QR-Zerlegung via Householder:  A = Q R  (P6)

    Rückgabe: Q (N×N orthogonal), R (N×n obere Dreiecksform)

    Beispiel (wie in P6):
        Q, R = QR(A)
        print(np.linalg.norm(Q @ R - A))              # ≈ 0
        print(np.linalg.norm(Q.T @ Q - np.eye(N)))   # ≈ 0
        print(np.linalg.norm(np.tril(R, -1)))         # ≈ 0
    """
    R = A.copy().astype(float)
    N, n = R.shape
    H = np.eye(N)
    for k in range(n):
        a = R[k:, k]
        Hk = np.eye(N)
        Hk[k:, k:] = householder(a)
        H = Hk @ H
        R = Hk @ R
    return H.T, R


def backsub(R, y):
    """
    Rückwärtseinsetzen für rechteckige obere Dreiecksmatrix R (N×n, N≥n).  (P6)

    Beispiel (wie in P6):
        x = backsub(R, Q.T @ b)
    """
    n = R.shape[1]
    x = np.zeros(n)
    x[-1] = y[n-1] / R[n-1, -1]
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    return x


def lstsq(A, b):
    """
    Lineares Ausgleichsproblem ‖Ap - b‖² → min via QR.  (P6, P8)

    Beispiel (wie in P6):
        p = lstsq(A, yi)
    """
    Q, R = QR(A)
    return backsub(R, Q.T @ b)


def gauss_newton(f_res, J_func, p0, maxit=10):
    """
    Gauss-Newton-Verfahren für nichtlineare Ausgleichsprobleme.  (P8)
    Minimiert ‖f(p)‖² → min.

    Parameter:
        f_res  – Residuenvektor f(p) ∈ ℝᴺ  (= Modell - Messung)
        J_func – Jacobimatrix J(p) ∈ ℝᴺˣⁿ
        p0     – Startwert (array)
        maxit  – Anzahl Iterationen (Standard: 10, wie in P8)

    Rückgabe: p (optimale Parameter)

    Beispiel (wie in P8):
        for k in range(10):
            delta = lstsq(J(*p), y - f(*p))
            p += delta
        # entspricht: p, _ = gauss_newton(...)
    """
    p = np.array(p0, dtype=float)
    for k in range(maxit):
        res   = np.array(f_res(p), dtype=float)
        Jp    = np.array(J_func(p), dtype=float)
        delta = lstsq(Jp, -res)
        p    += delta
    return p


def levenberg_marquardt(f_res, J_func, p0, mu0=1.0, beta0=0.25, beta1=0.75,
                        tol=1e-10, maxit=100):
    """
    Levenberg-Marquardt-Verfahren.  (P8)
    Robustere Alternative zu Gauss-Newton.

    Parameter:
        f_res  – Residuenvektor f(p)
        J_func – Jacobimatrix J(p)
        p0     – Startwert
        mu0    – Anfangsdämpfung (Standard: 1.0)
        beta0  – Untere ρ-Schwelle (Standard: 0.25)
        beta1  – Obere  ρ-Schwelle (Standard: 0.75)
        tol    – Abbruchtoleranz
        maxit  – Maximale Iterationszahl

    Rückgabe: p (optimale Parameter), history (‖f‖² pro Schritt)
    """
    p     = np.array(p0, dtype=float)
    mu    = float(mu0)
    n_p   = len(p)
    history = []
    for _ in range(maxit):
        fp   = np.array(f_res(p), dtype=float)
        Jp   = np.array(J_func(p), dtype=float)
        norm2 = float(fp @ fp)
        history.append(norm2)
        J_aug = np.vstack([Jp, mu * np.eye(n_p)])
        f_aug = np.concatenate([-fp, np.zeros(n_p)])
        delta = lstsq(J_aug, f_aug)
        fp_new   = np.array(f_res(p + delta), dtype=float)
        norm_new = float(fp_new @ fp_new)
        f_lin    = fp + Jp @ delta
        denom    = norm2 - float(f_lin @ f_lin)
        if abs(denom) < 1e-15:
            break
        rho = (norm2 - norm_new) / denom
        if rho < beta0:
            mu *= 2
        elif rho < beta1:
            p += delta
        else:
            mu /= 2
            p  += delta
        if np.linalg.norm(delta) < tol:
            break
    return p, history


# ════════════════════════════════════════════════════════════════
#  KAPITEL 2 – NEWTONVERFAHREN  (P7, P9, P10, P11)
# ════════════════════════════════════════════════════════════════

def newton(f, df, x, maxit=100):
    """
    Skalares Newtonverfahren.  (P9, P10, P11)
    Iterationsformel: x = x - f(x)/f'(x)

    Parameter:
        f     – Funktion f(x)
        df    – Ableitung f'(x)
        x     – Startwert
        maxit – Maximale Iterationszahl (Standard: 100, wie in P9/P10)

    Rückgabe: x (Nullstelle)

    Beispiel (wie in P9):
        def newton(f, df, x, maxit=100):
            for k in range(maxit):
                x = x - f(x) / df(x)
            return x
    """
    for k in range(maxit):
        x = x - f(x) / df(x)
    return x


def backwardKin(x_target, y_target, l1=2, l2=1, tol=1e-6, maxit=600):
    """
    Rückwärtskinematik für 2-Gelenk-Roboter via Newtonverfahren.  (P7)

    Parameter:
        x_target, y_target – Zielposition des Greifers
        l1, l2             – Linklängen (Standard: 2, 1 wie in P7)
        tol                – Abbruchtoleranz
        maxit              – Maximale Iterationszahl

    Rückgabe: th1, th2 (Gelenkwinkel in Radiant)
    Raises RuntimeError bei Fehler.

    Beispiel (wie in P7):
        th1, th2 = backwardKin(x, y)
    """
    assert l1 >= l2, "Annahme l1 >= l2 verletzt"
    r = np.hypot(x_target, y_target)
    if (l1 - l2 > r) or (r > l1 + l2):
        raise RuntimeError("Ausserhalb des erreichbaren Bereiches")

    f = lambda th1, th2: np.array([
        l1 * np.cos(th1) + l2 * np.cos(th1 + th2) - x_target,
        l1 * np.sin(th1) + l2 * np.sin(th1 + th2) - y_target])
    J = lambda th1, th2: np.array([
        [l1 * -np.sin(th1) + l2 * -np.sin(th1 + th2), l2 * -np.cos(th1 + th2)],
        [ l1 *  np.cos(th1) + l2 *  np.cos(th1 + th2), l2 *  np.cos(th1 + th2)]])

    th1, th2 = 0.5, 0.5
    it    = 0
    delta = np.array([1., 1.])
    while np.linalg.norm(delta) > tol and it < maxit:
        delta  = np.linalg.solve(J(th1, th2), -f(th1, th2))
        th1   += delta[0]
        th2   += delta[1]
        it    += 1
    if it >= maxit:
        raise RuntimeError("Maximale Iterationen überschritten")
    return th1, th2


# ════════════════════════════════════════════════════════════════
#  KAPITEL 4 – GEWÖHNLICHE DIFFERENTIALGLEICHUNGEN  (P9–P12)
# ════════════════════════════════════════════════════════════════
#
#  Alle Verfahren: y' = f(x, y),  y(x0) = y0
#  Rückgabe immer: x (ndarray), y (ndarray)
#  Implizite Verfahren brauchen df = ∂f/∂y

def eulerExplizit(f, x0, y0, xn, h):
    """
    Expliziter Euler (Ordnung 1).  (P9, P10, P11)
    y_{n+1} = y_n + h · f(x_n, y_n)

    Beispiel (wie in P9):
        f = lambda x, y: -4 * y
        xe, ye = eulerExplizit(f, 0, 1, 2, 0.1)
    """
    x = [x0]
    y = [y0]
    n_steps = int((xn - x0) / h)
    for _ in range(n_steps):
        x.append(x[-1] + h)
        y.append(y[-1] + h * f(x[-1], y[-1]))
    return np.array(x), np.array(y)


def eulerImplizit(f, df, x0, y0, xn, h):
    """
    Impliziter Euler (Ordnung 1).  (P9, P10, P11)
    Löst via Newton: F(s) = y_k + h·f(x_{k+1}, s) - s = 0
    df = ∂f/∂y

    Beispiel (wie in P9):
        f  = lambda x, y: -4 * y
        df = lambda x, y: -4
        xi, yi = eulerImplizit(f, df, 0, 1, 2, 0.1)
    """
    x = [x0]
    y = [y0]
    n_steps = int((xn - x0) / h)
    for k in range(n_steps):
        x_next = x[k] + h
        x.append(x_next)
        F  = lambda s: y[k] + h * f(x_next, s) - s
        dF = lambda s: h * df(x_next, s) - 1
        y_next = newton(F, dF, y[k])
        y.append(y_next)
    return np.array(x), np.array(y)


def Runge(f, x0, y0, xn, h):
    """
    Explizite Mittelpunktsregel / Verfahren von Runge (Ordnung 2).  (P10, P11)

    Butcher-Tableau:
        r1 = f(xk, yk)
        r2 = f(xk + h/2, yk + h/2·r1)
        y_{k+1} = yk + h·r2

    Beispiel (wie in P10/P11):
        xm, ym = Runge(f, x0, y0, xn, h)
    """
    n = int((xn - x0) / h)
    x = [x0]
    y = [y0]
    for k in range(n):
        r1 = f(x[-1], y[-1])
        r2 = f(x[-1] + h/2, y[-1] + h*r1/2)
        y.append(y[-1] + h * r2)
        x.append(x[-1] + h)
    return np.array(x), np.array(y)


def Heun(f, x0, y0, xn, h):
    """
    Verfahren von Heun / Explizite Trapezregel (Ordnung 2).  (P10, P11)

    Butcher-Tableau:
        r1 = f(xk, yk)
        r2 = f(xk + h, yk + h·r1)
        y_{k+1} = yk + h·(r1/2 + r2/2)

    Beispiel (wie in P10/P11):
        xh, yh = Heun(f, x0, y0, xn, h)
    """
    n = int((xn - x0) / h)
    x = [x0]
    y = [y0]
    for k in range(n):
        r1 = f(x[-1], y[-1])
        r2 = f(x[-1] + h, y[-1] + h * r1)
        y.append(y[-1] + h * (r1/2 + r2/2))
        x.append(x[-1] + h)
    return np.array(x), np.array(y)


def RK4(f, x0, y0, xn, h):
    """
    Klassisches Runge-Kutta 4. Ordnung.  (P10, P11, P12)

    Butcher-Tableau:
        r1 = f(xk,       yk)
        r2 = f(xk + h/2, yk + h/2·r1)
        r3 = f(xk + h/2, yk + h/2·r2)
        r4 = f(xk + h,   yk + h·r3)
        y_{k+1} = yk + (h/6)·(r1 + 2r2 + 2r3 + r4)

    Unterstützt auch vektorwertige y (z.B. P12: Phasenraumdarstellung).

    Beispiel (wie in P10/P11):
        xrk, yrk = RK4(f, x0, y0, xn, h)

    Beispiel vektoriell (wie in P12):
        f = lambda t, y: [y[1], (m*g - k*y[0]) / m]
        y_start = np.array([0, v0])
        xrk, yrk = RK4(f, 0, y_start, 200, 0.001)
        plt.plot(yrk[:,0], yrk[:,1])
    """
    n = int((xn - x0) / h)
    x = [x0]
    y = [y0]
    for k in range(n):
        r1 = np.array(f(x[-1], y[-1]))
        r2 = np.array(f(x[-1] + h/2, y[-1] + h*r1/2))
        r3 = np.array(f(x[-1] + h/2, y[-1] + h*r2/2))
        r4 = np.array(f(x[-1] + h,   y[-1] + h*r3))
        y.append(y[-1] + (h/6) * (r1 + 2*r2 + 2*r3 + r4))
        x.append(x[-1] + h)
    return np.array(x), np.array(y)


def Mittelpunktregel(f, df, x0, y0, xn, h):
    """
    Implizite Mittelpunktsregel (Ordnung 2).  (P10, P11)

    r1 = f(xk + h/2, yk + h/2·r1)  ← via Newton
    F(r1)  = r1 - f(xk + h/2, yk + h/2·r1) = 0
    F'(r1) = 1  - df(...) · h/2
    y_{k+1} = yk + h·r1
    df = ∂f/∂y

    Beispiel (wie in P10/P11):
        xm_i, ym_i = Mittelpunktregel(f, df, x0, y0, xn, h)
    """
    n = int((xn - x0) / h)
    x = [x0]
    y = [y0]
    for k in range(n):
        F  = lambda s: s - f(x[-1] + h/2, y[-1] + h*s/2)
        dF = lambda s: 1 - df(x[-1] + h/2, y[-1] + h*s/2) * h/2
        n_start = f(x[-1], y[-1])
        r1 = newton(F, dF, n_start)
        y.append(y[-1] + h * r1)
        x.append(x[-1] + h)
    return np.array(x), np.array(y)


def Trapezregel(f, df, x0, y0, xn, h):
    """
    Implizite Trapezregel (Ordnung 2).  (P10, P11)

    r1 = f(xk, yk)                             ← explizit
    r2 = f(xk+h, yk + h/2·r1 + h/2·r2)        ← via Newton
    F(r2)  = r2 - f(xk+h, yk + h/2·r1 + h/2·r2) = 0
    F'(r2) = 1  - df(...) · h/2
    y_{k+1} = yk + h·(r1/2 + r2/2)
    df = ∂f/∂y

    Beispiel (wie in P10/P11):
        xh_i, yh_i = Trapezregel(f, df, x0, y0, xn, h)
    """
    n = int((xn - x0) / h)
    x = [x0]
    y = [y0]
    for k in range(n):
        r1 = f(x[-1], y[-1])
        F  = lambda s: s - f(x[-1] + h, y[-1] + h*(r1/2 + s/2))
        dF = lambda s: 1 - df(x[-1] + h, y[-1] + h*(r1/2 + s/2)) * h/2
        r2 = newton(F, dF, r1)
        y.append(y[-1] + h * (r1/2 + r2/2))
        x.append(x[-1] + h)
    return np.array(x), np.array(y)


# ════════════════════════════════════════════════════════════════
#  KAPITEL 5 – PARTIELLE DIFFERENTIALGLEICHUNGEN  (P13)
# ════════════════════════════════════════════════════════════════

def ftcs(u, s, N, u0_func=None, uJ_func=None, T=None, diff_threshold=None):
    """
    FTCS-Schema (explizit) für Wärmeleitungsgleichung u_t = a·u_xx.  (P13)

    Exakt wie in P13 implementiert.
    u_j^{n+1} = s·(u_{j-1}^n + u_{j+1}^n) + (1-2s)·u_j^n
    Stabilitätsbedingung: s = a·τ/h² ≤ 0.5

    Parameter:
        u           – Anfangszustand shape (2, J+1), u[0] = Anfangsbedingung
        s           – Stabilitätsfaktor s = a·τ/h² (muss ≤ 0.5 sein!)
        N           – Anzahl Zeitschritte
        u0_func     – Randbedingung links u(0, t) als Funktion(n), oder None (Neumann)
        uJ_func     – Randbedingung rechts u(J, t) als Funktion(n), oder None (Neumann)
        T           – Zeitgitter (ndarray, shape N), für Randbedingungen benötigt
        diff_threshold – Abbruchschwelle für max. Änderung (wie in P13), oder None

    Rückgabe: u[1] (Endlösung), n (letzter Zeitschritt)

    Beispiel (wie in P13, Dirichlet):
        u = np.zeros((2, J+1))
        u[0] = phi(X)
        u[0, 0] = u0(0); u[0, -1] = uJ(0)
        u_end, n_end = ftcs(u, s=0.5, N=N, u0_func=u0, uJ_func=uJ, T=T)

    Beispiel (wie in P13, Neumann/adiabatisch):
        u_end, n_end = ftcs(u, s=0.49, N=N)   # u0_func=None → Neumann
    """
    if s > 0.5:
        raise ValueError(f"Stabilitätsbedingung verletzt: s={s:.4f} > 0.5")
    J = u.shape[1] - 1
    for n in range(N):
        # Dirichlet-Randbedingungen (falls gegeben)
        if u0_func is not None and T is not None:
            u[1, 0]  = u0_func(T[n])
        if uJ_func is not None and T is not None:
            u[1, -1] = uJ_func(T[n])

        # FTCS für innere Punkte
        u[1, 1:J] = s * (u[0, 2:J+1] + u[0, :J-1]) + (1 - 2*s) * u[0, 1:J]

        # Neumann-Randbedingungen (adiabatisch), falls keine Dirichlet-RB
        if u0_func is None:
            u[1, 0] = 2*s * (u[0, 1] - u[0, 0]) + u[0, 0]
        if uJ_func is None:
            u[1, J] = 2*s * (u[0, J-1] - u[0, J]) + u[0, J]

        diff = np.max(np.abs(u[1] - u[0]))
        u[0] = u[1].copy()

        if diff_threshold is not None and diff < diff_threshold:
            return u[1], n + 1

    return u[1], N


def crank_nicolson_P13(u, alpha, beta, gamma, J, N, u0_func, uJ_func, T,
                        diff_threshold=None, plot_interval=100):
    """
    Crank-Nicolson-Schema wie in P13 implementiert.  (P13)

    Verwendet scipy.linalg.solve_banded.

    Parameter:
        u         – Lösungsmatrix shape (N, J+1), u[0] = Anfangsbedingung
        alpha     – a / (2·h²)
        beta      – 1/τ + a/h²
        gamma     – 1/τ - a/h²
        J         – Anzahl Stützstellen - 1
        N         – Anzahl Zeitschritte
        u0_func   – Randbedingung links u(0, t)
        uJ_func   – Randbedingung rechts u(J, t)
        T         – Zeitgitter (ndarray)
        diff_threshold – Abbruchschwelle, oder None

    Rückgabe: u (vollständige Lösungsmatrix), n (letzter Zeitschritt)

    Hinweis: Braucht scipy. Wie in P13:
        alpha = a / (2*h**2)
        beta  = 1/tau + a/h**2
        gamma = 1/tau - a/h**2
        ab = np.array([-alpha*np.ones(J-1), beta*np.ones(J-1), -alpha*np.ones(J-1)])
        u_end, n_end = crank_nicolson_P13(u, alpha, beta, gamma, J, N, u0, uJ, T)
    """
    from scipy.linalg import solve_banded
    ab = np.array([-alpha * np.ones(J-1),
                    beta  * np.ones(J-1),
                   -alpha * np.ones(J-1)])
    for n in range(N - 1):
        b = alpha * (u[n, :J-1] + u[n, 2:J+1]) + gamma * u[n, 1:J]
        b[0]  += alpha * u[n+1, 0]
        b[-1] += alpha * u[n+1, J]
        u[n+1, 1:J] = solve_banded((1, 1), ab, b)
        diff = np.max(np.abs(u[n+1] - u[n]))
        if diff_threshold is not None and diff < diff_threshold:
            return u, n + 1
    return u, N - 1


# ════════════════════════════════════════════════════════════════
#  KAPITEL 6 – KUBISCHE SPLINE-INTERPOLATION  (P14)
# ════════════════════════════════════════════════════════════════

def thomas_algorithmus(lower, main, upper, r):
    """
    Thomas-Algorithmus (Alias, exakt wie in P14 benannt).  (P14)
    Identisch zu thomas(), aber mit dem Namen aus P14.
    """
    return thomas(lower, main, upper, r)


def cubic_spline_natural(x, y):
    """
    Natürlicher kubischer Spline: Koeffizienten a, b, c, d.  (P14)

    Randbedingung: S''(x_0) = S''(x_n) = 0.
    Auf Intervall [x_i, x_{i+1}]:
        S_i(t) = a_i + b_i·t + c_i·t² + d_i·t³,  t = x - x_i

    Parameter:
        x – Stützstellen (aufsteigend), shape (n+1,)
        y – Stützwerte,                  shape (n+1,)

    Rückgabe: a, b, c, d (je arrays der Länge n = Anzahl Intervalle)

    Beispiel (wie in P14):
        a, b, c, d = cubic_spline_natural(x_k, y_k)
        y_spline = spline_auswerten(x_k, a, b, c, d, x_fine)
    """
    n = len(x) - 1
    h = np.diff(x)

    # Rechte Seite
    r = np.zeros(n - 1)
    for i in range(n - 1):
        r[i] = 6.0 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])

    # Tridiagonalsystem
    diag_main  = np.array([2.0 * (h[i] + h[i+1]) for i in range(n - 1)])
    diag_upper = h[1:-1].copy()
    diag_lower = h[1:-1].copy()

    M_inner = thomas_algorithmus(diag_lower, diag_main, diag_upper, r)

    M = np.zeros(n + 1)
    M[1:n] = M_inner

    # Koeffizienten
    a = y[:-1].copy()
    b = np.zeros(n)
    c = M[:-1] / 2.0
    d = np.zeros(n)
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] / 6.0 * (2.0 * M[i] + M[i+1])
        d[i] = (M[i+1] - M[i]) / (6.0 * h[i])

    return a, b, c, d


def cubic_spline_periodisch(x, y):
    """
    Periodischer kubischer Spline (für geschlossene Kurven).  (P14)

    Voraussetzung: y[0] == y[-1] (geschlossene Kurve).
    Löst volles (n×n) System mit np.linalg.solve.

    Parameter:
        x – Stützstellen, shape (n+1,)
        y – Stützwerte (y[0]==y[-1]), shape (n+1,)

    Rückgabe: a, b, c, d (je arrays der Länge n)

    Beispiel (wie in P14):
        ax, bx, cx, dx = cubic_spline_periodisch(t, x_coords)
        ay, by, cy, dy = cubic_spline_periodisch(t, y_coords)
        x_fine = spline_auswerten(t, ax, bx, cx, dx, t_fine)
        y_fine = spline_auswerten(t, ay, by, cy, dy, t_fine)
    """
    n = len(x) - 1
    h = np.diff(x)
    delta = np.array([(y[i+1] - y[i]) / h[i] for i in range(n)])

    A = np.zeros((n, n))
    r = np.zeros(n)
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        A[i, im] += h[im % n]
        A[i, i]  += 2.0 * (h[im % n] + h[i % n])
        A[i, ip] += h[i % n]
        r[i] = 6.0 * (delta[i] - delta[im])

    M = np.linalg.solve(A, r)
    M = np.append(M, M[0])

    a = y[:-1].copy()
    b = np.zeros(n)
    c = M[:-1] / 2.0
    d = np.zeros(n)
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] / 6.0 * (2.0 * M[i] + M[i+1])
        d[i] = (M[i+1] - M[i]) / (6.0 * h[i])

    return a, b, c, d


def spline_auswerten(x_stuetz, a, b, c, d, x_neu):
    """
    Wertet den kubischen Spline an beliebigen Stellen aus.  (P14)

    S_i(x) = a_i + b_i·t + c_i·t² + d_i·t³,  t = x - x_i

    Parameter:
        x_stuetz – Stützstellen
        a, b, c, d – Koeffizienten (Ausgabe von cubic_spline_natural / _periodisch)
        x_neu    – Auswertestellen (array)

    Rückgabe: y_neu (Splinewerte)

    Beispiel (wie in P14):
        y_spline = spline_auswerten(x_k, a, b, c, d, x_fine)
    """
    y_neu = np.zeros_like(x_neu, dtype=float)
    for j, xj in enumerate(x_neu):
        i = np.searchsorted(x_stuetz, xj, side='right') - 1
        i = int(np.clip(i, 0, len(a) - 1))
        t = xj - x_stuetz[i]
        y_neu[j] = a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3
    return y_neu
