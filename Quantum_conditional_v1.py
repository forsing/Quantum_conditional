# Quantum_conditional 
# USLOVNE KVANTNE RASPODELE — GLOBALNO NORMALIZOVANA KOMBINACIJA
# uslovne raspodele i zavisni događaji 



# Sedam uslovnih sestokubitnih modela.
# Zavisnosti unutar rastuce kombinacije su prvog reda.
#
# Globalna normalizacija: proizvod pojedinacno normalizovanih legalnih nastavaka.
# proizvod Bornovih faktora, podeljen ukupnom masom
# SVIH legalnih kombinacija.
#
# Uci se zajednicka uslovna raspodela maksimalnom verodostojnoscu.
# Unitarni operatori ostaju unitarni tokom optimizacije.
# Qiskit proverava njihove amplitude i zavrsne operatore.
#
# Klasicni deo: lokalna simulacija, optimizator i dinamicko
# programiranje. Nema klasicnog regresora iza kvantnog modela.



import csv
import hashlib
import itertools
import math
import random
import re
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)

DATA_DIR = Path("/Users/4c/Desktop/GHQ/data")

WINDOWS = (1, 2, 3)
MAX_EPOCHS = 80
CHECK_EVERY = 5

N = 39
K = 7
D = 64


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def discover():
    pattern = re.compile(
        r"^loto7_(\d+)_k(\d+)"
        r"(?:_(loto|loto_plus)_(\d+))?\.csv$"
    )

    groups = {}

    for path in DATA_DIR.glob("loto7_*.csv"):
        match = pattern.fullmatch(path.name)

        if match:
            total, round_number, kind, count = match.groups()

            groups.setdefault(
                (int(total), int(round_number)), {}
            )[kind or "all"] = (
                path,
                int(count or total),
            )

    complete = [
        key
        for key, group in groups.items()
        if set(group) == {"all", "loto", "loto_plus"}
        and group["loto"][1] + group["loto_plus"][1] == key[0]
    ]

    require(
        bool(complete),
        "Nema potpunog usaglasenog skupa tri CSV fajla.",
    )

    chosen = groups[max(complete)]

    return [
        chosen[key]
        for key in ("all", "loto", "loto_plus")
    ]


def load(path, expected):
    raw = path.read_bytes()
    rows = []

    reader = csv.reader(
        raw.decode("utf-8-sig").splitlines()
    )

    for line, row in enumerate(reader, 1):
        require(
            len(row) == 7,
            f"{path.name}: red {line} nema sedam kolona.",
        )

        try:
            values = tuple(int(x.strip()) for x in row)
        except ValueError as exc:
            raise ValueError(
                f"{path.name}: necelobrojan red {line}."
            ) from exc

        require(
            len(set(values)) == 7
            and 1 <= min(values) <= max(values) <= 39,
            f"{path.name}: nelegalan red {line}.",
        )

        require(
            values == tuple(sorted(values)),
            f"{path.name}: red {line} nije rastuci.",
        )

        rows.append(values)

    require(
        len(rows) == expected and len(rows) >= 100,
        f"{path.name}: neusaglasen broj redova.",
    )

    return (
        np.array(rows, dtype=np.int64),
        hashlib.sha256(raw).hexdigest(),
    )


def history(data, times, window):
    times = np.asarray(times, dtype=int)

    require(
        len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravna vremena.",
    )

    h = np.zeros((len(times), 7 * window))

    for lag in range(1, window + 1):
        indices = times - lag
        valid = indices >= 0

        h[valid, 7 * (lag - 1):7 * lag] = (
            data[indices[valid]] - 20.0
        ) / 19.0

    # Nulti slotovi oznacavaju nedostupan deo ranog ulaza.
    # Ne dodaju se izmisljena kola ili brojevi.
    return h


def bases(h):
    result = []

    for j in range(7):
        b = np.zeros((len(h), 64))

        b[:, 0] = 1
        b[:, 1:1 + h.shape[1]] = h / np.sqrt(h.shape[1])
        b[:, 61] = (j + 1) / 7
        b[:, 63] = h[:, :7].mean(axis=1)

        prev = (
            np.array([0])
            if j == 0
            else np.arange(j, 33 + j)
        )

        norm = np.sqrt(
            np.sum(b * b, axis=1)[:, None]
            + (prev > 0)[None, :]
            + (prev[None, :] / 39) ** 2
        )

        mask = (
            (np.arange(1, 40)[None, :] > prev[:, None])
            & (np.arange(1, 40)[None, :] <= 33 + j)
        )

        result.append((b, prev, norm, mask))

    return result


def initial():
    # Deterministicki ortogonalan pocetni operator.
    # Nije pretpostavka raspodele loto podataka.
    i = np.arange(64)[:, None]
    j = np.arange(64)[None, :]

    u = (
        np.cos(np.pi * i * (j + 0.5) / 64)
        * np.sqrt(2.0 / 64)
    )

    u[0] /= np.sqrt(2.0)

    return np.repeat(u[None, :, :], 7, axis=0)


def circuit(u):
    qc = QuantumCircuit(6)
    qc.unitary(
        u,
        range(6),
        label="ConditionalU",
    )
    return qc


def potentials(units, features):
    tables = []
    cache = []

    for j, (b, prev, norm, mask) in enumerate(features):
        u = units[j]

        psi = np.repeat(
            (b @ u[:39].T)[:, None, :],
            len(prev),
            axis=1,
        )

        if j:
            psi += u[
                :39, 22 + prev - 1
            ].T[None, :, :]

        psi += (
            prev[None, :, None] / 39
        ) * u[None, None, :39, 62]

        psi /= norm[:, :, None]

        table = np.zeros((len(b), 40, 40))

        table[:, prev, 1:] = (
            psi * psi * mask[None, :, :]
        )

        tables.append(table)
        cache.append(psi)

    return tables, cache


def messages(tables):
    count = len(tables[0])

    alpha = np.zeros((count, 40))
    alpha[:, 0] = 1

    forwards = [alpha]
    scales = []
    logz = np.zeros(count)

    for table in tables:
        nxt = np.einsum(
            "np,npq->nq",
            alpha,
            table,
            optimize=True,
        )

        scale = nxt.sum(axis=1)

        require(
            np.all(np.isfinite(scale))
            and np.all(scale > 0),
            "Nulta ili nekonacna masa legalnih putanja.",
        )

        alpha = nxt / scale[:, None]

        forwards.append(alpha)
        scales.append(scale)
        logz += np.log(scale)

    return forwards, scales, logz


def objective(
    units,
    features,
    targets,
    gradient=False,
):
    tables, cache = potentials(units, features)

    forwards, scales, logz = messages(tables)

    count = len(targets)
    rows = np.arange(count)

    previous = np.zeros(count, dtype=int)
    logtrue = np.zeros(count)

    for j, table in enumerate(tables):
        actual = table[
            rows,
            previous,
            targets[:, j],
        ]

        if np.any(actual <= 0):
            if not gradient:
                return math.inf

            raise RuntimeError(
                "Nulta amplituda zabelezene kombinacije."
            )

        logtrue += np.log(actual)
        previous = targets[:, j]

    loss = float(np.mean(logz - logtrue))

    if not gradient:
        return loss

    gradients = np.zeros_like(units)
    backward = np.ones((count, 40))

    for j in range(6, -1, -1):
        b, prev, norm, mask = features[j]
        table = tables[j]
        psi = cache[j]

        posterior = (
            forwards[j][:, prev, None]
            * table[:, prev, 1:]
            * backward[:, None, 1:]
            / scales[j][:, None, None]
        )

        gs = np.divide(
            posterior,
            psi,
            out=np.zeros_like(psi),
            where=psi != 0,
        ) * 2 / count

        actual_prev = (
            np.zeros(count, dtype=int)
            if j == 0
            else targets[:, j - 1]
        )

        prev_index = np.searchsorted(
            prev,
            actual_prev,
        )

        observed = psi[
            rows,
            prev_index,
            targets[:, j] - 1,
        ]

        gs[
            rows,
            prev_index,
            targets[:, j] - 1,
        ] -= 2 / observed / count

        scaled = gs / norm[:, :, None]

        g = np.zeros((64, 64))

        g[:39] = (
            scaled.sum(axis=1).T @ b
        )

        if j:
            g[
                :39, 22 + prev - 1
            ] += scaled.sum(axis=0).T

        g[:39, 62] += np.einsum(
            "npq,p->q",
            scaled,
            prev / 39,
            optimize=True,
        )

        u = units[j]

        gradients[j] = (
            g
            - u @ (
                (u.T @ g + g.T @ u) / 2
            )
        )

        backward = np.einsum(
            "npq,nq->np",
            table,
            backward,
            optimize=True,
        ) / scales[j][:, None]

    require(
        np.isfinite(loss)
        and np.all(np.isfinite(gradients)),
        "Nekonacan gradijent.",
    )

    return loss, gradients


def retract(units, gradient, step):
    left, _, right = np.linalg.svd(
        units - step * gradient,
        full_matrices=False,
    )

    return left @ right


def fit(
    features,
    targets,
    epochs,
    label,
    validation=None,
):
    units = initial()

    previous_u = None
    previous_g = None

    best = math.inf
    best_epoch = None

    for epoch in range(1, epochs + 1):
        loss, g = objective(
            units,
            features,
            targets,
            True,
        )

        norm = float(np.sum(g * g))

        step = min(
            1.0,
            1.0 / max(np.sqrt(norm), 1.0),
        )

        if previous_u is not None:
            change = units - previous_u

            step = float(
                np.clip(
                    np.sum(change * change)
                    / (
                        abs(
                            np.sum(
                                change * (g - previous_g)
                            )
                        )
                        + 1e-16
                    ),
                    1e-6,
                    50,
                )
            )

        accepted = False

        for attempt in range(30):
            trial = retract(units, g, step)

            value = objective(
                trial,
                features,
                targets,
            )

            if (
                value
                <= loss
                - 1e-4 * step * norm
                + 1e-13
            ):
                accepted = True
                break

            step *= 0.5

        require(
            accepted or norm < 1e-18,
            "Nije nadjen opadajuci optimizacioni korak.",
        )

        previous_u = units.copy()
        previous_g = g.copy()

        if accepted:
            units = trial

        if validation is not None and (
            epoch % CHECK_EVERY == 0
            or epoch == epochs
        ):
            value = objective(
                units,
                *validation,
            )

            if value < best:
                best = value
                best_epoch = epoch

        if epoch % 20 == 0 or epoch == epochs:
            print(
                label,
                epoch,
                "NLL",
                objective(
                    units,
                    features,
                    targets,
                ),
                flush=True,
            )

    return units, best_epoch, best


def dataset(data, times, window):
    return (
        bases(history(data, times, window)),
        data[times],
    )


def decode(units, h):
    tables, _ = potentials(
        units,
        bases(h),
    )

    require(
        len(h) == 1,
        "Dekoder ocekuje jedan istorijski ulaz.",
    )

    _, _, logz = messages(tables)

    states = {0: (0.0, ())}

    for table in tables:
        nxt = {}

        for prev, (score, path) in states.items():
            for x in np.flatnonzero(
                table[0, prev] > 0
            ):
                candidate = (
                    score
                    + math.log(table[0, prev, x]),
                    path + (int(x),),
                )

                old = nxt.get(int(x))

                if (
                    old is None
                    or candidate[0] > old[0]
                    or (
                        candidate[0] == old[0]
                        and candidate[1] < old[1]
                    )
                ):
                    nxt[int(x)] = candidate

        require(
            bool(nxt),
            "Prazna pretraga maksimuma.",
        )

        states = nxt

    score, path = min(
        states.values(),
        key=lambda item: (
            -item[0],
            item[1],
        ),
    )

    # Nezavisna provera maksimuma unazad.
    tail = np.zeros(40)

    for table in reversed(tables):
        new = np.full(40, -np.inf)

        for prev in range(40):
            valid = np.flatnonzero(
                table[0, prev] > 0
            )

            if len(valid):
                new[prev] = np.max(
                    np.log(
                        table[0, prev, valid]
                    )
                    + tail[valid]
                )

        tail = new

    require(
        abs(score - tail[0]) < 1e-10,
        "Dve provere maksimuma se ne slazu.",
    )

    require(
        len(path) == 7
        and len(set(path)) == 7
        and path == tuple(sorted(path)),
        "Nelegalan maksimum.",
    )

    return path, float(score - logz[0])


def checks():
    data = np.array(
        [
            [1, 3, 6, 12, 19, 28, 39],
            [2, 5, 9, 16, 23, 31, 38],
            [4, 7, 11, 18, 26, 33, 37],
            [5, 14, 15, 17, 28, 30, 34],
        ],
        dtype=int,
    )

    units = initial()

    features, y = dataset(
        data,
        np.arange(1, 4),
        3,
    )

    value, g = objective(
        units,
        features,
        y,
        True,
    )

    for j in range(7):
        require(
            np.allclose(
                units[j].T @ units[j],
                np.eye(64),
                atol=1e-11,
            ),
            "Neunitaran operator.",
        )

        for i, k in (
            (0, 1),
            (4, 18),
            (25, 42),
            (38, 63),
        ):
            tangent = np.zeros_like(units)
            skew = np.zeros((64, 64))

            skew[i, k] = 1
            skew[k, i] = -1

            tangent[j] = units[j] @ skew

            eps = 1e-6

            fd = (
                objective(
                    retract(
                        units,
                        -tangent,
                        eps,
                    ),
                    features,
                    y,
                )
                - objective(
                    retract(
                        units,
                        tangent,
                        eps,
                    ),
                    features,
                    y,
                )
            ) / (2 * eps)

            require(
                abs(
                    fd
                    - np.sum(g * tangent)
                ) < 2e-5,
                "Gradijent ne prolazi numericku proveru.",
            )

    for j, (b, prev, norm, mask) in enumerate(features):
        a = b[0].copy()
        p = int(prev[-1])

        if p:
            a[22 + p - 1] += 1

        a[62] += p / 39
        a /= norm[0, -1]

        psi = Statevector(a).evolve(
            circuit(units[j])
        ).data

        tables, cache = potentials(
            units,
            features,
        )

        require(
            np.allclose(
                psi[:39],
                cache[j][0, -1],
                atol=1e-11,
            ),
            "Amplitudna simulacija se ne slaze sa Qiskitom.",
        )

    for w in WINDOWS:
        h = history(
            data,
            np.array([1]),
            w,
        )

        require(
            np.all(h[0, 7:] == 0),
            "Buduci podaci u ranom ulazu.",
        )

    # Mala iscrpna provera globalne particione funkcije.
    small = []

    for j in range(3):
        t = np.zeros((1, 40, 40))

        for p in (
            [0]
            if j == 0
            else range(j, 8 + j)
        ):
            for x in range(p + 1, 8 + j):
                t[0, p, x] = (
                    1 + (x + 2 * p + j) % 5
                ) / 10

        small.append(t)

    _, _, logz = messages(small)

    total = 0

    for comb in itertools.combinations(
        range(1, 10),
        3,
    ):
        p = 0
        weight = 1.0

        for j, x in enumerate(comb):
            weight *= small[j][0, p, x]
            p = x

        total += weight

    require(
        abs(
            math.exp(logz[0]) - total
        ) < 1e-11,
        "Pogresna globalna normalizacija.",
    )

    decode(
        units,
        history(
            data,
            np.array([len(data)]),
            3,
        ),
    )

    print(
        "PROVERE PROSLE: Qiskit amplitude, gradijenti, "
        "hronologija, globalna normalizacija i maksimum.",
        flush=True,
    )


def process(path, expected):
    data, digest = load(path, expected)

    n = len(data)
    a = int(0.6 * n)
    b = int(0.8 * n)

    print()
    print(
        "CSV:",
        path.name,
        "redova:",
        n,
        flush=True,
    )

    choices = []

    for w in WINDOWS:
        train = dataset(
            data,
            np.arange(1, a),
            w,
        )

        val = dataset(
            data,
            np.arange(a, b),
            w,
        )

        _, epoch, nll = fit(
            *train,
            MAX_EPOCHS,
            f"Validacija w={w}",
            val,
        )

        if epoch is not None and np.isfinite(nll):
            choices.append(
                (nll, w, epoch)
            )

    require(
        bool(choices),
        "Nema validiranog kandidata.",
    )

    val_nll, w, epochs = min(choices)

    units, _, _ = fit(
        *dataset(
            data,
            np.arange(1, b),
            w,
        ),
        epochs,
        "Pre testa",
    )

    test_nll = objective(
        units,
        *dataset(
            data,
            np.arange(b, n),
            w,
        ),
    )

    require(
        np.isfinite(test_nll),
        "Nekonacan test.",
    )

    print(
        "Izabran prozor:",
        w,
        "epoha:",
        epochs,
    )
    print(
        "Validacioni NLL:",
        val_nll,
    )
    print(
        "Hronoloski test NLL:",
        test_nll,
    )

    # Svih N-1 raspolozivih istorijskih prelaza.
    units, _, _ = fit(
        *dataset(
            data,
            np.arange(1, n),
            w,
        ),
        epochs,
        "Cela istorija",
    )

    require(
        all(
            np.allclose(
                u.T @ u,
                np.eye(64),
                atol=1e-10,
            )
            for u in units
        ),
        "Zavrsni operator nije unitaran.",
    )

    for u in units:
        require(
            np.allclose(
                Operator(circuit(u)).data,
                u,
                atol=1e-11,
            ),
            "Qiskit operator ne odgovara naucenom operatoru.",
        )

    path_result, logp = decode(
        units,
        history(
            data,
            np.array([n]),
            w,
        ),
    )

    print("NEXT:", *path_result)
    print(
        "Log-verovatnoca unutar modela:",
        logp,
    )

    longest = run = 1

    for left, right in zip(
        path_result,
        path_result[1:],
    ):
        run = (
            run + 1
            if right == left + 1
            else 1
        )

        longest = max(longest, run)

    if longest >= 5:
        print(
            "PROVERA MODELA: najmanje pet uzastopnih "
            "brojeva; rezultat nije filtriran."
        )

    return path.name, path_result, digest


def main():
    print(
        "USLOVNE KVANTNE RASPODELE — "
        "GLOBALNO NORMALIZOVANA KOMBINACIJA"
    )
    print(
        "Qiskit:",
        qiskit.__version__,
    )

    checks()

    results = [
        process(path, count)
        for path, count in discover()
    ]

    for left, right in itertools.combinations(
        results,
        2,
    ):
        if left[1] == right[1]:
            print(
                "PROVERA MODELA: isti NEXT:",
                left[0],
                right[0],
                "isti ulaz:",
                left[2] == right[2],
            )

    print(
        "ZAVRSEN RAD NAD SVA TRI CSV-a."
    )


if __name__ == "__main__":
    main()



"""
USLOVNE KVANTNE RASPODELE — GLOBALNO NORMALIZOVANA KOMBINACIJA
Qiskit: 1.4.4
PROVERE PROSLE: Qiskit amplitude, gradijenti, hronologija, globalna normalizacija i maksimum.

CSV: loto7_4714_k88.csv redova: 4714
Validacija w=1 20 NLL 20.443879941887683
Validacija w=1 40 NLL 19.141582314361315
Validacija w=1 60 NLL 17.98175334511109
Validacija w=1 80 NLL 17.064051655037883
Validacija w=2 20 NLL 22.298782576756338
Validacija w=2 40 NLL 20.06309530389756
Validacija w=2 60 NLL 18.712163816205834
Validacija w=2 80 NLL 17.818411895644356
Validacija w=3 20 NLL 22.479263858145014
Validacija w=3 40 NLL 21.062474186352862
Validacija w=3 60 NLL 20.231731793952452
Validacija w=3 80 NLL 19.32196095150342
Pre testa 20 NLL 21.803664227318116
Pre testa 40 NLL 20.39015667890964
Pre testa 60 NLL 19.8594653975934
Pre testa 80 NLL 19.384348412460188
Izabran prozor: 1 epoha: 80
Validacioni NLL: 17.695908916765475
Hronoloski test NLL: 19.85629363560119
Cela istorija 20 NLL 22.274895143416458
Cela istorija 40 NLL 20.858866865051063
Cela istorija 60 NLL 19.7816162961079
Cela istorija 80 NLL 18.890982167879574
NEXT: 5 9 22 29 34 38 39
Log-verovatnoca unutar modela: -10.816422190698994

CSV: loto7_4714_k88_loto_2979.csv redova: 2979
Validacija w=1 20 NLL 20.298928505379195
Validacija w=1 40 NLL 18.810328409591875
Validacija w=1 60 NLL 17.459672071354763
Validacija w=1 80 NLL 16.834817367747714
Validacija w=2 20 NLL 20.36333114604365
Validacija w=2 40 NLL 18.198643392014564
Validacija w=2 60 NLL 16.91707859919088
Validacija w=2 80 NLL 16.397347846538235
Validacija w=3 20 NLL 20.933918666185118
Validacija w=3 40 NLL 17.973227079951506
Validacija w=3 60 NLL 16.93572121950502
Validacija w=3 80 NLL 16.43599789423143
Pre testa 20 NLL 22.216884806637367
Pre testa 40 NLL 20.28092382426746
Pre testa 60 NLL 19.31966219338546
Pre testa 80 NLL 18.551965153627965
Izabran prozor: 3 epoha: 80
Validacioni NLL: 17.511237758174342
Hronoloski test NLL: 19.069673110587452
Cela istorija 20 NLL 22.131053204952046
Cela istorija 40 NLL 19.742417396724903
Cela istorija 60 NLL 18.86590952178325
Cela istorija 80 NLL 18.038770290997995
NEXT: 7 10 11 21 22 29 34
Log-verovatnoca unutar modela: -11.24979301905754

CSV: loto7_4714_k88_loto_plus_1735.csv redova: 1735
Validacija w=1 20 NLL 19.557153332537602
Validacija w=1 40 NLL 17.66238388550224
Validacija w=1 60 NLL 16.74475559055702
Validacija w=1 80 NLL 16.18430291036976
Validacija w=2 20 NLL 20.054024128550743
Validacija w=2 40 NLL 18.404346284394574
Validacija w=2 60 NLL 17.286597721501913
Validacija w=2 80 NLL 16.83061249201827
Validacija w=3 20 NLL 20.76957500616648
Validacija w=3 40 NLL 17.782554701978537
Validacija w=3 60 NLL 16.439673104527458
Validacija w=3 80 NLL 15.88158373040206
Pre testa 20 NLL 20.560716926222643
Pre testa 40 NLL 18.848817808941284
Pre testa 60 NLL 17.753925379866136
Izabran prozor: 3 epoha: 60
Validacioni NLL: 18.527294016627835
Hronoloski test NLL: 18.682797211497025
Cela istorija 20 NLL 21.099316643073706
Cela istorija 40 NLL 19.682478777300453
Cela istorija 60 NLL 18.239319504237024
NEXT: 1 2 12 16 17 21 22
Log-verovatnoca unutar modela: -10.036984127909221
ZAVRSEN RAD NAD SVA TRI CSV-a.
"""



"""
Sedam uslovnih kvantnih modela. 
Svaki uči raspodelu broja zavisnu od istorije i prethodnog broja iste kombinacije. 
Maksimum cele kombinacije računa se dinamičkim programiranjem.

- Završno učenje koristi svih N-1 prelaza.
- NLL se računa iz amplituda u logaritamskom obliku, bez nepotrebnog potpoticanja pri kvadriranju.
- Provera gradijenta obuhvata svih 12 parametara, bez ponovnog računanja gradijenta za svaki numerički korak.

Nezavisna matematička provera gradijenta prošla je sa najvećom razlikom 1,57e-11. 


Normalizuje se raspodela cele legalne kombinacije. To čuva uslovne zavisnosti i ne uvodi filtere.
Gradijenti i globalna normalizacija prošli su numeričke provere i poređenje sa Qiskitom.
Maksimum je potvrđen dvema nezavisnim pretragama. 

Prošle su provere amplituda, gradijenata, hronologije, globalne normalizacije i maksimuma.

lokalna normalizacija → normalizacija cele kombinacije; 
12 uglova → trenirajući unitarni operatori; 
"""
