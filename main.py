import argparse
import itertools as it
import json
import struct
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

@lru_cache(None)
def field(q):
    """Multiplication, addition, subtraction and inverse lookup tables."""
    # GF(256) uses the AES irreducible polynomial 0x11b.
    a = np.arange(q, dtype=np.int64)
    if q == 256:
        x = np.broadcast_to(a[:, None], (q, q)).copy()
        y = np.broadcast_to(a[None, :], (q, q)).copy()
        t = np.zeros_like(x)
        for _ in range(8):
            t ^= np.where(y & 1, x, 0)
            y >>= 1
            x = x << 1 ^ np.where(x & 128, 0x11b, 0)
        add = np.bitwise_xor(a[:, None], a[None, :])
        sub = add
    else:
        t = a[:, None] * a[None, :] % q
        add = (a[:, None] + a[None, :]) % q
        sub = (a[:, None] - a[None, :]) % q
    inv = np.zeros(q, dtype=np.int64)
    for i in range(1, q):
        inv[i] = np.flatnonzero(t[i] == 1)[0]
    return (t, add, sub, inv)

def wrap(meta, raw):
    return struct.pack('<I', len(raw)) + pack(meta, raw)

def rows_view(data, w, q):
    unit = 2 if q == 257 else 1
    width = w * unit
    count = len(data) // width * width
    rows = np.frombuffer(data[:count], dtype='<u2' if unit == 2 else 'u1').reshape(-1, w)
    if q != 256 and np.any(rows >= q):
        return (None, None)
    return (rows, data[count:])

@lru_cache(None)
def quadratic_roots(q):
    T, A, S, I = field(q)
    roots = np.full((q, q, 2), -1, np.int16)
    counts = np.zeros((q, q), np.uint8)
    for x in range(q):
        b = np.arange(q)
        c = S[0, A[T[x, x], T[b, x]]]
        slot = counts[b, c]
        roots[b, c, slot] = x
        counts[b, c] += 1
    return (roots, counts)

def packbits(x):
    return np.packbits(x, bitorder='little').tobytes()

def pack(meta, raw):
    """Serialize the equations and compress the context/ranks/exceptions."""
    header = json.dumps(meta, separators=(",", ":")).encode()
    return struct.pack("<H", len(header)) + header + zlib.compress(raw, 6)


def unpack(payload, nbytes):
    if len(payload) < 6:
        raise ValueError("truncated polynomial payload")
    rawlen, size = struct.unpack_from("<IH", payload)
    if rawlen > 3 * nbytes + 64 or size > len(payload) - 6:
        raise ValueError("invalid polynomial payload length")
    meta = json.loads(payload[6:6 + size])
    d = zlib.decompressobj()
    raw = d.decompress(payload[6 + size:], rawlen + 1)
    if len(raw) != rawlen or not d.eof or d.unused_data or d.unconsumed_tail:
        raise ValueError("invalid compressed body")
    return meta, raw


FIELDS = (256, 251, 257)
CONFIGS = ((2, 2), (1, 2), (3, 2), (4, 2), (2, 3), (6, 2), (1, 3), (3, 3), (4, 3))
MAX_EQ = 4

def dictionary(c, d):
    m = [()] + [t for j in range(1, d + 1) for t in it.combinations_with_replacement(range(c + 2), j)]
    return sorted(m, key=lambda t: (any((i >= c for i in t)), len(t), t))

def features(rows, mons, q):
    T = field(q)[0]
    cols = []
    for m in mons:
        a = np.ones(len(rows), np.int64)
        for j in m:
            a = T[a, rows[:, j]]
        cols.append(a)
    return np.column_stack(cols)

def nullspace(rows, mons, q):
    """Row-reduce the monomial matrix over GF(q) to find annihilators."""
    T, A, S, I = field(q)
    a = features(rows, mons, q)
    r = 0
    piv = []
    free = []
    for j in range(len(mons)):
        ids = np.flatnonzero(a[r:, j])
        if not len(ids):
            free.append(j)
            continue
        i = r + int(ids[0])
        a[[r, i]] = a[[i, r]]
        a[r] = T[a[r], I[a[r, j]]]
        ids = np.flatnonzero(a[:, j])
        ids = ids[ids != r]
        a[ids] = S[a[ids], T[a[ids, j, None], a[r, None, :]]]
        piv.append(j)
        r += 1
    return [[[list(mons[h]), int(S[0, a[i, j]])] for i, h in enumerate(piv) if a[i, j]] + [[list(mons[j]), 1]] for j in free]

def evaluate(rows, terms, q):
    T, A, _, _ = field(q)
    o = np.zeros(len(rows), np.int64)
    for m, v in terms:
        p = np.full(len(rows), v, np.int64)
        for j in m:
            p = T[p, rows[:, j]]
        o = A[o, p]
    return o

def equation_roots(ctx, terms, q):
    """For every possible x, return sorted y roots; count q means all y."""
    T, A, S, I = field(q)
    c = ctx.shape[1]
    n = len(ctx)
    co = np.zeros((3, n, q), np.int64)
    xx = np.arange(q)
    for m, v in terms:
        p = np.full(n, v, np.int64)
        for j in m:
            if j < c:
                p = T[p, ctx[:, j]]
        xp = np.ones(q, np.int64)
        for _ in range(m.count(c)):
            xp = T[xp, xx]
        e = m.count(c + 1)
        co[e] = A[co[e], T[p[:, None], xp[None, :]]]
    a, b, d = (co[2], co[1], co[0])
    count = np.zeros((n, q), np.int16)
    rr = np.full((n, q, 2), -1, np.int16)
    lin = (a == 0) & (b != 0)
    rr[:, :, 0][lin] = T[S[0, d[lin]], I[b[lin]]]
    count[lin] = 1
    free = (a == 0) & (b == 0) & (d == 0)
    count[free] = q
    quad = a != 0
    if np.any(quad):
        rt, ct = quadratic_roots(q)
        bn = T[b[quad], I[a[quad]]]
        dn = T[d[quad], I[a[quad]]]
        rr[quad] = rt[bn, dn]
        count[quad] = ct[bn, dn]
    return (rr, count)

def intersect(left, right, q):
    r, n = left
    s, m = right
    out = np.full_like(r, -1)
    cnt = np.zeros_like(n)
    af = n == q
    bf = m == q
    out[af] = s[af]
    cnt[af] = m[af]
    out[bf] = r[bf]
    cnt[bf] = n[bf]
    both = ~(af | bf)
    h0 = both & (r[:, :, 0] >= 0) & ((r[:, :, 0] == s[:, :, 0]) | (r[:, :, 0] == s[:, :, 1]))
    h1 = both & (r[:, :, 1] >= 0) & ((r[:, :, 1] == s[:, :, 0]) | (r[:, :, 1] == s[:, :, 1]))
    out[:, :, 0][h0] = r[:, :, 0][h0]
    out[:, :, 0][~h0 & h1] = r[:, :, 1][~h0 & h1]
    out[:, :, 1][h0 & h1] = r[:, :, 1][h0 & h1]
    cnt[both] = (h0.astype(np.int16) + h1.astype(np.int16))[both]
    return (out, cnt)

def solutions(ctx, eqs, q):
    itera = iter(eqs)
    res = equation_roots(ctx, next(itera), q)
    for eq in itera:
        res = intersect(res, equation_roots(ctx, eq, q), q)
    return res

def widths(n):
    w = np.zeros(len(n), np.int16)
    for k in range(1, 19):
        w[n > 1 << k - 1] = k
    return w

def classify(rows, sol, q):
    """Membership, solution count and lexicographic rank for each pair."""
    r, n = sol
    i = np.arange(len(rows))
    x, y = rows[:, -2:].T
    nr = n[i, x]
    rr = r[i, x]
    valid = (nr == q) | (y == rr[:, 0]) | (y == rr[:, 1])
    total = n.sum(axis=1)
    base = np.cumsum(n, axis=1)[i, x] - nr
    rank = base + np.where(nr == q, y, (y == rr[:, 1]).astype(np.int64))
    return (valid, total, rank)

def cost(rows, sol, q):
    valid, total, _ = classify(rows, sol, q)
    return (float(np.mean(np.where(valid, widths(total), 2 * (8 if q != 257 else 16) + 1))), float(np.mean(valid)))

def pack_indices(values, width):
    acc = bits = 0
    out = bytearray()
    for v, w in zip(values, width):
        acc |= int(v) << bits
        bits += int(w)
        while bits >= 8:
            out.append(acc & 255)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc)
    return bytes(out)

def unpack_indices(raw, width):
    acc = bits = p = 0
    for w in width:
        w = int(w)
        while bits < w:
            if p >= len(raw):
                raise ValueError('truncated rank stream')
            acc |= raw[p] << bits
            bits += 8
            p += 1
        yield (acc & (1 << w) - 1)
        acc >>= w
        bits -= w
    if p != len(raw) or acc:
        raise ValueError('extra rank bits')

def payload(rows, tail, q, eqs):
    c = rows.shape[1] - 2
    valid = []
    total = []
    rank = []
    for k in range(0, len(rows), 256):
        r = rows[k:k + 256]
        v, t, i = classify(r, solutions(r[:, :c], eqs, q), q)
        valid.extend(v)
        total.extend(t)
        rank.extend(i)
    valid = np.asarray(valid)
    total = np.asarray(total)
    rank = np.asarray(rank)
    w = widths(total[valid])
    rankbytes = pack_indices(rank[valid], w)
    dt = '<u2' if q == 257 else 'u1'
    mask = packbits(~valid) if not np.all(valid) else b''
    raw = rows[:, :c].astype(dt).tobytes() + mask + rows[~valid, -2:].astype(dt).tobytes() + rankbytes + tail
    meta = {'m': 'system', 'q': q, 'c': c, 'eq': eqs, 'n': len(rows), 'e': int((~valid).sum()), 'b': len(rankbytes), 'tail': len(tail)}
    return (wrap(meta, raw), meta, {'valid': float(valid.mean()), 'unique': float(np.mean(valid & (total == 1))), 'mean_roots': float(total.mean()), 'mean_rank_bits': float(w.sum() / len(rows)), 'rank_bytes': len(rankbytes)})

def discover(rows, q, d, windows=3):
    """Fit on training windows and greedily select on held-out records."""
    c = rows.shape[1] - 2
    mons = [m for m in dictionary(c, d) if m.count(c + 1) <= 2]
    size = len(mons) + max(24, len(mons) // 2)
    end = len(rows) * 3 // 4
    if end < size or len(rows) - end < 64:
        return
    val = rows[np.linspace(end, len(rows) - 1, min(256, len(rows) - end), dtype=int)]
    seen = set()
    models = []
    for start in np.unique(np.linspace(0, end - size, windows, dtype=int)):
        basis = nullspace(rows[start:start + size], mons, q)
        for eq in basis:
            if not any((any((i >= c for i in m)) for m, v in eq)):
                continue
            key = tuple(((tuple(m), v) for m, v in eq))
            if key in seen:
                continue
            seen.add(key)
            hit = np.mean(evaluate(val, eq, q) == 0)
            if hit >= 0.85:
                models.append((eq, hit))
    models.sort(key=lambda t: (-t[1], len(t[0])))
    models = models[:16]
    if not models:
        return
    cached = [equation_roots(val[:, :c], eq, q) for eq, h in models]
    selected = []
    sol = None
    old = 2 * (8 if q != 257 else 16)
    T, A, S, I = field(q)
    index = {m: i for i, m in enumerate(mons)}
    piv = {}
    for _ in range(MAX_EQ):
        opts = []
        for j, (eq, h) in enumerate(models):
            if j in selected:
                continue
            v = np.zeros(len(mons), np.int64)
            for m, a in eq:
                v[index[tuple(m)]] = a
            for k, p in sorted(piv.items()):
                if v[k]:
                    v = S[v, T[v[k], p]]
            nz = np.flatnonzero(v)
            if not len(nz):
                continue
            s = cached[j] if sol is None else intersect(sol, cached[j], q)
            co, hit = cost(val, s, q)
            if hit >= 0.85:
                opts.append((co, len(eq), j, s, v, int(nz[0]), hit))
        if not opts:
            break
        co, _, j, s, v, k, hit = min(opts, key=lambda x: x[:3])
        if co >= old - 0.05:
            break
        selected.append(j)
        sol = s
        old = co
        piv[k] = T[v, I[v[k]]]
        yield ([models[j][0] for j in selected], {'validation_bits': co, 'validation_hit': hit, 'basis_candidates': len(models)})

def candidates(data, configs=CONFIGS):
    if len(data) < 256:
        return
    for q in FIELDS:
        for c, d in configs:
            rows, tail = rows_view(data, c + 2, q)
            if rows is None:
                continue
            for eqs, info in discover(rows, q, d):
                p, m, stats = payload(rows, tail, q, eqs)
                yield (p, m, {**info, **stats, 'degree': d})

def best_model(data):
    best = None
    meta = None
    for p, m, st in candidates(data):
        if len(p) < len(data) and (best is None or len(p) < len(best)):
            best, meta = (p, m)
    return (best, meta)

def decode_system(meta, raw, nbytes):
    q = meta['q']
    c = meta['c']
    n = meta['n']
    e = meta['e']
    b = meta['b']
    tail = meta['tail']
    eq = meta['eq']
    if q not in FIELDS or not 1 <= c <= 6 or (not 1 <= len(eq) <= MAX_EQ) or (not 0 <= e <= n) or (not 0 <= b <= 3 * n):
        raise ValueError('system metadata')
    unit = 2 if q == 257 else 1
    dt = '<u2' if unit == 2 else 'u1'
    if n * (c + 2) * unit + tail != nbytes or not 0 <= tail < (c + 2) * unit:
        raise ValueError('system size')
    if any((not 1 <= len(f) <= 200 or any((len(m) > 3 or m.count(c + 1) > 2 or any((type(j) != int or not 0 <= j < c + 2 for j in m)) or (type(v) != int) or (not 0 < v < q) for m, v in f)) for f in eq)):
        raise ValueError('system terms')
    a = n * c * unit
    ms = (n + 7) // 8 if e else 0
    l = 2 * e * unit
    if a + ms + l + b + tail != len(raw):
        raise ValueError('system body')
    ctx = np.frombuffer(raw[:a], dtype=dt).reshape(n, c)
    mask = np.unpackbits(np.frombuffer(raw[a:a + ms], np.uint8), bitorder='little')[:n].astype(bool) if e else np.zeros(n, bool)
    lit = np.frombuffer(raw[a + ms:a + ms + l], dtype=dt).reshape(e, 2)
    if mask.sum() != e or np.any(ctx >= q) or np.any(lit >= q):
        raise ValueError('system literals')
    out = np.empty((n, c + 2), dtype=dt)
    out[:, :c] = ctx
    out[mask, -2:] = lit
    ids = np.flatnonzero(~mask)
    chunks = []
    ww = []
    for k in range(0, len(ids), 256):
        j = ids[k:k + 256]
        r, cnt = solutions(ctx[j], eq, q)
        tot = cnt.sum(axis=1)
        if np.any(tot == 0):
            raise ValueError('empty solution set')
        ww.extend(widths(tot))
    ranks = iter(unpack_indices(raw[a + ms + l:a + ms + l + b], ww))
    for k in range(0, len(ids), 256):
        j = ids[k:k + 256]
        r, cnt = solutions(ctx[j], eq, q)
        cum = np.cumsum(cnt, axis=1)
        for h, i in enumerate(j):
            rank = next(ranks)
            x = int(np.searchsorted(cum[h], rank, side='right'))
            if x >= q:
                raise ValueError('rank outside solution set')
            off = rank - (int(cum[h, x - 1]) if x else 0)
            out[i, -2] = x
            out[i, -1] = off if cnt[h, x] == q else r[h, x, off]
    try:
        next(ranks)
        raise ValueError('extra rank')
    except StopIteration:
        pass
    return out.tobytes() + (raw[-tail:] if tail else b'')

# The outer header includes the original size and the selected coding mode.
HEADER = struct.Struct("<4sQB")
MAGIC = b"PAN1"


def compress(data):
    """Try the polynomial search; fall back to zlib or raw bytes."""
    candidate, _ = best_model(data)
    options = [(0, data), (1, zlib.compress(data, 6))]
    if candidate is not None:
        options.append((2, candidate))
    mode, body = min(options, key=lambda item: len(item[1]))
    return HEADER.pack(MAGIC, len(data), mode) + body


def decompress(archive):
    magic, nbytes, mode = HEADER.unpack_from(archive)
    if magic != MAGIC:
        raise ValueError("invalid archive magic")
    body = archive[HEADER.size:]
    if mode == 0:
        data = body
    elif mode == 1:
        d = zlib.decompressobj()
        data = d.decompress(body, nbytes + 1)
        if not d.eof or d.unused_data or d.unconsumed_tail:
            raise ValueError("invalid zlib body")
    elif mode == 2:
        meta, raw = unpack(body, nbytes)
        data = decode_system(meta, raw, nbytes)
    else:
        raise ValueError("unknown coding mode")
    if len(data) != nbytes:
        raise ValueError("decoded length mismatch")
    return data


def model(archive):
    """Return the selected coding mode and, if present, its polynomial model."""
    magic, nbytes, mode = HEADER.unpack_from(archive)
    if magic != MAGIC:
        raise ValueError("invalid archive magic")
    if mode == 2:
        meta, _ = unpack(archive[HEADER.size:], nbytes)
        return meta
    return {"m": {0: "raw", 1: "zlib"}[mode]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("encode", "decode", "inspect"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    data = args.input.read_bytes()
    if args.action == "inspect":
        print(json.dumps(model(data), indent=2))
    else:
        if args.output is None:
            parser.error("encode/decode requires an output path")
        result = compress(data) if args.action == "encode" else decompress(data)
        args.output.write_bytes(result)


if __name__ == "__main__":
    main()
