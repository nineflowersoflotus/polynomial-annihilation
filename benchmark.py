"""Reproducible examples; all input files are generated in memory."""
import bz2
import gzip
import hashlib
import lzma
import zlib

import numpy as np

from annihilation import compress, decompress, field, model


def datasets(seed=20261124, size=16384):
    rng = np.random.default_rng(seed)
    T, A, _, I = field(256)
    x, y = rng.integers(0, 256, (2, size // 4))
    records = np.column_stack((A[x, y], T[x, y], x, y)).astype('u1')
    yield 'Sum and product', records.tobytes()
    noisy = records.copy()
    ids = rng.choice(len(noisy), len(noisy) // 100, replace=False)
    noisy[ids, 3] ^= 1
    yield 'Sum and product (1% noise)', noisy.tobytes()

    inputs = rng.integers(0, 256, (size // 4, 2))
    yield 'AES field inversion', np.column_stack((inputs, I[inputs])).astype('u1').tobytes()

    inputs = rng.integers(0, 256, (size // 8, 4))
    matrix = np.array([[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]])
    outputs = np.column_stack([np.bitwise_xor.reduce(T[inputs, row], axis=1) for row in matrix])
    yield 'AES MixColumns', np.column_stack((inputs, outputs)).astype('u1').tobytes()

    yield 'Uniform random bytes', rng.bytes(size)
    yield 'SHA-256 digests', b''.join(hashlib.sha256(i.to_bytes(8, 'big')).digest() for i in range(size // 32))


def entropy(data):
    counts = np.bincount(np.frombuffer(data, np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return -np.sum(p * np.log2(p))


def equations(meta):
    if meta['m'] != 'system':
        return ''
    names = [f'a{i}' for i in range(meta['c'])] + ['x', 'y']
    polys = []
    for eq in meta['eq']:
        terms = []
        for mon, coefficient in eq:
            term = '*'.join(names[i] for i in mon)
            terms.append(term if coefficient == 1 and term else str(coefficient) + ('*' + term if term else ''))
        polys.append(' + '.join(terms) + ' = 0')
    return '; '.join(polys)


def main():
    print('| Dataset | Bytes | Byte entropy | gzip | bzip2 | xz | zlib | Polynomial | Mode |', flush=True)
    print('|---|---:|---:|---:|---:|---:|---:|---:|---|', flush=True)
    found = []
    for name, data in datasets():
        archive = compress(data)
        assert decompress(archive) == data
        meta = model(archive)
        alternatives = [gzip.compress(data, compresslevel=6, mtime=0), bz2.compress(data), lzma.compress(data, preset=6), zlib.compress(data, 6)]
        assert [gzip.decompress(alternatives[0]), bz2.decompress(alternatives[1]), lzma.decompress(alternatives[2]), zlib.decompress(alternatives[3])] == [data] * 4
        sizes = ' | '.join(str(len(a)) for a in alternatives)
        print(f'| {name} | {len(data)} | {entropy(data):.4f} | {sizes} | {len(archive)} | {meta["m"]} |', flush=True)
        if meta['m'] == 'system':
            found.append(f'{name}: GF({meta["q"]}), {equations(meta)}')
    print('\nRecovered equations:')
    print('\n'.join(found))


if __name__ == '__main__':
    main()
