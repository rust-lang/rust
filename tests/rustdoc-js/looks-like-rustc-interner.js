// https://github.com/rust-lang/rust/pull/122247
// exact-check

const EXPECTED = [
    {
        'query': 'CanonicalVarKind, intoiterator -> intoiterator',
        'others': [],
    },
    {
        'query': '[CanonicalVarKind], interner<tys=intoiterator> -> intoiterator',
        'others': [
            { 'path': 'looks_like_rustc_interner::Interner', 'name': 'mk_canonical_var_kinds' },
        ],
    },
];
