// https://github.com/rust-lang/rust/pull/122247
// exact-check

const EXPECTED = {
    'query': 'canonicalvarinfo, intoiterator -> intoiterator',
    'others': [
        { 'path': 'looks_like_rustc_interner::Interner', 'name': 'mk_canonical_var_infos' },
    ],
};
