// This test ensures that search results are not duplicated between `std` and `core`.
// Regression test for <https://github.com/rust-lang/rust/issues/131670>.

// exact-check

const EXPECTED = [
    {
        'query': 'primitive:char',
        'others': [
            { 'path': 'std', 'name': 'char' },
        ],
    },
    {
        'query': 'char->u32',
        'others': [
            { 'path': 'std::u32', 'name': 'from' },
            { 'path': 'std::char', 'name': 'to_u32' },
            { 'path': 'std::char', 'name': 'to_digit' },
        ],
    },
];
