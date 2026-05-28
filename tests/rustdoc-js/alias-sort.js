// rank doc aliases lower than exact matches
// regression test for <https://github.com/rust-lang/rust/issues/140968>

const EXPECTED = {
    'query': 'foobazbar',
    'others': [
        { 'name': 'foobazbar' },
        { 'name': 'foo' },
    ],
};
