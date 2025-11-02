// rank doc aliases lower than exact matches
// regression test for <https://github.com/rust-lang/rust/issues/140968>

const EXPECTED = {
    'query': 'Foo',
    'others': [
        { 'path': 'alias_rank_lower', 'name': 'Foo' },
        { 'path': 'alias_rank_lower', 'name': 'Bar' },
    ],
};
