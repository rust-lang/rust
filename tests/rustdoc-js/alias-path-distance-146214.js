// exact-check

// consider path distance for doc aliases
// regression test for <https://github.com/rust-lang/rust/issues/146214>

const EXPECTED = {
    'query': 'Foo::zzz',
    'others': [{ 'path': 'alias_path_distance::Foo', 'name': 'baz' }],
};
