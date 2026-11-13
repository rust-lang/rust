// exact-check

// This test ensures that inlined reexport items keep the `#[doc(alias = "...")]`
// information.
// This is a regression test for <https://github.com/rust-lang/rust/issues/152939>.

const EXPECTED = {
    'query': 'answer',
    'others': [
        { 'path': 'foo', 'name': 'number', 'is_alias': true },
    ],
};
