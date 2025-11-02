// This test ensures that when filtering on `import`, `externcrate` items are also displayed.
// It also ensures that the opposite is not true.

const EXPECTED = [
    {
        'query': 'import:st',
        'others': [
            { 'path': 'foo', 'name': 'st', 'href': '../foo/index.html#reexport.st' },
            // FIXME: `href` is wrong: <https://github.com/rust-lang/rust/issues/148300>
            { 'path': 'foo', 'name': 'st2', 'href': '../st2/index.html' },
        ],
    },
    {
        'query': 'externcrate:st',
        'others': [
            // FIXME: `href` is wrong: <https://github.com/rust-lang/rust/issues/148300>
            { 'path': 'foo', 'name': 'st2', 'href': '../st2/index.html' },
        ],
    },
];
