// This test ensures that when filtering on `import`, `externcrate` items are also displayed.
// It also ensures that the opposite is not true.

const EXPECTED = [
    {
        'query': 'import:st',
        'others': [
            { 'path': 'foo', 'name': 'st', 'href': '../foo/index.html#reexport.st' },
            {
                'path': 'foo',
                'name': 'st2',
                'href': 'https://doc.rust-lang.org/nightly/std/index.html'
            },
        ],
    },
    {
        'query': 'externcrate:st',
        'others': [
            {
                'path': 'foo',
                'name': 'st2',
                'href': 'https://doc.rust-lang.org/nightly/std/index.html'
            },
        ],
    },
];
