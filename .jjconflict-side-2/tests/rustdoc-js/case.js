const EXPECTED = [
    {
        'query': 'Foo',
        'others': [
            { 'path': 'case', 'name': 'Foo', 'desc': 'Docs for Foo' },
            { 'path': 'case', 'name': 'foo', 'desc': 'Docs for foo' },
        ],
    },
    {
        'query': 'foo',
        'others': [
            // https://github.com/rust-lang/rust/issues/133017
            { 'path': 'case', 'name': 'Foo', 'desc': 'Docs for Foo' },
            { 'path': 'case', 'name': 'foo', 'desc': 'Docs for foo' },
        ],
    },
];
