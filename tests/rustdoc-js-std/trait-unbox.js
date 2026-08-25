// make sure type-based searches with traits get unboxed too

const EXPECTED = [
    {
        'query': 'any -> result<box>',
        'others': [
            { 'path': 'std::boxed::Box', 'name': 'downcast' },
        ],
    },
    {
        'query': 'split<bufread> -> option<result<vec<u8>>>',
        'others': [
            { 'path': 'std::io::Split', 'name': 'next' },
        ],
    },
];
