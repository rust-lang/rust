// exact-check
// ignore-order

// https://github.com/rust-lang/rust/issues/162334
const EXPECTED = [
    {
        'query': 'FooBar',
        'others': [
            {
                'path': 'upstream_type',
                'name': 'FooBar',
            },
        ],
        'in_args': [
            {
                'path': 'downstream_type',
                'name': 'downstream_fn',
                'desc': 'https://github.com/rust-lang/rust/issues/162334',
            },
        ],
        'returned': [
            {
                'path': 'upstream_type',
                'name': 'overlapping_name',
                'desc': 'Test case for overlapping function and struct name',
            },
        ],
    },
    {
        'query': 'overlapping_name',
        'others': [
            {
                'path': 'upstream_type',
                'name': 'overlapping_name',
                'ty': 5,
            },
            {
                'path': 'upstream_type',
                'name': 'overlapping_name',
                'ty': 7,
            },
        ],
        'returned': [],
        'in_args': [
            {
                'path': 'downstream_type',
                'name': 'with_overlap',
                'desc': '',
            },
        ]
    },
];
