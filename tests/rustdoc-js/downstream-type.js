// exact-check
// ignore-order

// The FooBar type is defined in upstream_type,
// but used in downstream_type. This means both crates'
// search indexes contain TypeData for it,
// but only upstream_type defines EntryData.
// This test case ensures we can merge them
// when running in CCI mode.
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
