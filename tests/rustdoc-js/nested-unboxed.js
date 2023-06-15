// exact-check

const EXPECTED = [
    {
        'query': '-> Result<Object, bool>',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Result<Object<i32, u32>, bool>',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Object, bool',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Object<i32, u32>, bool',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> i32, u32, bool',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Result<i32, u32, bool>',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Result<Object<i32>, bool>',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Result<Object<u32>, bool>',
        'others': [
            { 'path': 'nested_unboxed', 'name': 'something' },
        ],
    },
    {
        'query': '-> Result<Object<i32>, u32, bool>',
        'others': [],
    },
    {
        'query': '-> Result<i32, Object<u32>, bool>',
        'others': [],
    },
    {
        'query': '-> Result<i32, u32, Object<bool>>',
        'others': [],
    },
    {
        'query': '-> Result<Object<i32>, Object<u32>, bool>',
        'others': [],
    },
];
