// exact-check

const EXPECTED = [
    {
        'query': 'Inside<T> -> Out1<T>',
        'others': [
            { 'path': 'generics_unbox', 'name': 'alpha' },
        ],
    },
    {
        'query': 'Inside<T> -> Out3<T>',
        'others': [
            { 'path': 'generics_unbox', 'name': 'beta' },
        ],
    },
    {
        'query': 'Inside<T> -> Out4<T>',
        'others': [
            { 'path': 'generics_unbox', 'name': 'gamma' },
        ],
    },
    {
        'query': 'Inside<T> -> Out3<U, T>',
        'others': [
            { 'path': 'generics_unbox', 'name': 'gamma' },
        ],
    },
    {
        'query': 'Inside<T> -> Out4<U, T>',
        'others': [
            { 'path': 'generics_unbox', 'name': 'beta' },
        ],
    },
    {
        'query': '-> Sigma',
        'others': [
            { 'path': 'generics_unbox', 'name': 'delta' },
        ],
    },
];
