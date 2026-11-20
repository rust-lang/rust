// exact-check

const EXPECTED = [
    {
        'query': 'R<primitive:slice<P>>',
        'returned': [],
        'in_args': [
            { 'path': 'slice_array', 'name': 'alpha' },
        ],
    },
    {
        'query': 'primitive:slice<R<P>>',
        'returned': [
            { 'path': 'slice_array', 'name': 'alef' },
        ],
        'in_args': [],
    },
    {
        'query': 'R<primitive:slice<Q>>',
        'returned': [],
        'in_args': [],
    },
    {
        'query': 'primitive:slice<R<Q>>',
        'returned': [],
        'in_args': [],
    },
    {
        'query': 'R<primitive:array<Q>>',
        'returned': [
            { 'path': 'slice_array', 'name': 'bet' },
        ],
        'in_args': [],
    },
    {
        'query': 'primitive:array<R<Q>>',
        'returned': [],
        'in_args': [
            { 'path': 'slice_array', 'name': 'beta' },
        ],
    },
    {
        'query': 'primitive:array<TraitCat>',
        'in_args': [
            { 'path': 'slice_array', 'name': 'gamma' },
        ],
    },
    {
        'query': 'primitive:array<TraitDog>',
        'in_args': [
            { 'path': 'slice_array', 'name': 'gamma' },
        ],
    },
    {
        'query': '[TraitCat]',
        'in_args': [
            { 'path': 'slice_array', 'name': 'gamma' },
            { 'path': 'slice_array', 'name': 'epsilon' },
        ],
    },
    {
        'query': 'R<[Q]>',
        'returned': [
            { 'path': 'slice_array', 'name': 'bet' },
        ],
    },
    {
        'query': 'R<[P]>',
        'in_args': [
            { 'path': 'slice_array', 'name': 'alpha' },
        ],
    },
];
