// exact-check

const QUERY = [
    'R<primitive:slice<P>>',
    'primitive:slice<R<P>>',
    'R<primitive:slice<Q>>',
    'primitive:slice<R<Q>>',
    'R<primitive:array<Q>>',
    'primitive:array<R<Q>>',
    'primitive:array<TraitCat>',
    'primitive:array<TraitDog>',
];

const EXPECTED = [
    {
        // R<primitive:slice<P>>
        'returned': [],
        'in_args': [
            { 'path': 'slice_array', 'name': 'alpha' },
        ],
    },
    {
        // primitive:slice<R<P>>
        'returned': [
            { 'path': 'slice_array', 'name': 'alef' },
        ],
        'in_args': [],
    },
    {
        // R<primitive:slice<Q>>
        'returned': [],
        'in_args': [],
    },
    {
        // primitive:slice<R<Q>>
        'returned': [],
        'in_args': [],
    },
    {
        // R<primitive:array<Q>>
        'returned': [
            { 'path': 'slice_array', 'name': 'bet' },
        ],
        'in_args': [],
    },
    {
        // primitive:array<R<Q>>
        'returned': [],
        'in_args': [
            { 'path': 'slice_array', 'name': 'beta' },
        ],
    },
    {
        // primitive::array<TraitCat>
        'in_args': [
            { 'path': 'slice_array', 'name': 'gamma' },
        ],
    },
    {
        // primitive::array<TraitDog>
        'in_args': [
            { 'path': 'slice_array', 'name': 'gamma' },
        ],
    },
];
