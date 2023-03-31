// exact-check

const QUERY = [
    '-> Out<First<Second>>',
    '-> Out<Second<First>>',
    '-> Out<First, Second>',
    '-> Out<Second, First>',
];

const EXPECTED = [
    {
        // -> Out<First<Second>>
        'others': [
            { 'path': 'generics_nested', 'name': 'alef' },
        ],
    },
    {
        // -> Out<Second<First>>
        'others': [],
    },
    {
        // -> Out<First, Second>
        'others': [
            { 'path': 'generics_nested', 'name': 'bet' },
        ],
    },
    {
        // -> Out<Second, First>
        'others': [
            { 'path': 'generics_nested', 'name': 'bet' },
        ],
    },
];
