// exact-check

const EXPECTED = [
    {
        'query': '-> Out<First<Second>>',
        'others': [
            { 'path': 'generics_nested', 'name': 'alef' },
        ],
    },
    {
        'query': '-> Out<Second<First>>',
        'others': [],
    },
    {
        'query': '-> Out<First, Second>',
        'others': [
            { 'path': 'generics_nested', 'name': 'bet' },
        ],
    },
    {
        'query': '-> Out<Second, First>',
        'others': [
            { 'path': 'generics_nested', 'name': 'bet' },
        ],
    },
];
