// exact-check

const EXPECTED = [
    {
        'query': 'Subscriber',
        'others': [
            { 'path': 'reexport::fmt', 'name': 'Subscriber' },
            { 'path': 'reexport', 'name': 'FmtSubscriber' },
        ],
    },
    {
        'query': 'AnotherOne',
        'others': [
            { 'path': 'reexport', 'name': 'AnotherOne' },
        ],
    },
];
