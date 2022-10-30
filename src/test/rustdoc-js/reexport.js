// exact-check

const QUERY = ['Subscriber', 'AnotherOne'];

const EXPECTED = [
    {
        'others': [
            { 'path': 'reexport::fmt', 'name': 'Subscriber' },
            { 'path': 'reexport', 'name': 'FmtSubscriber' },
        ],
    },
    {
        'others': [
            { 'path': 'reexport', 'name': 'AnotherOne' },
        ],
    },
];
