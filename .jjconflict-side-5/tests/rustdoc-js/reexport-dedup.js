// exact-check

const EXPECTED = [
    {
        'query': 'Subscriber',
        'others': [
            { 'path': 'foo', 'name': 'Subscriber' },
        ],
    },
    {
        'query': 'fmt Subscriber',
        'others': [
            { 'path': 'foo::fmt', 'name': 'Subscriber' },
        ],
    },
    {
        'query': 'AnotherOne',
        'others': [
            { 'path': 'foo', 'name': 'AnotherOne' },
        ],
    },
];
