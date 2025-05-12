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
    {
        'query': 'fn:Equivalent::equivalent',
        'others': [
            // These results must never contain `reexport::equivalent::NotEquivalent`,
            // since that path does not exist.
            { 'path': 'equivalent::Equivalent', 'name': 'equivalent' },
            { 'path': 'reexport::NotEquivalent', 'name': 'equivalent' },
        ],
    },
];
