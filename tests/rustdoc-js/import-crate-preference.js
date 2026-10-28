// This test ensures that imports have lower ranking than other matches,
// even if those matches have shorter names.

const EXPECTED = [
    {
        'query': 'Eepy',
        'others': [
            { 'path': 'zzzzz', 'name': 'Eepy' },
        ],
    },
    {
        'query': 'zzzzz::Eepy',
        'others': [
            { 'path': 'zzzzz', 'name': 'Eepy' },
        ],
    },
    {
        'query': 'foo::Eepy',
        'others': [
            { 'path': 'foo', 'name': 'Eepy' },
        ],
    },
];
