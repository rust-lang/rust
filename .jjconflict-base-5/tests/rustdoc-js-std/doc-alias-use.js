// AsciiChar has a doc alias on its reexport and we
// want to make sure that actually works correctly,
// since apperently there are no other tests for this.

const EXPECTED = [
    {
        'query': 'AsciiChar',
        'others': [
            { 'path': 'core::ascii', 'name': 'Char' },
        ],
    },
];
