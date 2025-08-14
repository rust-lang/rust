// exact-check

// This test ensures that unstable items are sorted last.

const EXPECTED = [
    {
        'query': 'bar',
        'others': [
            { 'path': 'unstable', 'name': 'bar2' },
            { 'path': 'unstable', 'name': 'bar1' },
        ],
    },
];
