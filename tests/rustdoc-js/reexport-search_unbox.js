// exact-check

// This test ensures that `search_unbox` works even on inlined reexports.

const EXPECTED = [
    {
        'query': 'Inside<T> -> Out1<T>',
        'others': [
            { 'path': 'foo', 'name': 'alpha' },
        ],
    },
]
