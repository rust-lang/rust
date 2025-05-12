const EXPECTED = [
    {
        'query': 'write',
        'others': [
            { 'path': 'std::fmt', 'name': 'write' },
            { 'path': 'std::fs', 'name': 'write' },
            { 'path': 'std::ptr', 'name': 'write' },
            { 'path': 'std::fmt', 'name': 'Write' },
            { 'path': 'std::io', 'name': 'Write' },
            { 'path': 'std::hash::Hasher', 'name': 'write' },
        ],
    },
    {
        'query': 'Write',
        'others': [
            { 'path': 'std::fmt', 'name': 'Write' },
            { 'path': 'std::io', 'name': 'Write' },
            { 'path': 'std::fmt', 'name': 'write' },
            { 'path': 'std::fs', 'name': 'write' },
            { 'path': 'std::ptr', 'name': 'write' },
            { 'path': 'std::hash::Hasher', 'name': 'write' },
        ],
    },
];
