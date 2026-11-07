// This test ensures that aliases are also allowed to be partially matched.

// ignore-order

const EXPECTED = {
    // The full alias name is `getcwd`.
    'query': 'getcw',
    'others': [
        { 'path': 'std::env', 'name': 'current_dir', 'alias': 'getcwd' },
    ],
};
