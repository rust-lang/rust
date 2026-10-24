const EXPECTED = [
    {
        'query': 'waker_from',
        'others': [
            { 'path': 'substring::SuperWaker', 'name': 'local_waker_from_nonlocal' },
            { 'path': 'substring::SuperWakerTask', 'name': 'local_waker_from_nonlocal' },
        ],
    },
    {
        'query': 'my',
        'others': [
            { 'path': 'substring', 'name': 'm_y_substringmatching' },
        ],
    },
];
