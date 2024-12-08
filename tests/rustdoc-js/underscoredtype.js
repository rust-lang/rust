const EXPECTED = [
    {
        'query': 'pid_t',
        'correction': null,
        'proposeCorrectionFrom': null,
        'proposeCorrectionTo': null,
        'others': [
            { 'path': 'underscoredtype::unix', 'name': 'pid_t' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'set_pid' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'get_pid' },
        ],
    },
    {
        'query': 'pidt',
        'correction': 'pid_t',
        'proposeCorrectionFrom': null,
        'proposeCorrectionTo': null,
        'others': [
            { 'path': 'underscoredtype::unix', 'name': 'pid_t' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'set_pid' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'get_pid' },
        ],
    },
    {
        'query': 'unix::pid_t',
        'correction': null,
        'proposeCorrectionFrom': null,
        'proposeCorrectionTo': null,
        'others': [
            { 'path': 'underscoredtype::unix', 'name': 'pid_t' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'set_pid' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'get_pid' },
        ],
    },
    {
        'query': 'unix::pidt',
        'correction': 'pid_t',
        'proposeCorrectionFrom': null,
        'proposeCorrectionTo': null,
        'others': [
            { 'path': 'underscoredtype::unix', 'name': 'pid_t' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'set_pid' },
        ],
        'returned': [
            { 'path': 'underscoredtype::unix', 'name': 'get_pid' },
        ],
    },
];
