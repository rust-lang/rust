const EXPECTED = [
    {
        'query': 'Vec::new',
        'others': [
            { 'path': 'std::vec::Vec', 'name': 'new' },
            { 'path': 'std::vec::Vec', 'name': 'new_in' },
        ],
    },
    {
        'query': 'prelude::vec',
        'others': [
            { 'path': 'std::prelude::v1', 'name': 'Vec' },
        ],
    },
    {
        'query': 'Vec new',
        'others': [
            { 'path': 'std::vec::Vec', 'name': 'new' },
            { 'path': 'std::vec::Vec', 'name': 'new_in' },
        ],
    },
    {
        'query': 'std::Vec::new',
        'others': [
            { 'path': 'std::vec::Vec', 'name': 'new' },
            { 'path': 'std::vec::Vec', 'name': 'new_in' },
        ],
    },
    {
        'query': 'std Vec new',
        'others': [
            { 'path': 'std::vec::Vec', 'name': 'new' },
            { 'path': 'std::vec::Vec', 'name': 'new_in' },
        ],
    },
    {
        'query': 'alloc::Vec::new',
        'others': [
            { 'path': 'alloc::vec::Vec', 'name': 'new' },
            { 'path': 'alloc::vec::Vec', 'name': 'new_in' },
        ],
    },
    {
        'query': 'alloc Vec new',
        'others': [
            { 'path': 'alloc::vec::Vec', 'name': 'new' },
            { 'path': 'alloc::vec::Vec', 'name': 'new_in' },
        ],
    },
];
