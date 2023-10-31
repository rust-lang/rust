// exact-check

const EXPECTED = [
    {
        'query': 'Aaaaaaa -> u32',
        'others': [
            { 'path': 'generics_impl::Aaaaaaa', 'name': 'bbbbbbb' },
        ],
    },
    {
        'query': 'Aaaaaaa -> bool',
        'others': [
            { 'path': 'generics_impl::Aaaaaaa', 'name': 'ccccccc' },
        ],
    },
    {
        'query': 'Aaaaaaa -> usize',
        'others': [
            { 'path': 'generics_impl::Aaaaaaa', 'name': 'read' },
        ],
    },
    {
        'query': 'Read -> u64',
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'eeeeeee' },
            { 'path': 'generics_impl::Ddddddd', 'name': 'ggggggg' },
        ],
    },
    {
        'query': 'trait:Read -> u64',
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'eeeeeee' },
            { 'path': 'generics_impl::Ddddddd', 'name': 'ggggggg' },
        ],
    },
    {
        'query': 'struct:Read -> u64',
        'others': [],
    },
    {
        'query': 'bool -> u64',
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'fffffff' },
        ],
    },
    {
        'query': 'Ddddddd -> u64',
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'ggggggg' },
        ],
    },
    {
        'query': '-> Ddddddd',
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'hhhhhhh' },
        ],
    },
];
