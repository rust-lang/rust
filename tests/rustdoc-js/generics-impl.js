// exact-check

const QUERY = [
    'Aaaaaaa -> u32',
    'Aaaaaaa -> bool',
    'Aaaaaaa -> usize',
    'Read -> u64',
    'trait:Read -> u64',
    'struct:Read -> u64',
    'bool -> u64',
    'Ddddddd -> u64',
    '-> Ddddddd'
];

const EXPECTED = [
    {
        // Aaaaaaa -> u32
        'others': [
            { 'path': 'generics_impl::Aaaaaaa', 'name': 'bbbbbbb' },
        ],
    },
    {
        // Aaaaaaa -> bool
        'others': [
            { 'path': 'generics_impl::Aaaaaaa', 'name': 'ccccccc' },
        ],
    },
    {
        // Aaaaaaa -> usize
        'others': [
            { 'path': 'generics_impl::Aaaaaaa', 'name': 'read' },
        ],
    },
    {
        // Read -> u64
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'eeeeeee' },
            { 'path': 'generics_impl::Ddddddd', 'name': 'ggggggg' },
        ],
    },
    {
        // trait:Read -> u64
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'eeeeeee' },
            { 'path': 'generics_impl::Ddddddd', 'name': 'ggggggg' },
        ],
    },
    {
        // struct:Read -> u64
        'others': [],
    },
    {
        // bool -> u64
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'fffffff' },
        ],
    },
    {
        // Ddddddd -> u64
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'ggggggg' },
        ],
    },
    {
        // -> Ddddddd
        'others': [
            { 'path': 'generics_impl::Ddddddd', 'name': 'hhhhhhh' },
        ],
    },
];
