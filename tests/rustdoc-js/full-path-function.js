// exact-check

const EXPECTED = [
    {
        'query': 'sac -> usize',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'bar' },
            { 'path': 'full_path_function::b::Sac', 'name': 'len' },
            { 'path': 'full_path_function::sac::Sac', 'name': 'len' },
        ],
    },
    {
        'query': 'b::sac -> usize',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'bar' },
            { 'path': 'full_path_function::b::Sac', 'name': 'len' },
        ],
    },
    {
        'query': 'b::sac -> u32',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'bar2' },
        ],
    },
];
