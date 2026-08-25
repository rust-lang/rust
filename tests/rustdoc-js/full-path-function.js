// exact-check

const EXPECTED = [
    {
        'query': 'sac -> usize',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'len' },
            { 'path': 'full_path_function::sac::Sac', 'name': 'len' },
            { 'path': 'full_path_function::b::Sac', 'name': 'bar' },
        ],
    },
    {
        'query': 'b::sac -> usize',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'len' },
            { 'path': 'full_path_function::b::Sac', 'name': 'bar' },
        ],
    },
    {
        'query': 'b::sac -> u32',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'bar2' },
        ],
    },
    {
        'query': 'string::string -> u32',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'string' },
        ],
    },
    {
        'query': 'alloc::string::string -> u32',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'string' },
        ],
    },
    {
        'query': 'alloc::string -> u32',
        'others': [
            { 'path': 'full_path_function::b::Sac', 'name': 'string' },
        ],
    },
];
