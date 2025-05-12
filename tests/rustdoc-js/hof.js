// exact-check

const EXPECTED = [
    // not a HOF query
    {
        'query': 'u32 -> !',
        'others': [],
    },

    // ML-style higher-order function notation
    {
        'query': 'bool, (first<u32> -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
    {
        'query': 'u8, (second<u32> -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_once"},
        ],
    },
    {
        'query': 'i8, (third<u32> -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_mut"},
        ],
    },
    {
        'query': 'char, (u32 -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_"},
        ],
    },
    {
        'query': '(first<u32> -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
    {
        'query': '(second<u32> -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_once"},
        ],
    },
    {
        'query': '(third<u32> -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_mut"},
        ],
    },
    {
        'query': '(u32 -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_"},
        ],
    },
    {
        'query': '(str, str -> i8) -> ()',
        'others': [
            {"path": "hof", "name": "multiple"},
        ],
    },
    {
        'query': '(str ->) -> ()',
        'others': [
            {"path": "hof", "name": "multiple"},
        ],
    },
    {
        'query': '(-> i8) -> ()',
        'others': [
            {"path": "hof", "name": "multiple"},
        ],
    },
    {
        'query': '(str -> str) -> ()',
        // params and return are not the same
        'others': [],
    },
    {
        'query': '(i8 ->) -> ()',
        // params and return are not the same
        'others': [],
    },
    {
        'query': '(-> str) -> ()',
        // params and return are not the same
        'others': [],
    },

    // Rust-style higher-order function notation
    {
        'query': 'bool, fn(first<u32>) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
    {
        'query': 'u8, fnonce(second<u32>) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_once"},
        ],
    },
    {
        'query': 'u8, fn(second<u32>) -> ! -> ()',
        // fnonce != fn
        'others': [],
    },
    {
        'query': 'i8, fnmut(third<u32>) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_mut"},
        ],
    },
    {
        'query': 'i8, fn(third<u32>) -> ! -> ()',
        // fnmut != fn
        'others': [],
    },
    {
        'query': 'char, fn(u32) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_"},
        ],
    },
    {
        'query': 'char, fnmut(u32) -> ! -> ()',
        // fn != fnmut
        'others': [],
    },
    {
        'query': 'fn(first<u32>) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
    {
        'query': 'fnonce(second<u32>) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_once"},
        ],
    },
    {
        'query': 'fnmut(third<u32>) -> ! -> ()',
        'others': [
            {"path": "hof", "name": "fn_mut"},
        ],
    },
    {
        'query': 'fn() -> ! -> ()',
        'others': [
            // fn matches primitive:fn and trait:Fn
            {"path": "hof", "name": "fn_"},
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
    {
        'query': 'trait:fn() -> ! -> ()',
        'others': [
            // fn matches primitive:fn and trait:Fn
            {"path": "hof", "name": "fn_"},
        ],
    },
    {
        'query': 'primitive:fn() -> ! -> ()',
        'others': [
            // fn matches primitive:fn and trait:Fn
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
];
