// exact-check

const EXPECTED = [
    // ML-style higher-order function notation
    {
        'query': 'bool, (u32 -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_ptr"},
        ],
    },
    {
        'query': 'u8, (u32 -> !) -> ()',
        'others': [
            {"path": "hof", "name": "fn_once"},
        ],
    },
    {
        'query': 'i8, (u32 -> !) -> ()',
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
            {"path": "hof", "name": "fn_ptr"},
            {"path": "hof", "name": "fn_mut"},
            {"path": "hof", "name": "fn_once"},
        ],
    },
    {
        'query': 'u32 -> !',
        // not a HOF query
        'others': [],
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
];
