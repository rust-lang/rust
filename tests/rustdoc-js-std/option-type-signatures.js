// ignore-order

const FILTER_CRATE = "std";

const EXPECTED = [
    {
        'query': 'option, fnonce -> option',
        'others': [
            { 'path': 'std::option::Option', 'name': 'map' },
        ],
    },
    {
        'query': 'option -> default',
        'others': [
            { 'path': 'std::option::Option', 'name': 'unwrap_or_default' },
            { 'path': 'std::option::Option', 'name': 'get_or_insert_default' },
        ],
    },
    {
        'query': 'option -> []',
        'others': [
            { 'path': 'std::option::Option', 'name': 'as_slice' },
            { 'path': 'std::option::Option', 'name': 'as_mut_slice' },
        ],
    },
    {
        'query': 'option<t>, option<t> -> option<t>',
        'others': [
            { 'path': 'std::option::Option', 'name': 'or' },
            { 'path': 'std::option::Option', 'name': 'xor' },
        ],
    },
    {
        'query': 'option<t>, option<u> -> option<u>',
        'others': [
            { 'path': 'std::option::Option', 'name': 'and' },
            { 'path': 'std::option::Option', 'name': 'zip' },
        ],
    },
    {
        'query': 'option<t>, option<u> -> option<t>',
        'others': [
            { 'path': 'std::option::Option', 'name': 'and' },
            { 'path': 'std::option::Option', 'name': 'zip' },
        ],
    },
    {
        'query': 'option<t>, option<u> -> option<t, u>',
        'others': [
            { 'path': 'std::option::Option', 'name': 'zip' },
        ],
    },
    {
        'query': 'option<t>, e -> result<t, e>',
        'others': [
            { 'path': 'std::option::Option', 'name': 'ok_or' },
            { 'path': 'std::result::Result', 'name': 'transpose' },
        ],
    },
    {
        'query': 'result<option<t>, e> -> option<result<t, e>>',
        'others': [
            { 'path': 'std::result::Result', 'name': 'transpose' },
        ],
    },
    {
        'query': 'option<t>, option<t> -> bool',
        'others': [
            { 'path': 'std::option::Option', 'name': 'eq' },
        ],
    },
    {
        'query': 'option<option<t>> -> option<t>',
        'others': [
            { 'path': 'std::option::Option', 'name': 'flatten' },
        ],
    },
    {
        'query': 'option<t>',
        'returned': [
            { 'path': 'std::result::Result', 'name': 'ok' },
        ],
    },
];
