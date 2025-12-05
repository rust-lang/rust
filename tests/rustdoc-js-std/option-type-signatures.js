// ignore-order

const FILTER_CRATE = "std";

const EXPECTED = [
    {
        'query': 'option, fnonce -> option',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'map',
                'displayType': '`Option`<T>, F -> `Option`<U>',
                'displayWhereClause': "F: `FnOnce` (T) -> U",
            },
        ],
    },
    {
        'query': 'option<t>, fnonce -> option',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'map',
                'displayType': '`Option`<`T`>, F -> `Option`<U>',
                'displayWhereClause': "F: `FnOnce` (T) -> U",
            },
        ],
    },
    {
        'query': 'option -> default',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'unwrap_or_default',
                'displayType': '`Option`<T> -> `T`',
                'displayWhereClause': "T: `Default`",
            },
            {
                'path': 'std::option::Option',
                'name': 'get_or_insert_default',
                'displayType': '&mut `Option`<T> -> &mut `T`',
                'displayWhereClause': "T: `Default`",
            },
        ],
    },
    {
        'query': 'option -> []',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'as_slice',
                'displayType': '&`Option`<T> -> &`[`T`]`',
            },
            {
                'path': 'std::option::Option',
                'name': 'as_mut_slice',
                'displayType': '&mut `Option`<T> -> &mut `[`T`]`',
            },
        ],
    },
    {
        'query': 'option<t>, option<t> -> option<t>',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'or',
                'displayType': '`Option`<`T`>, `Option`<`T`> -> `Option`<`T`>',
            },
            {
                'path': 'std::option::Option',
                'name': 'xor',
                'displayType': '`Option`<`T`>, `Option`<`T`> -> `Option`<`T`>',
            },
        ],
    },
    {
        'query': 'option<t>, option<u> -> option<u>',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'and',
                'displayType': '`Option`<`T`>, `Option`<`U`> -> `Option`<`U`>',
            },
        ],
    },
    {
        'query': 'option<t>, option<u> -> option<t>',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'and',
                'displayType': '`Option`<`T`>, `Option`<`U`> -> `Option`<`U`>',
            },
            {
                'path': 'std::option::Option',
                'name': 'zip',
                'displayType': '`Option`<`T`>, `Option`<`U`> -> `Option`<(`T`, U)>',
            },
        ],
    },
    {
        'query': 'option<t>, option<u> -> option<(t, u)>',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'zip',
                'displayType': '`Option`<`T`>, `Option`<`U`> -> `Option`<`(T`, `U)`>',
            },
        ],
    },
    {
        'query': 'option<t>, e -> result<t, e>',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'ok_or',
                'displayType': '`Option`<`T`>, `E` -> `Result`<`T`, `E`>',
            },
            {
                'path': 'std::result::Result',
                'name': 'transpose',
                'displayType': 'Result<`Option`<`T`>, `E`> -> Option<`Result`<`T`, `E`>>',
            },
        ],
    },
    {
        'query': 'result<option<t>, e> -> option<result<t, e>>',
        'others': [
            {
                'path': 'std::result::Result',
                'name': 'transpose',
                'displayType': '`Result`<`Option`<`T`>, `E`> -> `Option`<`Result`<`T`, `E`>>',
            },
        ],
    },
    {
        'query': 'option<t>, option<t> -> bool',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'eq',
                'displayType': '&`Option`<`T`>, &`Option`<`T`> -> `bool`',
            },
        ],
    },
    {
        'query': 'option<option<t>> -> option<t>',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'flatten',
                'displayType': '`Option`<`Option`<`T`>> -> `Option`<`T`>',
            },
        ],
    },
    {
        'query': 'option<t>',
        'returned': [
            {
                'path': 'std::result::Result',
                'name': 'ok',
                'displayType': 'Result<T, E> -> `Option`<`T`>',
            },
        ],
    },
    {
        'query': 'option<t>, (fnonce () -> u) -> option',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'map',
                'displayType': '`Option`<`T`>, F -> `Option`<U>',
                'displayMappedNames': `t = T, u = U`,
                'displayWhereClause': "F: `FnOnce` (T) -> `U`",
            },
            {
                'path': 'std::option::Option',
                'name': 'and_then',
                'displayType': '`Option`<`T`>, F -> `Option`<U>',
                'displayMappedNames': `t = T, u = U`,
                'displayWhereClause': "F: `FnOnce` (T) -> Option<`U`>",
            },
            {
                'path': 'std::option::Option',
                'name': 'zip_with',
                'displayType': 'Option<T>, `Option`<`U`>, F -> `Option`<R>',
                'displayMappedNames': `t = U, u = R`,
                'displayWhereClause': "F: `FnOnce` (T, U) -> `R`",
            },
        ],
    },
    {
        'query': 'option<t>, (fnonce () -> option<u>) -> option',
        'others': [
            {
                'path': 'std::option::Option',
                'name': 'and_then',
                'displayType': '`Option`<`T`>, F -> `Option`<U>',
                'displayMappedNames': `t = T, u = U`,
                'displayWhereClause': "F: `FnOnce` (T) -> `Option`<`U`>",
            },
        ],
    },
];
