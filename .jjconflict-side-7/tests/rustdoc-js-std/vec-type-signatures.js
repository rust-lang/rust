// ignore-order

const FILTER_CRATE = "std";

const EXPECTED = [
    {
        'query': 'vec::intoiter<T> -> [T]',
        'others': [
            { 'path': 'std::vec::IntoIter', 'name': 'as_slice' },
            { 'path': 'std::vec::IntoIter', 'name': 'as_mut_slice' },
            { 'path': 'std::vec::IntoIter', 'name': 'next_chunk' },
        ],
    },
    {
        'query': 'vec::intoiter<T> -> []',
        'others': [
            { 'path': 'std::vec::IntoIter', 'name': 'as_slice' },
            { 'path': 'std::vec::IntoIter', 'name': 'as_mut_slice' },
            { 'path': 'std::vec::IntoIter', 'name': 'next_chunk' },
        ],
    },
    {
        'query': 'vec<T, Allocator> -> Box<[T]>',
        'others': [
            {
                'path': 'std::boxed::Box',
                'name': 'from',
                'displayType': '`Vec`<`T`, `A`> -> `Box`<`[T]`, A>',
                'displayMappedNames': `T = T`,
                'displayWhereClause': 'A: `Allocator`',
            },
        ],
    },
];
