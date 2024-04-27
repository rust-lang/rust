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
];
