// exact-check
const FILTER_CRATE = "std";
const EXPECTED = [
    {
        query: 'vec::intoiterator',
        // trait std::iter::IntoIterator is not the first result
        others: [],
    },
    {
        query: 'vec::iter',
        others: [
            // std::net::ToSocketAttrs::iter should not show up here
            { 'path': 'std::collections::VecDeque', 'name': 'iter' },
            { 'path': 'std::collections::VecDeque', 'name': 'iter_mut' },
            { 'path': 'std::vec::Vec', 'name': 'from_iter' },
            { 'path': 'std::vec', 'name': 'IntoIter' },
            { 'path': 'std::vec::Vec', 'name': 'into_iter' },
            { 'path': 'std::vec::ExtractIf', 'name': 'into_iter' },
            { 'path': 'std::vec::Drain', 'name': 'into_iter' },
            { 'path': 'std::vec::IntoIter', 'name': 'into_iter' },
            { 'path': 'std::vec::Splice', 'name': 'into_iter' },
            { 'path': 'std::collections::VecDeque', 'name': 'from_iter' },
            { 'path': 'std::collections::VecDeque', 'name': 'into_iter' },
        ],
    },
    {
        query: 'slice::itermut',
        others: [
            // std::collections::btree_map::itermut should not show up here
            { 'path': 'std::slice', 'name': 'IterMut' },
            { 'path': 'std::slice', 'name': 'iter_mut' },
        ],
    },
];
