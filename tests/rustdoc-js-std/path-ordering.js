const EXPECTED = [
    {
        query: 'hashset::insert',
        others: [
            // ensure hashset::insert comes first
            { 'path': 'std::collections::HashSet', 'name': 'insert' },
            { 'path': 'std::collections::HashSet', 'name': 'get_or_insert' },
            { 'path': 'std::collections::HashSet', 'name': 'get_or_insert_with' },
        ],
    },
    {
        query: 'hash::insert',
        others: [
            // ensure hashset/hashmap::insert come first
            { 'path': 'std::collections::HashMap', 'name': 'insert' },
            { 'path': 'std::collections::HashSet', 'name': 'insert' },
        ],
    },
];
