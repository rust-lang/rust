// test that `clone`-like functions are sorted lower when
// a search is based soley on return type

const FILTER_CRATE = "core";

const EXPECTED = [
    {
        'query': '-> AllocError',
        'others': [
            { 'path': 'core::alloc::Allocator', 'name': 'allocate' },
            { 'path': 'core::alloc::AllocError', 'name': 'clone' },
        ],
    },
    {
        'query': 'AllocError',
        'returned': [
            { 'path': 'core::alloc::Allocator', 'name': 'allocate' },
            { 'path': 'core::alloc::AllocError', 'name': 'clone' },
         ],
    },
    {
        'query': '-> &str',
        'others': [
            // type_name_of_val should not be consider clone-like
            { 'path': 'core::any', 'name': 'type_name_of_val' },
            // this returns `Option<&str>`, and thus should be sorted lower
            { 'path': 'core::str::Split', 'name': 'next' },
         ],
    },
]
