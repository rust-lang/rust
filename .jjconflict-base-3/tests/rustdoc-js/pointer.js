// exact-check

const EXPECTED = [
    // pinkie with explicit names
    {
        'query': 'usize, usize -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'pinky' },
        ],
    },
    {
        'query': 'pointer<usize>, usize -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'pinky' },
        ],
    },
    {
        'query': 'pointer<usize>, pointer<usize> -> ()',
        'others': [],
    },
    {
        'query': 'pointer<mut, usize>, usize -> ()',
        'others': [],
    },
    // thumb with explicit names
    {
        'query': 'thumb, thumb -> ()',
        'others': [
            { 'path': 'pointer::Thumb', 'name': 'up' },
        ],
    },
    {
        'query': 'pointer<thumb>, thumb -> ()',
        'others': [
            { 'path': 'pointer::Thumb', 'name': 'up' },
        ],
    },
    {
        'query': 'pointer<thumb>, pointer<thumb> -> ()',
        'others': [],
    },
    {
        'query': 'pointer<mut, thumb>, thumb -> ()',
        'others': [],
    },
    // index with explicit names
    {
        'query': 'index, index -> ()',
        'others': [
            { 'path': 'pointer::Index', 'name': 'point' },
        ],
    },
    {
        'query': 'pointer<index>, index -> ()',
        'others': [
            { 'path': 'pointer::Index', 'name': 'point' },
        ],
    },
    {
        'query': 'pointer<index>, pointer<index> -> ()',
        'others': [],
    },
    {
        'query': 'pointer<mut, index>, index -> ()',
        'others': [],
    },
    // ring with explicit names
    {
        'query': 'ring, ring -> ()',
        'others': [
            { 'path': 'pointer::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': 'pointer<ring>, ring -> ()',
        'others': [
            { 'path': 'pointer::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': 'pointer<ring>, pointer<ring> -> ()',
        // can't leave out the `mut`, because can't reorder like that
        'others': [],
    },
    {
        'query': 'pointer<mut, ring>, pointer<ring> -> ()',
        'others': [
            { 'path': 'pointer::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': 'pointer<mut, ring>, pointer<mut, ring> -> ()',
        'others': [],
    },
    // middle with explicit names
    {
        'query': 'middle, middle -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': 'pointer<middle>, pointer<middle> -> ()',
        // can't leave out the mut
        'others': [],
    },
    {
        'query': 'pointer<mut, middle>, pointer<mut, middle> -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': 'pointer<pointer<mut, middle>>, pointer<mut, pointer<middle>> -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': 'pointer<mut, pointer<middle>>, pointer<pointer<mut, middle>> -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': 'pointer<pointer<mut, middle>>, pointer<pointer<mut, middle>> -> ()',
        'others': [],
    },
    {
        'query': 'pointer<mut, pointer<middle>>, pointer<mut, pointer<middle>> -> ()',
        'others': [],
    },
    // pinkie with shorthand
    {
        'query': '*const usize, usize -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'pinky' },
        ],
    },
    // you can omit the `const`, if you want.
    {
        'query': '*usize, usize -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'pinky' },
        ],
    },
    {
        'query': '*const usize, *const usize -> ()',
        'others': [],
    },
    {
        'query': '*mut usize, usize -> ()',
        'others': [],
    },
    // thumb with shorthand
    {
        'query': '*const thumb, thumb -> ()',
        'others': [
            { 'path': 'pointer::Thumb', 'name': 'up' },
        ],
    },
    {
        'query': '*const thumb, *const thumb -> ()',
        'others': [],
    },
    {
        'query': '*mut thumb, thumb -> ()',
        'others': [],
    },
    // index with explicit names
    {
        'query': '*const index, index -> ()',
        'others': [
            { 'path': 'pointer::Index', 'name': 'point' },
        ],
    },
    {
        'query': '*const index, *const index -> ()',
        'others': [],
    },
    {
        'query': '*mut index, index -> ()',
        'others': [],
    },
    // ring with shorthand
    {
        'query': '*const ring, ring -> ()',
        'others': [
            { 'path': 'pointer::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': '*const ring, ring -> ()',
        'others': [
            { 'path': 'pointer::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': '*mut ring, *const ring -> ()',
        'others': [
            { 'path': 'pointer::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': '*mut ring, *mut ring -> ()',
        'others': [],
    },
    // middle with shorthand
    {
        'query': '*const middle, *const middle -> ()',
        // can't leave out the mut
        'others': [],
    },
    {
        'query': '*mut middle, *mut middle -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': '*const *mut middle, *mut *const middle -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': '*mut *const middle, *const *mut middle -> ()',
        'others': [
            { 'path': 'pointer', 'name': 'show' },
        ],
    },
    {
        'query': '*const *mut middle, *const *mut middle -> ()',
        'others': [],
    },
    {
        'query': '*mut *const middle, *mut *const middle -> ()',
        'others': [],
    },
];
