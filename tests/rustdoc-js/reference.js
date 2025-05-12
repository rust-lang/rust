// exact-check

const EXPECTED = [
    // pinkie with explicit names
    {
        'query': 'usize, usize -> ()',
        'others': [
            { 'path': 'reference', 'name': 'pinky' },
        ],
    },
    {
        'query': 'reference<usize>, usize -> ()',
        'others': [
            { 'path': 'reference', 'name': 'pinky' },
        ],
    },
    {
        'query': 'reference<usize>, reference<usize> -> ()',
        'others': [],
    },
    {
        'query': 'reference<mut, usize>, usize -> ()',
        'others': [],
    },
    // thumb with explicit names
    {
        'query': 'thumb, thumb -> ()',
        'others': [
            { 'path': 'reference::Thumb', 'name': 'up' },
        ],
    },
    {
        'query': 'reference<thumb>, thumb -> ()',
        'others': [
            { 'path': 'reference::Thumb', 'name': 'up' },
        ],
    },
    {
        'query': 'reference<thumb>, reference<thumb> -> ()',
        'others': [],
    },
    {
        'query': 'reference<mut, thumb>, thumb -> ()',
        'others': [],
    },
    // index with explicit names
    {
        'query': 'index, index -> ()',
        'others': [
            { 'path': 'reference::Index', 'name': 'point' },
        ],
    },
    {
        'query': 'reference<index>, index -> ()',
        'others': [
            { 'path': 'reference::Index', 'name': 'point' },
        ],
    },
    {
        'query': 'reference<index>, reference<index> -> ()',
        'others': [],
    },
    {
        'query': 'reference<mut, index>, index -> ()',
        'others': [],
    },
    // ring with explicit names
    {
        'query': 'ring, ring -> ()',
        'others': [
            { 'path': 'reference::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': 'reference<ring>, ring -> ()',
        'others': [
            { 'path': 'reference::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': 'reference<ring>, reference<ring> -> ()',
        // can't leave out the `mut`, because can't reorder like that
        'others': [],
    },
    {
        'query': 'reference<mut, ring>, reference<ring> -> ()',
        'others': [
            { 'path': 'reference::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': 'reference<mut, ring>, reference<mut, ring> -> ()',
        'others': [],
    },
    // middle with explicit names
    {
        'query': 'middle, middle -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': 'reference<middle>, reference<middle> -> ()',
        // can't leave out the mut
        'others': [],
    },
    {
        'query': 'reference<mut, middle>, reference<mut, middle> -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': 'reference<reference<mut, middle>>, reference<mut, reference<middle>> -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': 'reference<mut, reference<middle>>, reference<reference<mut, middle>> -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': 'reference<reference<mut, middle>>, reference<reference<mut, middle>> -> ()',
        'others': [],
    },
    {
        'query': 'reference<mut, reference<middle>>, reference<mut, reference<middle>> -> ()',
        'others': [],
    },
    // pinkie with shorthand
    {
        'query': '&usize, usize -> ()',
        'others': [
            { 'path': 'reference', 'name': 'pinky' },
        ],
    },
    {
        'query': '&usize, &usize -> ()',
        'others': [],
    },
    {
        'query': '&mut usize, usize -> ()',
        'others': [],
    },
    // thumb with shorthand
    {
        'query': '&thumb, thumb -> ()',
        'others': [
            { 'path': 'reference::Thumb', 'name': 'up' },
        ],
    },
    {
        'query': '&thumb, &thumb -> ()',
        'others': [],
    },
    {
        'query': '&mut thumb, thumb -> ()',
        'others': [],
    },
    // index with explicit names
    {
        'query': '&index, index -> ()',
        'others': [
            { 'path': 'reference::Index', 'name': 'point' },
        ],
    },
    {
        'query': '&index, &index -> ()',
        'others': [],
    },
    {
        'query': '&mut index, index -> ()',
        'others': [],
    },
    // ring with shorthand
    {
        'query': '&ring, ring -> ()',
        'others': [
            { 'path': 'reference::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': '&ring, ring -> ()',
        'others': [
            { 'path': 'reference::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': '&mut ring, &ring -> ()',
        'others': [
            { 'path': 'reference::Ring', 'name': 'wear' },
        ],
    },
    {
        'query': '&mut ring, &mut ring -> ()',
        'others': [],
    },
    // middle with shorthand
    {
        'query': '&middle, &middle -> ()',
        // can't leave out the mut
        'others': [],
    },
    {
        'query': '&mut middle, &mut middle -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': '&&mut middle, &mut &middle -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': '&mut &middle, &&mut middle -> ()',
        'others': [
            { 'path': 'reference', 'name': 'show' },
        ],
    },
    {
        'query': '&&mut middle, &&mut middle -> ()',
        'others': [],
    },
    {
        'query': '&mut &middle, &mut &middle -> ()',
        'others': [],
    },
];
