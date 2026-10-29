// Checks that `.` is considered the same as `::` but filters on
// `methods` and `fields` only.

// exact-check

const PARSED = [
    {
        'query': 'Another.end->a',
        'error': 'The `.` can only be used in queries with one element',
        'foundElems': 0,
        'returned': [],
        'userQuery': 'Another.end->a',
    },
    {
        'query': '->Another.end',
        'error': 'The `.` cannot be used in returned position',
        'foundElems': 0,
        'returned': [],
        'userQuery': '->Another.end',
    },
    {
        'query': 'a<Another.end>',
        'error': 'Unexpected `.` in generics (`<>`)',
        'foundElems': 0,
        'returned': [],
        'userQuery': 'a<Another.end>',
    },
];

const EXPECTED = [
    {
        'query': 'Bar.start',
        'others': [
            { 'path': 'dot::Bar', 'name': 'start', 'ty': 13 },
            { 'path': 'dot::Bar', 'name': 'start', 'ty': 14 },
        ],
    },
    {
        'query': 'Bar.end',
        'others': [
            { 'path': 'dot::Bar', 'name': 'end', 'ty': 13 },
            { 'path': 'dot::Bar', 'name': 'end', 'ty': 14 },
        ],
    },
    {
        'query': 'Another.start',
        'others': [
            { 'path': 'dot::Another', 'name': 'start', 'ty': 13 },
        ],
    },
    {
        'query': 'Another.end',
        'others': [
            { 'path': 'dot::Another', 'name': 'end', 'ty': 12 },
        ],
    },
];
