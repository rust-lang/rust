// This test is mostly to check that the parser still kinda outputs something
// (and doesn't enter an infinite loop!) even though the query is completely
// invalid.
const QUERY = [
    'a b',
    'a   b',
    'a,b(c)',
    'aaa,a',
    ',,,,',
    'mod    :',
    'mod\t:',
];

const PARSED = [
    {
        elems: [
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                typeFilter: -1,
            },
            {
                name: "b",
                fullPath: ["b"],
                pathWithoutLast: [],
                pathLast: "b",
                generics: [],
                typeFilter: -1,
            },
        ],
        foundElems: 2,
        original: "a b",
        returned: [],
        userQuery: "a b",
        error: null,
    },
    {
        elems: [
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                typeFilter: -1,
            },
            {
                name: "b",
                fullPath: ["b"],
                pathWithoutLast: [],
                pathLast: "b",
                generics: [],
                typeFilter: -1,
            },
        ],
        foundElems: 2,
        original: "a   b",
        returned: [],
        userQuery: "a   b",
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: "a,b(c)",
        returned: [],
        userQuery: "a,b(c)",
        error: "Unexpected `(`",
    },
    {
        elems: [
            {
                name: "aaa",
                fullPath: ["aaa"],
                pathWithoutLast: [],
                pathLast: "aaa",
                generics: [],
                typeFilter: -1,
            },
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                typeFilter: -1,
            },
        ],
        foundElems: 2,
        original: "aaa,a",
        returned: [],
        userQuery: "aaa,a",
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: ",,,,",
        returned: [],
        userQuery: ",,,,",
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: 'mod    :',
        returned: [],
        userQuery: 'mod    :',
        error: "Unexpected `:` (expected path after type filter)",
    },
    {
        elems: [],
        foundElems: 0,
        original: 'mod\t:',
        returned: [],
        userQuery: 'mod\t:',
        error: "Unexpected `:` (expected path after type filter)",
    },
];
