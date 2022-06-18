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
            },
            {
                name: "b",
                fullPath: ["b"],
                pathWithoutLast: [],
                pathLast: "b",
                generics: [],
            },
        ],
        foundElems: 2,
        original: "a b",
        returned: [],
        typeFilter: -1,
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
            },
            {
                name: "b",
                fullPath: ["b"],
                pathWithoutLast: [],
                pathLast: "b",
                generics: [],
            },
        ],
        foundElems: 2,
        original: "a   b",
        returned: [],
        typeFilter: -1,
        userQuery: "a   b",
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: "a,b(c)",
        returned: [],
        typeFilter: -1,
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
            },
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
            },
        ],
        foundElems: 2,
        original: "aaa,a",
        returned: [],
        typeFilter: -1,
        userQuery: "aaa,a",
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: ",,,,",
        returned: [],
        typeFilter: -1,
        userQuery: ",,,,",
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: 'mod    :',
        returned: [],
        typeFilter: 0,
        userQuery: 'mod    :',
        error: null,
    },
    {
        elems: [],
        foundElems: 0,
        original: 'mod\t:',
        returned: [],
        typeFilter: 0,
        userQuery: 'mod\t:',
        error: null,
    },
];
