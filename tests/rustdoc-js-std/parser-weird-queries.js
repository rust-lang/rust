// This test is mostly to check that the parser still kinda outputs something
// (and doesn't enter an infinite loop!) even though the query is completely
// invalid.

const PARSED = [
    {
        query: 'a b',
        elems: [
            {
                name: "a b",
                fullPath: ["a", "b"],
                pathWithoutLast: ["a"],
                pathLast: "b",
                generics: [],
            },
        ],
        foundElems: 1,
        original: "a b",
        returned: [],
        userQuery: "a b",
        error: null,
    },
    {
        query: 'a   b',
        elems: [
            {
                name: "a   b",
                fullPath: ["a", "b"],
                pathWithoutLast: ["a"],
                pathLast: "b",
                generics: [],
            },
        ],
        foundElems: 1,
        original: "a   b",
        returned: [],
        userQuery: "a   b",
        error: null,
    },
    {
        query: 'a,b(c)',
        elems: [],
        foundElems: 0,
        original: "a,b(c)",
        returned: [],
        userQuery: "a,b(c)",
        error: "Unexpected `(`",
    },
    {
        query: 'aaa,a',
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
        userQuery: "aaa,a",
        error: null,
    },
    {
        query: ',,,,',
        elems: [],
        foundElems: 0,
        original: ",,,,",
        returned: [],
        userQuery: ",,,,",
        error: null,
    },
    {
        query: 'mod    :',
        elems: [],
        foundElems: 0,
        original: 'mod    :',
        returned: [],
        userQuery: 'mod    :',
        error: "Unexpected `:` (expected path after type filter `mod:`)",
    },
    {
        query: 'mod\t:',
        elems: [],
        foundElems: 0,
        original: 'mod\t:',
        returned: [],
        userQuery: 'mod\t:',
        error: "Unexpected `:` (expected path after type filter `mod:`)",
    },
];
