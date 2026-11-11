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
        userQuery: "a b",
        returned: [],
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
        userQuery: "a   b",
        returned: [],
        error: null,
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
        userQuery: "aaa,a",
        returned: [],
        error: null,
    },
    {
        query: ',,,,',
        elems: [],
        foundElems: 0,
        userQuery: ",,,,",
        returned: [],
        error: null,
    },
    {
        query: 'mod    :',
        elems: [],
        foundElems: 0,
        userQuery: 'mod    :',
        returned: [],
        error: "Unexpected `:` (expected path after type filter `mod:`)",
    },
    {
        query: 'mod\t:',
        elems: [],
        foundElems: 0,
        userQuery: 'mod :',
        returned: [],
        error: "Unexpected `:` (expected path after type filter `mod:`)",
    },
];
