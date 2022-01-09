// This test is mostly to check that the parser still kinda outputs something
// (and doesn't enter an infinite loop!) even though the query is completely
// invalid.
const QUERY = ['a b', 'a   b', 'a,b(c)'];

const PARSED = [
    {
        args: [],
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
        args: [],
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
        args: [
            {
                name: "c",
                fullPath: ["c"],
                pathWithoutLast: [],
                pathLast: "c",
                generics: [],
            },
        ],
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
        foundElems: 3,
        original: "a,b(c)",
        returned: [],
        typeFilter: -1,
        userQuery: "a,b(c)",
        error: null,
    },
];
