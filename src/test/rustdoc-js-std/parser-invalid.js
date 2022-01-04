// This test is mostly to check that the parser still kinda outputs something
// (and doesn't enter an infinite loop!) even though the query is completely
// invalid.
const QUERY = ['-> <P> (p2)', '(p -> p2', 'a b', 'a,b(c)'];

const PARSED = [
    {
        args: [],
        elems: [],
        foundElems: 2,
        original: "-> <P> (p2)",
        returned: [
            {
                name: "",
                fullPath: [""],
                pathWithoutLast: [],
                pathLast: "",
                generics: [
                    {
                        name: "p",
                        fullPath: ["p"],
                        pathWithoutLast: [],
                        pathLast: "p",
                        generics: [],
                    },
                ],
            },
            {
                name: "p2",
                fullPath: ["p2"],
                pathWithoutLast: [],
                pathLast: "p2",
                generics: [],
            },
        ],
        typeFilter: -1,
        userQuery: "-> <p> (p2)",
        error: null,
    },
    {
        args: [
            {
                name: "p",
                fullPath: ["p"],
                pathWithoutLast: [],
                pathLast: "p",
                generics: [],
            },
            {
                name: "p2",
                fullPath: ["p2"],
                pathWithoutLast: [],
                pathLast: "p2",
                generics: [],
            },
        ],
        elems: [],
        foundElems: 2,
        original: "(p -> p2",
        returned: [],
        typeFilter: -1,
        userQuery: "(p -> p2",
        error: null,
    },
    {
        args: [],
        elems: [
            {
                name: "a b",
                fullPath: ["a b"],
                pathWithoutLast: [],
                pathLast: "a b",
                generics: [],
            },
        ],
        foundElems: 1,
        original: "a b",
        returned: [],
        typeFilter: -1,
        userQuery: "a b",
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
