const QUERY = ['-> "p"', '("p")'];

const PARSED = [
    {
        args: [],
        elems: [],
        foundElems: 1,
        original: "-> \"p\"",
        returned: [{
            name: "p",
            fullPath: ["p"],
            pathWithoutLast: [],
            pathLast: "p",
            generics: [],
        }],
        typeFilter: -1,
        userQuery: "-> \"p\"",
        error: null,
    },
    {
        args: [{
            name: "p",
            fullPath: ["p"],
            pathWithoutLast: [],
            pathLast: "p",
            generics: [],
        }],
        elems: [],
        foundElems: 1,
        original: "(\"p\")",
        returned: [],
        typeFilter: -1,
        userQuery: "(\"p\")",
        error: null,
    },
];
