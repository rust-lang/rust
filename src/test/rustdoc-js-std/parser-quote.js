const QUERY = ['-> "p"', '("p")'];

const PARSED = [
    {
        args: [],
        elemName: null,
        elems: [],
        foundElems: 1,
        id: "-> \"p\"",
        nameSplit: null,
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
        elemName: null,
        elems: [],
        foundElems: 1,
        id: "(\"p\")",
        nameSplit: null,
        original: "(\"p\")",
        returned: [],
        typeFilter: -1,
        userQuery: "(\"p\")",
        error: null,
    },
];
