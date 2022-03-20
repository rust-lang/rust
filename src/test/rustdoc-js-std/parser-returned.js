const QUERY = ['-> F<P>', '-> P'];

const PARSED = [
    {
        elems: [],
        foundElems: 1,
        original: "-> F<P>",
        returned: [{
            name: "f",
            fullPath: ["f"],
            pathWithoutLast: [],
            pathLast: "f",
            generics: [
                {
                    name: "p",
                    fullPath: ["p"],
                    pathWithoutLast: [],
                    pathLast: "p",
                    generics: [],
                },
            ],
        }],
        typeFilter: -1,
        userQuery: "-> f<p>",
        error: null,
    },
    {
        elems: [],
        foundElems: 1,
        original: "-> P",
        returned: [{
            name: "p",
            fullPath: ["p"],
            pathWithoutLast: [],
            pathLast: "p",
            generics: [],
        }],
        typeFilter: -1,
        userQuery: "-> p",
        error: null,
    },
];
