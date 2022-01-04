const QUERY = ['-> <P>', '-> P'];

const PARSED = [
    {
        args: [],
        elems: [],
        foundElems: 1,
        original: "-> <P>",
        returned: [{
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
        }],
        typeFilter: -1,
        userQuery: "-> <p>",
        error: null,
    },
    {
        args: [],
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
