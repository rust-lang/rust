const QUERY = ['(whatever)', '(<P>)'];

const PARSED = [
    {
        args: [{
            name: "whatever",
            fullPath: ["whatever"],
            pathWithoutLast: [],
            pathLast: "whatever",
            generics: [],
        }],
        elems: [],
        foundElems: 1,
        original: "(whatever)",
        returned: [],
        typeFilter: -1,
        userQuery: "(whatever)",
        error: null,
    },
    {
        args: [{
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
        elems: [],
        foundElems: 1,
        original: "(<P>)",
        returned: [],
        typeFilter: -1,
        userQuery: "(<p>)",
        error: null,
    },
];
