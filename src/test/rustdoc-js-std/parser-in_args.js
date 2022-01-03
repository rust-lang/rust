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
        elemName: null,
        elems: [],
        foundElems: 1,
        id: "(whatever)",
        nameSplit: null,
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
        elemName: null,
        elems: [],
        foundElems: 1,
        id: "(<P>)",
        nameSplit: null,
        original: "(<P>)",
        returned: [],
        typeFilter: -1,
        userQuery: "(<p>)",
        error: null,
    },
];
