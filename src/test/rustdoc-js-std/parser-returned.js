const QUERY = ['-> <P>', '-> P'];

const PARSED = [
    {
        args: [],
        elemName: null,
        elems: [],
        foundElems: 1,
        id: "-> <P>",
        nameSplit: null,
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
        val: "-> <p>",
        error: null,
    },
    {
        args: [],
        elemName: null,
        elems: [],
        foundElems: 1,
        id: "-> P",
        nameSplit: null,
        original: "-> P",
        returned: [{
            name: "p",
            fullPath: ["p"],
            pathWithoutLast: [],
            pathLast: "p",
            generics: [],
        }],
        typeFilter: -1,
        val: "-> p",
        error: null,
    },
];
