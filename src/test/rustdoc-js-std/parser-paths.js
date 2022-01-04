const QUERY = ['A::B', '::A::B', 'A::B::,C',  'A::B::<f>,C'];

const PARSED = [
    {
        args: [],
        elems: [{
            name: "a::b",
            fullPath: ["a", "b"],
            pathWithoutLast: ["a"],
            pathLast: "b",
            generics: [],
        }],
        foundElems: 1,
        original: "A::B",
        returned: [],
        typeFilter: -1,
        userQuery: "a::b",
        error: null,
    },
    {
        args: [],
        elems: [{
            name: "::a::b",
            fullPath: ["a", "b"],
            pathWithoutLast: ["a"],
            pathLast: "b",
            generics: [],
        }],
        foundElems: 1,
        original: '::A::B',
        returned: [],
        typeFilter: -1,
        userQuery: '::a::b',
        error: null,
    },
    {
        args: [],
        elems: [
            {
                name: "a::b::",
                fullPath: ["a", "b"],
                pathWithoutLast: ["a"],
                pathLast: "b",
                generics: [],
            },
            {
                name: "c",
                fullPath: ["c"],
                pathWithoutLast: [],
                pathLast: "c",
                generics: [],
            },
        ],
        foundElems: 2,
        original: 'A::B::,C',
        returned: [],
        typeFilter: -1,
        userQuery: 'a::b::,c',
        error: null,
    },
    {
        args: [],
        elems: [
            {
                name: "a::b::",
                fullPath: ["a", "b"],
                pathWithoutLast: ["a"],
                pathLast: "b",
                generics: [
                    {
                        name: "f",
                        fullPath: ["f"],
                        pathWithoutLast: [],
                        pathLast: "f",
                        generics: [],
                    },
                ],
            },
            {
                name: "c",
                fullPath: ["c"],
                pathWithoutLast: [],
                pathLast: "c",
                generics: [],
            },
        ],
        foundElems: 2,
        original: 'A::B::<f>,C',
        returned: [],
        typeFilter: -1,
        userQuery: 'a::b::<f>,c',
        error: null,
    },
];
