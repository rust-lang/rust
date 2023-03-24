const QUERY = ['A::B', 'A::B,C',  'A::B<f>,C', 'mod::a'];

const PARSED = [
    {
        elems: [{
            name: "a::b",
            fullPath: ["a", "b"],
            pathWithoutLast: ["a"],
            pathLast: "b",
            generics: [],
            typeFilter: -1,
        }],
        foundElems: 1,
        original: "A::B",
        returned: [],
        userQuery: "a::b",
        error: null,
    },
    {
        elems: [
            {
                name: "a::b",
                fullPath: ["a", "b"],
                pathWithoutLast: ["a"],
                pathLast: "b",
                generics: [],
                typeFilter: -1,
            },
            {
                name: "c",
                fullPath: ["c"],
                pathWithoutLast: [],
                pathLast: "c",
                generics: [],
                typeFilter: -1,
            },
        ],
        foundElems: 2,
        original: 'A::B,C',
        returned: [],
        userQuery: 'a::b,c',
        error: null,
    },
    {
        elems: [
            {
                name: "a::b",
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
                typeFilter: -1,
            },
            {
                name: "c",
                fullPath: ["c"],
                pathWithoutLast: [],
                pathLast: "c",
                generics: [],
                typeFilter: -1,
            },
        ],
        foundElems: 2,
        original: 'A::B<f>,C',
        returned: [],
        userQuery: 'a::b<f>,c',
        error: null,
    },
    {
        elems: [{
            name: "mod::a",
            fullPath: ["mod", "a"],
            pathWithoutLast: ["mod"],
            pathLast: "a",
            generics: [],
            typeFilter: -1,
        }],
        foundElems: 1,
        original: "mod::a",
        returned: [],
        userQuery: "mod::a",
        error: null,
    },
];
