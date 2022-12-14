const QUERY = ['A::B', 'A::B,C',  'A::B<f>,C', 'mod::a'];

const PARSED = [
    {
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
        elems: [
            {
                name: "a::b",
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
        original: 'A::B,C',
        returned: [],
        typeFilter: -1,
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
        original: 'A::B<f>,C',
        returned: [],
        typeFilter: -1,
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
        }],
        foundElems: 1,
        original: "mod::a",
        returned: [],
        typeFilter: -1,
        userQuery: "mod::a",
        error: null,
    },
];
