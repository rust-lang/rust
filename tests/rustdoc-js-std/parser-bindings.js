const PARSED = [
    {
        query: 'A<B=C>',
        elems: [
            {
                name: "A",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                normalizedPathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [
                            {
                                name: "C",
                                fullPath: ["c"],
                                pathWithoutLast: [],
                                pathLast: "c",
                                normalizedPathLast: "c",
                                generics: [],
                                typeFilter: -1,
                            },
                        ]
                    ],
                ],
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        userQuery: 'A<B=C>',
        returned: [],
        error: null,
    },
    {
        query: 'A<B = C>',
        elems: [
            {
                name: "A",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [{
                            name: "C",
                            fullPath: ["c"],
                            pathWithoutLast: [],
                            pathLast: "c",
                            generics: [],
                            typeFilter: -1,
                        }]
                    ],
                ],
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        userQuery: 'A<B = C>',
        returned: [],
        error: null,
    },
    {
        query: 'A<B=!>',
        elems: [
            {
                name: "A",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [{
                            name: "never",
                            fullPath: ["never"],
                            pathWithoutLast: [],
                            pathLast: "never",
                            generics: [],
                            typeFilter: 1,
                        }]
                    ],
                ],
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        userQuery: 'A<B=!>',
        returned: [],
        error: null,
    },
    {
        query: 'A<B=[]>',
        elems: [
            {
                name: "A",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [{
                            name: "[]",
                            fullPath: ["[]"],
                            pathWithoutLast: [],
                            pathLast: "[]",
                            generics: [],
                            typeFilter: 1,
                        }]
                    ],
                ],
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        userQuery: 'A<B=[]>',
        returned: [],
        error: null,
    },
    {
        query: 'A<B=[!]>',
        elems: [
            {
                name: "A",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [{
                            name: "[]",
                            fullPath: ["[]"],
                            pathWithoutLast: [],
                            pathLast: "[]",
                            generics: [
                                {
                                    name: "never",
                                    fullPath: ["never"],
                                    pathWithoutLast: [],
                                    pathLast: "never",
                                    generics: [],
                                    typeFilter: 1,
                                },
                            ],
                            typeFilter: 1,
                        }]
                    ],
                ],
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        userQuery: 'A<B=[!]>',
        returned: [],
        error: null,
    },
    {
        query: 'A<B=C=>',
        elems: [],
        foundElems: 0,
        userQuery: 'A<B=C=>',
        returned: [],
        error: "Cannot write `=` twice in a binding",
    },
    {
        query: 'A<B=>',
        elems: [],
        foundElems: 0,
        userQuery: 'A<B=>',
        returned: [],
        error: "Unexpected `>` after `=`",
    },
    {
        query: 'B=C',
        elems: [],
        foundElems: 0,
        userQuery: 'B=C',
        returned: [],
        error: "Type parameter `=` must be within generics list",
    },
    {
        query: '[B=C]',
        elems: [],
        foundElems: 0,
        userQuery: '[B=C]',
        returned: [],
        error: "Type parameter `=` cannot be within slice `[]`",
    },
    {
        query: 'A<B<X>=C>',
        elems: [
            {
                name: "A",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [
                            {
                                name: "C",
                                fullPath: ["c"],
                                pathWithoutLast: [],
                                pathLast: "c",
                                generics: [],
                                typeFilter: -1,
                            },
                            {
                                name: "X",
                                fullPath: ["x"],
                                pathWithoutLast: [],
                                pathLast: "x",
                                generics: [],
                                typeFilter: -1,
                            },
                        ],
                    ],
                ],
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        userQuery: 'A<B<X>=C>',
        returned: [],
        error: null,
    },
];
