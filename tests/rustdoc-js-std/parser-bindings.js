const PARSED = [
    {
        query: 'A<B=C>',
        elems: [
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [
                            {
                                name: "c",
                                fullPath: ["c"],
                                pathWithoutLast: [],
                                pathLast: "c",
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
        original: 'A<B=C>',
        returned: [],
        userQuery: 'a<b=c>',
        error: null,
    },
    {
        query: 'A<B = C>',
        elems: [
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [{
                            name: "c",
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
        original: 'A<B = C>',
        returned: [],
        userQuery: 'a<b = c>',
        error: null,
    },
    {
        query: 'A<B=!>',
        elems: [
            {
                name: "a",
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
        original: 'A<B=!>',
        returned: [],
        userQuery: 'a<b=!>',
        error: null,
    },
    {
        query: 'A<B=[]>',
        elems: [
            {
                name: "a",
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
        original: 'A<B=[]>',
        returned: [],
        userQuery: 'a<b=[]>',
        error: null,
    },
    {
        query: 'A<B=[!]>',
        elems: [
            {
                name: "a",
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
        original: 'A<B=[!]>',
        returned: [],
        userQuery: 'a<b=[!]>',
        error: null,
    },
    {
        query: 'A<B=C=>',
        elems: [],
        foundElems: 0,
        original: 'A<B=C=>',
        returned: [],
        userQuery: 'a<b=c=>',
        error: "Cannot write `=` twice in a binding",
    },
    {
        query: 'A<B=>',
        elems: [],
        foundElems: 0,
        original: 'A<B=>',
        returned: [],
        userQuery: 'a<b=>',
        error: "Unexpected `>` after `=`",
    },
    {
        query: 'B=C',
        elems: [],
        foundElems: 0,
        original: 'B=C',
        returned: [],
        userQuery: 'b=c',
        error: "Type parameter `=` must be within generics list",
    },
    {
        query: '[B=C]',
        elems: [],
        foundElems: 0,
        original: '[B=C]',
        returned: [],
        userQuery: '[b=c]',
        error: "Type parameter `=` cannot be within slice `[]`",
    },
    {
        query: 'A<B<X>=C>',
        elems: [
            {
                name: "a",
                fullPath: ["a"],
                pathWithoutLast: [],
                pathLast: "a",
                generics: [],
                bindings: [
                    [
                        'b',
                        [
                            {
                                name: "c",
                                fullPath: ["c"],
                                pathWithoutLast: [],
                                pathLast: "c",
                                generics: [],
                                typeFilter: -1,
                            },
                            {
                                name: "x",
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
        original: 'A<B<X>=C>',
        returned: [],
        userQuery: 'a<b<x>=c>',
        error: null,
    },
];
