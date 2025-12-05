const PARSED = [
    {
        query: 'A<B<C<D>,  E>',
        elems: [],
        foundElems: 0,
        userQuery: 'A<B<C<D>,  E>',
        returned: [],
        error: 'Unclosed `<`',
    },
    {
        query: 'p<>,u8',
        elems: [
            {
                name: "p",
                fullPath: ["p"],
                pathWithoutLast: [],
                pathLast: "p",
                generics: [],
                typeFilter: null,
            },
            {
                name: "u8",
                fullPath: ["u8"],
                pathWithoutLast: [],
                pathLast: "u8",
                generics: [],
                typeFilter: null,
            },
        ],
        foundElems: 2,
        userQuery: "p<>,u8",
        returned: [],
        error: null,
    },
    {
        query: '"p"<a>',
        elems: [
            {
                name: "p",
                fullPath: ["p"],
                pathWithoutLast: [],
                pathLast: "p",
                generics: [
                    {
                        name: "a",
                        fullPath: ["a"],
                        pathWithoutLast: [],
                        pathLast: "a",
                        generics: [],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: '"p"<a>',
        returned: [],
        error: null,
    },
    {
        query: 'p<u<x>>',
        elems: [
            {
                name: "p",
                fullPath: ["p"],
                pathWithoutLast: [],
                pathLast: "p",
                generics: [
                    {
                        name: "u",
                        fullPath: ["u"],
                        pathWithoutLast: [],
                        pathLast: "u",
                        generics: [
                            {
                                name: "x",
                                fullPath: ["x"],
                                pathWithoutLast: [],
                                pathLast: "x",
                                generics: [],
                            },
                        ],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: 'p<u<x>>',
        returned: [],
        error: null,
    },
    {
        query: 'p<u<x>, r>',
        elems: [
            {
                name: "p",
                fullPath: ["p"],
                pathWithoutLast: [],
                pathLast: "p",
                generics: [
                    {
                        name: "u",
                        fullPath: ["u"],
                        pathWithoutLast: [],
                        pathLast: "u",
                        generics: [
                            {
                                name: "x",
                                fullPath: ["x"],
                                pathWithoutLast: [],
                                pathLast: "x",
                                generics: [],
                            },
                        ],
                    },
                    {
                        name: "r",
                        fullPath: ["r"],
                        pathWithoutLast: [],
                        pathLast: "r",
                        generics: [],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: 'p<u<x>, r>',
        returned: [],
        error: null,
    },
    {
        query: 'p<u<x, r>>',
        elems: [
            {
                name: "p",
                fullPath: ["p"],
                pathWithoutLast: [],
                pathLast: "p",
                generics: [
                    {
                        name: "u",
                        fullPath: ["u"],
                        pathWithoutLast: [],
                        pathLast: "u",
                        generics: [
                            {
                                name: "x",
                                fullPath: ["x"],
                                pathWithoutLast: [],
                                pathLast: "x",
                                generics: [],
                            },
                            {
                                name: "r",
                                fullPath: ["r"],
                                pathWithoutLast: [],
                                pathLast: "r",
                                generics: [],
                            },
                        ],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: 'p<u<x, r>>',
        returned: [],
        error: null,
    },
];
