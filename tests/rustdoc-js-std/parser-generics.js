const PARSED = [
    {
        query: 'A<B<C<D>,  E>',
        elems: [],
        foundElems: 0,
        original: 'A<B<C<D>,  E>',
        returned: [],
        userQuery: 'a<b<c<d>,  e>',
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
                typeFilter: -1,
            },
            {
                name: "u8",
                fullPath: ["u8"],
                pathWithoutLast: [],
                pathLast: "u8",
                generics: [],
                typeFilter: -1,
            },
        ],
        foundElems: 2,
        original: "p<>,u8",
        returned: [],
        userQuery: "p<>,u8",
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
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        original: '"p"<a>',
        returned: [],
        userQuery: '"p"<a>',
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
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        original: 'p<u<x>>',
        returned: [],
        userQuery: 'p<u<x>>',
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
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        original: 'p<u<x>, r>',
        returned: [],
        userQuery: 'p<u<x>, r>',
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
                typeFilter: -1,
            },
        ],
        foundElems: 1,
        original: 'p<u<x, r>>',
        returned: [],
        userQuery: 'p<u<x, r>>',
        error: null,
    },
];
