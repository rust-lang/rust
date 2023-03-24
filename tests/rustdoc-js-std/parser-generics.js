const QUERY = ['A<B<C<D>,  E>', 'p<> u8', '"p"<a>'];

const PARSED = [
    {
        elems: [],
        foundElems: 0,
        original: 'A<B<C<D>,  E>',
        returned: [],
        userQuery: 'a<b<c<d>,  e>',
        error: 'Unexpected `<` after `<`',
    },
    {
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
        original: "p<> u8",
        returned: [],
        userQuery: "p<> u8",
        error: null,
    },
    {
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
];
