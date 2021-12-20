const QUERY = ['<P>', 'A<B<C<D>, E>'];

const PARSED = [
    {
        args: [],
        elemName: null,
        elems: [{
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
        foundElems: 1,
        id: "<P>",
        nameSplit: null,
        original: "<P>",
        returned: [],
        typeFilter: -1,
        val: "<p>",
        error: null,
    },
    {
        args: [],
        elemName: null,
        elems: [{
            name: "a",
            fullPath: ["a"],
            pathWithoutLast: [],
            pathLast: "a",
            generics: [
                {
                    name: "b",
                    fullPath: ["b"],
                    pathWithoutLast: [],
                    pathLast: "b",
                    generics: [
                        {
                            name: "c",
                            fullPath: ["c"],
                            pathWithoutLast: [],
                            pathLast: "c",
                            generics: [
                                {
                                    name: "d",
                                    fullPath: ["d"],
                                    pathWithoutLast: [],
                                    pathLast: "d",
                                    generics: [],
                                },
                            ],
                        },
                        {
                            name: "e",
                            fullPath: ["e"],
                            pathWithoutLast: [],
                            pathLast: "e",
                            generics: [],
                        },
                    ],
                },
            ],
        }],
        foundElems: 1,
        id: 'A<B<C<D>, E>',
        nameSplit: null,
        original: 'A<B<C<D>, E>',
        returned: [],
        typeFilter: -1,
        val: 'a<b<c<d>, e>',
        error: null,
    }
];
