// ignore-tidy-tab

const PARSED = [
    {
        query: 'aaaaaa	b',
        elems: [
            {
                name: 'aaaaaa b',
                fullPath: ['aaaaaa', 'b'],
                pathWithoutLast: ['aaaaaa'],
                pathLast: 'b',
                generics: [],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: "aaaaaa b",
        returned: [],
        error: null,
    },
    {
        query: "aaaaaa,	b",
        elems: [
            {
                name: 'aaaaaa',
                fullPath: ['aaaaaa'],
                pathWithoutLast: [],
                pathLast: 'aaaaaa',
                generics: [],
                typeFilter: null,
            },
            {
                name: 'b',
                fullPath: ['b'],
                pathWithoutLast: [],
                pathLast: 'b',
                generics: [],
                typeFilter: null,
            },
        ],
        foundElems: 2,
        userQuery: "aaaaaa, b",
        returned: [],
        error: null,
    },
    {
        query: 'a b',
        elems: [
            {
                name: 'a b',
                fullPath: ['a', 'b'],
                pathWithoutLast: ['a'],
                pathLast: 'b',
                generics: [],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: "a b",
        returned: [],
        error: null,
    },
    {
        query: 'a,b',
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [],
                typeFilter: null,
            },
            {
                name: 'b',
                fullPath: ['b'],
                pathWithoutLast: [],
                pathLast: 'b',
                generics: [],
                typeFilter: null,
            },
        ],
        foundElems: 2,
        userQuery: "a,b",
        returned: [],
        error: null,
    },
    {
        query: 'a\tb',
        elems: [
            {
                name: 'a b',
                fullPath: ['a', 'b'],
                pathWithoutLast: ['a'],
                pathLast: 'b',
                generics: [],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: "a b",
        returned: [],
        error: null,
    },
    {
        query: 'a<b c>',
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [
                    {
                        name: 'b c',
                        fullPath: ['b', 'c'],
                        pathWithoutLast: ['b'],
                        pathLast: 'c',
                        generics: [],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: "a<b c>",
        returned: [],
        error: null,
    },
    {
        query: 'a<b,c>',
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [
                    {
                        name: 'b',
                        fullPath: ['b'],
                        pathWithoutLast: [],
                        pathLast: 'b',
                        generics: [],
                    },
                    {
                        name: 'c',
                        fullPath: ['c'],
                        pathWithoutLast: [],
                        pathLast: 'c',
                        generics: [],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: "a<b,c>",
        returned: [],
        error: null,
    },
    {
        query: 'a<b\tc>',
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [
                    {
                        name: 'b c',
                        fullPath: ['b', 'c'],
                        pathWithoutLast: ['b'],
                        pathLast: 'c',
                        generics: [],
                    },
                ],
                typeFilter: null,
            },
        ],
        foundElems: 1,
        userQuery: "a<b c>",
        returned: [],
        error: null,
    },
];
