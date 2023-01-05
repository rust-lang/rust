// ignore-tidy-tab

const QUERY = [
    'aaaaaa	b',
    'a b',
    'a,b',
    'a\tb',
    'a<b c>',
    'a<b,c>',
    'a<b\tc>',
];

const PARSED = [
    {
        elems: [
            {
                name: 'aaaaaa',
                fullPath: ['aaaaaa'],
                pathWithoutLast: [],
                pathLast: 'aaaaaa',
                generics: [],
            },
            {
                name: 'b',
                fullPath: ['b'],
                pathWithoutLast: [],
                pathLast: 'b',
                generics: [],
            },
        ],
        foundElems: 2,
        original: "aaaaaa	b",
        returned: [],
        typeFilter: -1,
        userQuery: "aaaaaa	b",
        error: null,
    },
    {
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [],
            },
            {
                name: 'b',
                fullPath: ['b'],
                pathWithoutLast: [],
                pathLast: 'b',
                generics: [],
            },
        ],
        foundElems: 2,
        original: "a b",
        returned: [],
        typeFilter: -1,
        userQuery: "a b",
        error: null,
    },
    {
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [],
            },
            {
                name: 'b',
                fullPath: ['b'],
                pathWithoutLast: [],
                pathLast: 'b',
                generics: [],
            },
        ],
        foundElems: 2,
        original: "a,b",
        returned: [],
        typeFilter: -1,
        userQuery: "a,b",
        error: null,
    },
    {
        elems: [
            {
                name: 'a',
                fullPath: ['a'],
                pathWithoutLast: [],
                pathLast: 'a',
                generics: [],
            },
            {
                name: 'b',
                fullPath: ['b'],
                pathWithoutLast: [],
                pathLast: 'b',
                generics: [],
            },
        ],
        foundElems: 2,
        original: "a\tb",
        returned: [],
        typeFilter: -1,
        userQuery: "a\tb",
        error: null,
    },
    {
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
            },
        ],
        foundElems: 1,
        original: "a<b c>",
        returned: [],
        typeFilter: -1,
        userQuery: "a<b c>",
        error: null,
    },
    {
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
            },
        ],
        foundElems: 1,
        original: "a<b,c>",
        returned: [],
        typeFilter: -1,
        userQuery: "a<b,c>",
        error: null,
    },
    {
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
            },
        ],
        foundElems: 1,
        original: "a<b\tc>",
        returned: [],
        typeFilter: -1,
        userQuery: "a<b\tc>",
        error: null,
    },
];
