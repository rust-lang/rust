const QUERY = ['fn:foo', 'enum : foo', 'macro<f>:foo'];

const PARSED = [
    {
        args: [],
        elems: [{
            name: "foo",
            fullPath: ["foo"],
            pathWithoutLast: [],
            pathLast: "foo",
            generics: [],
        }],
        foundElems: 1,
        original: "fn:foo",
        returned: [],
        typeFilter: 5,
        userQuery: "fn:foo",
        error: null,
    },
    {
        args: [],
        elems: [{
            name: "foo",
            fullPath: ["foo"],
            pathWithoutLast: [],
            pathLast: "foo",
            generics: [],
        }],
        foundElems: 1,
        original: "enum : foo",
        returned: [],
        typeFilter: 4,
        userQuery: "enum : foo",
        error: null,
    },
    {
        args: [],
        elems: [{
            name: "foo",
            fullPath: ["foo"],
            pathWithoutLast: [],
            pathLast: "foo",
            generics: [],
        }],
        foundElems: 1,
        original: "macro<f>:foo",
        returned: [],
        typeFilter: 14,
        userQuery: "macro<f>:foo",
        error: null,
    },
];
