const QUERY = ['fn:foo', 'enum : foo', 'macro<f>:foo'];

const PARSED = [
    {
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
        elems: [],
        foundElems: 0,
        original: "macro<f>:foo",
        returned: [],
        typeFilter: -1,
        userQuery: "macro<f>:foo",
        error: "Unexpected `:`",
    },
];
