const PARSED = [
    {
        query: '中文',
        elems: [{
            name: "中文",
            fullPath: ["中文"],
            pathWithoutLast: [],
            pathLast: "中文",
            generics: [],
            typeFilter: null,
        }],
        returned: [],
        foundElems: 1,
        userQuery: "中文",
        error: null,
    },
    {
        query: '_0Mixed中英文',
        elems: [{
            name: "_0Mixed中英文",
            fullPath: ["_0mixed中英文"],
            pathWithoutLast: [],
            pathLast: "_0mixed中英文",
            normalizedPathLast: "0mixed中英文",
            generics: [],
            typeFilter: null,
        }],
        foundElems: 1,
        userQuery: "_0Mixed中英文",
        returned: [],
        error: null,
    },
    {
        query: 'my_crate::中文API',
        elems: [{
            name: "my_crate::中文API",
            fullPath: ["my_crate", "中文api"],
            pathWithoutLast: ["my_crate"],
            pathLast: "中文api",
            generics: [],
            typeFilter: null,
        }],
        foundElems: 1,
        userQuery: "my_crate::中文API",
        returned: [],
        error: null,
    },
    {
        query: '类型A,类型B<约束C>->返回类型<关联类型=路径::约束D>',
        elems: [{
            name: "类型A",
            fullPath: ["类型a"],
            pathWithoutLast: [],
            pathLast: "类型a",
            generics: [],
        }, {
            name: "类型B",
            fullPath: ["类型b"],
            pathWithoutLast: [],
            pathLast: "类型b",
            generics: [{
                name: "约束C",
                fullPath: ["约束c"],
                pathWithoutLast: [],
                pathLast: "约束c",
                generics: [],
            }],
        }],
        foundElems: 3,
        totalElems: 5,
        literalSearch: true,
        userQuery: "类型A,类型B<约束C>->返回类型<关联类型=路径::约束D>",
        returned: [{
            name: "返回类型",
            fullPath: ["返回类型"],
            pathWithoutLast: [],
            pathLast: "返回类型",
            generics: [],
            bindings: [["关联类型", [{
                name: "路径::约束D",
                fullPath: ["路径", "约束d"],
                pathWithoutLast: ["路径"],
                pathLast: "约束d",
                generics: [],
            }]]],
        }],
        error: null,
    },
    {
        query: 'my_crate 中文宏!',
        elems: [{
            name: "my_crate 中文宏",
            fullPath: ["my_crate", "中文宏"],
            pathWithoutLast: ["my_crate"],
            pathLast: "中文宏",
            generics: [],
            typeFilter: "macro",
        }],
        foundElems: 1,
        userQuery: "my_crate 中文宏!",
        returned: [],
        error: null,
    },
    {
        query: '非法符号——',
        elems: [],
        foundElems: 0,
        userQuery: "非法符号——",
        returned: [],
        error: "Unexpected `—` after `号` (not a valid identifier)",
    }
]
const EXPECTED = [
    {
        query: '加法',
        others: [
            {
                name: "加法",
                path: "non_english_identifier",
                href: "../non_english_identifier/trait.加法.html",
                desc: "Add"
            },
            {
                name: "add",
                path: "non_english_identifier",
                is_alias: true,
                alias: "加法",
                href: "../non_english_identifier/fn.add.html"
            },
            {
                name: "add",
                path: "non_english_identifier",
                is_alias: true,
                alias: "加法",
                href: "../non_english_identifier/macro.add.html"
            },
        ],
        in_args: [{
            name: "加上",
            path: "non_english_identifier::加法",
            href: "../non_english_identifier/trait.加法.html#tymethod.加上",
        }],
        returned: [],
    },
    { // levensthein and substring checking only kick in at three characters
        query: '加法宏',
        others: [
            {
                name: "中文名称的加法宏",
                path: "non_english_identifier",
                href: "../non_english_identifier/macro.中文名称的加法宏.html",
            }],
        in_args: [],
        returned: [],
    },
    { // levensthein and substring checking only kick in at three characters
        query: '加法A',
        others: [
            {
                name: "中文名称的加法API",
                path: "non_english_identifier",
                href: "../non_english_identifier/fn.中文名称的加法API.html",
            }],
        in_args: [],
        returned: [],
    },
    { // Extensive type-based search is still buggy, experimental & work-in-progress.
        query: '可迭代->可选',
        others: [{
            name: "总计",
            path: "non_english_identifier",
            href: "../non_english_identifier/fn.总计.html",
            desc: "“sum”"
        }],
        in_args: [],
        returned: [],
    },
];
