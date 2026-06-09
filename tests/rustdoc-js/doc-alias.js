const EXPECTED = [
    {
        'query': 'StructItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Struct',
                'desc': 'Doc for <code>Struct</code>',
                'alias': 'StructItem',
                'href': '../doc_alias/struct.Struct.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'StructFieldItem',
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'field',
                'desc': 'Doc for <code>Struct</code>’s <code>field</code>',
                'alias': 'StructFieldItem',
                'href': '../doc_alias/struct.Struct.html#structfield.field',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'StructMethodItem',
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'method',
                'desc': 'Doc for <code>Struct::method</code>',
                'alias': 'StructMethodItem',
                'href': '../doc_alias/struct.Struct.html#method.method',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'ImplTraitItem',
        'others': [],
    },
    {
        'query': 'StructImplConstItem',
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'ImplConstItem',
                'desc': 'Doc for <code>Struct::ImplConstItem</code>',
                'alias': 'StructImplConstItem',
                'href': '../doc_alias/struct.Struct.html#associatedconstant.ImplConstItem',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'ImplTraitFunction',
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'function',
                'desc': 'Doc for <code>Trait::function</code> implemented for Struct',
                'alias': 'ImplTraitFunction',
                'href': '../doc_alias/struct.Struct.html#method.function',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'EnumItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Enum',
                'desc': 'Doc for <code>Enum</code>',
                'alias': 'EnumItem',
                'href': '../doc_alias/enum.Enum.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'VariantItem',
        'others': [
            {
                'path': 'doc_alias::Enum',
                'name': 'Variant',
                'desc': 'Doc for <code>Enum::Variant</code>',
                'alias': 'VariantItem',
                'href': '../doc_alias/enum.Enum.html#variant.Variant',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'EnumMethodItem',
        'others': [
            {
                'path': 'doc_alias::Enum',
                'name': 'method',
                'desc': 'Doc for <code>Enum::method</code>',
                'alias': 'EnumMethodItem',
                'href': '../doc_alias/enum.Enum.html#method.method',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'TypedefItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Typedef',
                'desc': 'Doc for type alias <code>Typedef</code>',
                'alias': 'TypedefItem',
                'href': '../doc_alias/type.Typedef.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'TraitItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Trait',
                'desc': 'Doc for <code>Trait</code>',
                'alias': 'TraitItem',
                'href': '../doc_alias/trait.Trait.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'TraitTypeItem',
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'Target',
                'desc': 'Doc for <code>Trait::Target</code>',
                'alias': 'TraitTypeItem',
                'href': '../doc_alias/trait.Trait.html#associatedtype.Target',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'AssociatedConstItem',
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'AssociatedConst',
                'desc': 'Doc for <code>Trait::AssociatedConst</code>',
                'alias': 'AssociatedConstItem',
                'href': '../doc_alias/trait.Trait.html#associatedconstant.AssociatedConst',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'TraitFunctionItem',
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'function',
                'desc': 'Doc for <code>Trait::function</code>',
                'alias': 'TraitFunctionItem',
                'href': '../doc_alias/trait.Trait.html#tymethod.function',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'FunctionItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'function',
                'desc': 'Doc for <code>function</code>',
                'alias': 'FunctionItem',
                'href': '../doc_alias/fn.function.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'ModuleItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Module',
                'desc': 'Doc for <code>Module</code>',
                'alias': 'ModuleItem',
                'href': '../doc_alias/Module/index.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'ConstItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Const',
                'desc': 'Doc for <code>Const</code>',
                'alias': 'ConstItem',
                'href': '../doc_alias/constant.Const.html',
                'is_alias': true
            },
            {
                'path': 'doc_alias::Struct',
                'name': 'ImplConstItem',
            },
        ],
    },
    {
        'query': 'StaticItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Static',
                'desc': 'Doc for <code>Static</code>',
                'alias': 'StaticItem',
                'href': '../doc_alias/static.Static.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'UnionItem',
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'union_item',
                'desc': 'Doc for <code>Union::union_item</code>',
                'href': '../doc_alias/union.Union.html#structfield.union_item'
            },
            {
                'path': 'doc_alias',
                'name': 'Union',
                'desc': 'Doc for <code>Union</code>',
                'alias': 'UnionItem',
                'href': '../doc_alias/union.Union.html',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'UnionFieldItem',
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'union_item',
                'desc': 'Doc for <code>Union::union_item</code>',
                'alias': 'UnionFieldItem',
                'href': '../doc_alias/union.Union.html#structfield.union_item',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'UnionMethodItem',
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'method',
                'desc': 'Doc for <code>Union::method</code>',
                'alias': 'UnionMethodItem',
                'href': '../doc_alias/union.Union.html#method.method',
                'is_alias': true
            },
        ],
    },
    {
        'query': 'MacroItem',
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Macro',
                'desc': 'Doc for <code>Macro</code>',
                'alias': 'MacroItem',
                'href': '../doc_alias/macro.Macro.html',
                'is_alias': true
            },
        ],
    },
];
