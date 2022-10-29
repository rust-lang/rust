const QUERY = [
    'StructItem',
    'StructFieldItem',
    'StructMethodItem',
    'ImplTraitItem',
    'StructImplConstItem',
    'ImplTraitFunction',
    'EnumItem',
    'VariantItem',
    'EnumMethodItem',
    'TypedefItem',
    'TraitItem',
    'TraitTypeItem',
    'AssociatedConstItem',
    'TraitFunctionItem',
    'FunctionItem',
    'ModuleItem',
    'ConstItem',
    'StaticItem',
    'UnionItem',
    'UnionFieldItem',
    'UnionMethodItem',
    'MacroItem',
];

const EXPECTED = [
    {
        // StructItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Struct',
                'alias': 'StructItem',
                'href': '../doc_alias/struct.Struct.html',
                'is_alias': true
            },
        ],
    },
    {
        // StructFieldItem
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'field',
                'alias': 'StructFieldItem',
                'href': '../doc_alias/struct.Struct.html#structfield.field',
                'is_alias': true
            },
        ],
    },
    {
        // StructMethodItem
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'method',
                'alias': 'StructMethodItem',
                'href': '../doc_alias/struct.Struct.html#method.method',
                'is_alias': true
            },
        ],
    },
    {
        // ImplTraitItem
        'others': [],
    },
    {
        // StructImplConstItem
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'ImplConstItem',
                'alias': 'StructImplConstItem',
                'href': '../doc_alias/struct.Struct.html#associatedconstant.ImplConstItem',
                'is_alias': true
            },
        ],
    },
    {
        // ImplTraitFunction
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'function',
                'alias': 'ImplTraitFunction',
                'href': '../doc_alias/struct.Struct.html#method.function',
                'is_alias': true
            },
        ],
    },
    {
        // EnumItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Enum',
                'alias': 'EnumItem',
                'href': '../doc_alias/enum.Enum.html',
                'is_alias': true
            },
        ],
    },
    {
        // VariantItem
        'others': [
            {
                'path': 'doc_alias::Enum',
                'name': 'Variant',
                'alias': 'VariantItem',
                'href': '../doc_alias/enum.Enum.html#variant.Variant',
                'is_alias': true
            },
        ],
    },
    {
        // EnumMethodItem
        'others': [
            {
                'path': 'doc_alias::Enum',
                'name': 'method',
                'alias': 'EnumMethodItem',
                'href': '../doc_alias/enum.Enum.html#method.method',
                'is_alias': true
            },
        ],
    },
    {
        // TypedefItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Typedef',
                'alias': 'TypedefItem',
                'href': '../doc_alias/type.Typedef.html',
                'is_alias': true
            },
        ],
    },
    {
        // TraitItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Trait',
                'alias': 'TraitItem',
                'href': '../doc_alias/trait.Trait.html',
                'is_alias': true
            },
        ],
    },
    {
        // TraitTypeItem
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'Target',
                'alias': 'TraitTypeItem',
                'href': '../doc_alias/trait.Trait.html#associatedtype.Target',
                'is_alias': true
            },
        ],
    },
    {
        // AssociatedConstItem
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'AssociatedConst',
                'alias': 'AssociatedConstItem',
                'href': '../doc_alias/trait.Trait.html#associatedconstant.AssociatedConst',
                'is_alias': true
            },
        ],
    },
    {
        // TraitFunctionItem
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'function',
                'alias': 'TraitFunctionItem',
                'href': '../doc_alias/trait.Trait.html#tymethod.function',
                'is_alias': true
            },
        ],
    },
    {
        // FunctionItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'function',
                'alias': 'FunctionItem',
                'href': '../doc_alias/fn.function.html',
                'is_alias': true
            },
        ],
    },
    {
        // ModuleItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Module',
                'alias': 'ModuleItem',
                'href': '../doc_alias/Module/index.html',
                'is_alias': true
            },
        ],
    },
    {
        // ConstItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Const',
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
        // StaticItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Static',
                'alias': 'StaticItem',
                'href': '../doc_alias/static.Static.html',
                'is_alias': true
            },
        ],
    },
    {
        // UnionItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Union',
                'alias': 'UnionItem',
                'href': '../doc_alias/union.Union.html',
                'is_alias': true
            },
            // Not an alias!
            {
                'path': 'doc_alias::Union',
                'name': 'union_item',
                'href': '../doc_alias/union.Union.html#structfield.union_item'
            },
        ],
    },
    {
        // UnionFieldItem
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'union_item',
                'alias': 'UnionFieldItem',
                'href': '../doc_alias/union.Union.html#structfield.union_item',
                'is_alias': true
            },
        ],
    },
    {
        // UnionMethodItem
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'method',
                'alias': 'UnionMethodItem',
                'href': '../doc_alias/union.Union.html#method.method',
                'is_alias': true
            },
        ],
    },
    {
        // MacroItem
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Macro',
                'alias': 'MacroItem',
                'href': '../doc_alias/macro.Macro.html',
                'is_alias': true
            },
        ],
    },
];
