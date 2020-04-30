// exact-check

const QUERY = [
    'StructItem',
    'StructFieldItem',
    'StructMethodItem',
    'ImplTraitItem',
    'ImplAssociatedConstItem',
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
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Struct',
                'alias': 'StructItem',
                'href': '../doc_alias/struct.Struct.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'field',
                'alias': 'StructFieldItem',
                'href': '../doc_alias/struct.Struct.html#structfield.field'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Struct',
                'name': 'method',
                'alias': 'StructMethodItem',
                'href': '../doc_alias/struct.Struct.html#method.method'
            },
        ],
    },
    {
        // ImplTraitItem
        'others': [],
    },
    {
        // ImplAssociatedConstItem
        'others': [],
    },
    {
        // ImplTraitFunction
        'others': [],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Enum',
                'alias': 'EnumItem',
                'href': '../doc_alias/enum.Enum.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Enum',
                'name': 'Variant',
                'alias': 'VariantItem',
                'href': '../doc_alias/enum.Enum.html#variant.Variant'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Enum',
                'name': 'method',
                'alias': 'EnumMethodItem',
                'href': '../doc_alias/enum.Enum.html#method.method'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Typedef',
                'alias': 'TypedefItem',
                'href': '../doc_alias/type.Typedef.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Trait',
                'alias': 'TraitItem',
                'href': '../doc_alias/trait.Trait.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'Target',
                'alias': 'TraitTypeItem',
                'href': '../doc_alias/trait.Trait.html#associatedtype.Target'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'AssociatedConst',
                'alias': 'AssociatedConstItem',
                'href': '../doc_alias/trait.Trait.html#associatedconstant.AssociatedConst'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Trait',
                'name': 'function',
                'alias': 'TraitFunctionItem',
                'href': '../doc_alias/trait.Trait.html#tymethod.function'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'function',
                'alias': 'FunctionItem',
                'href': '../doc_alias/fn.function.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Module',
                'alias': 'ModuleItem',
                'href': '../doc_alias/Module/index.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Const',
                'alias': 'ConstItem',
                'href': '../doc_alias/constant.Const.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Static',
                'alias': 'StaticItem',
                'href': '../doc_alias/static.Static.html'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Union',
                'alias': 'UnionItem',
                'href': '../doc_alias/union.Union.html'
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
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'union_item',
                'alias': 'UnionFieldItem',
                'href': '../doc_alias/union.Union.html#structfield.union_item'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias::Union',
                'name': 'method',
                'alias': 'UnionMethodItem',
                'href': '../doc_alias/union.Union.html#method.method'
            },
        ],
    },
    {
        'others': [
            {
                'path': 'doc_alias',
                'name': 'Macro',
                'alias': 'MacroItem',
                'href': '../doc_alias/macro.Macro.html'
            },
        ],
    },
];
