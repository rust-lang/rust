//! Defines input for code generation process.

pub(crate) struct KindsSrc<'a> {
    pub(crate) punct: &'a [(&'a str, &'a str)],
    pub(crate) keywords: &'a [&'a str],
    pub(crate) contextual_keywords: &'a [&'a str],
    pub(crate) literals: &'a [&'a str],
    pub(crate) tokens: &'a [&'a str],
    pub(crate) nodes: &'a [&'a str],
}

pub(crate) const KINDS_SRC: KindsSrc = KindsSrc {
    punct: &[
        (";", "SEMICOLON"),
        (",", "COMMA"),
        ("(", "L_PAREN"),
        (")", "R_PAREN"),
        ("{", "L_CURLY"),
        ("}", "R_CURLY"),
        ("[", "L_BRACK"),
        ("]", "R_BRACK"),
        ("<", "L_ANGLE"),
        (">", "R_ANGLE"),
        ("@", "AT"),
        ("#", "POUND"),
        ("~", "TILDE"),
        ("?", "QUESTION"),
        ("$", "DOLLAR"),
        ("&", "AMP"),
        ("|", "PIPE"),
        ("+", "PLUS"),
        ("*", "STAR"),
        ("/", "SLASH"),
        ("^", "CARET"),
        ("%", "PERCENT"),
        ("_", "UNDERSCORE"),
        (".", "DOT"),
        ("..", "DOT2"),
        ("...", "DOT3"),
        ("..=", "DOT2EQ"),
        (":", "COLON"),
        ("::", "COLON2"),
        ("=", "EQ"),
        ("==", "EQ2"),
        ("=>", "FAT_ARROW"),
        ("!", "BANG"),
        ("!=", "NEQ"),
        ("-", "MINUS"),
        ("->", "THIN_ARROW"),
        ("<=", "LTEQ"),
        (">=", "GTEQ"),
        ("+=", "PLUSEQ"),
        ("-=", "MINUSEQ"),
        ("|=", "PIPEEQ"),
        ("&=", "AMPEQ"),
        ("^=", "CARETEQ"),
        ("/=", "SLASHEQ"),
        ("*=", "STAREQ"),
        ("%=", "PERCENTEQ"),
        ("&&", "AMP2"),
        ("||", "PIPE2"),
        ("<<", "SHL"),
        (">>", "SHR"),
        ("<<=", "SHLEQ"),
        (">>=", "SHREQ"),
    ],
    keywords: &[
        "as", "async", "await", "box", "break", "const", "continue", "crate", "dyn", "else",
        "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "macro",
        "match", "mod", "move", "mut", "pub", "ref", "return", "self", "static", "struct", "super",
        "trait", "true", "try", "type", "unsafe", "use", "where", "while",
    ],
    contextual_keywords: &["auto", "default", "existential", "union", "raw"],
    literals: &[
        "INT_NUMBER",
        "FLOAT_NUMBER",
        "CHAR",
        "BYTE",
        "STRING",
        "RAW_STRING",
        "BYTE_STRING",
        "RAW_BYTE_STRING",
    ],
    tokens: &[
        "ERROR",
        "IDENT",
        "WHITESPACE",
        "LIFETIME",
        "COMMENT",
        "SHEBANG",
        "L_DOLLAR",
        "R_DOLLAR",
    ],
    nodes: &[
        "SOURCE_FILE",
        "STRUCT_DEF",
        "UNION_DEF",
        "ENUM_DEF",
        "FN_DEF",
        "RET_TYPE",
        "EXTERN_CRATE_ITEM",
        "MODULE",
        "USE_ITEM",
        "STATIC_DEF",
        "CONST_DEF",
        "TRAIT_DEF",
        "IMPL_DEF",
        "TYPE_ALIAS_DEF",
        "MACRO_CALL",
        "TOKEN_TREE",
        "MACRO_DEF",
        "PAREN_TYPE",
        "TUPLE_TYPE",
        "NEVER_TYPE",
        "PATH_TYPE",
        "POINTER_TYPE",
        "ARRAY_TYPE",
        "SLICE_TYPE",
        "REFERENCE_TYPE",
        "PLACEHOLDER_TYPE",
        "FN_POINTER_TYPE",
        "FOR_TYPE",
        "IMPL_TRAIT_TYPE",
        "DYN_TRAIT_TYPE",
        "OR_PAT",
        "PAREN_PAT",
        "REF_PAT",
        "BOX_PAT",
        "BIND_PAT",
        "PLACEHOLDER_PAT",
        "DOT_DOT_PAT",
        "PATH_PAT",
        "RECORD_PAT",
        "RECORD_FIELD_PAT_LIST",
        "RECORD_FIELD_PAT",
        "TUPLE_STRUCT_PAT",
        "TUPLE_PAT",
        "SLICE_PAT",
        "RANGE_PAT",
        "LITERAL_PAT",
        "MACRO_PAT",
        // atoms
        "TUPLE_EXPR",
        "ARRAY_EXPR",
        "PAREN_EXPR",
        "PATH_EXPR",
        "LAMBDA_EXPR",
        "IF_EXPR",
        "WHILE_EXPR",
        "CONDITION",
        "LOOP_EXPR",
        "FOR_EXPR",
        "CONTINUE_EXPR",
        "BREAK_EXPR",
        "LABEL",
        "BLOCK_EXPR",
        "RETURN_EXPR",
        "MATCH_EXPR",
        "MATCH_ARM_LIST",
        "MATCH_ARM",
        "MATCH_GUARD",
        "RECORD_LIT",
        "RECORD_FIELD_LIST",
        "RECORD_FIELD",
        "EFFECT_EXPR",
        "BOX_EXPR",
        // postfix
        "CALL_EXPR",
        "INDEX_EXPR",
        "METHOD_CALL_EXPR",
        "FIELD_EXPR",
        "AWAIT_EXPR",
        "TRY_EXPR",
        "CAST_EXPR",
        // unary
        "REF_EXPR",
        "PREFIX_EXPR",
        "RANGE_EXPR", // just weird
        "BIN_EXPR",
        "EXTERN_BLOCK",
        "EXTERN_ITEM_LIST",
        "ENUM_VARIANT",
        "RECORD_FIELD_DEF_LIST",
        "RECORD_FIELD_DEF",
        "TUPLE_FIELD_DEF_LIST",
        "TUPLE_FIELD_DEF",
        "ENUM_VARIANT_LIST",
        "ITEM_LIST",
        "ATTR",
        "META_ITEM", // not an item actually
        "USE_TREE",
        "USE_TREE_LIST",
        "PATH",
        "PATH_SEGMENT",
        "LITERAL",
        "ALIAS",
        "VISIBILITY",
        "WHERE_CLAUSE",
        "WHERE_PRED",
        "ABI",
        "NAME",
        "NAME_REF",
        "LET_STMT",
        "EXPR_STMT",
        "TYPE_PARAM_LIST",
        "LIFETIME_PARAM",
        "TYPE_PARAM",
        "CONST_PARAM",
        "TYPE_ARG_LIST",
        "LIFETIME_ARG",
        "TYPE_ARG",
        "ASSOC_TYPE_ARG",
        "CONST_ARG",
        "PARAM_LIST",
        "PARAM",
        "SELF_PARAM",
        "ARG_LIST",
        "TYPE_BOUND",
        "TYPE_BOUND_LIST",
        // macro related
        "MACRO_ITEMS",
        "MACRO_STMTS",
    ],
};

pub(crate) struct AstSrc<'a> {
    pub(crate) tokens: &'a [&'a str],
    pub(crate) nodes: &'a [AstNodeSrc<'a>],
    pub(crate) enums: &'a [AstEnumSrc<'a>],
}

pub(crate) struct AstNodeSrc<'a> {
    pub(crate) name: &'a str,
    pub(crate) traits: &'a [&'a str],
    pub(crate) fields: &'a [Field<'a>],
}

pub(crate) enum Field<'a> {
    Token(&'a str),
    Node { name: &'a str, src: FieldSrc<'a> },
}

pub(crate) enum FieldSrc<'a> {
    Shorthand,
    Optional(&'a str),
    Many(&'a str),
}

pub(crate) struct AstEnumSrc<'a> {
    pub(crate) name: &'a str,
    pub(crate) traits: &'a [&'a str],
    pub(crate) variants: &'a [&'a str],
}

macro_rules! ast_nodes {
    ($(
        struct $name:ident$(: $($trait:ident),*)? {
            $($field_name:ident $(![$token:tt])? $(: $ty:tt)?),*$(,)?
        }
    )*) => {
        [$(
            AstNodeSrc {
                name: stringify!($name),
                traits: &[$($(stringify!($trait)),*)?],
                fields: &[
                    $(field!($(T![$token])? $field_name $($ty)?)),*
                ],

            }
        ),*]
    };
}

macro_rules! field {
    (T![$token:tt] T) => {
        Field::Token(stringify!($token))
    };
    ($field_name:ident) => {
        Field::Node { name: stringify!($field_name), src: FieldSrc::Shorthand }
    };
    ($field_name:ident [$ty:ident]) => {
        Field::Node { name: stringify!($field_name), src: FieldSrc::Many(stringify!($ty)) }
    };
    ($field_name:ident $ty:ident) => {
        Field::Node { name: stringify!($field_name), src: FieldSrc::Optional(stringify!($ty)) }
    };
}

macro_rules! ast_enums {
    ($(
        enum $name:ident $(: $($trait:ident),*)? {
            $($variant:ident),*$(,)?
        }
    )*) => {
        [$(
            AstEnumSrc {
                name: stringify!($name),
                traits: &[$($(stringify!($trait)),*)?],
                variants: &[$(stringify!($variant)),*],
            }
        ),*]
    };
}

pub(crate) const AST_SRC: AstSrc = AstSrc {
    tokens: &["Whitespace", "Comment", "String", "RawString"],
    nodes: &ast_nodes! {
        struct SourceFile: ModuleItemOwner, AttrsOwner, DocCommentsOwner {
            modules: [Module],
        }

        struct FnDef: VisibilityOwner, NameOwner, TypeParamsOwner, DocCommentsOwner, AttrsOwner {
            Abi,
            T![const],
            T![default],
            T![async],
            T![unsafe],
            T![fn],
            ParamList,
            RetType,
            body: BlockExpr,
            T![;]
        }

        struct RetType { T![->], TypeRef }

        struct StructDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![struct],
            FieldDefList,
            T![;]
        }

        struct UnionDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![union],
            RecordFieldDefList,
        }

        struct RecordFieldDefList { T!['{'], fields: [RecordFieldDef], T!['}'] }
        struct RecordFieldDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner { }

        struct TupleFieldDefList { T!['('], fields: [TupleFieldDef], T![')'] }
        struct TupleFieldDef: VisibilityOwner, AttrsOwner {
            TypeRef,
        }

        struct EnumDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![enum],
            variant_list: EnumVariantList,
        }
        struct EnumVariantList {
            T!['{'],
            variants: [EnumVariant],
            T!['}']
        }
        struct EnumVariant: VisibilityOwner, NameOwner, DocCommentsOwner, AttrsOwner {
            FieldDefList,
            T![=],
            Expr
        }

        struct TraitDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeParamsOwner, TypeBoundsOwner {
            T![unsafe],
            T![auto],
            T![trait],
            ItemList,
        }

        struct Module: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner {
            T![mod],
            ItemList,
            T![;]
        }

        struct ItemList: ModuleItemOwner {
            T!['{'],
            assoc_items: [AssocItem],
            T!['}']
        }

        struct ConstDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            T![default],
            T![const],
            T![=],
            body: Expr,
            T![;]
        }

        struct StaticDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            T![static],
            T![mut],
            T![=],
            body: Expr,
            T![;]
        }

        struct TypeAliasDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeBoundsOwner {
            T![default],
            T![type],
            T![=],
            TypeRef,
            T![;]
        }

        struct ImplDef: TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![default],
            T![const],
            T![unsafe],
            T![impl],
            T![!],
            T![for],
            ItemList,
        }

        struct ParenType { T!['('], TypeRef, T![')'] }
        struct TupleType { T!['('], fields: [TypeRef], T![')'] }
        struct NeverType { T![!] }
        struct PathType { Path }
        struct PointerType { T![*], T![const], T![mut], TypeRef }
        struct ArrayType { T!['['], TypeRef, T![;], Expr, T![']'] }
        struct SliceType { T!['['], TypeRef, T![']'] }
        struct ReferenceType { T![&], T![lifetime], T![mut], TypeRef }
        struct PlaceholderType { T![_] }
        struct FnPointerType { Abi, T![unsafe], T![fn], ParamList, RetType }
        struct ForType { T![for], TypeParamList, TypeRef }
        struct ImplTraitType: TypeBoundsOwner { T![impl] }
        struct DynTraitType: TypeBoundsOwner { T![dyn] }

        struct TupleExpr: AttrsOwner { T!['('], exprs: [Expr], T![')'] }
        struct ArrayExpr: AttrsOwner { T!['['], exprs: [Expr], T![;], T![']'] }
        struct ParenExpr: AttrsOwner { T!['('], Expr, T![')'] }
        struct PathExpr  { Path }
        struct LambdaExpr: AttrsOwner {
            T![static],
            T![async],
            T![move],
            ParamList,
            RetType,
            body: Expr,
        }
        struct IfExpr: AttrsOwner { T![if], Condition }
        struct LoopExpr: AttrsOwner, LoopBodyOwner { T![loop] }
        struct EffectExpr: AttrsOwner { Label, T![try], T![unsafe], T![async], BlockExpr }
        struct ForExpr: AttrsOwner, LoopBodyOwner {
            T![for],
            Pat,
            T![in],
            iterable: Expr,
        }
        struct WhileExpr: AttrsOwner, LoopBodyOwner { T![while], Condition }
        struct ContinueExpr: AttrsOwner { T![continue], T![lifetime] }
        struct BreakExpr: AttrsOwner { T![break], T![lifetime], Expr }
        struct Label { T![lifetime] }
        struct BlockExpr: AttrsOwner, ModuleItemOwner {
            T!['{'], statements: [Stmt], Expr, T!['}'],
        }
        struct ReturnExpr: AttrsOwner { Expr }
        struct CallExpr: ArgListOwner { Expr }
        struct MethodCallExpr: AttrsOwner, ArgListOwner {
            Expr, T![.], NameRef, TypeArgList,
        }
        struct IndexExpr: AttrsOwner { T!['['], T![']'] }
        struct FieldExpr: AttrsOwner { Expr, T![.], NameRef }
        struct AwaitExpr: AttrsOwner { Expr, T![.], T![await] }
        struct TryExpr: AttrsOwner { Expr, T![?] }
        struct CastExpr: AttrsOwner { Expr, T![as], TypeRef }
        struct RefExpr: AttrsOwner { T![&], T![raw], T![mut], Expr }
        struct PrefixExpr: AttrsOwner { /*PrefixOp,*/ Expr }
        struct BoxExpr: AttrsOwner { T![box], Expr }
        struct RangeExpr: AttrsOwner { /*RangeOp*/ }
        struct BinExpr: AttrsOwner { /*BinOp*/ }
        struct Literal { /*LiteralToken*/ }

        struct MatchExpr: AttrsOwner { T![match], Expr, MatchArmList }
        struct MatchArmList: AttrsOwner { T!['{'], arms: [MatchArm], T!['}'] }
        struct MatchArm: AttrsOwner {
            pat: Pat,
            guard: MatchGuard,
            T![=>],
            Expr,
        }
        struct MatchGuard { T![if], Expr }

        struct RecordLit { Path, RecordFieldList}
        struct RecordFieldList {
            T!['{'],
            fields: [RecordField],
            T![..],
            spread: Expr,
            T!['}']
        }
        struct RecordField: AttrsOwner { NameRef, T![:], Expr }

        struct OrPat { pats: [Pat] }
        struct ParenPat { T!['('], Pat, T![')'] }
        struct RefPat { T![&], T![mut], Pat }
        struct BoxPat { T![box], Pat }
        struct BindPat: AttrsOwner, NameOwner { T![ref], T![mut], T![@], Pat }
        struct PlaceholderPat { T![_] }
        struct DotDotPat { T![..] }
        struct PathPat { Path }
        struct SlicePat { T!['['], args: [Pat], T![']'] }
        struct RangePat { /*RangeSeparator*/ }
        struct LiteralPat { Literal }
        struct MacroPat { MacroCall }

        struct RecordPat { RecordFieldPatList, Path }
        struct RecordFieldPatList {
            T!['{'],
            pats: [RecordInnerPat],
            record_field_pats: [RecordFieldPat],
            bind_pats: [BindPat],
            T![..],
            T!['}']
        }
        struct RecordFieldPat: AttrsOwner { NameRef, T![:], Pat }

        struct TupleStructPat { Path, T!['('], args: [Pat], T![')'] }
        struct TuplePat { T!['('], args: [Pat], T![')'] }

        struct Visibility { T![pub], T![super], T![self], T![crate] }
        struct Name { T![ident] }
        struct NameRef { /*NameRefToken*/ }

        struct MacroCall: NameOwner, AttrsOwner,DocCommentsOwner {
            Path, T![!], TokenTree, T![;]
        }
        struct Attr { T![#], T![!], T!['['], Path, T![=], input: AttrInput, T![']'] }
        struct TokenTree {}
        struct TypeParamList {
            T![<],
            generic_params: [GenericParam],
            type_params: [TypeParam],
            lifetime_params: [LifetimeParam],
            const_params: [ConstParam],
            T![>]
        }
        struct TypeParam: NameOwner, AttrsOwner, TypeBoundsOwner {
            T![=],
            default_type: TypeRef,
        }
        struct ConstParam: NameOwner, AttrsOwner, TypeAscriptionOwner {
            T![=],
            default_val: Expr,
        }
        struct LifetimeParam: AttrsOwner { T![lifetime] }
        struct TypeBound { T![lifetime], /* Question,  */ T![const], /* Question,  */ TypeRef}
        struct TypeBoundList { bounds: [TypeBound] }
        struct WherePred: TypeBoundsOwner { T![lifetime], TypeRef }
        struct WhereClause { T![where], predicates: [WherePred] }
        struct Abi { /*String*/ }
        struct ExprStmt: AttrsOwner { Expr, T![;] }
        struct LetStmt: AttrsOwner, TypeAscriptionOwner {
            T![let],
            Pat,
            T![=],
            initializer: Expr,
            T![;],
        }
        struct Condition { T![let], Pat, T![=], Expr }
        struct ParamList {
            T!['('],
            SelfParam,
            params: [Param],
            T![')']
        }
        struct SelfParam: TypeAscriptionOwner, AttrsOwner { T![&], T![mut], T![lifetime], T![self] }
        struct Param: TypeAscriptionOwner, AttrsOwner {
            Pat,
            T![...]
        }
        struct UseItem: AttrsOwner, VisibilityOwner {
            T![use],
            UseTree,
        }
        struct UseTree {
            Path, T![*], UseTreeList, Alias
        }
        struct Alias: NameOwner { T![as] }
        struct UseTreeList { T!['{'], use_trees: [UseTree], T!['}'] }
        struct ExternCrateItem: AttrsOwner, VisibilityOwner {
            T![extern], T![crate], NameRef, Alias,
        }
        struct ArgList {
            T!['('],
            args: [Expr],
            T![')']
        }
        struct Path {
            segment: PathSegment,
            qualifier: Path,
        }
        struct PathSegment {
            T![::], T![crate], T![self], T![super], T![<], NameRef, TypeArgList, ParamList, RetType, PathType, T![>]
        }
        struct TypeArgList {
            T![::],
            T![<],
            generic_args: [GenericArg],
            type_args: [TypeArg],
            lifetime_args: [LifetimeArg],
            assoc_type_args: [AssocTypeArg],
            const_args: [ConstArg],
            T![>]
        }
        struct TypeArg { TypeRef }
        struct AssocTypeArg : TypeBoundsOwner { NameRef, T![=], TypeRef }
        struct LifetimeArg { T![lifetime] }
        struct ConstArg { Literal, T![=], BlockExpr }

        struct MacroItems: ModuleItemOwner{ }

        struct MacroStmts {
            statements: [Stmt],
            Expr,
        }

        struct ExternItemList: ModuleItemOwner {
            T!['{'],
            extern_items: [ExternItem],
            T!['}']
        }

        struct ExternBlock {
            Abi,
            ExternItemList
        }

        struct MetaItem {
            Path, T![=], AttrInput, nested_meta_items: [MetaItem]
        }

        struct MacroDef {
            Name, TokenTree
        }
    },
    enums: &ast_enums! {
        enum NominalDef: NameOwner, TypeParamsOwner, AttrsOwner {
            StructDef, EnumDef, UnionDef,
        }

        enum GenericParam {
            LifetimeParam,
            TypeParam,
            ConstParam
        }

        enum GenericArg {
            LifetimeArg,
            TypeArg,
            ConstArg,
            AssocTypeArg
        }

        enum TypeRef {
            ParenType,
            TupleType,
            NeverType,
            PathType,
            PointerType,
            ArrayType,
            SliceType,
            ReferenceType,
            PlaceholderType,
            FnPointerType,
            ForType,
            ImplTraitType,
            DynTraitType,
        }

        enum ModuleItem: NameOwner, AttrsOwner, VisibilityOwner {
            StructDef,
            UnionDef,
            EnumDef,
            FnDef,
            TraitDef,
            TypeAliasDef,
            ImplDef,
            UseItem,
            ExternCrateItem,
            ConstDef,
            StaticDef,
            Module,
            MacroCall,
            ExternBlock
        }

        /* impl blocks can also contain MacroCall */
        enum AssocItem: NameOwner, AttrsOwner {
            FnDef, TypeAliasDef, ConstDef
        }

        /* extern blocks can also contain MacroCall */
        enum ExternItem: NameOwner, AttrsOwner, VisibilityOwner {
            FnDef, StaticDef
        }

        enum Expr: AttrsOwner {
            TupleExpr,
            ArrayExpr,
            ParenExpr,
            PathExpr,
            LambdaExpr,
            IfExpr,
            LoopExpr,
            ForExpr,
            WhileExpr,
            ContinueExpr,
            BreakExpr,
            Label,
            BlockExpr,
            ReturnExpr,
            MatchExpr,
            RecordLit,
            CallExpr,
            IndexExpr,
            MethodCallExpr,
            FieldExpr,
            AwaitExpr,
            TryExpr,
            EffectExpr,
            CastExpr,
            RefExpr,
            PrefixExpr,
            RangeExpr,
            BinExpr,
            Literal,
            MacroCall,
            BoxExpr,
        }

        enum Pat {
            OrPat,
            ParenPat,
            RefPat,
            BoxPat,
            BindPat,
            PlaceholderPat,
            DotDotPat,
            PathPat,
            RecordPat,
            TupleStructPat,
            TuplePat,
            SlicePat,
            RangePat,
            LiteralPat,
            MacroPat,
        }

        enum RecordInnerPat {
            RecordFieldPat,
            BindPat
        }

        enum AttrInput { Literal, TokenTree }
        enum Stmt {
            LetStmt,
            ExprStmt,
            // macro calls are parsed as expression statements */
        }

        enum FieldDefList {
            RecordFieldDefList,
            TupleFieldDefList,
        }
    },
};
