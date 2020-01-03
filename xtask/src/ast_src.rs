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
        (";", "SEMI"),
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
        ("..", "DOTDOT"),
        ("...", "DOTDOTDOT"),
        ("..=", "DOTDOTEQ"),
        (":", "COLON"),
        ("::", "COLONCOLON"),
        ("=", "EQ"),
        ("==", "EQEQ"),
        ("=>", "FAT_ARROW"),
        ("!", "EXCL"),
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
        ("&&", "AMPAMP"),
        ("||", "PIPEPIPE"),
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
    contextual_keywords: &["auto", "default", "existential", "union"],
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
        "IMPL_BLOCK",
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
        "TRY_BLOCK_EXPR",
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
        "BLOCK",
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
    pub(crate) nodes: &'a [AstNodeSrc<'a>],
    pub(crate) enums: &'a [AstEnumSrc<'a>],
}

pub(crate) struct AstNodeSrc<'a> {
    pub(crate) name: &'a str,
    pub(crate) traits: &'a [&'a str],
    pub(crate) fields: &'a [(&'a str, FieldSrc<&'a str>)],
}

pub(crate) enum FieldSrc<T> {
    Shorthand,
    Optional(T),
    Many(T),
}

pub(crate) struct AstEnumSrc<'a> {
    pub(crate) name: &'a str,
    pub(crate) traits: &'a [&'a str],
    pub(crate) variants: &'a [&'a str],
}

macro_rules! ast_nodes {
    ($(
        struct $name:ident$(: $($trait:ident),*)? {
            $($field_name:ident $(: $ty:tt)?),*$(,)?
        }
    )*) => {
        [$(
            AstNodeSrc {
                name: stringify!($name),
                traits: &[$($(stringify!($trait)),*)?],
                fields: &[$(
                    (stringify!($field_name), field_ty!($field_name $($ty)?))
                ),*],

            }
        ),*]
    };
}

macro_rules! field_ty {
    ($field_name:ident) => {
        FieldSrc::Shorthand
    };
    ($field_name:ident [$ty:ident]) => {
        FieldSrc::Many(stringify!($ty))
    };
    ($field_name:ident $ty:ident) => {
        FieldSrc::Optional(stringify!($ty))
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
    nodes: &ast_nodes! {
        struct SourceFile: ModuleItemOwner, FnDefOwner {
            modules: [Module],
        }

        struct FnDef: VisibilityOwner, NameOwner, TypeParamsOwner, DocCommentsOwner, AttrsOwner {
            ParamList,
            RetType,
            body: BlockExpr,
        }

        struct RetType { TypeRef }

        struct StructDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
        }

        struct UnionDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            RecordFieldDefList,
        }

        struct RecordFieldDefList { fields: [RecordFieldDef] }
        struct RecordFieldDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner { }

        struct TupleFieldDefList { fields: [TupleFieldDef] }
        struct TupleFieldDef: VisibilityOwner, AttrsOwner {
            TypeRef,
        }

        struct EnumDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            variant_list: EnumVariantList,
        }
        struct EnumVariantList {
            variants: [EnumVariant],
        }
        struct EnumVariant: NameOwner, DocCommentsOwner, AttrsOwner {
            Expr
        }

        struct TraitDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeParamsOwner, TypeBoundsOwner {
            ItemList,
        }

        struct Module: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner {
            ItemList,
        }

        struct ItemList: FnDefOwner, ModuleItemOwner {
            impl_items: [ImplItem],
        }

        struct ConstDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            body: Expr,
        }

        struct StaticDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            body: Expr,
        }

        struct TypeAliasDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeBoundsOwner {
            TypeRef,
        }

        struct ImplBlock: TypeParamsOwner, AttrsOwner {
            ItemList,
        }

        struct ParenType { TypeRef }
        struct TupleType { fields: [TypeRef] }
        struct NeverType { }
        struct PathType { Path }
        struct PointerType { TypeRef }
        struct ArrayType { TypeRef, Expr }
        struct SliceType { TypeRef }
        struct ReferenceType { TypeRef }
        struct PlaceholderType {  }
        struct FnPointerType { ParamList, RetType }
        struct ForType { TypeRef }
        struct ImplTraitType: TypeBoundsOwner {}
        struct DynTraitType: TypeBoundsOwner {}

        struct TupleExpr { exprs: [Expr] }
        struct ArrayExpr { exprs: [Expr] }
        struct ParenExpr { Expr }
        struct PathExpr  { Path }
        struct LambdaExpr {
            ParamList,
            RetType,
            body: Expr,
        }
        struct IfExpr { Condition }
        struct LoopExpr: LoopBodyOwner { }
        struct TryBlockExpr { body: BlockExpr }
        struct ForExpr: LoopBodyOwner {
            Pat,
            iterable: Expr,
        }
        struct WhileExpr: LoopBodyOwner { Condition }
        struct ContinueExpr {}
        struct BreakExpr { Expr }
        struct Label {}
        struct BlockExpr { Block  }
        struct ReturnExpr { Expr }
        struct CallExpr: ArgListOwner { Expr }
        struct MethodCallExpr: ArgListOwner {
            Expr, NameRef, TypeArgList,
        }
        struct IndexExpr {}
        struct FieldExpr { Expr, NameRef }
        struct AwaitExpr { Expr }
        struct TryExpr { Expr }
        struct CastExpr { Expr, TypeRef }
        struct RefExpr { Expr }
        struct PrefixExpr { Expr }
        struct BoxExpr { Expr }
        struct RangeExpr {}
        struct BinExpr {}
        struct Literal {}

        struct MatchExpr { Expr, MatchArmList }
        struct MatchArmList: AttrsOwner { arms: [MatchArm] }
        struct MatchArm: AttrsOwner {
            pats: [Pat],
            guard: MatchGuard,
            Expr,
         }
        struct MatchGuard { Expr }

        struct RecordLit { Path, RecordFieldList }
        struct RecordFieldList {
            fields: [RecordField],
            spread: Expr,
         }
        struct RecordField { NameRef, Expr }

        struct RefPat { Pat }
        struct BoxPat { Pat }
        struct BindPat: NameOwner { Pat }
        struct PlaceholderPat { }
        struct DotDotPat { }
        struct PathPat {  Path }
        struct SlicePat {}
        struct RangePat {}
        struct LiteralPat { Literal }

        struct RecordPat { RecordFieldPatList, Path }
        struct RecordFieldPatList {
            record_field_pats: [RecordFieldPat],
            bind_pats: [BindPat],
        }
        struct RecordFieldPat: NameOwner { Pat }

        struct TupleStructPat { Path, args: [Pat] }
        struct TuplePat { args: [Pat] }

        struct Visibility {}
        struct Name {}
        struct NameRef {}

        struct MacroCall: NameOwner, AttrsOwner,DocCommentsOwner {
            TokenTree, Path
        }
        struct Attr { Path, input: AttrInput }
        struct TokenTree {}
        struct TypeParamList {
            type_params: [TypeParam],
            lifetime_params: [LifetimeParam],
        }
        struct TypeParam: NameOwner, AttrsOwner, TypeBoundsOwner {
            default_type: TypeRef,
        }
        struct ConstParam: NameOwner, AttrsOwner, TypeAscriptionOwner {
            default_val: Expr,
        }
        struct LifetimeParam: AttrsOwner { }
        struct TypeBound { TypeRef}
        struct TypeBoundList { bounds: [TypeBound] }
        struct WherePred: TypeBoundsOwner { TypeRef }
        struct WhereClause { predicates: [WherePred] }
        struct ExprStmt { Expr }
        struct LetStmt: TypeAscriptionOwner {
            Pat,
            initializer: Expr,
        }
        struct Condition { Pat, Expr }
        struct Block: AttrsOwner, ModuleItemOwner {
            statements: [Stmt],
            Expr,
        }
        struct ParamList {
            SelfParam,
            params: [Param],
        }
        struct SelfParam: TypeAscriptionOwner, AttrsOwner { }
        struct Param: TypeAscriptionOwner, AttrsOwner {
            Pat,
        }
        struct UseItem: AttrsOwner, VisibilityOwner {
            UseTree,
        }
        struct UseTree {
            Path, UseTreeList, Alias
        }
        struct Alias: NameOwner { }
        struct UseTreeList { use_trees: [UseTree] }
        struct ExternCrateItem: AttrsOwner, VisibilityOwner {
            NameRef, Alias,
        }
        struct ArgList {
            args: [Expr],
        }
        struct Path {
            segment: PathSegment,
            qualifier: Path,
        }
        struct PathSegment {
            NameRef, TypeArgList, ParamList, RetType, PathType,
        }
        struct TypeArgList {
            type_args: [TypeArg],
            lifetime_args: [LifetimeArg],
            assoc_type_args: [AssocTypeArg],
        }
        struct TypeArg { TypeRef }
        struct AssocTypeArg { NameRef, TypeRef }
        struct LifetimeArg {}

        struct MacroItems: ModuleItemOwner, FnDefOwner { }

        struct MacroStmts {
            statements: [Stmt],
            Expr,
        }
    },
    enums: &ast_enums! {
        enum NominalDef: NameOwner, TypeParamsOwner, AttrsOwner {
            StructDef, EnumDef, UnionDef,
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

        enum ModuleItem: AttrsOwner, VisibilityOwner {
            StructDef,
            UnionDef,
            EnumDef,
            FnDef,
            TraitDef,
            TypeAliasDef,
            ImplBlock,
            UseItem,
            ExternCrateItem,
            ConstDef,
            StaticDef,
            Module,
        }

        enum ImplItem: AttrsOwner {
            FnDef, TypeAliasDef, ConstDef,
        }

        enum Expr {
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
            TryBlockExpr,
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
        }

        enum AttrInput { Literal, TokenTree }
        enum Stmt { ExprStmt, LetStmt }
    },
};
