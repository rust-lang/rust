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
    pub(crate) nodes: &'a [AstNodeSrc<'a>],
    pub(crate) enums: &'a [AstEnumSrc<'a>],
    pub(crate) token_enums: &'a [AstEnumSrc<'a>],
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
        struct SourceFile: ModuleItemOwner, FnDefOwner, AttrsOwner {
            modules: [Module],
        }

        struct FnDef: VisibilityOwner, NameOwner, TypeParamsOwner, DocCommentsOwner, AttrsOwner {
            Abi,
            ConstKw,
            DefaultKw,
            AsyncKw,
            UnsafeKw,
            FnKw,
            ParamList,
            RetType,
            body: BlockExpr,
            Semi
        }

        struct RetType { ThinArrow, TypeRef }

        struct StructDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            StructKw,
            FieldDefList,
            Semi
        }

        struct UnionDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            UnionKw,
            RecordFieldDefList,
        }

        struct RecordFieldDefList { LCurly, fields: [RecordFieldDef], RCurly }
        struct RecordFieldDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner { }

        struct TupleFieldDefList { LParen, fields: [TupleFieldDef], RParen }
        struct TupleFieldDef: VisibilityOwner, AttrsOwner {
            TypeRef,
        }

        struct EnumDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            EnumKw,
            variant_list: EnumVariantList,
        }
        struct EnumVariantList {
            LCurly,
            variants: [EnumVariant],
            RCurly
        }
        struct EnumVariant: VisibilityOwner, NameOwner, DocCommentsOwner, AttrsOwner {
            FieldDefList,
            Eq,
            Expr
        }

        struct TraitDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeParamsOwner, TypeBoundsOwner {
            UnsafeKw,
            AutoKw,
            TraitKw,
            ItemList,
        }

        struct Module: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner {
            ModKw,
            ItemList,
            Semi
        }

        struct ItemList: FnDefOwner, ModuleItemOwner {
            LCurly,
            impl_items: [ImplItem],
            RCurly
        }

        struct ConstDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            DefaultKw,
            ConstKw,
            Eq,
            body: Expr,
            Semi
        }

        struct StaticDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            StaticKw,
            MutKw,
            Eq,
            body: Expr,
            Semi
        }

        struct TypeAliasDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeBoundsOwner {
            DefaultKw,
            TypeKw,
            Eq,
            TypeRef,
            Semi
        }

        struct ImplDef: TypeParamsOwner, AttrsOwner {
            DefaultKw,
            ConstKw,
            UnsafeKw,
            ImplKw,
            Excl,
            ForKw,
            ItemList,
        }

        struct ParenType { LParen, TypeRef, RParen }
        struct TupleType { LParen, fields: [TypeRef], RParen }
        struct NeverType { Excl }
        struct PathType { Path }
        struct PointerType { Star, ConstKw, MutKw, TypeRef }
        struct ArrayType { LBrack, TypeRef, Semi, Expr, RBrack }
        struct SliceType { LBrack, TypeRef, RBrack }
        struct ReferenceType { Amp, Lifetime, MutKw, TypeRef }
        struct PlaceholderType { Underscore }
        struct FnPointerType { Abi, UnsafeKw, FnKw, ParamList, RetType }
        struct ForType { ForKw, TypeParamList, TypeRef }
        struct ImplTraitType: TypeBoundsOwner { ImplKw }
        struct DynTraitType: TypeBoundsOwner { DynKw }

        struct TupleExpr: AttrsOwner { LParen, exprs: [Expr], RParen }
        struct ArrayExpr: AttrsOwner { LBrack, exprs: [Expr], Semi, RBrack }
        struct ParenExpr: AttrsOwner { LParen, Expr, RParen }
        struct PathExpr  { Path }
        struct LambdaExpr: AttrsOwner {
            StaticKw,
            AsyncKw,
            MoveKw,
            ParamList,
            RetType,
            body: Expr,
        }
        struct IfExpr: AttrsOwner { IfKw, Condition }
        struct LoopExpr: AttrsOwner, LoopBodyOwner { LoopKw }
        struct TryBlockExpr: AttrsOwner { TryKw, body: BlockExpr }
        struct ForExpr: AttrsOwner, LoopBodyOwner {
            ForKw,
            Pat,
            InKw,
            iterable: Expr,
        }
        struct WhileExpr: AttrsOwner, LoopBodyOwner { WhileKw, Condition }
        struct ContinueExpr: AttrsOwner { ContinueKw, Lifetime }
        struct BreakExpr: AttrsOwner { BreakKw, Lifetime, Expr }
        struct Label { Lifetime }
        struct BlockExpr: AttrsOwner { Label, UnsafeKw, Block  }
        struct ReturnExpr: AttrsOwner { Expr }
        struct CallExpr: ArgListOwner { Expr }
        struct MethodCallExpr: AttrsOwner, ArgListOwner {
            Expr, Dot, NameRef, TypeArgList,
        }
        struct IndexExpr: AttrsOwner { LBrack, RBrack }
        struct FieldExpr: AttrsOwner { Expr, Dot, NameRef }
        struct AwaitExpr: AttrsOwner { Expr, Dot, AwaitKw }
        struct TryExpr: AttrsOwner { TryKw, Expr }
        struct CastExpr: AttrsOwner { Expr, AsKw, TypeRef }
        struct RefExpr: AttrsOwner { Amp, RawKw, MutKw, Expr }
        struct PrefixExpr: AttrsOwner { PrefixOp, Expr }
        struct BoxExpr: AttrsOwner { BoxKw, Expr }
        struct RangeExpr: AttrsOwner { RangeOp }
        struct BinExpr: AttrsOwner { BinOp }
        struct Literal { LiteralToken }

        struct MatchExpr: AttrsOwner { MatchKw, Expr, MatchArmList }
        struct MatchArmList: AttrsOwner { LCurly, arms: [MatchArm], RCurly }
        struct MatchArm: AttrsOwner {
            pat: Pat,
            guard: MatchGuard,
            FatArrow,
            Expr,
        }
        struct MatchGuard { IfKw, Expr }

        struct RecordLit { Path, RecordFieldList}
        struct RecordFieldList {
            LCurly,
            fields: [RecordField],
            Dotdot,
            spread: Expr,
            RCurly
        }
        struct RecordField: AttrsOwner { NameRef, Colon, Expr }

        struct OrPat { pats: [Pat] }
        struct ParenPat { LParen, Pat, RParen }
        struct RefPat { Amp, MutKw, Pat }
        struct BoxPat { BoxKw, Pat }
        struct BindPat: AttrsOwner, NameOwner { RefKw, MutKw, At, Pat }
        struct PlaceholderPat { Underscore }
        struct DotDotPat { Dotdot }
        struct PathPat { Path }
        struct SlicePat { LBrack, args: [Pat], RBrack }
        struct RangePat { RangeSeparator }
        struct LiteralPat { Literal }
        struct MacroPat { MacroCall }

        struct RecordPat { RecordFieldPatList, Path }
        struct RecordFieldPatList {
            LCurly,
            pats: [RecordInnerPat],
            record_field_pats: [RecordFieldPat],
            bind_pats: [BindPat],
            Dotdot,
            RCurly
        }
        struct RecordFieldPat: AttrsOwner, NameOwner { Colon, Pat }

        struct TupleStructPat { Path, LParen, args: [Pat], RParen }
        struct TuplePat { LParen, args: [Pat], RParen }

        struct Visibility { PubKw, SuperKw, SelfKw, CrateKw }
        struct Name { Ident }
        struct NameRef { NameRefToken }

        struct MacroCall: NameOwner, AttrsOwner,DocCommentsOwner {
            Path, Excl, TokenTree, Semi
        }
        struct Attr { Pound, Excl, LBrack, Path, Eq, input: AttrInput, RBrack }
        struct TokenTree {}
        struct TypeParamList {
            LAngle,
            generic_params: [GenericParam],
            type_params: [TypeParam],
            lifetime_params: [LifetimeParam],
            const_params: [ConstParam],
            RAngle
        }
        struct TypeParam: NameOwner, AttrsOwner, TypeBoundsOwner {
            Eq,
            default_type: TypeRef,
        }
        struct ConstParam: NameOwner, AttrsOwner, TypeAscriptionOwner {
            Eq,
            default_val: Expr,
        }
        struct LifetimeParam: AttrsOwner { Lifetime}
        struct TypeBound { Lifetime, /* Question,  */ ConstKw, /* Question,  */ TypeRef}
        struct TypeBoundList { bounds: [TypeBound] }
        struct WherePred: TypeBoundsOwner { Lifetime, TypeRef }
        struct WhereClause { WhereKw, predicates: [WherePred] }
        struct Abi { String }
        struct ExprStmt: AttrsOwner { Expr, Semi }
        struct LetStmt: AttrsOwner, TypeAscriptionOwner {
            LetKw,
            Pat,
            Eq,
            initializer: Expr,
            Semi,
        }
        struct Condition { LetKw, Pat, Eq, Expr }
        struct Block: AttrsOwner, ModuleItemOwner {
            LCurly,
            statements: [Stmt],
            Expr,
            RCurly,
        }
        struct ParamList {
            LParen,
            SelfParam,
            params: [Param],
            RParen
        }
        struct SelfParam: TypeAscriptionOwner, AttrsOwner { Amp, Lifetime, SelfKw }
        struct Param: TypeAscriptionOwner, AttrsOwner {
            Pat,
            Dotdotdot
        }
        struct UseItem: AttrsOwner, VisibilityOwner {
            UseKw,
            UseTree,
        }
        struct UseTree {
            Path, Star, UseTreeList, Alias
        }
        struct Alias: NameOwner { AsKw }
        struct UseTreeList { LCurly, use_trees: [UseTree], RCurly }
        struct ExternCrateItem: AttrsOwner, VisibilityOwner {
            ExternKw, CrateKw, NameRef, Alias,
        }
        struct ArgList {
            LParen,
            args: [Expr],
            RParen
        }
        struct Path {
            segment: PathSegment,
            qualifier: Path,
        }
        struct PathSegment {
            Coloncolon, LAngle, NameRef, TypeArgList, ParamList, RetType, PathType, RAngle
        }
        struct TypeArgList {
            Coloncolon,
            LAngle,
            generic_args: [GenericArg],
            type_args: [TypeArg],
            lifetime_args: [LifetimeArg],
            assoc_type_args: [AssocTypeArg],
            const_args: [ConstArg],
            RAngle
        }
        struct TypeArg { TypeRef }
        struct AssocTypeArg : TypeBoundsOwner { NameRef, Eq, TypeRef }
        struct LifetimeArg { Lifetime }
        struct ConstArg { Literal, Eq, BlockExpr }

        struct MacroItems: ModuleItemOwner, FnDefOwner { }

        struct MacroStmts {
            statements: [Stmt],
            Expr,
        }

        struct ExternItemList: FnDefOwner, ModuleItemOwner {
            LCurly,
            extern_items: [ExternItem],
            RCurly
        }

        struct ExternBlock {
            Abi,
            ExternItemList
        }

        struct MetaItem {
            Path, Eq, AttrInput, nested_meta_items: [MetaItem]
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
        enum ImplItem: NameOwner, AttrsOwner {
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

    token_enums: &ast_enums! {
        enum LeftDelimiter { LParen, LBrack, LCurly }
        enum RightDelimiter { RParen, RBrack, RCurly }
        enum RangeSeparator { Dotdot, Dotdotdot, Dotdoteq}

        enum BinOp {
            Pipepipe,
            Ampamp,
            Eqeq,
            Neq,
            Lteq,
            Gteq,
            LAngle,
            RAngle,
            Plus,
            Star,
            Minus,
            Slash,
            Percent,
            Shl,
            Shr,
            Caret,
            Pipe,
            Amp,
            Eq,
            Pluseq,
            Slasheq,
            Stareq,
            Percenteq,
            Shreq,
            Shleq,
            Minuseq,
            Pipeeq,
            Ampeq,
            Careteq,
        }

        enum PrefixOp {
            Minus,
            Excl,
            Star
        }

        enum RangeOp {
            Dotdot,
            Dotdoteq
        }

        enum LiteralToken {
            IntNumber,
            FloatNumber,
            String,
            RawString,
            TrueKw,
            FalseKw,
            ByteString,
            RawByteString,
            Char,
            Byte
        }

        enum NameRefToken {
            Ident,
            IntNumber
        }
    },
};
