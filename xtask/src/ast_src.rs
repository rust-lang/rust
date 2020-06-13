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
    pub(crate) doc: &'a [&'a str],
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
    pub(crate) doc: &'a [&'a str],
    pub(crate) name: &'a str,
    pub(crate) traits: &'a [&'a str],
    pub(crate) variants: &'a [&'a str],
}

macro_rules! ast_nodes {
    ($(
        $(#[doc = $doc:expr])+
        struct $name:ident$(: $($trait:ident),*)? {
            $($field_name:ident $(![$token:tt])? $(: $ty:tt)?),*$(,)?
        }
    )*) => {
        [$(
            AstNodeSrc {
                doc: &[$($doc),*],
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
        $(#[doc = $doc:expr])+
        enum $name:ident $(: $($trait:ident),*)? {
            $($variant:ident),*$(,)?
        }
    )*) => {
        [$(
            AstEnumSrc {
                doc: &[$($doc),*],
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
        /// The entire Rust source file. Includes all top-level inner attributes and module items.
        ///
        /// [Reference](https://doc.rust-lang.org/reference/crates-and-source-files.html)
        struct SourceFile: ModuleItemOwner, AttrsOwner, DocCommentsOwner {
            modules: [Module],
        }

        /// Function definition either with body or not.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub extern "C" fn foo<T>(#[attr] Patern {p}: Pattern) -> u32
        ///     where
        ///         T: Debug
        ///     {
        ///         42
        ///     }
        /// ❱
        ///
        /// extern "C" {
        ///     ❰ fn fn_decl(also_variadic_ffi: u32, ...) -> u32; ❱
        /// }
        /// ```
        ///
        /// - [Reference](https://doc.rust-lang.org/reference/items/functions.html)
        /// - [Nomicon](https://doc.rust-lang.org/nomicon/ffi.html#variadic-functions)
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

        /// Return type annotation.
        ///
        /// ```
        /// fn foo(a: u32) ❰ -> Option<u32> ❱ { Some(a) }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/functions.html)
        struct RetType { T![->], TypeRef }

        /// Struct definition.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     struct Foo<T> where T: Debug {
        ///         /// Docs
        ///         #[attr]
        ///         pub a: u32,
        ///         b: T,
        ///     }
        /// ❱
        ///
        /// ❰ struct Foo; ❱
        /// ❰ struct Foo<T>(#[attr] T) where T: Debug; ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
        struct StructDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![struct],
            FieldDefList,
            T![;]
        }

        /// Union definition.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub union Foo<T> where T: Debug {
        ///         /// Docs
        ///         #[attr]
        ///         a: T,
        ///         b: u32,
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/unions.html)
        struct UnionDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![union],
            RecordFieldDefList,
        }

        /// Record field definition list including enclosing curly braces.
        ///
        /// ```
        /// struct Foo // same for union
        /// ❰
        ///     {
        ///         a: u32,
        ///         b: bool,
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
        struct RecordFieldDefList { T!['{'], fields: [RecordFieldDef], T!['}'] }

        /// Record field definition including its attributes and doc comments.
        ///
        /// ` ``
        /// same for union
        /// struct Foo {
        ///      ❰
        ///          /// Docs
        ///          #[attr]
        ///          pub a: u32
        ///      ❱
        ///
        ///      ❰ b: bool ❱
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
        struct RecordFieldDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner { }

        /// Tuple field definition list including enclosing parens.
        ///
        /// ```
        /// struct Foo ❰ (u32, String, Vec<u32>) ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
        struct TupleFieldDefList { T!['('], fields: [TupleFieldDef], T![')'] }

        /// Tuple field definition including its attributes.
        ///
        /// ```
        /// struct Foo(❰ #[attr] u32 ❱);
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/structs.html)
        struct TupleFieldDef: VisibilityOwner, AttrsOwner {
            TypeRef,
        }

        /// Enum definition.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub enum Foo<T> where T: Debug {
        ///         /// Docs
        ///         #[attr]
        ///         Bar,
        ///         Baz(#[attr] u32),
        ///         Bruh {
        ///             a: u32,
        ///             /// Docs
        ///             #[attr]
        ///             b: T,
        ///         }
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/enumerations.html)
        struct EnumDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![enum],
            variant_list: EnumVariantList,
        }

        /// Enum variant definition list including enclosing curly braces.
        ///
        /// ```
        /// enum Foo
        /// ❰
        ///     {
        ///         Bar,
        ///         Baz(u32),
        ///         Bruh {
        ///             a: u32
        ///         }
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/enumerations.html)
        struct EnumVariantList {
            T!['{'],
            variants: [EnumVariant],
            T!['}']
        }

        /// Enum variant definition including its attributes and discriminant value definition.
        ///
        /// ```
        /// enum Foo {
        ///     ❰
        ///         /// Docs
        ///         #[attr]
        ///         Bar
        ///     ❱
        ///
        ///     // same for tuple and record variants
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/enumerations.html)
        struct EnumVariant: VisibilityOwner, NameOwner, DocCommentsOwner, AttrsOwner {
            FieldDefList,
            T![=],
            Expr
        }

        /// Trait definition.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub unsafe trait Foo<T>: Debug where T: Debug {
        ///         // ...
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/traits.html)
        struct TraitDef: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner, TypeParamsOwner, TypeBoundsOwner {
            T![unsafe],
            T![auto],
            T![trait],
            ItemList,
        }

        /// Module definition either with body or not.
        /// Includes all of its inner and outer attributes, module items, doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub mod foo;
        /// ❱
        ///
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub mod bar {
        ///        //! Inner docs
        ///        #![inner_attr]
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/modules.html)
        struct Module: VisibilityOwner, NameOwner, AttrsOwner, DocCommentsOwner {
            T![mod],
            ItemList,
            T![;]
        }

        /// Item defintion list.
        /// This is used for both top-level items and impl block items.
        ///
        /// ```
        /// ❰
        ///     fn foo {}
        ///     struct Bar;
        ///     enum Baz;
        ///     trait Bruh;
        ///     const BRUUH: u32 = 42;
        /// ❱
        ///
        /// impl Foo
        /// ❰
        ///     {
        ///         fn bar() {}
        ///         const BAZ: u32 = 42;
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items.html)
        struct ItemList: ModuleItemOwner {
            T!['{'],
            assoc_items: [AssocItem],
            T!['}']
        }

        /// Constant variable definition.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub const FOO: u32 = 42;
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/constant-items.html)
        struct ConstDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            T![default],
            T![const],
            T![=],
            body: Expr,
            T![;]
        }


        /// Static variable definition.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub static mut FOO: u32 = 42;
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/static-items.html)
        struct StaticDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeAscriptionOwner {
            T![static],
            T![mut],
            T![=],
            body: Expr,
            T![;]
        }

        /// Type alias definition.
        /// Includes associated type clauses with type bounds.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     pub type Foo<T> where T: Debug = T;
        /// ❱
        ///
        /// trait Bar {
        ///     ❰ type Baz: Debug; ❱
        ///     ❰ type Bruh = String; ❱
        ///     ❰ type Bruuh: Debug = u32; ❱
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/type-aliases.html)
        struct TypeAliasDef: VisibilityOwner, NameOwner, TypeParamsOwner, AttrsOwner, DocCommentsOwner, TypeBoundsOwner {
            T![default],
            T![type],
            T![=],
            TypeRef,
            T![;]
        }

        /// Inherent and trait impl definition.
        /// Includes all of its inner and outer attributes.
        ///
        /// ```
        /// ❰
        ///     #[attr]
        ///     unsafe impl<T> const !Foo for Bar where T: Debug {
        ///         #![inner_attr]
        ///         // ...
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/implementations.html)
        struct ImplDef: TypeParamsOwner, AttrsOwner, DocCommentsOwner {
            T![default],
            T![const],
            T![unsafe],
            T![impl],
            T![!],
            T![for],
            ItemList,
        }


        /// Parenthesized type reference.
        /// Note: parens are only used for grouping, this is not a tuple type.
        ///
        /// ```
        /// // This is effectively just `u32`.
        /// // Single-item tuple must be defined with a trailing comma: `(u32,)`
        /// type Foo = ❰ (u32) ❱;
        ///
        /// let bar: &'static ❰ (dyn Debug) ❱ = "bruh";
        /// ```
        struct ParenType { T!['('], TypeRef, T![')'] }

        /// Unnamed tuple type.
        ///
        /// ```
        /// let foo: ❰ (u32, bool) ❱ = (42, true);
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/tuple.html)
        struct TupleType { T!['('], fields: [TypeRef], T![')'] }

        /// The never type (i.e. the exclamation point).
        ///
        /// ```
        /// type T = ❰ ! ❱;
        ///
        /// fn no_return() -> ❰ ! ❱ {
        ///     loop {}
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/never.html)
        struct NeverType { T![!] }

        /// Path to a type.
        /// Includes single identifier type names and elaborate paths with
        /// generic parameters.
        ///
        /// ```
        /// type Foo = ❰ String ❱;
        /// type Bar = ❰ std::vec::Vec<T> ❱;
        /// type Baz = ❰ ::bruh::<Bruuh as Iterator>::Item ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/paths.html)
        struct PathType { Path }

        /// Raw pointer type.
        ///
        /// ```
        /// type Foo = ❰ *const u32 ❱;
        /// type Bar = ❰ *mut u32 ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/pointer.html#raw-pointers-const-and-mut)
        struct PointerType { T![*], T![const], T![mut], TypeRef }

        /// Array type.
        ///
        /// ```
        /// type Foo = ❰ [u32; 24 - 3] ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/array.html)
        struct ArrayType { T!['['], TypeRef, T![;], Expr, T![']'] }

        /// Slice type.
        ///
        /// ```
        /// type Foo = ❰ [u8] ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/slice.html)
        struct SliceType { T!['['], TypeRef, T![']'] }

        /// Reference type.
        ///
        /// ```
        /// type Foo = ❰ &'static str ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/pointer.html)
        struct ReferenceType { T![&], T![lifetime], T![mut], TypeRef }

        /// Placeholder type (i.e. the underscore).
        ///
        /// ```
        /// let foo: ❰ _ ❱ = 42_u32;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/inferred.html)
        struct PlaceholderType { T![_] }

        /// Function pointer type (not to be confused with `Fn*` family of traits).
        ///
        /// ```
        /// type Foo = ❰ async fn(#[attr] u32, named: bool) -> u32 ❱;
        ///
        /// type Bar = ❰ extern "C" fn(variadic: u32, #[attr] ...) ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/function-pointer.html)
        struct FnPointerType { Abi, T![unsafe], T![fn], ParamList, RetType }

        /// Higher order type.
        ///
        /// ```
        /// type Foo = ❰ for<'a> fn(&'a str) ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/nomicon/hrtb.html)
        struct ForType { T![for], TypeParamList, TypeRef }

        /// Opaque `impl Trait` type.
        ///
        /// ```
        /// fn foo(bar: ❰ impl Debug + Eq ❱) {}
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/impl-trait.html)
        struct ImplTraitType: TypeBoundsOwner { T![impl] }

        /// Trait object type.
        ///
        /// ```
        /// type Foo = ❰ dyn Debug ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/types/trait-object.html)
        struct DynTraitType: TypeBoundsOwner { T![dyn] }

        /// Tuple literal.
        ///
        /// ```
        /// ❰ (42, true) ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/tuple-expr.html)
        struct TupleExpr: AttrsOwner { T!['('], exprs: [Expr], T![')'] }

        /// Array literal.
        ///
        /// ```
        /// ❰ [#![inner_attr] true, false, true] ❱;
        ///
        /// ❰ ["baz"; 24] ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/array-expr.html)
        struct ArrayExpr: AttrsOwner { T!['['], exprs: [Expr], T![;], T![']'] }

        /// Parenthesized expression.
        /// Note: parens are only used for grouping, this is not a tuple literal.
        ///
        /// ```
        /// ❰ (#![inner_attr] 2 + 2) ❱ * 2;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/grouped-expr.html)
        struct ParenExpr: AttrsOwner { T!['('], Expr, T![')'] }

        /// Path to a symbol in expression context.
        /// Includes single identifier variable names and elaborate paths with
        /// generic parameters.
        ///
        /// ```
        /// ❰ Some::<i32> ❱;
        /// ❰ foo ❱ + 42;
        /// ❰ Vec::<i32>::push ❱;
        /// ❰ <[i32]>::reverse ❱;
        /// ❰ <String as std::borrow::Borrow<str>>::borrow ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/path-expr.html)
        struct PathExpr { Path }

        /// Anonymous callable object literal a.k.a. closure, lambda or functor.
        ///
        /// ```
        /// ❰ || 42 ❱;
        /// ❰ |a: u32| val + 1 ❱;
        /// ❰ async |#[attr] Pattern(_): Pattern| { bar } ❱;
        /// ❰ move || baz ❱;
        /// ❰ || -> u32 { closure_with_ret_type_annotation_requires_block_expr } ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/closure-expr.html)
        struct LambdaExpr: AttrsOwner {
            T![static], // Note(@matklad): I belive this is (used to be?) syntax for generators
            T![async],
            T![move],
            ParamList,
            RetType,
            body: Expr,
        }

        /// If expression. Includes both regular `if` and `if let` forms.
        /// Beware that `else if` is a special case syntax sugar, because in general
        /// there has to be block expression after `else`.
        ///
        /// ```
        /// ❰ if bool_cond { 42 } ❱
        /// ❰ if bool_cond { 42 } else { 24 } ❱
        /// ❰ if bool_cond { 42 } else if bool_cond2 { 42 } ❱
        ///
        /// ❰
        ///     if let Pattern(foo) = bar {
        ///         foo
        ///     } else {
        ///         panic!();
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/if-expr.html)
        struct IfExpr: AttrsOwner { T![if], Condition }

        /// Unconditional loop expression.
        ///
        /// ```
        /// ❰
        ///     loop {
        ///         // yeah, it's that simple...
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html)
        struct LoopExpr: AttrsOwner, LoopBodyOwner { T![loop] }

        /// Block expression with an optional prefix (label, try ketword,
        /// unsafe keyword, async keyword...).
        ///
        /// ```
        /// ❰
        ///     'label: try {
        ///         None?
        ///     }
        /// ❱
        /// ```
        ///
        /// - [try block](https://doc.rust-lang.org/unstable-book/language-features/try-blocks.html)
        /// - [unsafe block](https://doc.rust-lang.org/reference/expressions/block-expr.html#unsafe-blocks)
        /// - [async block](https://doc.rust-lang.org/reference/expressions/block-expr.html#async-blocks)
        struct EffectExpr: AttrsOwner { Label, T![try], T![unsafe], T![async], BlockExpr }


        /// For loop expression.
        /// Note: record struct literals are not valid as iterable expression
        /// due to ambiguity.
        ///
        /// ```
        /// ❰
        /// for i in (0..4) {
        ///     dbg!(i);
        /// }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#iterator-loops)
        struct ForExpr: AttrsOwner, LoopBodyOwner {
            T![for],
            Pat,
            T![in],
            iterable: Expr,
        }

        /// While loop expression. Includes both regular `while` and `while let` forms.
        ///
        /// ```
        /// ❰
        ///     while bool_cond {
        ///         42;
        ///     }
        /// ❱
        /// ❰
        ///     while let Pattern(foo) = bar {
        ///         bar += 1;
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#predicate-loops)
        struct WhileExpr: AttrsOwner, LoopBodyOwner { T![while], Condition }

        /// Continue expression.
        ///
        /// ```
        /// while bool_cond {
        ///     ❰ continue ❱;
        /// }
        ///
        /// 'outer: loop {
        ///     loop {
        ///         ❰ continue 'outer ❱;
        ///     }
        /// }
        ///
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#continue-expressions)
        struct ContinueExpr: AttrsOwner { T![continue], T![lifetime] }

        /// Break expression.
        ///
        /// ```
        /// while bool_cond {
        ///     ❰ break ❱;
        /// }
        /// 'outer: loop {
        ///     for foo in bar {
        ///         ❰ break 'outer ❱;
        ///     }
        /// }
        /// 'outer: loop {
        ///     loop {
        ///         ❰ break 'outer 42 ❱;
        ///     }
        /// }
        /// ```
        ///
        /// [Refernce](https://doc.rust-lang.org/reference/expressions/loop-expr.html#break-expressions)
        struct BreakExpr: AttrsOwner { T![break], T![lifetime], Expr }

        /// Label.
        ///
        /// ```
        /// ❰ 'outer: ❱ loop {}
        ///
        /// let foo = ❰ 'bar: ❱ loop {}
        ///
        /// ❰ 'baz: ❱ {
        ///     break 'baz;
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html?highlight=label#loop-labels)
        /// [Labels for blocks RFC](https://github.com/rust-lang/rfcs/blob/master/text/2046-label-break-value.md)
        struct Label { T![lifetime] }

        /// Block expression. Includes unsafe blocks and block labels.
        ///
        /// ```
        ///     let foo = ❰
        ///         {
        ///             #![inner_attr]
        ///             ❰ { } ❱
        ///
        ///             ❰ 'label: { break 'label } ❱
        ///         }
        ///     ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/block-expr.html)
        /// [Labels for blocks RFC](https://github.com/rust-lang/rfcs/blob/master/text/2046-label-break-value.md)
        struct BlockExpr: AttrsOwner, ModuleItemOwner {
            Label, T!['{'], statements: [Stmt], Expr, T!['}'],
        }

        /// Return expression.
        ///
        /// ```
        /// || ❰ return 42 ❱;
        ///
        /// fn bar() {
        ///     ❰ return ❱;
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/return-expr.html)
        struct ReturnExpr: AttrsOwner { Expr }

        /// Call expression (not to be confused with method call expression, it is
        /// a separate ast node).
        ///
        /// ```
        /// ❰ foo() ❱;
        /// ❰ &str::len("bar") ❱;
        /// ❰ <&str as PartialEq<&str>>::eq(&"", &"") ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/call-expr.html)
        struct CallExpr: ArgListOwner { Expr }

        /// Method call expression.
        ///
        /// ```
        /// ❰ receiver_expr.method() ❱;
        /// ❰ receiver_expr.method::<T>(42, true) ❱;
        ///
        /// ❰ ❰ ❰ foo.bar() ❱ .baz() ❱ .bruh() ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/method-call-expr.html)
        struct MethodCallExpr: AttrsOwner, ArgListOwner {
            Expr, T![.], NameRef, TypeArgList,
        }

        /// Index expression a.k.a. subscript operator call.
        ///
        /// ```
        /// ❰ foo[42] ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/array-expr.html)
        struct IndexExpr: AttrsOwner { T!['['], T![']'] }

        /// Field access expression.
        ///
        /// ```
        /// ❰ expr.bar ❱;
        ///
        /// ❰ ❰ ❰ foo.bar ❱ .baz ❱ .bruh ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/field-expr.html)
        struct FieldExpr: AttrsOwner { Expr, T![.], NameRef }

        /// Await operator call expression.
        ///
        /// ```
        /// ❰ expr.await ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/await-expr.html)
        struct AwaitExpr: AttrsOwner { Expr, T![.], T![await] }

        /// The question mark operator call.
        ///
        /// ```
        /// ❰ expr? ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#the-question-mark-operator)
        struct TryExpr: AttrsOwner { Expr, T![?] }

        /// Type cast expression.
        ///
        /// ```
        /// ❰ expr as T ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions)
        struct CastExpr: AttrsOwner { Expr, T![as], TypeRef }


        /// Borrow operator call.
        ///
        /// ```
        /// ❰ &foo ❱;
        /// ❰ &mut bar ❱;
        /// ❰ &raw const bar ❱;
        /// ❰ &raw mut bar ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#borrow-operators)
        struct RefExpr: AttrsOwner { T![&], T![raw], T![mut], T![const], Expr }

        /// Prefix operator call. This is either `!` or `*` or `-`.
        ///
        /// ```
        /// ❰ !foo ❱;
        /// ❰ *bar ❱;
        /// ❰ -42 ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html)
        struct PrefixExpr: AttrsOwner { /*PrefixOp,*/ Expr }

        /// Box operator call.
        ///
        /// ```
        /// ❰ box 42 ❱;
        /// ```
        ///
        /// [RFC](https://github.com/rust-lang/rfcs/blob/0806be4f282144cfcd55b1d20284b43f87cbe1c6/text/0809-box-and-in-for-stdlib.md)
        struct BoxExpr: AttrsOwner { T![box], Expr }

        /// Range operator call.
        ///
        /// ```
        /// ❰ 0..42 ❱;
        /// ❰ ..42 ❱;
        /// ❰ 0.. ❱;
        /// ❰ .. ❱;
        /// ❰ 0..=42 ❱;
        /// ❰ ..=42 ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/range-expr.html)
        struct RangeExpr: AttrsOwner { /*RangeOp*/ }


        /// Binary operator call.
        /// Includes all arithmetic, logic, bitwise and assignment operators.
        ///
        /// ```
        /// ❰ 2 + ❰ 2 * 2 ❱ ❱;
        /// ❰ ❰ true && false ❱ || true ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators)
        struct BinExpr: AttrsOwner { /*BinOp*/ }


        /// [Raw] string, [raw] byte string, char, byte, integer, float or bool literal.
        ///
        /// ```
        /// ❰ "str" ❱;
        /// ❰ br##"raw byte str"## ❱;
        /// ❰ 'c' ❱;
        /// ❰ b'c' ❱;
        /// ❰ 42 ❱;
        /// ❰ 1e9 ❱;
        /// ❰ true ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/literal-expr.html)
        struct Literal { /*LiteralToken*/ }

        /// Match expression.
        ///
        /// ```
        /// ❰
        ///     match expr {
        ///         Pat1 => {}
        ///         Pat2(_) => 42,
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html)
        struct MatchExpr: AttrsOwner { T![match], Expr, MatchArmList }

        /// Match arm list part of match expression. Includes its inner attributes.
        ///
        /// ```
        /// match expr
        /// ❰
        ///     {
        ///         #![inner_attr]
        ///         Pat1 => {}
        ///         Pat2(_) => 42,
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html)
        struct MatchArmList: AttrsOwner { T!['{'], arms: [MatchArm], T!['}'] }


        /// Match arm.
        /// Note: record struct literals are not valid as target match expression
        /// due to ambiguity.
        /// ```
        /// match expr {
        ///     ❰ #[attr] Pattern(it) if bool_cond => it ❱,
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html)
        struct MatchArm: AttrsOwner {
            pat: Pat,
            guard: MatchGuard,
            T![=>],
            Expr,
        }

        /// Match guard.
        ///
        /// ```
        /// match expr {
        ///     Pattern(it) ❰ if bool_cond ❱ => it,
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/match-expr.html#match-guards)
        struct MatchGuard { T![if], Expr }

        /// Record literal expression. The same syntax is used for structs,
        /// unions and record enum variants.
        ///
        /// ```
        /// ❰
        ///     foo::Bar {
        ///         #![inner_attr]
        ///         baz: 42,
        ///         bruh: true,
        ///         ..spread
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/struct-expr.html)
        struct RecordLit { Path, RecordFieldList}

        /// Record field list including enclosing curly braces.
        ///
        /// foo::Bar ❰
        ///     {
        ///         baz: 42,
        ///         ..spread
        ///     }
        /// ❱
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/struct-expr.html)
        struct RecordFieldList {
            T!['{'],
            fields: [RecordField],
            T![..],
            spread: Expr,
            T!['}']
        }

        /// Record field.
        ///
        /// ```
        /// foo::Bar {
        ///     ❰ #[attr] baz: 42 ❱
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/struct-expr.html)
        struct RecordField: AttrsOwner { NameRef, T![:], Expr }

        /// Disjunction of patterns.
        ///
        /// ```
        /// let ❰ Foo(it) | Bar(it) | Baz(it) ❱ = bruh;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html)
        struct OrPat { pats: [Pat] }

        /// Parenthesized pattern.
        /// Note: parens are only used for grouping, this is not a tuple pattern.
        ///
        /// ```
        /// if let ❰ &(0..=42) ❱ = foo {}
        /// ```
        ///
        /// https://doc.rust-lang.org/reference/patterns.html#grouped-patterns
        struct ParenPat { T!['('], Pat, T![')'] }

        /// Reference pattern.
        /// Note: this has nothing to do with `ref` keyword, the latter is used in bind patterns.
        ///
        /// ```
        /// let ❰ &mut foo ❱ = bar;
        ///
        /// let ❰ & ❰ &mut ❰ &_ ❱ ❱ ❱ = baz;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#reference-patterns)
        struct RefPat { T![&], T![mut], Pat }

        /// Box pattern.
        ///
        /// ```
        /// let ❰ box foo ❱ = box 42;
        /// ```
        ///
        /// [Unstable book](https://doc.rust-lang.org/unstable-book/language-features/box-patterns.html)
        struct BoxPat { T![box], Pat }

        /// Bind pattern.
        ///
        /// ```
        /// match foo {
        ///     Some(❰ ref mut bar ❱) => {}
        ///     ❰ baz @ None ❱ => {}
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#identifier-patterns)
        struct BindPat: AttrsOwner, NameOwner { T![ref], T![mut], T![@], Pat }

        /// Placeholder pattern a.k.a. the wildcard pattern or the underscore.
        ///
        /// ```
        /// let ❰ _ ❱ = foo;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#wildcard-pattern)
        struct PlaceholderPat { T![_] }

        /// Rest-of-the record/tuple pattern.
        /// Note: this is not the unbonded range pattern (even more: it doesn't exist).
        ///
        /// ```
        /// let Foo { bar, ❰ .. ❱ } = baz;
        /// let (❰ .. ❱, bruh) = (42, 24, 42);
        /// let Bruuh(❰ .. ❱) = bruuuh;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
        struct DotDotPat { T![..] }

        /// Path pattern.
        /// Doesn't include the underscore pattern (it is a special case, namely `PlaceholderPat`).
        ///
        /// ```
        /// let ❰ foo::bar::Baz ❱ { .. } = bruh;
        /// if let ❰ CONST ❱ = 42 {}
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#path-patterns)
        struct PathPat { Path }

        /// Slice pattern.
        ///
        /// ```
        /// let ❰ [foo, bar, baz] ❱ = [1, 2, 3];
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#slice-patterns)
        struct SlicePat { T!['['], args: [Pat], T![']'] }

        /// Range pattern.
        ///
        /// ```
        /// match foo {
        ///     ❰ 0..42 ❱ => {}
        ///     ❰ 0..=42 ❱ => {}
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#range-patterns)
        struct RangePat { } // FIXME(@matklad): here should be T![..], T![..=] I think, if we don't already have an accessor in expresions_ext

        /// Literal pattern.
        /// Includes only bool, number, char, and string literals.
        ///
        /// ```
        /// match foo {
        ///     Number(❰ 42 ❱) => {}
        ///     String(❰ "42" ❱) => {}
        ///     Bool(❰ true ❱) => {}
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#literal-patterns)
        struct LiteralPat { Literal }

        /// Macro invocation in pattern position.
        ///
        /// ```
        /// let ❰ foo!(my custom syntax) ❱ = baz;
        ///
        /// ```
        /// [Reference](https://doc.rust-lang.org/reference/macros.html#macro-invocation)
        struct MacroPat { MacroCall }

        /// Record literal pattern.
        ///
        /// ```
        /// let ❰ foo::Bar { baz, .. } ❱ = bruh;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
        struct RecordPat { RecordFieldPatList, Path }

        /// Record literal's field patterns list including enclosing curly braces.
        ///
        /// ```
        /// let foo::Bar ❰ { baz, bind @ bruh, .. } ❱ = bruuh;
        /// ``
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
        struct RecordFieldPatList {
            T!['{'],
            pats: [RecordInnerPat],
            record_field_pats: [RecordFieldPat],
            bind_pats: [BindPat],
            T![..],
            T!['}']
        }

        /// Record literal's field pattern.
        /// Note: record literal can also match tuple structs.
        ///
        /// ```
        /// let Foo { ❰ bar: _ ❱ } = baz;
        /// let TupleStruct { ❰ 0: _ ❱ } = bruh;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#struct-patterns)
        struct RecordFieldPat: AttrsOwner { NameRef, T![:], Pat }

        /// Tuple struct literal pattern.
        ///
        /// ```
        /// let ❰ foo::Bar(baz, bruh) ❱ = bruuh;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#tuple-struct-patterns)
        struct TupleStructPat { Path, T!['('], args: [Pat], T![')'] }

        /// Tuple pattern.
        /// Note: this doesn't include tuple structs (see `TupleStructPat`)
        ///
        /// ```
        /// let ❰ (foo, bar, .., baz) ❱ = bruh;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/patterns.html#tuple-patterns)
        struct TuplePat { T!['('], args: [Pat], T![')'] }

        /// Visibility.
        ///
        /// ```
        /// ❰ pub mod ❱ foo;
        /// ❰ pub(crate) ❱ struct Bar;
        /// ❰ pub(self) ❱ enum Baz {}
        /// ❰ pub(super) ❱ fn bruh() {}
        /// ❰ pub(in bruuh::bruuuh) ❱ type T = u64;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/visibility-and-privacy.html)
        struct Visibility { T![pub], T![super], T![self], T![crate] }

        /// Single identifier.
        /// Note(@matklad): `Name` is for things that install a new name into the scope,
        /// `NameRef` is a usage of a name. Most of the time, this definition/reference
        /// distinction can be determined purely syntactically, ie in
        /// ```
        /// fn foo() { foo() }
        /// ```
        /// the first foo is `Name`, the second one is `NameRef`.
        /// The notable exception are patterns, where in
        /// ``
        /// let x = 92
        /// ```
        /// `x` can be semantically either a name or a name ref, depeding on
        /// wether there's an `x` constant in scope.
        /// We use `Name` for patterns, and disambiguate semantically (see `NameClass` in ide_db).
        ///
        /// ```
        /// let ❰ foo ❱ = bar;
        /// struct ❰ Baz ❱;
        /// fn ❰ bruh ❱() {}
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/identifiers.html)
        struct Name { T![ident] }

        /// Reference to a name.
        /// See the explanation on the difference between `Name` and `NameRef`
        /// in `Name` ast node docs.
        ///
        /// ```
        /// let foo = ❰ bar ❱(❰ Baz(❰ bruh ❱) ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/identifiers.html)
        struct NameRef { }

        /// Macro call.
        /// Includes all of its attributes and doc comments.
        ///
        /// ```
        /// ❰
        ///     /// Docs
        ///     #[attr]
        ///     macro_rules! foo {   // macro rules is also a macro call
        ///         ($bar: tt) => {}
        ///     }
        /// ❱
        ///
        /// // semicolon is a part of `MacroCall` when it is used in item positions
        /// ❰ foo!(); ❱
        ///
        /// fn main() {
        ///     ❰ foo!() ❱; // macro call in expression positions doesn't include the semi
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/macros.html)
        struct MacroCall: NameOwner, AttrsOwner, DocCommentsOwner {
            Path, T![!], TokenTree, T![;]
        }

        /// Attribute.
        ///
        /// ```
        /// ❰ #![inner_attr] ❱
        ///
        /// ❰ #[attr] ❱
        /// ❰ #[foo = "bar"] ❱
        /// ❰ #[baz(bruh::bruuh = "42")] ❱
        /// struct Foo;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/attributes.html)
        struct Attr { T![#], T![!], T!['['], Path, T![=], input: AttrInput, T![']'] }

        /// Stores a list of lexer tokens and other `TokenTree`s.
        /// It appears in attributes, macro_rules and macro call (foo!)
        ///
        /// ```
        /// macro_call! ❰ { my syntax here } ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/macros.html)
        struct TokenTree {}

        /// Generic lifetime, type and constants parameters list **declaration**.
        ///
        /// ```
        /// fn foo❰ <'a, 'b, T, U, const BAR: u64> ❱() {}
        ///
        /// struct Baz❰ <T> ❱(T);
        ///
        /// impl❰ <T> ❱ Bruh<T> {}
        ///
        /// type Bruuh = for❰ <'a> ❱ fn(&'a str) -> &'a str;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/generics.html)
        struct TypeParamList {
            T![<],
            generic_params: [GenericParam],
            type_params: [TypeParam],
            lifetime_params: [LifetimeParam],
            const_params: [ConstParam],
            T![>]
        }

        /// Single type parameter **declaration**.
        ///
        /// ```
        /// fn foo<❰ K ❱, ❰ I ❱, ❰ E: Debug ❱, ❰ V = DefaultType ❱>() {}
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/generics.html)
        struct TypeParam: NameOwner, AttrsOwner, TypeBoundsOwner {
            T![=],
            default_type: TypeRef,
        }

        /// Const generic parameter **declaration**.
        /// ```
        /// fn foo<T, U, ❰ const BAR: usize ❱, ❰ const BAZ: bool ❱>() {}
        /// ```
        ///
        /// [RFC](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md#declaring-a-const-parameter)
        struct ConstParam: NameOwner, AttrsOwner, TypeAscriptionOwner {
            T![=],
            default_val: Expr,
        }

        /// Lifetime parameter **declaration**.
        ///
        /// ```
        /// fn foo<❰ 'a ❱, ❰ 'b ❱, V, G, D>(bar: &'a str, baz: &'b mut str) {}
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/generics.html)
        struct LifetimeParam: AttrsOwner { T![lifetime] }

        /// Type bound declaration clause.
        ///
        /// ```
        /// fn foo<T: ❰ ?Sized ❱ + ❰ Debug ❱>() {}
        ///
        /// trait Bar<T>
        /// where
        ///     T: ❰ Send ❱ + ❰ Sync ❱
        /// {
        ///     type Baz: ❰ !Sync ❱ + ❰ Debug ❱ + ❰ ?const Add ❱;
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/trait-bounds.html)
        struct TypeBound { T![lifetime], /* Question,  */ T![const], /* Question, */ TypeRef }

        /// Type bounds list.
        ///
        /// ```
        ///
        /// fn foo<T: ❰ ?Sized + Debug ❱>() {}
        ///
        /// trait Bar<T>
        /// where
        ///     T: ❰ Send + Sync ❱
        /// {
        ///     type Baz: ❰ !Sync + Debug ❱;
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/trait-bounds.html)
        struct TypeBoundList { bounds: [TypeBound] }

        /// Single where predicate.
        ///
        /// ```
        /// trait Foo<'a, 'b, T>
        /// where
        ///     ❰ 'a: 'b ❱,
        ///     ❰ T: IntoIterator ❱,
        ///     ❰ for<'c> <T as IntoIterator>::Item: Bar<'c> ❱
        /// {}
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/generics.html#where-clauses)
        struct WherePred: TypeBoundsOwner { T![for], TypeParamList, T![lifetime], TypeRef }

        /// Where clause.
        ///
        /// ```
        /// trait Foo<'a, T> ❰ where 'a: 'static, T: Debug ❱ {}
        ///
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/generics.html#where-clauses)
        struct WhereClause { T![where], predicates: [WherePred] }

        /// Abi declaration.
        /// Note: the abi string is optional.
        ///
        /// ```
        /// ❰ extern "C" ❱ {
        ///     fn foo() {}
        /// }
        ///
        /// type Bar = ❰ extern ❱ fn() -> u32;
        ///
        /// type Baz = ❰ extern r#"stdcall"# ❱ fn() -> bool;
        /// ```
        ///
        /// - [Extern blocks reference](https://doc.rust-lang.org/reference/items/external-blocks.html)
        /// - [FFI function pointers reference](https://doc.rust-lang.org/reference/items/functions.html#functions)
        struct Abi { /*String*/ }

        /// Expression statement.
        ///
        /// ```
        /// ❰ 42; ❱
        /// ❰ foo(); ❱
        /// ❰ (); ❱
        /// ❰ {}; ❱
        ///
        /// // constructions with trailing curly brace can omit the semicolon
        /// // but only when there are satements immediately after them (this is important!)
        /// ❰ if bool_cond { } ❱
        /// ❰ loop {} ❱
        /// ❰ somestatment; ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/statements.html)
        struct ExprStmt: AttrsOwner { Expr, T![;] }

        /// Let statement.
        ///
        /// ```
        /// ❰ #[attr] let foo; ❱
        /// ❰ let bar: u64; ❱
        /// ❰ let baz = 42; ❱
        /// ❰ let bruh: bool = true; ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/statements.html#let-statements)
        struct LetStmt: AttrsOwner, TypeAscriptionOwner {
            T![let],
            Pat,
            T![=],
            initializer: Expr,
            T![;],
        }

        /// Condition of `if` or `while` expression.
        ///
        /// ```
        /// if ❰ true ❱ {}
        /// if ❰ let Pat(foo) = bar ❱ {}
        ///
        /// while ❰ true ❱ {}
        /// while ❰ let Pat(baz) = bruh ❱ {}
        /// ```
        ///
        /// [If expression reference](https://doc.rust-lang.org/reference/expressions/if-expr.html)
        /// [While expression reference](https://doc.rust-lang.org/reference/expressions/loop-expr.html#predicate-loops)
        struct Condition { T![let], Pat, T![=], Expr }

        /// Parameter list **declaration**.
        ///
        /// ```
        /// fn foo❰ (a: u32, b: bool) ❱ -> u32 {}
        /// let bar = ❰ |a, b| ❱ {};
        ///
        /// impl Baz {
        ///     fn bruh❰ (&self, a: u32) ❱ {}
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/functions.html)ocs to codegen script
        struct ParamList { // FIXME: this node is used by closure expressions too, but hey use pipes instead of parens...
            T!['('],
            SelfParam,
            params: [Param],
            T![')']
        }

        /// Self parameter **declaration**.
        ///
        /// ```
        /// impl Bruh {
        ///     fn foo(❰ self ❱) {}
        ///     fn bar(❰ &self ❱) {}
        ///     fn baz(❰ &mut self ❱) {}
        ///     fn blah<'a>(❰ &'a self ❱) {}
        ///     fn blin(❰ self: Box<Self> ❱) {}
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/functions.html)
        struct SelfParam: TypeAscriptionOwner, AttrsOwner { T![&], T![mut], T![lifetime], T![self] }

        /// Parameter **declaration**.
        ///
        /// ```
        /// fn foo(❰ #[attr] Pat(bar): Pat(u32) ❱, ❰ #[attr] _: bool ❱) {}
        ///
        /// extern "C" {
        ///     fn bar(❰ baz: u32 ❱, ❰ ... ❱) -> u32;
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/functions.html)
        struct Param: TypeAscriptionOwner, AttrsOwner {
            Pat,
            T![...]
        }

        /// Use declaration.
        ///
        /// ```
        /// ❰ #[attr] pub use foo; ❱
        /// ❰ use bar as baz; ❱
        /// ❰ use bruh::{self, bruuh}; ❱
        /// ❰ use { blin::blen, blah::* };
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
        struct UseItem: AttrsOwner, VisibilityOwner {
            T![use],
            UseTree,
        }

        /// Use tree.
        ///
        /// ```
        /// pub use ❰ foo::❰ * ❱ ❱;
        /// use ❰ bar as baz ❱;
        /// use ❰ bruh::bruuh::{ ❰ self ❱, ❰ blin ❱ } ❱;
        /// use ❰ { ❰ blin::blen ❱ } ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
        struct UseTree {
            Path, T![*], UseTreeList, Alias
        }

        /// Item alias.
        /// Note: this is not the type alias.
        ///
        /// ```
        /// use foo ❰ as bar ❱;
        /// use baz::{bruh ❰ as _ ❱};
        /// extern crate bruuh ❰ as blin ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
        struct Alias: NameOwner { T![as] }

        /// Sublist of use trees.
        ///
        /// ```
        /// use bruh::bruuh::❰ { ❰ self ❱, ❰ blin ❱ } ❱;
        /// use ❰ { blin::blen::❰ {} ❱ } ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/use-declarations.html)
        struct UseTreeList { T!['{'], use_trees: [UseTree], T!['}'] }

        /// Extern crate item.
        ///
        /// ```
        /// ❰ #[attr] pub extern crate foo; ❱
        /// ❰ extern crate self as bar; ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/extern-crates.html)
        struct ExternCrateItem: AttrsOwner, VisibilityOwner {
            T![extern], T![crate], NameRef, Alias,
        }

        /// Call site arguments list.
        ///
        /// ```
        /// foo::<T, U>❰ (42, true) ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/expressions/call-expr.html)
        struct ArgList {
            T!['('],
            args: [Expr],
            T![')']
        }

        /// Path to a symbol. Includes single identifier names and elaborate paths with
        /// generic parameters.
        ///
        /// ```
        /// (0..10).❰ ❰ collect ❱ ::<Vec<_>> ❱();
        /// ❰ ❰ ❰ Vec ❱ ::<u8> ❱ ::with_capacity ❱(1024);
        /// ❰ ❰ <❰ Foo ❱ as ❰ ❰ bar ❱ ::Bar ❱> ❱ ::baz ❱();
        /// ❰ ❰ <❰ bruh ❱> ❱ ::bruuh ❱();
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/paths.html)
        struct Path {
            segment: PathSegment,
            T![::],
            qualifier: Path,
        }

        /// Segment of the path to a symbol.
        /// Only path segment of an absolute path holds the `::` token,
        /// all other `::` tokens that connect path segments reside under `Path` itself.`
        ///
        /// ```
        /// (0..10).❰ collect ❱ :: ❰ <Vec<_>> ❱();
        /// ❰ Vec ❱ :: ❰ <u8> ❱ :: ❰ with_capacity ❱(1024);
        /// ❰ <❰ Foo ❱ as ❰ bar ❱ :: ❰ Bar ❱> ❱ :: ❰ baz ❱();
        /// ❰ <❰ bruh ❱> ❱ :: ❰ bruuh ❱();
        ///
        /// // Note that only in this case `::` token is inlcuded:
        /// ❰ ::foo ❱;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/paths.html)
        struct PathSegment {
            T![::], T![crate], T![self], T![super], T![<], NameRef, TypeArgList, ParamList, RetType, PathType, T![>]
        }

        /// List of type arguments that are passed at generic instantiation site.
        ///
        /// ```
        /// type _ = Foo ❰ ::<'a, u64, Item = Bar, 42, {true}> ❱::Bar;
        ///
        /// Vec❰ ::<bool> ❱::();
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/paths.html#paths-in-expressions)
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

        /// Type argument that is passed at generic instantiation site.
        ///
        /// ```
        /// type _ = Foo::<'a, ❰ u64 ❱, ❰ bool ❱, Item = Bar, 42>::Baz;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/paths.html#paths-in-expressions)
        struct TypeArg { TypeRef }

        /// Associated type argument that is passed at generic instantiation site.
        /// ```
        /// type Foo = Bar::<'a, u64, bool, ❰ Item = Baz ❱, 42>::Bruh;
        ///
        /// trait Bruh<T>: Iterator<❰ Item: Debug ❱> {}
        /// ```
        ///
        struct AssocTypeArg : TypeBoundsOwner { NameRef, T![=], TypeRef }

        /// Lifetime argument that is passed at generic instantiation site.
        ///
        /// ```
        /// fn foo<'a>(s: &'a str) {
        ///     bar::<❰ 'a ❱>(s);
        /// }
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/paths.html#paths-in-expressions)
        struct LifetimeArg { T![lifetime] }

        /// Constant value argument that is passed at generic instantiation site.
        ///
        /// ```
        /// foo::<u32, ❰ { true } ❱>();
        ///
        /// bar::<❰ { 2 + 2} ❱>();
        /// ```
        ///
        /// [RFC](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md#declaring-a-const-parameter)
        struct ConstArg { Literal, BlockExpr }


        /// FIXME: (@edwin0cheng) Remove it to use ItemList instead
        /// https://github.com/rust-analyzer/rust-analyzer/pull/4083#discussion_r422666243
        ///
        /// [Reference](https://doc.rust-lang.org/reference/macros.html)
        struct MacroItems: ModuleItemOwner { }

        /// FIXME: (@edwin0cheng) add some documentation here. As per the writing
        /// of this comment this ast node is not used.
        ///
        /// ```
        /// // FIXME: example here
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/macros.html)
        struct MacroStmts {
            statements: [Stmt],
            Expr,
        }

        /// List of items in an extern block.
        ///
        /// ```
        /// extern "C" ❰
        ///     {
        ///         fn foo();
        ///         static var: u32;
        ///     }
        /// ❱
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/external-blocks.html)
        struct ExternItemList: ModuleItemOwner {
            T!['{'],
            extern_items: [ExternItem],
            T!['}']
        }

        /// Extern block.
        ///
        /// ```
        /// ❰
        ///     extern "C" {
        ///         fn foo();
        ///     }
        /// ❱
        ///
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/items/external-blocks.html)
        struct ExternBlock {
            Abi,
            ExternItemList
        }

        /// Meta item in an attribute.
        ///
        /// ```
        /// #[❰ bar::baz = "42" ❱]
        /// #[❰ bruh(bruuh("true")) ❱]
        /// struct Foo;
        /// ```
        ///
        /// [Reference](https://doc.rust-lang.org/reference/attributes.html?highlight=meta,item#meta-item-attribute-syntax)
        struct MetaItem {
            Path, T![=], AttrInput, nested_meta_items: [MetaItem]
        }

        /// Macro 2.0 definition.
        /// Their syntax is still WIP by rustc team...
        /// ```
        /// ❰
        ///     macro foo { }
        /// ❱
        /// ```
        ///
        /// [RFC](https://github.com/rust-lang/rfcs/blob/master/text/1584-macros.md)
        struct MacroDef {
            Name, TokenTree
        }
    },
    enums: &ast_enums! {
        /// Any kind of nominal type definition.
        enum NominalDef: NameOwner, TypeParamsOwner, AttrsOwner {
            StructDef, EnumDef, UnionDef,
        }

        /// Any kind of **declared** generic parameter
        enum GenericParam {
            LifetimeParam,
            TypeParam,
            ConstParam
        }

        /// Any kind of generic argument passed at instantiation site
        enum GenericArg {
            LifetimeArg,
            TypeArg,
            ConstArg,
            AssocTypeArg
        }

        /// Any kind of construct valid in type context
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

        /// Any kind of top-level item that may appear in a module
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



        /// Any kind of item that may appear in an impl block
        ///
        /// // FIXME: impl blocks can also contain MacroCall
        enum AssocItem: NameOwner, AttrsOwner {
            FnDef, TypeAliasDef, ConstDef
        }

        /// Any kind of item that may appear in an extern block
        ///
        /// // FIXME: extern blocks can also contain MacroCall
        enum ExternItem: NameOwner, AttrsOwner, VisibilityOwner {
            FnDef, StaticDef
        }

        /// Any kind of expression
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

        /// Any kind of pattern
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

        /// Any kind of pattern that appears directly inside of the curly
        /// braces of a record pattern
        enum RecordInnerPat {
            RecordFieldPat,
            BindPat
        }

        /// Any kind of input to an attribute
        enum AttrInput { Literal, TokenTree }

        /// Any kind of statement
        /// Note: there are no empty statements, these are just represented as
        /// bare semicolons without a dedicated statement ast node.
        enum Stmt {
            LetStmt,
            ExprStmt,
            // macro calls are parsed as expression statements
        }

        /// Any kind of fields list (record or tuple field lists)
        enum FieldDefList {
            RecordFieldDefList,
            TupleFieldDefList,
        }
    },
};
