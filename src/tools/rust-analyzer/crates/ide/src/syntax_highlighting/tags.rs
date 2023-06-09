//! Defines token tags we use for syntax highlighting.
//! A tag is not unlike a CSS class.

use std::{
    fmt::{self, Write},
    ops,
};

use ide_db::SymbolKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Highlight {
    pub tag: HlTag,
    pub mods: HlMods,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HlMods(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HlTag {
    Symbol(SymbolKind),

    AttributeBracket,
    BoolLiteral,
    BuiltinType,
    ByteLiteral,
    CharLiteral,
    Comment,
    EscapeSequence,
    FormatSpecifier,
    Keyword,
    NumericLiteral,
    Operator(HlOperator),
    Punctuation(HlPunct),
    StringLiteral,
    UnresolvedReference,

    // For things which don't have a specific highlight.
    None,
}

// Don't forget to adjust the feature description in crates/ide/src/syntax_highlighting.rs.
// And make sure to use the lsp strings used when converting to the protocol in crates\rust-analyzer\src\semantic_tokens.rs, not the names of the variants here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum HlMod {
    /// Used for items in traits and impls.
    Associated = 0,
    /// Used with keywords like `async` and `await`.
    Async,
    /// Used to differentiate individual elements within attribute calls.
    Attribute,
    /// Callable item or value.
    Callable,
    /// Value that is being consumed in a function call
    Consuming,
    /// Used with keywords like `if` and `break`.
    ControlFlow,
    /// Used for crate names, like `serde`.
    CrateRoot,
    /// Used for items from built-in crates (std, core, alloc, test and proc_macro).
    DefaultLibrary,
    /// `foo` in `fn foo(x: i32)` is a definition, `foo` in `foo(90 + 2)` is
    /// not.
    Definition,
    /// Doc-strings like this one.
    Documentation,
    /// Highlighting injection like rust code in doc strings or ra_fixture.
    Injected,
    /// Used for intra doc links in doc injection.
    IntraDocLink,
    /// Used for items from other crates.
    Library,
    /// Used to differentiate individual elements within macro calls.
    Macro,
    /// Mutable binding.
    Mutable,
    /// Used for public items.
    Public,
    /// Immutable reference.
    Reference,
    /// Used for associated functions.
    Static,
    /// Used for items in traits and trait impls.
    Trait,
    // Keep this last!
    /// Used for unsafe functions, unsafe traits, mutable statics, union accesses and unsafe operations.
    Unsafe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HlPunct {
    /// []
    Bracket,
    /// {}
    Brace,
    /// ()
    Parenthesis,
    /// <>
    Angle,
    /// ,
    Comma,
    /// .
    Dot,
    /// :
    Colon,
    /// ;
    Semi,
    /// ! (only for macro calls)
    MacroBang,
    ///
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HlOperator {
    /// |, &, !, ^, |=, &=, ^=
    Bitwise,
    /// +, -, *, /, +=, -=, *=, /=
    Arithmetic,
    /// &&, ||, !
    Logical,
    /// >, <, ==, >=, <=, !=
    Comparison,
    ///
    Other,
}

impl HlTag {
    fn as_str(self) -> &'static str {
        match self {
            HlTag::Symbol(symbol) => match symbol {
                SymbolKind::Attribute => "attribute",
                SymbolKind::BuiltinAttr => "builtin_attr",
                SymbolKind::Const => "constant",
                SymbolKind::ConstParam => "const_param",
                SymbolKind::Derive => "derive",
                SymbolKind::DeriveHelper => "derive_helper",
                SymbolKind::Enum => "enum",
                SymbolKind::Field => "field",
                SymbolKind::Function => "function",
                SymbolKind::Impl => "self_type",
                SymbolKind::Label => "label",
                SymbolKind::LifetimeParam => "lifetime",
                SymbolKind::Local => "variable",
                SymbolKind::Macro => "macro",
                SymbolKind::Module => "module",
                SymbolKind::SelfParam => "self_keyword",
                SymbolKind::SelfType => "self_type_keyword",
                SymbolKind::Static => "static",
                SymbolKind::Struct => "struct",
                SymbolKind::ToolModule => "tool_module",
                SymbolKind::Trait => "trait",
                SymbolKind::TraitAlias => "trait_alias",
                SymbolKind::TypeAlias => "type_alias",
                SymbolKind::TypeParam => "type_param",
                SymbolKind::Union => "union",
                SymbolKind::ValueParam => "value_param",
                SymbolKind::Variant => "enum_variant",
            },
            HlTag::AttributeBracket => "attribute_bracket",
            HlTag::BoolLiteral => "bool_literal",
            HlTag::BuiltinType => "builtin_type",
            HlTag::ByteLiteral => "byte_literal",
            HlTag::CharLiteral => "char_literal",
            HlTag::Comment => "comment",
            HlTag::EscapeSequence => "escape_sequence",
            HlTag::FormatSpecifier => "format_specifier",
            HlTag::Keyword => "keyword",
            HlTag::Punctuation(punct) => match punct {
                HlPunct::Bracket => "bracket",
                HlPunct::Brace => "brace",
                HlPunct::Parenthesis => "parenthesis",
                HlPunct::Angle => "angle",
                HlPunct::Comma => "comma",
                HlPunct::Dot => "dot",
                HlPunct::Colon => "colon",
                HlPunct::Semi => "semicolon",
                HlPunct::MacroBang => "macro_bang",
                HlPunct::Other => "punctuation",
            },
            HlTag::NumericLiteral => "numeric_literal",
            HlTag::Operator(op) => match op {
                HlOperator::Bitwise => "bitwise",
                HlOperator::Arithmetic => "arithmetic",
                HlOperator::Logical => "logical",
                HlOperator::Comparison => "comparison",
                HlOperator::Other => "operator",
            },
            HlTag::StringLiteral => "string_literal",
            HlTag::UnresolvedReference => "unresolved_reference",
            HlTag::None => "none",
        }
    }
}

impl fmt::Display for HlTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl HlMod {
    const ALL: &'static [HlMod; HlMod::Unsafe as usize + 1] = &[
        HlMod::Associated,
        HlMod::Async,
        HlMod::Attribute,
        HlMod::Callable,
        HlMod::Consuming,
        HlMod::ControlFlow,
        HlMod::CrateRoot,
        HlMod::DefaultLibrary,
        HlMod::Definition,
        HlMod::Documentation,
        HlMod::Injected,
        HlMod::IntraDocLink,
        HlMod::Library,
        HlMod::Macro,
        HlMod::Mutable,
        HlMod::Public,
        HlMod::Reference,
        HlMod::Static,
        HlMod::Trait,
        HlMod::Unsafe,
    ];

    fn as_str(self) -> &'static str {
        match self {
            HlMod::Associated => "associated",
            HlMod::Async => "async",
            HlMod::Attribute => "attribute",
            HlMod::Callable => "callable",
            HlMod::Consuming => "consuming",
            HlMod::ControlFlow => "control",
            HlMod::CrateRoot => "crate_root",
            HlMod::DefaultLibrary => "default_library",
            HlMod::Definition => "declaration",
            HlMod::Documentation => "documentation",
            HlMod::Injected => "injected",
            HlMod::IntraDocLink => "intra_doc_link",
            HlMod::Library => "library",
            HlMod::Macro => "macro",
            HlMod::Mutable => "mutable",
            HlMod::Public => "public",
            HlMod::Reference => "reference",
            HlMod::Static => "static",
            HlMod::Trait => "trait",
            HlMod::Unsafe => "unsafe",
        }
    }

    fn mask(self) -> u32 {
        1 << (self as u32)
    }
}

impl fmt::Display for HlMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl fmt::Display for Highlight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.tag.fmt(f)?;
        for modifier in self.mods.iter() {
            f.write_char('.')?;
            modifier.fmt(f)?;
        }
        Ok(())
    }
}

impl From<HlTag> for Highlight {
    fn from(tag: HlTag) -> Highlight {
        Highlight::new(tag)
    }
}

impl From<HlOperator> for Highlight {
    fn from(op: HlOperator) -> Highlight {
        Highlight::new(HlTag::Operator(op))
    }
}

impl From<HlPunct> for Highlight {
    fn from(punct: HlPunct) -> Highlight {
        Highlight::new(HlTag::Punctuation(punct))
    }
}

impl From<SymbolKind> for Highlight {
    fn from(sym: SymbolKind) -> Highlight {
        Highlight::new(HlTag::Symbol(sym))
    }
}

impl Highlight {
    pub(crate) fn new(tag: HlTag) -> Highlight {
        Highlight { tag, mods: HlMods::default() }
    }
    pub fn is_empty(&self) -> bool {
        self.tag == HlTag::None && self.mods.is_empty()
    }
}

impl ops::BitOr<HlMod> for HlTag {
    type Output = Highlight;

    fn bitor(self, rhs: HlMod) -> Highlight {
        Highlight::new(self) | rhs
    }
}

impl ops::BitOrAssign<HlMod> for HlMods {
    fn bitor_assign(&mut self, rhs: HlMod) {
        self.0 |= rhs.mask();
    }
}

impl ops::BitOrAssign<HlMod> for Highlight {
    fn bitor_assign(&mut self, rhs: HlMod) {
        self.mods |= rhs;
    }
}

impl ops::BitOr<HlMod> for Highlight {
    type Output = Highlight;

    fn bitor(mut self, rhs: HlMod) -> Highlight {
        self |= rhs;
        self
    }
}

impl HlMods {
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, m: HlMod) -> bool {
        self.0 & m.mask() == m.mask()
    }

    pub fn iter(self) -> impl Iterator<Item = HlMod> {
        HlMod::ALL.iter().copied().filter(move |it| self.0 & it.mask() == it.mask())
    }
}
