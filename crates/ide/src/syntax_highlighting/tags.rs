//! Defines token tags we use for syntax highlighting.
//! A tag is not unlike a CSS class.

use std::{fmt, ops};

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

    Attribute,
    BoolLiteral,
    BuiltinAttr,
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
    /// Used to differentiate individual elements within attributes.
    Attribute,
    /// Callable item or value.
    Callable,
    /// Value that is being consumed in a function call
    Consuming,
    /// Used with keywords like `if` and `break`.
    ControlFlow,
    /// `foo` in `fn foo(x: i32)` is a definition, `foo` in `foo(90 + 2)` is
    /// not.
    Definition,
    /// Doc-strings like this one.
    Documentation,
    /// Highlighting injection like rust code in doc strings or ra_fixture.
    Injected,
    /// Used for intra doc links in doc injection.
    IntraDocLink,
    /// Mutable binding.
    Mutable,
    /// Used for associated functions.
    Static,
    /// Used for items in traits and trait impls.
    Trait,
    /// Used with keywords like `async` and `await`.
    Async,
    /// Used for items from other crates.
    Library,
    /// Used for public items.
    Public,
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
                SymbolKind::Const => "constant",
                SymbolKind::Static => "static",
                SymbolKind::Enum => "enum",
                SymbolKind::Variant => "enum_variant",
                SymbolKind::Struct => "struct",
                SymbolKind::Union => "union",
                SymbolKind::Field => "field",
                SymbolKind::Module => "module",
                SymbolKind::Trait => "trait",
                SymbolKind::Function => "function",
                SymbolKind::TypeAlias => "type_alias",
                SymbolKind::TypeParam => "type_param",
                SymbolKind::ConstParam => "const_param",
                SymbolKind::LifetimeParam => "lifetime",
                SymbolKind::Macro => "macro",
                SymbolKind::Local => "variable",
                SymbolKind::Label => "label",
                SymbolKind::ValueParam => "value_param",
                SymbolKind::SelfParam => "self_keyword",
                SymbolKind::Impl => "self_type",
            },
            HlTag::Attribute => "attribute",
            HlTag::BoolLiteral => "bool_literal",
            HlTag::BuiltinAttr => "builtin_attr",
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
    const ALL: &'static [HlMod; HlMod::Unsafe as u8 as usize + 1] = &[
        HlMod::Associated,
        HlMod::Attribute,
        HlMod::Callable,
        HlMod::Consuming,
        HlMod::ControlFlow,
        HlMod::Definition,
        HlMod::Documentation,
        HlMod::Injected,
        HlMod::IntraDocLink,
        HlMod::Mutable,
        HlMod::Static,
        HlMod::Trait,
        HlMod::Async,
        HlMod::Library,
        HlMod::Public,
        HlMod::Unsafe,
    ];

    fn as_str(self) -> &'static str {
        match self {
            HlMod::Associated => "associated",
            HlMod::Attribute => "attribute",
            HlMod::Callable => "callable",
            HlMod::Consuming => "consuming",
            HlMod::ControlFlow => "control",
            HlMod::Definition => "declaration",
            HlMod::Documentation => "documentation",
            HlMod::Injected => "injected",
            HlMod::IntraDocLink => "intra_doc_link",
            HlMod::Mutable => "mutable",
            HlMod::Static => "static",
            HlMod::Trait => "trait",
            HlMod::Async => "async",
            HlMod::Library => "library",
            HlMod::Public => "public",
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
        write!(f, "{}", self.tag)?;
        for modifier in self.mods.iter() {
            write!(f, ".{}", modifier)?
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
        self.tag == HlTag::None && self.mods == HlMods::default()
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
    pub fn contains(self, m: HlMod) -> bool {
        self.0 & m.mask() == m.mask()
    }

    pub fn iter(self) -> impl Iterator<Item = HlMod> {
        HlMod::ALL.iter().copied().filter(move |it| self.0 & it.mask() == it.mask())
    }
}
