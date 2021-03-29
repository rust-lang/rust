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
    BuiltinType,
    ByteLiteral,
    CharLiteral,
    Comment,
    EscapeSequence,
    FormatSpecifier,
    Keyword,
    NumericLiteral,
    Operator,
    Punctuation(HlPunct),
    StringLiteral,
    UnresolvedReference,

    // For things which don't have a specific highlight.
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum HlMod {
    /// Used to differentiate individual elements within attributes.
    Attribute = 0,
    /// Used with keywords like `if` and `break`.
    ControlFlow,
    /// `foo` in `fn foo(x: i32)` is a definition, `foo` in `foo(90 + 2)` is
    /// not.
    Definition,
    Documentation,
    Injected,
    Mutable,
    Consuming,
    Callable,
    /// Used for associated functions
    Static,
    /// Used for items in impls&traits.
    Associated,
    /// Used for intra doc links in doc injection.
    IntraDocLink,
    /// Used for trait items in impls.
    Trait,

    /// Keep this last!
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
            HlTag::Operator => "operator",
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
        HlMod::Attribute,
        HlMod::ControlFlow,
        HlMod::Definition,
        HlMod::Documentation,
        HlMod::IntraDocLink,
        HlMod::Injected,
        HlMod::Mutable,
        HlMod::Consuming,
        HlMod::Callable,
        HlMod::Static,
        HlMod::Associated,
        HlMod::Trait,
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
