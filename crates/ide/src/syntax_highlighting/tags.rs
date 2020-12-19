//! Defines token tags we use for syntax highlighting.
//! A tag is not unlike a CSS class.

use std::{fmt, ops};

use crate::SymbolKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Highlight {
    pub tag: HighlightTag,
    pub modifiers: HighlightModifiers,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HighlightModifiers(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HighlightTag {
    Symbol(SymbolKind),

    BoolLiteral,
    BuiltinType,
    ByteLiteral,
    CharLiteral,
    NumericLiteral,
    StringLiteral,
    Attribute,
    Comment,
    EscapeSequence,
    FormatSpecifier,
    Keyword,
    Punctuation,
    Operator,
    UnresolvedReference,

    // For things which don't have proper Tag, but want to use modifiers.
    Dummy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum HighlightModifier {
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

    /// Keep this last!
    Unsafe,
}

impl HighlightTag {
    fn as_str(self) -> &'static str {
        match self {
            HighlightTag::Symbol(symbol) => match symbol {
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
                SymbolKind::LifetimeParam => "lifetime",
                SymbolKind::Macro => "macro",
                SymbolKind::Local => "variable",
                SymbolKind::ValueParam => "value_param",
                SymbolKind::SelfParam => "self_keyword",
                SymbolKind::Impl => "self_type",
            },
            HighlightTag::Attribute => "attribute",
            HighlightTag::BoolLiteral => "bool_literal",
            HighlightTag::BuiltinType => "builtin_type",
            HighlightTag::ByteLiteral => "byte_literal",
            HighlightTag::CharLiteral => "char_literal",
            HighlightTag::Comment => "comment",
            HighlightTag::EscapeSequence => "escape_sequence",
            HighlightTag::FormatSpecifier => "format_specifier",
            HighlightTag::Dummy => "dummy",
            HighlightTag::Keyword => "keyword",
            HighlightTag::Punctuation => "punctuation",
            HighlightTag::NumericLiteral => "numeric_literal",
            HighlightTag::Operator => "operator",
            HighlightTag::StringLiteral => "string_literal",
            HighlightTag::UnresolvedReference => "unresolved_reference",
        }
    }
}

impl fmt::Display for HighlightTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl HighlightModifier {
    const ALL: &'static [HighlightModifier; HighlightModifier::Unsafe as u8 as usize + 1] = &[
        HighlightModifier::Attribute,
        HighlightModifier::ControlFlow,
        HighlightModifier::Definition,
        HighlightModifier::Documentation,
        HighlightModifier::Injected,
        HighlightModifier::Mutable,
        HighlightModifier::Consuming,
        HighlightModifier::Callable,
        HighlightModifier::Static,
        HighlightModifier::Associated,
        HighlightModifier::Unsafe,
    ];

    fn as_str(self) -> &'static str {
        match self {
            HighlightModifier::Attribute => "attribute",
            HighlightModifier::ControlFlow => "control",
            HighlightModifier::Definition => "declaration",
            HighlightModifier::Documentation => "documentation",
            HighlightModifier::Injected => "injected",
            HighlightModifier::Mutable => "mutable",
            HighlightModifier::Consuming => "consuming",
            HighlightModifier::Unsafe => "unsafe",
            HighlightModifier::Callable => "callable",
            HighlightModifier::Static => "static",
            HighlightModifier::Associated => "associated",
        }
    }

    fn mask(self) -> u32 {
        1 << (self as u32)
    }
}

impl fmt::Display for HighlightModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl fmt::Display for Highlight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag)?;
        for modifier in self.modifiers.iter() {
            write!(f, ".{}", modifier)?
        }
        Ok(())
    }
}

impl From<HighlightTag> for Highlight {
    fn from(tag: HighlightTag) -> Highlight {
        Highlight::new(tag)
    }
}

impl Highlight {
    pub(crate) fn new(tag: HighlightTag) -> Highlight {
        Highlight { tag, modifiers: HighlightModifiers::default() }
    }
}

impl ops::BitOr<HighlightModifier> for HighlightTag {
    type Output = Highlight;

    fn bitor(self, rhs: HighlightModifier) -> Highlight {
        Highlight::new(self) | rhs
    }
}

impl ops::BitOrAssign<HighlightModifier> for HighlightModifiers {
    fn bitor_assign(&mut self, rhs: HighlightModifier) {
        self.0 |= rhs.mask();
    }
}

impl ops::BitOrAssign<HighlightModifier> for Highlight {
    fn bitor_assign(&mut self, rhs: HighlightModifier) {
        self.modifiers |= rhs;
    }
}

impl ops::BitOr<HighlightModifier> for Highlight {
    type Output = Highlight;

    fn bitor(mut self, rhs: HighlightModifier) -> Highlight {
        self |= rhs;
        self
    }
}

impl HighlightModifiers {
    pub fn contains(self, m: HighlightModifier) -> bool {
        self.0 & m.mask() == m.mask()
    }

    pub fn iter(self) -> impl Iterator<Item = HighlightModifier> {
        HighlightModifier::ALL.iter().copied().filter(move |it| self.0 & it.mask() == it.mask())
    }
}
