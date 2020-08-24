//! Defines token tags we use for syntax highlighting.
//! A tag is not unlike a CSS class.

use std::{fmt, ops};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Highlight {
    pub tag: HighlightTag,
    pub modifiers: HighlightModifiers,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HighlightModifiers(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HighlightTag {
    Attribute,
    BoolLiteral,
    BuiltinType,
    ByteLiteral,
    CharLiteral,
    Comment,
    Constant,
    Enum,
    EnumVariant,
    EscapeSequence,
    Field,
    Function,
    Generic,
    Keyword,
    Lifetime,
    Macro,
    Module,
    NumericLiteral,
    Punctuation,
    SelfKeyword,
    SelfType,
    Static,
    StringLiteral,
    Struct,
    Trait,
    TypeAlias,
    TypeParam,
    Union,
    ValueParam,
    Local,
    UnresolvedReference,
    FormatSpecifier,
    Operator,
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
    Unsafe,
}

impl HighlightTag {
    fn as_str(self) -> &'static str {
        match self {
            HighlightTag::Attribute => "attribute",
            HighlightTag::BoolLiteral => "bool_literal",
            HighlightTag::BuiltinType => "builtin_type",
            HighlightTag::ByteLiteral => "byte_literal",
            HighlightTag::CharLiteral => "char_literal",
            HighlightTag::Comment => "comment",
            HighlightTag::Constant => "constant",
            HighlightTag::Enum => "enum",
            HighlightTag::EnumVariant => "enum_variant",
            HighlightTag::EscapeSequence => "escape_sequence",
            HighlightTag::Field => "field",
            HighlightTag::FormatSpecifier => "format_specifier",
            HighlightTag::Function => "function",
            HighlightTag::Generic => "generic",
            HighlightTag::Keyword => "keyword",
            HighlightTag::Lifetime => "lifetime",
            HighlightTag::Punctuation => "punctuation",
            HighlightTag::Macro => "macro",
            HighlightTag::Module => "module",
            HighlightTag::NumericLiteral => "numeric_literal",
            HighlightTag::Operator => "operator",
            HighlightTag::SelfKeyword => "self_keyword",
            HighlightTag::SelfType => "self_type",
            HighlightTag::Static => "static",
            HighlightTag::StringLiteral => "string_literal",
            HighlightTag::Struct => "struct",
            HighlightTag::Trait => "trait",
            HighlightTag::TypeAlias => "type_alias",
            HighlightTag::TypeParam => "type_param",
            HighlightTag::Union => "union",
            HighlightTag::ValueParam => "value_param",
            HighlightTag::Local => "variable",
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
    const ALL: &'static [HighlightModifier] = &[
        HighlightModifier::Attribute,
        HighlightModifier::ControlFlow,
        HighlightModifier::Definition,
        HighlightModifier::Documentation,
        HighlightModifier::Injected,
        HighlightModifier::Mutable,
        HighlightModifier::Consuming,
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
    pub fn iter(self) -> impl Iterator<Item = HighlightModifier> {
        HighlightModifier::ALL.iter().copied().filter(move |it| self.0 & it.mask() == it.mask())
    }
}
