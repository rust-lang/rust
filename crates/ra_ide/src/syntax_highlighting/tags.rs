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
    Struct,
    Enum,
    Union,
    Trait,
    TypeAlias,
    BuiltinType,

    Field,
    Function,
    Module,
    Constant,
    Macro,
    Variable,

    TypeSelf,
    TypeParam,
    TypeLifetime,

    LiteralByte,
    LiteralNumeric,
    LiteralChar,

    Comment,
    LiteralString,
    Attribute,

    Keyword,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum HighlightModifier {
    Mutable = 0,
    Unsafe,
    /// Used with keywords like `if` and `break`.
    Control,
}

impl HighlightTag {
    fn as_str(self) -> &'static str {
        match self {
            HighlightTag::Struct => "struct",
            HighlightTag::Enum => "enum",
            HighlightTag::Union => "union",
            HighlightTag::Trait => "trait",
            HighlightTag::TypeAlias => "type_alias",
            HighlightTag::BuiltinType => "builtin_type",

            HighlightTag::Field => "field",
            HighlightTag::Function => "function",
            HighlightTag::Module => "module",
            HighlightTag::Constant => "constant",
            HighlightTag::Macro => "macro",
            HighlightTag::Variable => "variable",
            HighlightTag::TypeSelf => "type.self",
            HighlightTag::TypeParam => "type.param",
            HighlightTag::TypeLifetime => "type.lifetime",
            HighlightTag::LiteralByte => "literal.byte",
            HighlightTag::LiteralNumeric => "literal.numeric",
            HighlightTag::LiteralChar => "literal.char",
            HighlightTag::Comment => "comment",
            HighlightTag::LiteralString => "string",
            HighlightTag::Attribute => "attribute",
            HighlightTag::Keyword => "keyword",
        }
    }
}

impl fmt::Display for HighlightTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl HighlightModifier {
    const ALL: &'static [HighlightModifier] =
        &[HighlightModifier::Mutable, HighlightModifier::Unsafe, HighlightModifier::Control];

    fn as_str(self) -> &'static str {
        match self {
            HighlightModifier::Mutable => "mutable",
            HighlightModifier::Unsafe => "unsafe",
            HighlightModifier::Control => "control",
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
