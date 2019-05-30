use std::fmt;

use ra_syntax::{ast, SmolStr};

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygiene info, but we are
/// not there yet!
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name {
    text: SmolStr,
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.text, f)
    }
}

impl Name {
    /// Note: this is private to make creating name from random string hard.
    /// Hopefully, this should allow us to integrate hygiene cleaner in the
    /// future, and to switch to interned representation of names.
    fn new(text: SmolStr) -> Name {
        Name { text }
    }

    pub(crate) fn missing() -> Name {
        Name::new("[missing name]".into())
    }

    pub(crate) fn self_param() -> Name {
        Name::new("self".into())
    }

    pub(crate) fn self_type() -> Name {
        Name::new("Self".into())
    }

    pub(crate) fn tuple_field_name(idx: usize) -> Name {
        Name::new(idx.to_string().into())
    }

    // There's should be no way to extract a string out of `Name`: `Name` in the
    // future, `Name` will include hygiene information, and you can't encode
    // hygiene into a String.
    //
    // If you need to compare something with `Name`, compare `Name`s directly.
    //
    // If you need to render `Name` for the user, use the `Display` impl, but be
    // aware that it strips hygiene info.
    #[deprecated(note = "use to_string instead")]
    pub fn as_smolstr(&self) -> &SmolStr {
        &self.text
    }

    pub(crate) fn as_known_name(&self) -> Option<KnownName> {
        let name = match self.text.as_str() {
            "isize" => KnownName::Isize,
            "i8" => KnownName::I8,
            "i16" => KnownName::I16,
            "i32" => KnownName::I32,
            "i64" => KnownName::I64,
            "i128" => KnownName::I128,
            "usize" => KnownName::Usize,
            "u8" => KnownName::U8,
            "u16" => KnownName::U16,
            "u32" => KnownName::U32,
            "u64" => KnownName::U64,
            "u128" => KnownName::U128,
            "f32" => KnownName::F32,
            "f64" => KnownName::F64,
            "bool" => KnownName::Bool,
            "char" => KnownName::Char,
            "str" => KnownName::Str,
            "Self" => KnownName::SelfType,
            "self" => KnownName::SelfParam,
            "macro_rules" => KnownName::MacroRules,
            _ => return None,
        };
        Some(name)
    }
}

pub(crate) trait AsName {
    fn as_name(&self) -> Name;
}

impl AsName for ast::NameRef {
    fn as_name(&self) -> Name {
        Name::new(self.text().clone())
    }
}

impl AsName for ast::Name {
    fn as_name(&self) -> Name {
        Name::new(self.text().clone())
    }
}

impl<'a> AsName for ast::FieldKind<'a> {
    fn as_name(&self) -> Name {
        match self {
            ast::FieldKind::Name(nr) => nr.as_name(),
            ast::FieldKind::Index(idx) => Name::new(idx.text().clone()),
        }
    }
}

impl AsName for ra_db::Dependency {
    fn as_name(&self) -> Name {
        Name::new(self.name.clone())
    }
}

// Ideally, should be replaced with
// ```
// const ISIZE: Name = Name::new("isize")
// ```
// but const-fn is not that powerful yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KnownName {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,

    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,

    F32,
    F64,

    Bool,
    Char,
    Str,

    SelfType,
    SelfParam,

    MacroRules,
}

impl AsName for KnownName {
    fn as_name(&self) -> Name {
        let s = match self {
            KnownName::Isize => "isize",
            KnownName::I8 => "i8",
            KnownName::I16 => "i16",
            KnownName::I32 => "i32",
            KnownName::I64 => "i64",
            KnownName::I128 => "i128",
            KnownName::Usize => "usize",
            KnownName::U8 => "u8",
            KnownName::U16 => "u16",
            KnownName::U32 => "u32",
            KnownName::U64 => "u64",
            KnownName::U128 => "u128",
            KnownName::F32 => "f32",
            KnownName::F64 => "f64",
            KnownName::Bool => "bool",
            KnownName::Char => "char",
            KnownName::Str => "str",
            KnownName::SelfType => "Self",
            KnownName::SelfParam => "self",
            KnownName::MacroRules => "macro_rules",
        };
        Name::new(s.into())
    }
}
