use std::fmt;

use ra_syntax::{ast, SmolStr};

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygiene info, but we are
/// not there yet!
#[derive(Clone, PartialEq, Eq, Hash)]
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
    pub(crate) fn new(text: SmolStr) -> Name {
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
#[derive(Debug, PartialEq, Eq)]
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
