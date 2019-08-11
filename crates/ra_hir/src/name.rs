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
    const fn new(text: SmolStr) -> Name {
        Name { text }
    }

    pub(crate) fn missing() -> Name {
        Name::new("[missing name]".into())
    }

    pub(crate) fn tuple_field_name(idx: usize) -> Name {
        Name::new(idx.to_string().into())
    }

    // Needed for Deref
    pub(crate) fn target() -> Name {
        Name::new("Target".into())
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
}

pub(crate) trait AsName {
    fn as_name(&self) -> Name;
}

impl AsName for ast::NameRef {
    fn as_name(&self) -> Name {
        let name = resolve_name(self.text());
        Name::new(name)
    }
}

impl AsName for ast::Name {
    fn as_name(&self) -> Name {
        let name = resolve_name(self.text());
        Name::new(name)
    }
}

impl AsName for ast::FieldKind {
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

pub(crate) const ISIZE: Name = Name::new(SmolStr::new_inline_from_ascii(5, b"isize"));
pub(crate) const I8: Name = Name::new(SmolStr::new_inline_from_ascii(2, b"i8"));
pub(crate) const I16: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"i16"));
pub(crate) const I32: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"i32"));
pub(crate) const I64: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"i64"));
pub(crate) const I128: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"i128"));
pub(crate) const USIZE: Name = Name::new(SmolStr::new_inline_from_ascii(5, b"usize"));
pub(crate) const U8: Name = Name::new(SmolStr::new_inline_from_ascii(2, b"u8"));
pub(crate) const U16: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"u16"));
pub(crate) const U32: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"u32"));
pub(crate) const U64: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"u64"));
pub(crate) const U128: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"u128"));
pub(crate) const F32: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"f32"));
pub(crate) const F64: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"f64"));
pub(crate) const BOOL: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"bool"));
pub(crate) const CHAR: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"char"));
pub(crate) const STR: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"str"));
pub(crate) const SELF_PARAM: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"self"));
pub(crate) const SELF_TYPE: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"Self"));
pub(crate) const MACRO_RULES: Name = Name::new(SmolStr::new_inline_from_ascii(11, b"macro_rules"));
pub(crate) const STD: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"std"));
pub(crate) const ITER: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"iter"));
pub(crate) const INTO_ITERATOR: Name =
    Name::new(SmolStr::new_inline_from_ascii(12, b"IntoIterator"));
pub(crate) const ITEM: Name = Name::new(SmolStr::new_inline_from_ascii(4, b"Item"));
pub(crate) const OPS: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"ops"));
pub(crate) const TRY: Name = Name::new(SmolStr::new_inline_from_ascii(3, b"Try"));
pub(crate) const OK: Name = Name::new(SmolStr::new_inline_from_ascii(2, b"Ok"));
pub(crate) const FUTURE_MOD: Name = Name::new(SmolStr::new_inline_from_ascii(6, b"future"));
pub(crate) const FUTURE_TYPE: Name = Name::new(SmolStr::new_inline_from_ascii(6, b"Future"));
pub(crate) const RESULT_MOD: Name = Name::new(SmolStr::new_inline_from_ascii(6, b"result"));
pub(crate) const RESULT_TYPE: Name = Name::new(SmolStr::new_inline_from_ascii(6, b"Result"));
pub(crate) const OUTPUT: Name = Name::new(SmolStr::new_inline_from_ascii(6, b"Output"));

fn resolve_name(text: &SmolStr) -> SmolStr {
    let raw_start = "r#";
    if text.as_str().starts_with(raw_start) {
        SmolStr::new(&text[raw_start.len()..])
    } else {
        text.clone()
    }
}
