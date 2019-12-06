//! FIXME: write short doc here

use std::fmt;

use ra_syntax::{ast, SmolStr};

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygiene info, but we are
/// not there yet!
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name(Repr);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Repr {
    Text(SmolStr),
    TupleField(usize),
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            Repr::Text(text) => fmt::Display::fmt(&text, f),
            Repr::TupleField(idx) => fmt::Display::fmt(&idx, f),
        }
    }
}

impl Name {
    /// Note: this is private to make creating name from random string hard.
    /// Hopefully, this should allow us to integrate hygiene cleaner in the
    /// future, and to switch to interned representation of names.
    const fn new_text(text: SmolStr) -> Name {
        Name(Repr::Text(text))
    }

    pub fn new_tuple_field(idx: usize) -> Name {
        Name(Repr::TupleField(idx))
    }

    /// Shortcut to create inline plain text name
    const fn new_inline_ascii(text: &[u8]) -> Name {
        Name::new_text(SmolStr::new_inline_from_ascii(text.len(), text))
    }

    /// Resolve a name from the text of token.
    fn resolve(raw_text: &SmolStr) -> Name {
        let raw_start = "r#";
        if raw_text.as_str().starts_with(raw_start) {
            Name::new_text(SmolStr::new(&raw_text[raw_start.len()..]))
        } else {
            Name::new_text(raw_text.clone())
        }
    }

    pub fn missing() -> Name {
        Name::new_text("[missing name]".into())
    }

    pub fn as_tuple_index(&self) -> Option<usize> {
        match self.0 {
            Repr::TupleField(idx) => Some(idx),
            _ => None,
        }
    }
}

pub trait AsName {
    fn as_name(&self) -> Name;
}

impl AsName for ast::NameRef {
    fn as_name(&self) -> Name {
        match self.as_tuple_field() {
            Some(idx) => Name::new_tuple_field(idx),
            None => Name::resolve(self.text()),
        }
    }
}

impl AsName for ast::Name {
    fn as_name(&self) -> Name {
        Name::resolve(self.text())
    }
}

impl AsName for tt::Ident {
    fn as_name(&self) -> Name {
        Name::resolve(&self.text)
    }
}

impl AsName for ast::FieldKind {
    fn as_name(&self) -> Name {
        match self {
            ast::FieldKind::Name(nr) => nr.as_name(),
            ast::FieldKind::Index(idx) => Name::new_tuple_field(idx.text().parse().unwrap()),
        }
    }
}

impl AsName for ra_db::Dependency {
    fn as_name(&self) -> Name {
        Name::new_text(self.name.clone())
    }
}

// Primitives
pub const ISIZE: Name = Name::new_inline_ascii(b"isize");
pub const I8: Name = Name::new_inline_ascii(b"i8");
pub const I16: Name = Name::new_inline_ascii(b"i16");
pub const I32: Name = Name::new_inline_ascii(b"i32");
pub const I64: Name = Name::new_inline_ascii(b"i64");
pub const I128: Name = Name::new_inline_ascii(b"i128");
pub const USIZE: Name = Name::new_inline_ascii(b"usize");
pub const U8: Name = Name::new_inline_ascii(b"u8");
pub const U16: Name = Name::new_inline_ascii(b"u16");
pub const U32: Name = Name::new_inline_ascii(b"u32");
pub const U64: Name = Name::new_inline_ascii(b"u64");
pub const U128: Name = Name::new_inline_ascii(b"u128");
pub const F32: Name = Name::new_inline_ascii(b"f32");
pub const F64: Name = Name::new_inline_ascii(b"f64");
pub const BOOL: Name = Name::new_inline_ascii(b"bool");
pub const CHAR: Name = Name::new_inline_ascii(b"char");
pub const STR: Name = Name::new_inline_ascii(b"str");

// Special names
pub const SELF_PARAM: Name = Name::new_inline_ascii(b"self");
pub const SELF_TYPE: Name = Name::new_inline_ascii(b"Self");
pub const MACRO_RULES: Name = Name::new_inline_ascii(b"macro_rules");

// Components of known path (value or mod name)
pub const STD: Name = Name::new_inline_ascii(b"std");
pub const ITER: Name = Name::new_inline_ascii(b"iter");
pub const OPS: Name = Name::new_inline_ascii(b"ops");
pub const FUTURE: Name = Name::new_inline_ascii(b"future");
pub const RESULT: Name = Name::new_inline_ascii(b"result");
pub const BOXED: Name = Name::new_inline_ascii(b"boxed");

// Components of known path (type name)
pub const INTO_ITERATOR_TYPE: Name = Name::new_inline_ascii(b"IntoIterator");
pub const ITEM_TYPE: Name = Name::new_inline_ascii(b"Item");
pub const TRY_TYPE: Name = Name::new_inline_ascii(b"Try");
pub const OK_TYPE: Name = Name::new_inline_ascii(b"Ok");
pub const FUTURE_TYPE: Name = Name::new_inline_ascii(b"Future");
pub const RESULT_TYPE: Name = Name::new_inline_ascii(b"Result");
pub const OUTPUT_TYPE: Name = Name::new_inline_ascii(b"Output");
pub const TARGET_TYPE: Name = Name::new_inline_ascii(b"Target");
pub const BOX_TYPE: Name = Name::new_inline_ascii(b"Box");
pub const RANGE_FROM_TYPE: Name = Name::new_inline_ascii(b"RangeFrom");
pub const RANGE_FULL_TYPE: Name = Name::new_inline_ascii(b"RangeFull");
pub const RANGE_INCLUSIVE_TYPE: Name = Name::new_inline_ascii(b"RangeInclusive");
pub const RANGE_TO_INCLUSIVE_TYPE: Name = Name::new_inline_ascii(b"RangeToInclusive");
pub const RANGE_TO_TYPE: Name = Name::new_inline_ascii(b"RangeTo");
pub const RANGE_TYPE: Name = Name::new_inline_ascii(b"Range");

// Builtin Macros
pub const FILE_MACRO: Name = Name::new_inline_ascii(b"file");
pub const COLUMN_MACRO: Name = Name::new_inline_ascii(b"column");
pub const COMPILE_ERROR_MACRO: Name = Name::new_inline_ascii(b"compile_error");
pub const LINE_MACRO: Name = Name::new_inline_ascii(b"line");
pub const STRINGIFY_MACRO: Name = Name::new_inline_ascii(b"stringify");
pub const FORMAT_ARGS_MACRO: Name = Name::new_inline_ascii(b"format_args");
pub const FORMAT_ARGS_NL_MACRO: Name = Name::new_inline_ascii(b"format_args_nl");

// Builtin derives
pub const COPY_TRAIT: Name = Name::new_inline_ascii(b"Copy");
pub const CLONE_TRAIT: Name = Name::new_inline_ascii(b"Clone");
pub const DEFAULT_TRAIT: Name = Name::new_inline_ascii(b"Default");
pub const DEBUG_TRAIT: Name = Name::new_inline_ascii(b"Debug");
pub const HASH_TRAIT: Name = Name::new_inline_ascii(b"Hash");
pub const ORD_TRAIT: Name = Name::new_inline_ascii(b"Ord");
pub const PARTIAL_ORD_TRAIT: Name = Name::new_inline_ascii(b"PartialOrd");
pub const EQ_TRAIT: Name = Name::new_inline_ascii(b"Eq");
pub const PARTIAL_EQ_TRAIT: Name = Name::new_inline_ascii(b"PartialEq");
