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

pub mod known {
    macro_rules! known_names {
        ($($ident:ident),* $(,)?) => {
            $(
                #[allow(bad_style)]
                pub const $ident: super::Name =
                    super::Name::new_inline_ascii(stringify!($ident).as_bytes());
            )*
        };
    }

    known_names!(
        // Primitives
        isize,
        i8,
        i16,
        i32,
        i64,
        i128,
        usize,
        u8,
        u16,
        u32,
        u64,
        u128,
        f32,
        f64,
        bool,
        char,
        str,
        // Special names
        macro_rules,
        // Components of known path (value or mod name)
        std,
        iter,
        ops,
        future,
        result,
        boxed,
        // Components of known path (type name)
        IntoIterator,
        Item,
        Try,
        Ok,
        Future,
        Result,
        Output,
        Target,
        Box,
        RangeFrom,
        RangeFull,
        RangeInclusive,
        RangeToInclusive,
        RangeTo,
        Range,
        Neg,
        Not,
        Index,
        // Builtin macros
        file,
        column,
        compile_error,
        line,
        stringify,
        format_args,
        format_args_nl,
        env,
        option_env,
        // Builtin derives
        Copy,
        Clone,
        Default,
        Debug,
        Hash,
        Ord,
        PartialOrd,
        Eq,
        PartialEq,
    );

    // self/Self cannot be used as an identifier
    pub const SELF_PARAM: super::Name = super::Name::new_inline_ascii(b"self");
    pub const SELF_TYPE: super::Name = super::Name::new_inline_ascii(b"Self");

    #[macro_export]
    macro_rules! name {
        (self) => {
            $crate::name::known::SELF_PARAM
        };
        (Self) => {
            $crate::name::known::SELF_TYPE
        };
        ($ident:ident) => {
            $crate::name::known::$ident
        };
    }
}

pub use crate::name;
