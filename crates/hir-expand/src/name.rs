//! See [`Name`].

use std::fmt;

use syntax::{ast, utils::is_raw_identifier, SmolStr};

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygiene info, but we are
/// not there yet!
///
/// Note that `Name` holds and prints escaped name i.e. prefixed with "r#" when it
/// is a raw identifier. Use [`unescaped()`][Name::unescaped] when you need the
/// name without "r#".
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Name(Repr);

/// Wrapper of `Name` to print the name without "r#" even when it is a raw identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnescapedName<'a>(&'a Name);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Repr {
    Text(SmolStr),
    TupleField(usize),
}

impl UnescapedName<'_> {
    /// Returns the textual representation of this name as a [`SmolStr`]. Prefer using this over
    /// [`ToString::to_string`] if possible as this conversion is cheaper in the general case.
    pub fn to_smol_str(&self) -> SmolStr {
        match &self.0 .0 {
            Repr::Text(it) => {
                if let Some(stripped) = it.strip_prefix("r#") {
                    SmolStr::new(stripped)
                } else {
                    it.clone()
                }
            }
            Repr::TupleField(it) => SmolStr::new(it.to_string()),
        }
    }

    pub fn display(&self, db: &dyn crate::db::ExpandDatabase) -> impl fmt::Display + '_ {
        _ = db;
        UnescapedDisplay { name: self }
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

    pub fn new_lifetime(lt: &ast::Lifetime) -> Name {
        Self::new_text(lt.text().into())
    }

    /// Shortcut to create inline plain text name. Panics if `text.len() > 22`
    const fn new_inline(text: &str) -> Name {
        Name::new_text(SmolStr::new_inline(text))
    }

    /// Resolve a name from the text of token.
    fn resolve(raw_text: &str) -> Name {
        match raw_text.strip_prefix("r#") {
            // When `raw_text` starts with "r#" but the name does not coincide with any
            // keyword, we never need the prefix so we strip it.
            Some(text) if !is_raw_identifier(text) => Name::new_text(SmolStr::new(text)),
            // Keywords (in the current edition) *can* be used as a name in earlier editions of
            // Rust, e.g. "try" in Rust 2015. Even in such cases, we keep track of them in their
            // escaped form.
            None if is_raw_identifier(raw_text) => {
                Name::new_text(SmolStr::from_iter(["r#", raw_text]))
            }
            _ => Name::new_text(raw_text.into()),
        }
    }

    /// A fake name for things missing in the source code.
    ///
    /// For example, `impl Foo for {}` should be treated as a trait impl for a
    /// type with a missing name. Similarly, `struct S { : u32 }` should have a
    /// single field with a missing name.
    ///
    /// Ideally, we want a `gensym` semantics for missing names -- each missing
    /// name is equal only to itself. It's not clear how to implement this in
    /// salsa though, so we punt on that bit for a moment.
    pub const fn missing() -> Name {
        Name::new_inline("[missing name]")
    }

    /// Returns true if this is a fake name for things missing in the source code. See
    /// [`missing()`][Self::missing] for details.
    ///
    /// Use this method instead of comparing with `Self::missing()` as missing names
    /// (ideally should) have a `gensym` semantics.
    pub fn is_missing(&self) -> bool {
        self == &Name::missing()
    }

    /// Generates a new name which is only equal to itself, by incrementing a counter. Due
    /// its implementation, it should not be used in things that salsa considers, like
    /// type names or field names, and it should be only used in names of local variables
    /// and labels and similar things.
    pub fn generate_new_name() -> Name {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CNT: AtomicUsize = AtomicUsize::new(0);
        let c = CNT.fetch_add(1, Ordering::Relaxed);
        Name::new_text(format!("<ra@gennew>{c}").into())
    }

    /// Returns the tuple index this name represents if it is a tuple field.
    pub fn as_tuple_index(&self) -> Option<usize> {
        match self.0 {
            Repr::TupleField(idx) => Some(idx),
            _ => None,
        }
    }

    /// Returns the text this name represents if it isn't a tuple field.
    pub fn as_text(&self) -> Option<SmolStr> {
        match &self.0 {
            Repr::Text(it) => Some(it.clone()),
            _ => None,
        }
    }

    /// Returns the text this name represents if it isn't a tuple field.
    pub fn as_str(&self) -> Option<&str> {
        match &self.0 {
            Repr::Text(it) => Some(it),
            _ => None,
        }
    }

    /// Returns the textual representation of this name as a [`SmolStr`].
    /// Prefer using this over [`ToString::to_string`] if possible as this conversion is cheaper in
    /// the general case.
    pub fn to_smol_str(&self) -> SmolStr {
        match &self.0 {
            Repr::Text(it) => it.clone(),
            Repr::TupleField(it) => SmolStr::new(it.to_string()),
        }
    }

    pub fn unescaped(&self) -> UnescapedName<'_> {
        UnescapedName(self)
    }

    pub fn is_escaped(&self) -> bool {
        match &self.0 {
            Repr::Text(it) => it.starts_with("r#"),
            Repr::TupleField(_) => false,
        }
    }

    pub fn display<'a>(&'a self, db: &dyn crate::db::ExpandDatabase) -> impl fmt::Display + 'a {
        _ = db;
        Display { name: self }
    }
}

struct Display<'a> {
    name: &'a Name,
}

impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name.0 {
            Repr::Text(text) => fmt::Display::fmt(&text, f),
            Repr::TupleField(idx) => fmt::Display::fmt(&idx, f),
        }
    }
}

struct UnescapedDisplay<'a> {
    name: &'a UnescapedName<'a>,
}

impl fmt::Display for UnescapedDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name.0 .0 {
            Repr::Text(text) => {
                let text = text.strip_prefix("r#").unwrap_or(text);
                fmt::Display::fmt(&text, f)
            }
            Repr::TupleField(idx) => fmt::Display::fmt(&idx, f),
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
            None => Name::resolve(&self.text()),
        }
    }
}

impl AsName for ast::Name {
    fn as_name(&self) -> Name {
        Name::resolve(&self.text())
    }
}

impl AsName for ast::NameOrNameRef {
    fn as_name(&self) -> Name {
        match self {
            ast::NameOrNameRef::Name(it) => it.as_name(),
            ast::NameOrNameRef::NameRef(it) => it.as_name(),
        }
    }
}

impl<Span> AsName for tt::Ident<Span> {
    fn as_name(&self) -> Name {
        Name::resolve(&self.text)
    }
}

impl AsName for ast::FieldKind {
    fn as_name(&self) -> Name {
        match self {
            ast::FieldKind::Name(nr) => nr.as_name(),
            ast::FieldKind::Index(idx) => {
                let idx = idx.text().parse::<usize>().unwrap_or(0);
                Name::new_tuple_field(idx)
            }
        }
    }
}

impl AsName for base_db::Dependency {
    fn as_name(&self) -> Name {
        Name::new_text(SmolStr::new(&*self.name))
    }
}

pub mod known {
    macro_rules! known_names {
        ($($ident:ident),* $(,)?) => {
            $(
                #[allow(bad_style)]
                pub const $ident: super::Name =
                    super::Name::new_inline(stringify!($ident));
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
        doc,
        cfg,
        cfg_attr,
        register_attr,
        register_tool,
        // Components of known path (value or mod name)
        std,
        core,
        alloc,
        iter,
        ops,
        fmt,
        future,
        result,
        string,
        boxed,
        option,
        prelude,
        rust_2015,
        rust_2018,
        rust_2021,
        v1,
        // Components of known path (type name)
        Iterator,
        IntoIterator,
        Item,
        IntoIter,
        Try,
        Ok,
        Future,
        IntoFuture,
        Result,
        Option,
        Output,
        Target,
        Box,
        RangeFrom,
        RangeFull,
        RangeInclusive,
        RangeToInclusive,
        RangeTo,
        Range,
        String,
        Neg,
        Not,
        None,
        Index,
        // Components of known path (function name)
        filter_map,
        next,
        iter_mut,
        len,
        is_empty,
        as_str,
        new,
        // Builtin macros
        asm,
        assert,
        column,
        compile_error,
        concat_idents,
        concat_bytes,
        concat,
        const_format_args,
        core_panic,
        env,
        file,
        format,
        format_args_nl,
        format_args,
        global_asm,
        include_bytes,
        include_str,
        include,
        line,
        llvm_asm,
        log_syntax,
        module_path,
        option_env,
        std_panic,
        stringify,
        trace_macros,
        unreachable,
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
        // Builtin attributes
        bench,
        cfg_accessible,
        cfg_eval,
        crate_type,
        derive,
        derive_const,
        global_allocator,
        no_core,
        no_std,
        test,
        test_case,
        recursion_limit,
        feature,
        // known methods of lang items
        call_once,
        call_mut,
        call,
        eq,
        ne,
        ge,
        gt,
        le,
        lt,
        // known fields of lang items
        pieces,
        // lang items
        add_assign,
        add,
        bitand_assign,
        bitand,
        bitor_assign,
        bitor,
        bitxor_assign,
        bitxor,
        branch,
        deref_mut,
        deref,
        div_assign,
        div,
        drop,
        fn_mut,
        fn_once,
        future_trait,
        index,
        index_mut,
        into_future,
        mul_assign,
        mul,
        neg,
        not,
        owned_box,
        partial_ord,
        poll,
        r#fn,
        rem_assign,
        rem,
        shl_assign,
        shl,
        shr_assign,
        shr,
        sub_assign,
        sub,
        unsafe_cell,
        va_list
    );

    // self/Self cannot be used as an identifier
    pub const SELF_PARAM: super::Name = super::Name::new_inline("self");
    pub const SELF_TYPE: super::Name = super::Name::new_inline("Self");

    pub const STATIC_LIFETIME: super::Name = super::Name::new_inline("'static");

    #[macro_export]
    macro_rules! name {
        (self) => {
            $crate::name::known::SELF_PARAM
        };
        (Self) => {
            $crate::name::known::SELF_TYPE
        };
        ('static) => {
            $crate::name::known::STATIC_LIFETIME
        };
        ($ident:ident) => {
            $crate::name::known::$ident
        };
    }
}

pub use crate::name;
