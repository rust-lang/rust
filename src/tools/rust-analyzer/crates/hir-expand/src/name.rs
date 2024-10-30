//! See [`Name`].

use std::fmt;

use intern::{sym, Symbol};
use span::{Edition, SyntaxContextId};
use syntax::ast;
use syntax::utils::is_raw_identifier;

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygiene info, but we are
/// not there yet!
///
/// Note that the rawness (`r#`) of names does not depend on whether they are written raw.
/// This is because we want to show (in completions etc.) names as raw depending on the needs
/// of the current crate, for example if it is edition 2021 complete `gen` even if the defining
/// crate is in edition 2024 and wrote `r#gen`, and the opposite holds as well.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Name {
    symbol: Symbol,
    // If you are making this carry actual hygiene, beware that the special handling for variables and labels
    // in bodies can go.
    ctx: (),
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Name")
            .field("symbol", &self.symbol.as_str())
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl Ord for Name {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.symbol.as_str().cmp(other.symbol.as_str())
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// No need to strip `r#`, all comparisons are done against well-known symbols.
impl PartialEq<Symbol> for Name {
    fn eq(&self, sym: &Symbol) -> bool {
        self.symbol == *sym
    }
}

impl PartialEq<Name> for Symbol {
    fn eq(&self, name: &Name) -> bool {
        *self == name.symbol
    }
}

/// Wrapper of `Name` to print the name without "r#" even when it is a raw identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnescapedName<'a>(&'a Name);

impl<'a> UnescapedName<'a> {
    pub fn display(self, db: &dyn crate::db::ExpandDatabase) -> impl fmt::Display + 'a {
        _ = db;
        UnescapedDisplay { name: self }
    }
    #[doc(hidden)]
    pub fn display_no_db(self) -> impl fmt::Display + 'a {
        UnescapedDisplay { name: self }
    }
}

impl Name {
    /// Note: this is private to make creating name from random string hard.
    /// Hopefully, this should allow us to integrate hygiene cleaner in the
    /// future, and to switch to interned representation of names.
    fn new_text(text: &str) -> Name {
        Name { symbol: Symbol::intern(text), ctx: () }
    }

    pub fn new(text: &str, ctx: SyntaxContextId) -> Name {
        _ = ctx;
        Self::new_text(text)
    }

    pub fn new_tuple_field(idx: usize) -> Name {
        Name { symbol: Symbol::intern(&idx.to_string()), ctx: () }
    }

    pub fn new_lifetime(lt: &ast::Lifetime) -> Name {
        Name { symbol: Symbol::intern(lt.text().as_str()), ctx: () }
    }

    /// Resolve a name from the text of token.
    fn resolve(raw_text: &str) -> Name {
        Name::new_text(raw_text.trim_start_matches("r#"))
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
    pub fn missing() -> Name {
        Name { symbol: sym::MISSING_NAME.clone(), ctx: () }
    }

    /// Returns true if this is a fake name for things missing in the source code. See
    /// [`missing()`][Self::missing] for details.
    ///
    /// Use this method instead of comparing with `Self::missing()` as missing names
    /// (ideally should) have a `gensym` semantics.
    pub fn is_missing(&self) -> bool {
        self == &Name::missing()
    }

    /// Generates a new name that attempts to be unique. Should only be used when body lowering and
    /// creating desugared locals and labels. The caller is responsible for picking an index
    /// that is stable across re-executions
    pub fn generate_new_name(idx: usize) -> Name {
        Name::new_text(&format!("<ra@gennew>{idx}"))
    }

    /// Returns the tuple index this name represents if it is a tuple field.
    pub fn as_tuple_index(&self) -> Option<usize> {
        self.symbol.as_str().parse().ok()
    }

    /// Returns the text this name represents if it isn't a tuple field.
    pub fn as_str(&self) -> &str {
        self.symbol.as_str()
    }

    pub fn unescaped(&self) -> UnescapedName<'_> {
        UnescapedName(self)
    }

    pub fn is_escaped(&self, edition: Edition) -> bool {
        is_raw_identifier(self.symbol.as_str(), edition)
    }

    pub fn display<'a>(
        &'a self,
        db: &dyn crate::db::ExpandDatabase,
        edition: Edition,
    ) -> impl fmt::Display + 'a {
        _ = db;
        self.display_no_db(edition)
    }

    // FIXME: Remove this
    #[doc(hidden)]
    pub fn display_no_db(&self, edition: Edition) -> impl fmt::Display + '_ {
        Display { name: self, needs_escaping: is_raw_identifier(self.symbol.as_str(), edition) }
    }

    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }

    pub const fn new_symbol(symbol: Symbol, ctx: SyntaxContextId) -> Self {
        _ = ctx;
        Self { symbol, ctx: () }
    }

    // FIXME: This needs to go once we have hygiene
    pub const fn new_symbol_root(sym: Symbol) -> Self {
        Self { symbol: sym, ctx: () }
    }

    #[inline]
    pub fn eq_ident(&self, ident: &str) -> bool {
        self.as_str() == ident.trim_start_matches("r#")
    }
}

struct Display<'a> {
    name: &'a Name,
    needs_escaping: bool,
}

impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.needs_escaping {
            write!(f, "r#")?;
        }
        fmt::Display::fmt(self.name.symbol.as_str(), f)
    }
}

struct UnescapedDisplay<'a> {
    name: UnescapedName<'a>,
}

impl fmt::Display for UnescapedDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = self.name.0.symbol.as_str();
        fmt::Display::fmt(symbol, f)
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
        Name::resolve(self.sym.as_str())
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
        Name::new_text(&self.name)
    }
}
