//! See [`Name`].

use std::fmt;

use intern::{Symbol, sym};
use span::{Edition, SyntaxContext};
use syntax::utils::is_raw_identifier;
use syntax::{ast, format_smolstr};

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygiene info, but we are
/// not there yet!
///
/// Note that the rawness (`r#`) of names is not preserved. Names are always stored without a `r#` prefix.
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

impl PartialEq<&Symbol> for Name {
    fn eq(&self, &sym: &&Symbol) -> bool {
        self.symbol == *sym
    }
}

impl PartialEq<Name> for Symbol {
    fn eq(&self, name: &Name) -> bool {
        *self == name.symbol
    }
}

impl PartialEq<Name> for &Symbol {
    fn eq(&self, name: &Name) -> bool {
        **self == name.symbol
    }
}

impl Name {
    fn new_text(text: &str) -> Name {
        Name { symbol: Symbol::intern(text), ctx: () }
    }

    pub fn new(text: &str, mut ctx: SyntaxContext) -> Name {
        // For comparisons etc. we remove the edition, because sometimes we search for some `Name`
        // and we don't know which edition it came from.
        // Can't do that for all `SyntaxContextId`s because it breaks Salsa.
        ctx.remove_root_edition();
        _ = ctx;
        match text.strip_prefix("r#") {
            Some(text) => Self::new_text(text),
            None => Self::new_text(text),
        }
    }

    pub fn new_root(text: &str) -> Name {
        // The edition doesn't matter for hygiene.
        Self::new(text, SyntaxContext::root(Edition::Edition2015))
    }

    pub fn new_tuple_field(idx: usize) -> Name {
        let symbol = match idx {
            0 => sym::INTEGER_0,
            1 => sym::INTEGER_1,
            2 => sym::INTEGER_2,
            3 => sym::INTEGER_3,
            4 => sym::INTEGER_4,
            5 => sym::INTEGER_5,
            6 => sym::INTEGER_6,
            7 => sym::INTEGER_7,
            8 => sym::INTEGER_8,
            9 => sym::INTEGER_9,
            10 => sym::INTEGER_10,
            11 => sym::INTEGER_11,
            12 => sym::INTEGER_12,
            13 => sym::INTEGER_13,
            14 => sym::INTEGER_14,
            15 => sym::INTEGER_15,
            _ => Symbol::intern(&idx.to_string()),
        };
        Name { symbol, ctx: () }
    }

    pub fn new_lifetime(lt: &str) -> Name {
        match lt.strip_prefix("'r#") {
            Some(lt) => Self::new_text(&format_smolstr!("'{lt}")),
            None => Self::new_text(lt),
        }
    }

    pub fn new_symbol(symbol: Symbol, ctx: SyntaxContext) -> Self {
        debug_assert!(!symbol.as_str().starts_with("r#"));
        _ = ctx;
        Self { symbol, ctx: () }
    }

    // FIXME: This needs to go once we have hygiene
    pub fn new_symbol_root(sym: Symbol) -> Self {
        Self::new_symbol(sym, SyntaxContext::root(Edition::Edition2015))
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
        Name { symbol: sym::MISSING_NAME, ctx: () }
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

    /// Whether this name needs to be escaped in the given edition via `r#`.
    pub fn needs_escape(&self, edition: Edition) -> bool {
        is_raw_identifier(self.symbol.as_str(), edition)
    }

    /// Returns the text this name represents if it isn't a tuple field.
    ///
    /// Do not use this for user-facing text, use `display` instead to handle editions properly.
    // FIXME: This should take a database argument to hide the interning
    pub fn as_str(&self) -> &str {
        self.symbol.as_str()
    }

    pub fn display<'a>(
        &'a self,
        db: &dyn crate::db::ExpandDatabase,
        edition: Edition,
    ) -> impl fmt::Display + 'a {
        _ = db;
        self.display_no_db(edition)
    }

    // FIXME: Remove this in favor of `display`, see fixme on `as_str`
    #[doc(hidden)]
    pub fn display_no_db(&self, edition: Edition) -> impl fmt::Display + '_ {
        Display { name: self, edition }
    }

    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }
}

struct Display<'a> {
    name: &'a Name,
    edition: Edition,
}

impl fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut symbol = self.name.symbol.as_str();

        if symbol == "'static" {
            // FIXME: '`static` can also be a label, and there it does need escaping.
            // But knowing where it is will require adding a parameter to `display()`,
            // and that is an infectious change.
            return f.write_str(symbol);
        }

        if let Some(s) = symbol.strip_prefix('\'') {
            f.write_str("'")?;
            symbol = s;
        }
        if is_raw_identifier(symbol, self.edition) {
            f.write_str("r#")?;
        }
        f.write_str(symbol)
    }
}

pub trait AsName {
    fn as_name(&self) -> Name;
}

impl AsName for ast::NameRef {
    fn as_name(&self) -> Name {
        match self.as_tuple_field() {
            Some(idx) => Name::new_tuple_field(idx),
            None => Name::new_root(&self.text()),
        }
    }
}

impl AsName for ast::Name {
    fn as_name(&self) -> Name {
        Name::new_root(&self.text())
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
        Name::new_root(self.sym.as_str())
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

impl AsName for base_db::BuiltDependency {
    fn as_name(&self) -> Name {
        Name::new_symbol_root((*self.name).clone())
    }
}
