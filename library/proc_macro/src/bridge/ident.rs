use std::{fmt, hash};

#[derive(Copy, Clone)]
pub struct Ident<Span, Symbol> {
    pub sym: Symbol,
    pub is_raw: bool,
    pub span: Span,
}

impl<Span, Symbol: fmt::Display, T> PartialEq<T> for Ident<Span, Symbol>
where
    Symbol: PartialEq<str>,
    T: AsRef<str> + ?Sized,
{
    fn eq(&self, other: &T) -> bool {
        self.to_string() == other.as_ref()
    }
}

impl<Span, Symbol: hash::Hash> hash::Hash for Ident<Span, Symbol> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.sym.hash(state);
        self.is_raw.hash(state);
    }
}

/// Prints the identifier as a string that should be losslessly convertible back
/// into the same identifier.
#[stable(feature = "proc_macro_lib2", since = "1.29.0")]
impl<Span, Symbol: fmt::Display> fmt::Display for Ident<Span, Symbol> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_raw {
            f.write_str("r#")?;
        }
        fmt::Display::fmt(&self.sym, f)
    }
}
