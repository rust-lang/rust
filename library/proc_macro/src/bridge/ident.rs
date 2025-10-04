use std::{cmp, hash};

#[derive(Copy, Clone)]
pub struct Ident<Span, Symbol> {
    pub sym: Symbol,
    pub is_raw: bool,
    pub span: Span,
}

impl<Span, Symbol: PartialEq> PartialEq<Self> for Ident<Span, Symbol> {
    fn eq(&self, other: &Self) -> bool {
        self.sym == other.sym && self.is_raw == other.is_raw
    }
}

impl<Span, Symbol, T> PartialEq<T> for Ident<Span, Symbol>
where
    Symbol: PartialEq<str>,
    T: AsRef<str> + ?Sized,
{
    fn eq(&self, other: &T) -> bool {
        if self.is_raw {
            if let Some(inner) = other.as_ref().strip_prefix("r#") {
                self.sym == *inner
            } else {
                false
            }
        } else {
            self.sym == *other.as_ref()
        }
    }
}

impl<Span, Symbol: Eq> Eq for Ident<Span, Symbol> {}

impl<Span, Symbol: Ord> PartialOrd for Ident<Span, Symbol> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Span, Symbol: Ord> Ord for Ident<Span, Symbol> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.sym.cmp(&other.sym).then_with(|| self.is_raw.cmp(&other.is_raw))
    }
}

impl<Span, Symbol: hash::Hash> hash::Hash for Ident<Span, Symbol> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.sym.hash(state);
        self.is_raw.hash(state);
    }
}
