//! The condition expression used in `#[cfg(..)]` attributes.
//!
//! See: <https://doc.rust-lang.org/reference/conditional-compilation.html#conditional-compilation>

use std::fmt;

use intern::Symbol;

/// A simple configuration value passed in from the outside.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CfgAtom {
    /// eg. `#[cfg(test)]`
    Flag(Symbol),
    /// eg. `#[cfg(target_os = "linux")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[cfg(target_feature = "sse")]` and `#[cfg(target_feature = "sse2")]`.
    KeyValue { key: Symbol, value: Symbol },
}

impl fmt::Display for CfgAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgAtom::Flag(name) => name.fmt(f),
            CfgAtom::KeyValue { key, value } => write!(f, "{key} = {value:?}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(derive_arbitrary::Arbitrary))]
pub enum CfgExpr {
    Invalid,
    Atom(CfgAtom),
    All(Box<[CfgExpr]>),
    Any(Box<[CfgExpr]>),
    Not(Box<CfgExpr>),
}

impl From<CfgAtom> for CfgExpr {
    fn from(atom: CfgAtom) -> Self {
        CfgExpr::Atom(atom)
    }
}

impl CfgExpr {
    #[cfg(feature = "tt")]
    pub fn parse<S>(tt: &tt::Subtree<S>) -> CfgExpr {
        next_cfg_expr(&mut tt.token_trees.iter()).unwrap_or(CfgExpr::Invalid)
    }

    /// Fold the cfg by querying all basic `Atom` and `KeyValue` predicates.
    pub fn fold(&self, query: &dyn Fn(&CfgAtom) -> bool) -> Option<bool> {
        match self {
            CfgExpr::Invalid => None,
            CfgExpr::Atom(atom) => Some(query(atom)),
            CfgExpr::All(preds) => {
                preds.iter().try_fold(true, |s, pred| Some(s && pred.fold(query)?))
            }
            CfgExpr::Any(preds) => {
                preds.iter().try_fold(false, |s, pred| Some(s || pred.fold(query)?))
            }
            CfgExpr::Not(pred) => pred.fold(query).map(|s| !s),
        }
    }
}

#[cfg(feature = "tt")]
fn next_cfg_expr<S>(it: &mut std::slice::Iter<'_, tt::TokenTree<S>>) -> Option<CfgExpr> {
    use intern::sym;

    let name = match it.next() {
        None => return None,
        Some(tt::TokenTree::Leaf(tt::Leaf::Ident(ident))) => ident.sym.clone(),
        Some(_) => return Some(CfgExpr::Invalid),
    };

    // Peek
    let ret = match it.as_slice().first() {
        Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) if punct.char == '=' => {
            match it.as_slice().get(1) {
                Some(tt::TokenTree::Leaf(tt::Leaf::Literal(literal))) => {
                    it.next();
                    it.next();
                    CfgAtom::KeyValue { key: name, value: literal.symbol.clone() }.into()
                }
                _ => return Some(CfgExpr::Invalid),
            }
        }
        Some(tt::TokenTree::Subtree(subtree)) => {
            it.next();
            let mut sub_it = subtree.token_trees.iter();
            let mut subs = std::iter::from_fn(|| next_cfg_expr(&mut sub_it));
            match name {
                s if s == sym::all => CfgExpr::All(subs.collect()),
                s if s == sym::any => CfgExpr::Any(subs.collect()),
                s if s == sym::not => {
                    CfgExpr::Not(Box::new(subs.next().unwrap_or(CfgExpr::Invalid)))
                }
                _ => CfgExpr::Invalid,
            }
        }
        _ => CfgAtom::Flag(name).into(),
    };

    // Eat comma separator
    if let Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) = it.as_slice().first() {
        if punct.char == ',' {
            it.next();
        }
    }
    Some(ret)
}

#[cfg(test)]
impl arbitrary::Arbitrary<'_> for CfgAtom {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        if u.arbitrary()? {
            Ok(CfgAtom::Flag(Symbol::intern(<_>::arbitrary(u)?)))
        } else {
            Ok(CfgAtom::KeyValue {
                key: Symbol::intern(<_>::arbitrary(u)?),
                value: Symbol::intern(<_>::arbitrary(u)?),
            })
        }
    }
}
