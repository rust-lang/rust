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

impl PartialOrd for CfgAtom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CfgAtom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (CfgAtom::Flag(a), CfgAtom::Flag(b)) => a.as_str().cmp(b.as_str()),
            (CfgAtom::Flag(_), CfgAtom::KeyValue { .. }) => std::cmp::Ordering::Less,
            (CfgAtom::KeyValue { .. }, CfgAtom::Flag(_)) => std::cmp::Ordering::Greater,
            (CfgAtom::KeyValue { key, value }, CfgAtom::KeyValue { key: key2, value: value2 }) => {
                key.as_str().cmp(key2.as_str()).then(value.as_str().cmp(value2.as_str()))
            }
        }
    }
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
    pub fn parse<S: Copy>(tt: &tt::TopSubtree<S>) -> CfgExpr {
        next_cfg_expr(&mut tt.iter()).unwrap_or(CfgExpr::Invalid)
    }

    #[cfg(feature = "tt")]
    pub fn parse_from_iter<S: Copy>(tt: &mut tt::iter::TtIter<'_, S>) -> CfgExpr {
        next_cfg_expr(tt).unwrap_or(CfgExpr::Invalid)
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
fn next_cfg_expr<S: Copy>(it: &mut tt::iter::TtIter<'_, S>) -> Option<CfgExpr> {
    use intern::sym;
    use tt::iter::TtElement;

    let name = match it.next() {
        None => return None,
        Some(TtElement::Leaf(tt::Leaf::Ident(ident))) => ident.sym.clone(),
        Some(_) => return Some(CfgExpr::Invalid),
    };

    let ret = match it.peek() {
        Some(TtElement::Leaf(tt::Leaf::Punct(punct)))
            // Don't consume on e.g. `=>`.
            if punct.char == '='
                && (punct.spacing == tt::Spacing::Alone
                    || it.remaining().flat_tokens().get(1).is_none_or(|peek2| {
                        !matches!(peek2, tt::TokenTree::Leaf(tt::Leaf::Punct(_)))
                    })) =>
        {
            match it.remaining().flat_tokens().get(1) {
                Some(tt::TokenTree::Leaf(tt::Leaf::Literal(literal))) => {
                    it.next();
                    it.next();
                    CfgAtom::KeyValue { key: name, value: literal.symbol.clone() }.into()
                }
                _ => return Some(CfgExpr::Invalid),
            }
        }
        Some(TtElement::Subtree(_, mut sub_it)) => {
            it.next();
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
    if let Some(TtElement::Leaf(tt::Leaf::Punct(punct))) = it.peek()
        && punct.char == ','
    {
        it.next();
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
