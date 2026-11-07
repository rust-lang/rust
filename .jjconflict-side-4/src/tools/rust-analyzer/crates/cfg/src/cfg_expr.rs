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
#[cfg_attr(test, derive(arbitrary::Arbitrary))]
pub enum CfgExpr {
    Invalid,
    Atom(CfgAtom),
    All(Box<[CfgExpr]>),
    Any(Box<[CfgExpr]>),
    Not(Box<CfgExpr>),
}

impl fmt::Display for CfgExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgExpr::Atom(atom) => atom.fmt(f),
            CfgExpr::All(exprs) => {
                write!(f, "all(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    expr.fmt(f)?;
                }
                write!(f, ")")
            }
            CfgExpr::Any(exprs) => {
                write!(f, "any(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    expr.fmt(f)?;
                }
                write!(f, ")")
            }
            CfgExpr::Not(expr) => write!(f, "not({})", expr),
            CfgExpr::Invalid => write!(f, "invalid"),
        }
    }
}

impl From<CfgAtom> for CfgExpr {
    fn from(atom: CfgAtom) -> Self {
        CfgExpr::Atom(atom)
    }
}

impl CfgExpr {
    // FIXME: Parsing from `tt` is only used in a handful of places, reconsider
    // if we should switch them to AST.
    #[cfg(feature = "tt")]
    pub fn parse(tt: &tt::TopSubtree) -> CfgExpr {
        next_cfg_expr(&mut tt.iter()).unwrap_or(CfgExpr::Invalid)
    }

    #[cfg(feature = "tt")]
    pub fn parse_from_iter(tt: &mut tt::iter::TtIter<'_>) -> CfgExpr {
        next_cfg_expr(tt).unwrap_or(CfgExpr::Invalid)
    }

    #[cfg(feature = "syntax")]
    pub fn parse_from_ast(ast: syntax::ast::CfgPredicate) -> CfgExpr {
        use intern::sym;
        use syntax::ast::{self, AstToken};

        match ast {
            ast::CfgPredicate::CfgAtom(atom) => {
                let atom = match atom.key() {
                    Some(ast::CfgAtomKey::True) => CfgAtom::Flag(sym::true_),
                    Some(ast::CfgAtomKey::False) => CfgAtom::Flag(sym::false_),
                    Some(ast::CfgAtomKey::Ident(key)) => {
                        let key = Symbol::intern(key.text());
                        match atom.string_token().and_then(ast::String::cast) {
                            Some(value) => {
                                if let Ok(value) = value.value() {
                                    CfgAtom::KeyValue { key, value: Symbol::intern(&value) }
                                } else {
                                    return CfgExpr::Invalid;
                                }
                            }
                            None => CfgAtom::Flag(key),
                        }
                    }
                    None => return CfgExpr::Invalid,
                };
                CfgExpr::Atom(atom)
            }
            ast::CfgPredicate::CfgComposite(composite) => {
                let Some(keyword) = composite.keyword() else {
                    return CfgExpr::Invalid;
                };
                match keyword.text() {
                    "all" => CfgExpr::All(
                        composite.cfg_predicates().map(CfgExpr::parse_from_ast).collect(),
                    ),
                    "any" => CfgExpr::Any(
                        composite.cfg_predicates().map(CfgExpr::parse_from_ast).collect(),
                    ),
                    "not" => {
                        let mut inner = composite.cfg_predicates();
                        let (Some(inner), None) = (inner.next(), inner.next()) else {
                            return CfgExpr::Invalid;
                        };
                        CfgExpr::Not(Box::new(CfgExpr::parse_from_ast(inner)))
                    }
                    _ => CfgExpr::Invalid,
                }
            }
        }
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
fn next_cfg_expr(it: &mut tt::iter::TtIter<'_>) -> Option<CfgExpr> {
    use intern::sym;
    use tt::iter::TtElement;

    let name = match it.next() {
        None => return None,
        Some(TtElement::Leaf(tt::Leaf::Ident(ident))) => ident.sym.clone(),
        Some(_) => return Some(CfgExpr::Invalid),
    };

    let mut it_clone = it.clone();
    let ret = match it_clone.next() {
        Some(TtElement::Leaf(tt::Leaf::Punct(punct)))
            // Don't consume on e.g. `=>`.
            if punct.char == '='
                && (punct.spacing == tt::Spacing::Alone
                    || it_clone.peek().is_none_or(|peek2| {
                        !matches!(peek2, tt::TtElement::Leaf(tt::Leaf::Punct(_)))
                    })) =>
        {
            match it_clone.next() {
                Some(tt::TtElement::Leaf(tt::Leaf::Literal(literal))) => {
                    it.next();
                    it.next();
                    CfgAtom::KeyValue { key: name, value: Symbol::intern(literal.text()) }.into()
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
