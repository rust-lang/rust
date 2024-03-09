//! The condition expression used in `#[cfg(..)]` attributes.
//!
//! See: <https://doc.rust-lang.org/reference/conditional-compilation.html#conditional-compilation>

use std::{fmt, iter::Peekable, slice::Iter as SliceIter};

use syntax::{
    ast::{self, Meta},
    NodeOrToken,
};
use tt::SmolStr;

/// A simple configuration value passed in from the outside.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum CfgAtom {
    /// eg. `#[cfg(test)]`
    Flag(SmolStr),
    /// eg. `#[cfg(target_os = "linux")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[cfg(target_feature = "sse")]` and `#[cfg(target_feature = "sse2")]`.
    KeyValue { key: SmolStr, value: SmolStr },
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
    All(Vec<CfgExpr>),
    Any(Vec<CfgExpr>),
    Not(Box<CfgExpr>),
}

impl From<CfgAtom> for CfgExpr {
    fn from(atom: CfgAtom) -> Self {
        CfgExpr::Atom(atom)
    }
}

impl CfgExpr {
    pub fn parse<S>(tt: &tt::Subtree<S>) -> CfgExpr {
        next_cfg_expr(&mut tt.token_trees.iter()).unwrap_or(CfgExpr::Invalid)
    }
    /// Parses a `cfg` attribute from the meta
    pub fn parse_from_attr_meta(meta: Meta) -> Option<CfgExpr> {
        let tt = meta.token_tree()?;
        let mut iter = tt.token_trees_and_tokens().skip(1).peekable();
        next_cfg_expr_from_syntax(&mut iter)
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
fn next_cfg_expr_from_syntax<I>(iter: &mut Peekable<I>) -> Option<CfgExpr>
where
    I: Iterator<Item = NodeOrToken<ast::TokenTree, syntax::SyntaxToken>>,
{
    let name = match iter.next() {
        None => return None,
        Some(NodeOrToken::Token(element)) => match element.kind() {
            syntax::T![ident] => SmolStr::new(element.text()),
            _ => return Some(CfgExpr::Invalid),
        },
        Some(_) => return Some(CfgExpr::Invalid),
    };
    let result = match name.as_str() {
        "all" | "any" | "not" => {
            let mut preds = Vec::new();
            let Some(NodeOrToken::Node(tree)) = iter.next() else {
                return Some(CfgExpr::Invalid);
            };
            let mut tree_iter = tree.token_trees_and_tokens().skip(1).peekable();
            while tree_iter
                .peek()
                .filter(
                    |element| matches!(element, NodeOrToken::Token(token) if (token.kind() != syntax::T![')'])),
                )
                .is_some()
            {
                let pred = next_cfg_expr_from_syntax(&mut tree_iter);
                if let Some(pred) = pred {
                    preds.push(pred);
                }
            }
            let group = match name.as_str() {
                "all" => CfgExpr::All(preds),
                "any" => CfgExpr::Any(preds),
                "not" => CfgExpr::Not(Box::new(preds.pop().unwrap_or(CfgExpr::Invalid))),
                _ => unreachable!(),
            };
            Some(group)
        }
        _ => match iter.peek() {
            Some(NodeOrToken::Token(element)) if (element.kind() == syntax::T![=]) => {
                iter.next();
                match iter.next() {
                    Some(NodeOrToken::Token(value_token))
                        if (value_token.kind() == syntax::SyntaxKind::STRING) =>
                    {
                        let value = value_token.text();
                        let value = SmolStr::new(value.trim_matches('"'));
                        Some(CfgExpr::Atom(CfgAtom::KeyValue { key: name, value }))
                    }
                    _ => None,
                }
            }
            _ => Some(CfgExpr::Atom(CfgAtom::Flag(name))),
        },
    };
    if let Some(NodeOrToken::Token(element)) = iter.peek() {
        if element.kind() == syntax::T![,] {
            iter.next();
        }
    }
    result
}

fn next_cfg_expr<S>(it: &mut SliceIter<'_, tt::TokenTree<S>>) -> Option<CfgExpr> {
    let name = match it.next() {
        None => return None,
        Some(tt::TokenTree::Leaf(tt::Leaf::Ident(ident))) => ident.text.clone(),
        Some(_) => return Some(CfgExpr::Invalid),
    };

    // Peek
    let ret = match it.as_slice().first() {
        Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) if punct.char == '=' => {
            match it.as_slice().get(1) {
                Some(tt::TokenTree::Leaf(tt::Leaf::Literal(literal))) => {
                    it.next();
                    it.next();
                    // FIXME: escape? raw string?
                    let value =
                        SmolStr::new(literal.text.trim_start_matches('"').trim_end_matches('"'));
                    CfgAtom::KeyValue { key: name, value }.into()
                }
                _ => return Some(CfgExpr::Invalid),
            }
        }
        Some(tt::TokenTree::Subtree(subtree)) => {
            it.next();
            let mut sub_it = subtree.token_trees.iter();
            let mut subs = std::iter::from_fn(|| next_cfg_expr(&mut sub_it)).collect();
            match name.as_str() {
                "all" => CfgExpr::All(subs),
                "any" => CfgExpr::Any(subs),
                "not" => CfgExpr::Not(Box::new(subs.pop().unwrap_or(CfgExpr::Invalid))),
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
            Ok(CfgAtom::Flag(String::arbitrary(u)?.into()))
        } else {
            Ok(CfgAtom::KeyValue {
                key: String::arbitrary(u)?.into(),
                value: String::arbitrary(u)?.into(),
            })
        }
    }
}
