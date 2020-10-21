//! The condition expression used in `#[cfg(..)]` attributes.
//!
//! See: https://doc.rust-lang.org/reference/conditional-compilation.html#conditional-compilation

use std::slice::Iter as SliceIter;

use tt::SmolStr;

/// A simple configuration value passed in from the outside.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CfgAtom {
    /// eg. `#[cfg(test)]`
    Flag(SmolStr),
    /// eg. `#[cfg(target_os = "linux")]`
    ///
    /// Note that a key can have multiple values that are all considered "active" at the same time.
    /// For example, `#[cfg(target_feature = "sse")]` and `#[cfg(target_feature = "sse2")]`.
    KeyValue { key: SmolStr, value: SmolStr },
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn parse(tt: &tt::Subtree) -> CfgExpr {
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

fn next_cfg_expr(it: &mut SliceIter<tt::TokenTree>) -> Option<CfgExpr> {
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
mod tests {
    use super::*;

    use mbe::ast_to_token_tree;
    use syntax::ast::{self, AstNode};

    fn assert_parse_result(input: &str, expected: CfgExpr) {
        let (tt, _) = {
            let source_file = ast::SourceFile::parse(input).ok().unwrap();
            let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
            ast_to_token_tree(&tt).unwrap()
        };
        let cfg = CfgExpr::parse(&tt);
        assert_eq!(cfg, expected);
    }

    #[test]
    fn test_cfg_expr_parser() {
        assert_parse_result("#![cfg(foo)]", CfgAtom::Flag("foo".into()).into());
        assert_parse_result("#![cfg(foo,)]", CfgAtom::Flag("foo".into()).into());
        assert_parse_result(
            "#![cfg(not(foo))]",
            CfgExpr::Not(Box::new(CfgAtom::Flag("foo".into()).into())),
        );
        assert_parse_result("#![cfg(foo(bar))]", CfgExpr::Invalid);

        // Only take the first
        assert_parse_result(r#"#![cfg(foo, bar = "baz")]"#, CfgAtom::Flag("foo".into()).into());

        assert_parse_result(
            r#"#![cfg(all(foo, bar = "baz"))]"#,
            CfgExpr::All(vec![
                CfgAtom::Flag("foo".into()).into(),
                CfgAtom::KeyValue { key: "bar".into(), value: "baz".into() }.into(),
            ]),
        );

        assert_parse_result(
            r#"#![cfg(any(not(), all(), , bar = "baz",))]"#,
            CfgExpr::Any(vec![
                CfgExpr::Not(Box::new(CfgExpr::Invalid)),
                CfgExpr::All(vec![]),
                CfgExpr::Invalid,
                CfgAtom::KeyValue { key: "bar".into(), value: "baz".into() }.into(),
            ]),
        );
    }
}
