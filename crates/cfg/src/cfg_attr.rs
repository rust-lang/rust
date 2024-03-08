use std::{
    fmt::{self, Debug},
    slice::Iter as SliceIter,
};

use crate::{cfg_expr::next_cfg_expr, CfgAtom, CfgExpr};
use tt::{Delimiter, SmolStr, Span};
/// Represents a `#[cfg_attr(.., my_attr)]` attribute.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CfgAttr<S> {
    /// Expression in `cfg_attr` attribute.
    pub cfg_expr: CfgExpr,
    /// Inner attribute.
    pub attr: tt::Subtree<S>,
}

impl<S: Clone + Span + Debug> CfgAttr<S> {
    /// Parses a sub tree in the form of (cfg_expr, inner_attribute)
    pub fn parse(tt: &tt::Subtree<S>) -> Option<CfgAttr<S>> {
        let mut iter = tt.token_trees.iter();
        let cfg_expr = next_cfg_expr(&mut iter).unwrap_or(CfgExpr::Invalid);
        // FIXME: This is probably not the right way to do this
        // Get's the span of the next token tree
        let first_span = iter.as_slice().first().map(|tt| tt.first_span())?;
        let attr = tt::Subtree {
            delimiter: Delimiter::invisible_spanned(first_span),
            token_trees: iter.cloned().collect(),
        };
        Some(CfgAttr { cfg_expr, attr: attr })
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use mbe::{syntax_node_to_token_tree, DummyTestSpanMap, DUMMY};
    use syntax::{ast, AstNode};

    use crate::{CfgAttr, DnfExpr};

    fn check_dnf(input: &str, expected_dnf: Expect, expected_attrs: Expect) {
        let source_file = ast::SourceFile::parse(input).ok().unwrap();
        let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        let tt = syntax_node_to_token_tree(tt.syntax(), DummyTestSpanMap, DUMMY);
        let Some(CfgAttr { cfg_expr, attr }) = CfgAttr::parse(&tt) else {
            assert!(false, "failed to parse cfg_attr");
            return;
        };

        let actual = format!("#![cfg({})]", DnfExpr::new(cfg_expr));
        expected_dnf.assert_eq(&actual);
        let actual_attrs = format!("#![{}]", attr);
        expected_attrs.assert_eq(&actual_attrs);
    }

    #[test]
    fn smoke() {
        check_dnf(
            r#"#![cfg_attr(feature = "nightly", feature(slice_split_at_unchecked))]"#,
            expect![[r#"#![cfg(feature = "nightly")]"#]],
            expect![r#"#![feature (slice_split_at_unchecked)]"#],
        );

        check_dnf(
            r#"#![cfg_attr(not(feature = "std"), no_std)]"#,
            expect![[r#"#![cfg(not(feature = "std"))]"#]],
            expect![r#"#![no_std]"#],
        );
    }
}
