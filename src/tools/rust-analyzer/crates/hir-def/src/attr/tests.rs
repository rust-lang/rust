//! This module contains tests for doc-expression parsing.
//! Currently, it tests `#[doc(hidden)]` and `#[doc(alias)]`.

use triomphe::Arc;

use base_db::FileId;
use hir_expand::span_map::{RealSpanMap, SpanMap};
use mbe::syntax_node_to_token_tree;
use syntax::{ast, AstNode, TextRange};

use crate::attr::{DocAtom, DocExpr};

fn assert_parse_result(input: &str, expected: DocExpr) {
    let source_file = ast::SourceFile::parse(input).ok().unwrap();
    let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
    let map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(FileId::from_raw(0))));
    let tt = syntax_node_to_token_tree(
        tt.syntax(),
        map.as_ref(),
        map.span_for_range(TextRange::empty(0.into())),
    );
    let cfg = DocExpr::parse(&tt);
    assert_eq!(cfg, expected);
}

#[test]
fn test_doc_expr_parser() {
    assert_parse_result("#![doc(hidden)]", DocAtom::Flag("hidden".into()).into());

    assert_parse_result(
        r#"#![doc(alias = "foo")]"#,
        DocAtom::KeyValue { key: "alias".into(), value: "foo".into() }.into(),
    );

    assert_parse_result(r#"#![doc(alias("foo"))]"#, DocExpr::Alias(["foo".into()].into()));
    assert_parse_result(
        r#"#![doc(alias("foo", "bar", "baz"))]"#,
        DocExpr::Alias(["foo".into(), "bar".into(), "baz".into()].into()),
    );

    assert_parse_result(
        r#"
        #[doc(alias("Bar", "Qux"))]
        struct Foo;"#,
        DocExpr::Alias(["Bar".into(), "Qux".into()].into()),
    );
}
