//! This module contains tests for doc-expression parsing.
//! Currently, it tests `#[doc(hidden)]` and `#[doc(alias)]`.

use mbe::syntax_node_to_token_tree;
use syntax::{ast, AstNode};

use crate::attr::{DocAtom, DocExpr};

fn assert_parse_result(input: &str, expected: DocExpr) {
    let (tt, _) = {
        let source_file = ast::SourceFile::parse(input).ok().unwrap();
        let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        syntax_node_to_token_tree(tt.syntax())
    };
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
