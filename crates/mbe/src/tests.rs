use std::iter;

use syntax::{ast, AstNode};

use super::*;

#[test]
fn test_node_to_tt_censor() {
    use syntax::ast::{HasAttrs, HasModuleItem};

    let source = r##"
#[attr0]
#[attr1]
#[attr2]
struct Struct {
    field: ()
}
"##;
    let source_file = ast::SourceFile::parse(source).ok().unwrap();
    let item = source_file.items().next().unwrap();
    let attr = item.attrs().nth(1).unwrap();

    let (tt, _) = syntax_node_to_token_tree_censored(
        item.syntax(),
        &iter::once(attr.syntax().clone()).collect(),
    );
    expect_test::expect![[r##"# [attr0] # [attr2] struct Struct {field : ()}"##]]
        .assert_eq(&tt.to_string());

    let source = r##"
#[attr0]
#[derive(Derive0)]
#[attr1]
#[derive(Derive1)]
#[attr2]
#[derive(Derive2)]
#[attr3]
struct Struct {
    field: ()
}
"##;
    let source_file = ast::SourceFile::parse(source).ok().unwrap();
    let item = source_file.items().next().unwrap();
    let derive_attr_index = 3;
    let censor = item
        .attrs()
        .take(derive_attr_index as usize + 1)
        .filter(|attr| attr.simple_name().as_deref() == Some("derive"))
        .map(|it| it.syntax().clone())
        .collect();

    let (tt, _) = syntax_node_to_token_tree_censored(item.syntax(), &censor);
    expect_test::expect![[r##"# [attr0] # [attr1] # [attr2] # [derive (Derive2)] # [attr3] struct Struct {field : ()}"##]]
        .assert_eq(&tt.to_string());
}
