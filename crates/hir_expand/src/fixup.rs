use mbe::{SyntheticToken, SyntheticTokenId, TokenMap};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, AstNode},
    match_ast, SyntaxKind, SyntaxNode, TextRange,
};
use tt::Subtree;

#[derive(Debug)]
pub struct SyntaxFixups {
    pub append: FxHashMap<SyntaxNode, Vec<SyntheticToken>>,
    pub replace: FxHashMap<SyntaxNode, Vec<SyntheticToken>>,
}

pub fn fixup_syntax(node: &SyntaxNode) -> SyntaxFixups {
    let mut append = FxHashMap::default();
    let mut replace = FxHashMap::default();
    let mut preorder = node.preorder();
    let empty_id = SyntheticTokenId(0);
    while let Some(event) = preorder.next() {
        let node = match event {
            syntax::WalkEvent::Enter(node) => node,
            syntax::WalkEvent::Leave(_) => continue,
        };
        if node.kind() == SyntaxKind::ERROR {
            // TODO this might not be helpful
            replace.insert(node, Vec::new());
            preorder.skip_subtree();
            continue;
        }
        let end_range = TextRange::empty(node.text_range().end());
        match_ast! {
            match node {
                ast::FieldExpr(it) => {
                    if it.name_ref().is_none() {
                        // incomplete field access: some_expr.|
                        append.insert(node.clone(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::IDENT,
                                text: "__ra_fixup".into(),
                                range: end_range,
                                id: empty_id,
                            },
                        ]);
                    }
                },
                ast::ExprStmt(it) => {
                    if it.semicolon_token().is_none() {
                        append.insert(node.clone(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::SEMICOLON,
                                text: ";".into(),
                                range: end_range,
                                id: empty_id,
                            },
                        ]);
                    }
                },
                _ => (),
            }
        }
    }
    SyntaxFixups { append, replace }
}

pub fn reverse_fixups(tt: &mut Subtree, token_map: &TokenMap) {
    eprintln!("token_map: {:?}", token_map);
    tt.token_trees.retain(|tt| match tt {
        tt::TokenTree::Leaf(leaf) => token_map.synthetic_token_id(leaf.id()).is_none(),
        _ => true,
    });
    tt.token_trees.iter_mut().for_each(|tt| match tt {
        tt::TokenTree::Subtree(tt) => reverse_fixups(tt, token_map),
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use super::reverse_fixups;

    #[track_caller]
    fn check(ra_fixture: &str, mut expect: Expect) {
        let parsed = syntax::SourceFile::parse(ra_fixture);
        let fixups = super::fixup_syntax(&parsed.syntax_node());
        let (mut tt, tmap) = mbe::syntax_node_to_token_tree_censored(
            &parsed.syntax_node(),
            fixups.replace,
            fixups.append,
        );

        let mut actual = tt.to_string();
        actual.push_str("\n");

        expect.indent(false);
        expect.assert_eq(&actual);

        // the fixed-up tree should be syntactically valid
        let (parse, _) = mbe::token_tree_to_syntax_node(&tt, ::mbe::TopEntryPoint::MacroItems);
        assert_eq!(
            parse.errors(),
            &[],
            "parse has syntax errors. parse tree:\n{:#?}",
            parse.syntax_node()
        );

        reverse_fixups(&mut tt, &tmap);

        // the fixed-up + reversed version should be equivalent to the original input
        // (but token IDs don't matter)
        let (original_as_tt, _) = mbe::syntax_node_to_token_tree(&parsed.syntax_node());
        assert_eq!(tt.to_string(), original_as_tt.to_string());
    }

    #[test]
    fn incomplete_field_expr_1() {
        check(
            r#"
fn foo() {
    a.
}
"#,
            expect![[r#"
fn foo () {a . __ra_fixup}
"#]],
        )
    }

    #[test]
    fn incomplete_field_expr_2() {
        check(
            r#"
fn foo() {
    a. ;
}
"#,
            expect![[r#"
fn foo () {a . __ra_fixup ;}
"#]],
        )
    }

    #[test]
    fn incomplete_field_expr_3() {
        check(
            r#"
fn foo() {
    a. ;
    bar();
}
"#,
            expect![[r#"
fn foo () {a . __ra_fixup ; bar () ;}
"#]],
        )
    }

    #[test]
    fn field_expr_before_call() {
        // another case that easily happens while typing
        check(
            r#"
fn foo() {
    a.b
    bar();
}
"#,
            expect![[r#"
fn foo () {a . b ; bar () ;}
"#]],
        )
    }
}
