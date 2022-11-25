//! To make attribute macros work reliably when typing, we need to take care to
//! fix up syntax errors in the code we're passing to them.
use std::mem;

use mbe::{SyntheticToken, SyntheticTokenId, TokenMap};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, AstNode, HasLoopBody},
    match_ast, SyntaxElement, SyntaxKind, SyntaxNode, TextRange,
};
use tt::Subtree;

/// The result of calculating fixes for a syntax node -- a bunch of changes
/// (appending to and replacing nodes), the information that is needed to
/// reverse those changes afterwards, and a token map.
#[derive(Debug)]
pub(crate) struct SyntaxFixups {
    pub(crate) append: FxHashMap<SyntaxElement, Vec<SyntheticToken>>,
    pub(crate) replace: FxHashMap<SyntaxElement, Vec<SyntheticToken>>,
    pub(crate) undo_info: SyntaxFixupUndoInfo,
    pub(crate) token_map: TokenMap,
    pub(crate) next_id: u32,
}

/// This is the information needed to reverse the fixups.
#[derive(Debug, PartialEq, Eq)]
pub struct SyntaxFixupUndoInfo {
    original: Vec<Subtree>,
}

const EMPTY_ID: SyntheticTokenId = SyntheticTokenId(!0);

pub(crate) fn fixup_syntax(node: &SyntaxNode) -> SyntaxFixups {
    let mut append = FxHashMap::<SyntaxElement, _>::default();
    let mut replace = FxHashMap::<SyntaxElement, _>::default();
    let mut preorder = node.preorder();
    let mut original = Vec::new();
    let mut token_map = TokenMap::default();
    let mut next_id = 0;
    while let Some(event) = preorder.next() {
        let node = match event {
            syntax::WalkEvent::Enter(node) => node,
            syntax::WalkEvent::Leave(_) => continue,
        };

        if can_handle_error(&node) && has_error_to_handle(&node) {
            // the node contains an error node, we have to completely replace it by something valid
            let (original_tree, new_tmap, new_next_id) =
                mbe::syntax_node_to_token_tree_with_modifications(
                    &node,
                    mem::take(&mut token_map),
                    next_id,
                    Default::default(),
                    Default::default(),
                );
            token_map = new_tmap;
            next_id = new_next_id;
            let idx = original.len() as u32;
            original.push(original_tree);
            let replacement = SyntheticToken {
                kind: SyntaxKind::IDENT,
                text: "__ra_fixup".into(),
                range: node.text_range(),
                id: SyntheticTokenId(idx),
            };
            replace.insert(node.clone().into(), vec![replacement]);
            preorder.skip_subtree();
            continue;
        }

        // In some other situations, we can fix things by just appending some tokens.
        let end_range = TextRange::empty(node.text_range().end());
        match_ast! {
            match node {
                ast::FieldExpr(it) => {
                    if it.name_ref().is_none() {
                        // incomplete field access: some_expr.|
                        append.insert(node.clone().into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::IDENT,
                                text: "__ra_fixup".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                },
                ast::ExprStmt(it) => {
                    if it.semicolon_token().is_none() {
                        append.insert(node.clone().into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::SEMICOLON,
                                text: ";".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                },
                ast::LetStmt(it) => {
                    if it.semicolon_token().is_none() {
                        append.insert(node.clone().into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::SEMICOLON,
                                text: ";".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                },
                ast::IfExpr(it) => {
                    if it.condition().is_none() {
                        // insert placeholder token after the if token
                        let if_token = match it.if_token() {
                            Some(t) => t,
                            None => continue,
                        };
                        append.insert(if_token.into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::IDENT,
                                text: "__ra_fixup".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                    if it.then_branch().is_none() {
                        append.insert(node.clone().into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::L_CURLY,
                                text: "{".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                            SyntheticToken {
                                kind: SyntaxKind::R_CURLY,
                                text: "}".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                },
                ast::WhileExpr(it) => {
                    if it.condition().is_none() {
                        // insert placeholder token after the while token
                        let while_token = match it.while_token() {
                            Some(t) => t,
                            None => continue,
                        };
                        append.insert(while_token.into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::IDENT,
                                text: "__ra_fixup".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                    if it.loop_body().is_none() {
                        append.insert(node.clone().into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::L_CURLY,
                                text: "{".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                            SyntheticToken {
                                kind: SyntaxKind::R_CURLY,
                                text: "}".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                },
                ast::LoopExpr(it) => {
                    if it.loop_body().is_none() {
                        append.insert(node.clone().into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::L_CURLY,
                                text: "{".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                            SyntheticToken {
                                kind: SyntaxKind::R_CURLY,
                                text: "}".into(),
                                range: end_range,
                                id: EMPTY_ID,
                            },
                        ]);
                    }
                },
                // FIXME: foo::
                // FIXME: for, match etc.
                _ => (),
            }
        }
    }
    SyntaxFixups {
        append,
        replace,
        token_map,
        next_id,
        undo_info: SyntaxFixupUndoInfo { original },
    }
}

fn has_error(node: &SyntaxNode) -> bool {
    node.children().any(|c| c.kind() == SyntaxKind::ERROR)
}

fn can_handle_error(node: &SyntaxNode) -> bool {
    ast::Expr::can_cast(node.kind())
}

fn has_error_to_handle(node: &SyntaxNode) -> bool {
    has_error(node) || node.children().any(|c| !can_handle_error(&c) && has_error_to_handle(&c))
}

pub(crate) fn reverse_fixups(
    tt: &mut Subtree,
    token_map: &TokenMap,
    undo_info: &SyntaxFixupUndoInfo,
) {
    tt.token_trees.retain(|tt| match tt {
        tt::TokenTree::Leaf(leaf) => {
            token_map.synthetic_token_id(leaf.id()).is_none()
                || token_map.synthetic_token_id(leaf.id()) != Some(EMPTY_ID)
        }
        tt::TokenTree::Subtree(st) => st.delimiter.map_or(true, |d| {
            token_map.synthetic_token_id(d.id).is_none()
                || token_map.synthetic_token_id(d.id) != Some(EMPTY_ID)
        }),
    });
    tt.token_trees.iter_mut().for_each(|tt| match tt {
        tt::TokenTree::Subtree(tt) => reverse_fixups(tt, token_map, undo_info),
        tt::TokenTree::Leaf(leaf) => {
            if let Some(id) = token_map.synthetic_token_id(leaf.id()) {
                let original = &undo_info.original[id.0 as usize];
                *tt = tt::TokenTree::Subtree(original.clone());
            }
        }
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
        let (mut tt, tmap, _) = mbe::syntax_node_to_token_tree_with_modifications(
            &parsed.syntax_node(),
            fixups.token_map,
            fixups.next_id,
            fixups.replace,
            fixups.append,
        );

        let mut actual = tt.to_string();
        actual.push('\n');

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

        reverse_fixups(&mut tt, &tmap, &fixups.undo_info);

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
    fn incomplete_let() {
        check(
            r#"
fn foo() {
    let x = a
}
"#,
            expect![[r#"
fn foo () {let x = a ;}
"#]],
        )
    }

    #[test]
    fn incomplete_field_expr_in_let() {
        check(
            r#"
fn foo() {
    let x = a.
}
"#,
            expect![[r#"
fn foo () {let x = a . __ra_fixup ;}
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

    #[test]
    fn extraneous_comma() {
        check(
            r#"
fn foo() {
    bar(,);
}
"#,
            expect![[r#"
fn foo () {__ra_fixup ;}
"#]],
        )
    }

    #[test]
    fn fixup_if_1() {
        check(
            r#"
fn foo() {
    if a
}
"#,
            expect![[r#"
fn foo () {if a {}}
"#]],
        )
    }

    #[test]
    fn fixup_if_2() {
        check(
            r#"
fn foo() {
    if
}
"#,
            expect![[r#"
fn foo () {if __ra_fixup {}}
"#]],
        )
    }

    #[test]
    fn fixup_if_3() {
        check(
            r#"
fn foo() {
    if {}
}
"#,
            // the {} gets parsed as the condition, I think?
            expect![[r#"
fn foo () {if {} {}}
"#]],
        )
    }

    #[test]
    fn fixup_while_1() {
        check(
            r#"
fn foo() {
    while
}
"#,
            expect![[r#"
fn foo () {while __ra_fixup {}}
"#]],
        )
    }

    #[test]
    fn fixup_while_2() {
        check(
            r#"
fn foo() {
    while foo
}
"#,
            expect![[r#"
fn foo () {while foo {}}
"#]],
        )
    }
    #[test]
    fn fixup_while_3() {
        check(
            r#"
fn foo() {
    while {}
}
"#,
            expect![[r#"
fn foo () {while __ra_fixup {}}
"#]],
        )
    }

    #[test]
    fn fixup_loop() {
        check(
            r#"
fn foo() {
    loop
}
"#,
            expect![[r#"
fn foo () {loop {}}
"#]],
        )
    }
}
