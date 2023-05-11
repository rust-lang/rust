//! To make attribute macros work reliably when typing, we need to take care to
//! fix up syntax errors in the code we're passing to them.
use std::mem;

use mbe::{SyntheticToken, SyntheticTokenId, TokenMap};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use syntax::{
    ast::{self, AstNode, HasLoopBody},
    match_ast, SyntaxElement, SyntaxKind, SyntaxNode, TextRange,
};
use tt::token_id::Subtree;

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
                ast::MatchExpr(it) => {
                    if it.expr().is_none() {
                        let match_token = match it.match_token() {
                            Some(t) => t,
                            None => continue
                        };
                        append.insert(match_token.into(), vec![
                            SyntheticToken {
                                kind: SyntaxKind::IDENT,
                                text: "__ra_fixup".into(),
                                range: end_range,
                                id: EMPTY_ID
                            },
                        ]);
                    }
                    if it.match_arm_list().is_none() {
                        // No match arms
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
                ast::ForExpr(it) => {
                    let for_token = match it.for_token() {
                        Some(token) => token,
                        None => continue
                    };

                    let [pat, in_token, iter] = [
                        (SyntaxKind::UNDERSCORE, "_"),
                        (SyntaxKind::IN_KW, "in"),
                        (SyntaxKind::IDENT, "__ra_fixup")
                    ].map(|(kind, text)| SyntheticToken { kind, text: text.into(), range: end_range, id: EMPTY_ID});

                    if it.pat().is_none() && it.in_token().is_none() && it.iterable().is_none() {
                        append.insert(for_token.into(), vec![pat, in_token, iter]);
                    // does something funky -- see test case for_no_pat
                    } else if it.pat().is_none() {
                        append.insert(for_token.into(), vec![pat]);
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
    let tts = std::mem::take(&mut tt.token_trees);
    tt.token_trees = tts
        .into_iter()
        .filter(|tt| match tt {
            tt::TokenTree::Leaf(leaf) => {
                token_map.synthetic_token_id(*leaf.span()) != Some(EMPTY_ID)
            }
            tt::TokenTree::Subtree(st) => {
                token_map.synthetic_token_id(st.delimiter.open) != Some(EMPTY_ID)
            }
        })
        .flat_map(|tt| match tt {
            tt::TokenTree::Subtree(mut tt) => {
                reverse_fixups(&mut tt, token_map, undo_info);
                SmallVec::from_const([tt.into()])
            }
            tt::TokenTree::Leaf(leaf) => {
                if let Some(id) = token_map.synthetic_token_id(*leaf.span()) {
                    let original = undo_info.original[id.0 as usize].clone();
                    if original.delimiter.kind == tt::DelimiterKind::Invisible {
                        original.token_trees.into()
                    } else {
                        SmallVec::from_const([original.into()])
                    }
                } else {
                    SmallVec::from_const([leaf.into()])
                }
            }
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tt;

    use super::reverse_fixups;

    // The following three functions are only meant to check partial structural equivalence of
    // `TokenTree`s, see the last assertion in `check()`.
    fn check_leaf_eq(a: &tt::Leaf, b: &tt::Leaf) -> bool {
        match (a, b) {
            (tt::Leaf::Literal(a), tt::Leaf::Literal(b)) => a.text == b.text,
            (tt::Leaf::Punct(a), tt::Leaf::Punct(b)) => a.char == b.char,
            (tt::Leaf::Ident(a), tt::Leaf::Ident(b)) => a.text == b.text,
            _ => false,
        }
    }

    fn check_subtree_eq(a: &tt::Subtree, b: &tt::Subtree) -> bool {
        a.delimiter.kind == b.delimiter.kind
            && a.token_trees.len() == b.token_trees.len()
            && a.token_trees.iter().zip(&b.token_trees).all(|(a, b)| check_tt_eq(a, b))
    }

    fn check_tt_eq(a: &tt::TokenTree, b: &tt::TokenTree) -> bool {
        match (a, b) {
            (tt::TokenTree::Leaf(a), tt::TokenTree::Leaf(b)) => check_leaf_eq(a, b),
            (tt::TokenTree::Subtree(a), tt::TokenTree::Subtree(b)) => check_subtree_eq(a, b),
            _ => false,
        }
    }

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

        let actual = format!("{tt}\n");

        expect.indent(false);
        expect.assert_eq(&actual);

        // the fixed-up tree should be syntactically valid
        let (parse, _) = mbe::token_tree_to_syntax_node(&tt, ::mbe::TopEntryPoint::MacroItems);
        assert!(
            parse.errors().is_empty(),
            "parse has syntax errors. parse tree:\n{:#?}",
            parse.syntax_node()
        );

        reverse_fixups(&mut tt, &tmap, &fixups.undo_info);

        // the fixed-up + reversed version should be equivalent to the original input
        // modulo token IDs and `Punct`s' spacing.
        let (original_as_tt, _) = mbe::syntax_node_to_token_tree(&parsed.syntax_node());
        assert!(
            check_subtree_eq(&tt, &original_as_tt),
            "different token tree: {tt:?},\n{original_as_tt:?}"
        );
    }

    #[test]
    fn just_for_token() {
        check(
            r#"
fn foo() {
    for
}
"#,
            expect![[r#"
fn foo () {for _ in __ra_fixup {}}
"#]],
        )
    }

    #[test]
    fn for_no_iter_pattern() {
        check(
            r#"
fn foo() {
    for {}
}
"#,
            expect![[r#"
fn foo () {for _ in __ra_fixup {}}
"#]],
        )
    }

    #[test]
    fn for_no_body() {
        check(
            r#"
fn foo() {
    for bar in qux
}
"#,
            expect![[r#"
fn foo () {for bar in qux {}}
"#]],
        )
    }

    // FIXME: https://github.com/rust-lang/rust-analyzer/pull/12937#discussion_r937633695
    #[test]
    fn for_no_pat() {
        check(
            r#"
fn foo() {
    for in qux {

    }
}
"#,
            expect![[r#"
fn foo () {__ra_fixup}
"#]],
        )
    }

    #[test]
    fn match_no_expr_no_arms() {
        check(
            r#"
fn foo() {
    match
}
"#,
            expect![[r#"
fn foo () {match __ra_fixup {}}
"#]],
        )
    }

    #[test]
    fn match_expr_no_arms() {
        check(
            r#"
fn foo() {
    match x {

    }
}
"#,
            expect![[r#"
fn foo () {match x {}}
"#]],
        )
    }

    #[test]
    fn match_no_expr() {
        check(
            r#"
fn foo() {
    match {
        _ => {}
    }
}
"#,
            expect![[r#"
fn foo () {match __ra_fixup {}}
"#]],
        )
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
    a.;
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
    a.;
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
            expect![[r#"
fn foo () {if __ra_fixup {} {}}
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
