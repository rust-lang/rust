//! To make attribute macros work reliably when typing, we need to take care to
//! fix up syntax errors in the code we're passing to them.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use span::{ErasedFileAstId, Span, SpanAnchor, FIXUP_ERASED_FILE_AST_ID_MARKER};
use stdx::never;
use syntax::{
    ast::{self, AstNode, HasLoopBody},
    match_ast, SyntaxElement, SyntaxKind, SyntaxNode, TextRange, TextSize,
};
use triomphe::Arc;
use tt::Spacing;

use crate::{
    span_map::SpanMapRef,
    tt::{Ident, Leaf, Punct, Subtree},
};

/// The result of calculating fixes for a syntax node -- a bunch of changes
/// (appending to and replacing nodes), the information that is needed to
/// reverse those changes afterwards, and a token map.
#[derive(Debug, Default)]
pub(crate) struct SyntaxFixups {
    pub(crate) append: FxHashMap<SyntaxElement, Vec<Leaf>>,
    pub(crate) remove: FxHashSet<SyntaxElement>,
    pub(crate) undo_info: SyntaxFixupUndoInfo,
}

/// This is the information needed to reverse the fixups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyntaxFixupUndoInfo {
    // FIXME: ThinArc<[Subtree]>
    original: Option<Arc<Box<[Subtree]>>>,
}

impl SyntaxFixupUndoInfo {
    pub(crate) const NONE: Self = SyntaxFixupUndoInfo { original: None };
}

// We mark spans with `FIXUP_DUMMY_AST_ID` to indicate that they are fake.
const FIXUP_DUMMY_AST_ID: ErasedFileAstId = FIXUP_ERASED_FILE_AST_ID_MARKER;
const FIXUP_DUMMY_RANGE: TextRange = TextRange::empty(TextSize::new(0));
// If the fake span has this range end, that means that the range start is an index into the
// `original` list in `SyntaxFixupUndoInfo`.
const FIXUP_DUMMY_RANGE_END: TextSize = TextSize::new(!0);

pub(crate) fn fixup_syntax(
    span_map: SpanMapRef<'_>,
    node: &SyntaxNode,
    call_site: Span,
) -> SyntaxFixups {
    let mut append = FxHashMap::<SyntaxElement, _>::default();
    let mut remove = FxHashSet::<SyntaxElement>::default();
    let mut preorder = node.preorder();
    let mut original = Vec::new();
    let dummy_range = FIXUP_DUMMY_RANGE;
    let fake_span = |range| {
        let span = span_map.span_for_range(range);
        Span {
            range: dummy_range,
            anchor: SpanAnchor { ast_id: FIXUP_DUMMY_AST_ID, ..span.anchor },
            ctx: span.ctx,
        }
    };
    while let Some(event) = preorder.next() {
        let syntax::WalkEvent::Enter(node) = event else { continue };

        let node_range = node.text_range();
        if can_handle_error(&node) && has_error_to_handle(&node) {
            remove.insert(node.clone().into());
            // the node contains an error node, we have to completely replace it by something valid
            let original_tree = mbe::syntax_node_to_token_tree(&node, span_map, call_site);
            let idx = original.len() as u32;
            original.push(original_tree);
            let span = span_map.span_for_range(node_range);
            let replacement = Leaf::Ident(Ident {
                text: "__ra_fixup".into(),
                span: Span {
                    range: TextRange::new(TextSize::new(idx), FIXUP_DUMMY_RANGE_END),
                    anchor: SpanAnchor { ast_id: FIXUP_DUMMY_AST_ID, ..span.anchor },
                    ctx: span.ctx,
                },
            });
            append.insert(node.clone().into(), vec![replacement]);
            preorder.skip_subtree();
            continue;
        }

        // In some other situations, we can fix things by just appending some tokens.
        match_ast! {
            match node {
                ast::FieldExpr(it) => {
                    if it.name_ref().is_none() {
                        // incomplete field access: some_expr.|
                        append.insert(node.clone().into(), vec![
                            Leaf::Ident(Ident {
                                text: "__ra_fixup".into(),
                                span: fake_span(node_range),
                            }),
                        ]);
                    }
                },
                ast::ExprStmt(it) => {
                    if it.semicolon_token().is_none() {
                        append.insert(node.clone().into(), vec![
                            Leaf::Punct(Punct {
                                char: ';',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range),
                            }),
                        ]);
                    }
                },
                ast::LetStmt(it) => {
                    if it.semicolon_token().is_none() {
                        append.insert(node.clone().into(), vec![
                            Leaf::Punct(Punct {
                                char: ';',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
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
                            Leaf::Ident(Ident {
                                text: "__ra_fixup".into(),
                                span: fake_span(node_range)
                            }),
                        ]);
                    }
                    if it.then_branch().is_none() {
                        append.insert(node.clone().into(), vec![
                            // FIXME: THis should be a subtree no?
                            Leaf::Punct(Punct {
                                char: '{',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                            Leaf::Punct(Punct {
                                char: '}',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
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
                            Leaf::Ident(Ident {
                                text: "__ra_fixup".into(),
                                span: fake_span(node_range)
                            }),
                        ]);
                    }
                    if it.loop_body().is_none() {
                        append.insert(node.clone().into(), vec![
                            // FIXME: THis should be a subtree no?
                            Leaf::Punct(Punct {
                                char: '{',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                            Leaf::Punct(Punct {
                                char: '}',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                        ]);
                    }
                },
                ast::LoopExpr(it) => {
                    if it.loop_body().is_none() {
                        append.insert(node.clone().into(), vec![
                            // FIXME: THis should be a subtree no?
                            Leaf::Punct(Punct {
                                char: '{',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                            Leaf::Punct(Punct {
                                char: '}',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
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
                            Leaf::Ident(Ident {
                                text: "__ra_fixup".into(),
                                span: fake_span(node_range)
                            }),
                        ]);
                    }
                    if it.match_arm_list().is_none() {
                        // No match arms
                        append.insert(node.clone().into(), vec![
                            // FIXME: THis should be a subtree no?
                            Leaf::Punct(Punct {
                                char: '{',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                            Leaf::Punct(Punct {
                                char: '}',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                        ]);
                    }
                },
                ast::ForExpr(it) => {
                    let for_token = match it.for_token() {
                        Some(token) => token,
                        None => continue
                    };

                    let [pat, in_token, iter] = [
                         "_",
                         "in",
                         "__ra_fixup"
                    ].map(|text|
                        Leaf::Ident(Ident {
                            text: text.into(),
                            span: fake_span(node_range)
                        }),
                    );

                    if it.pat().is_none() && it.in_token().is_none() && it.iterable().is_none() {
                        append.insert(for_token.into(), vec![pat, in_token, iter]);
                    // does something funky -- see test case for_no_pat
                    } else if it.pat().is_none() {
                        append.insert(for_token.into(), vec![pat]);
                    }

                    if it.loop_body().is_none() {
                        append.insert(node.clone().into(), vec![
                            // FIXME: THis should be a subtree no?
                            Leaf::Punct(Punct {
                                char: '{',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                            Leaf::Punct(Punct {
                                char: '}',
                                spacing: Spacing::Alone,
                                span: fake_span(node_range)
                            }),
                        ]);
                    }
                },
                _ => (),
            }
        }
    }
    let needs_fixups = !append.is_empty() || !original.is_empty();
    SyntaxFixups {
        append,
        remove,
        undo_info: SyntaxFixupUndoInfo {
            original: needs_fixups.then(|| Arc::new(original.into_boxed_slice())),
        },
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

pub(crate) fn reverse_fixups(tt: &mut Subtree, undo_info: &SyntaxFixupUndoInfo) {
    let Some(undo_info) = undo_info.original.as_deref() else { return };
    let undo_info = &**undo_info;
    #[allow(deprecated)]
    if never!(
        tt.delimiter.close.anchor.ast_id == FIXUP_DUMMY_AST_ID
            || tt.delimiter.open.anchor.ast_id == FIXUP_DUMMY_AST_ID
    ) {
        tt.delimiter.close = Span::DUMMY;
        tt.delimiter.open = Span::DUMMY;
    }
    reverse_fixups_(tt, undo_info);
}

fn reverse_fixups_(tt: &mut Subtree, undo_info: &[Subtree]) {
    let tts = std::mem::take(&mut tt.token_trees).into_vec();
    tt.token_trees = tts
        .into_iter()
        // delete all fake nodes
        .filter(|tt| match tt {
            tt::TokenTree::Leaf(leaf) => {
                let span = leaf.span();
                let is_real_leaf = span.anchor.ast_id != FIXUP_DUMMY_AST_ID;
                let is_replaced_node = span.range.end() == FIXUP_DUMMY_RANGE_END;
                is_real_leaf || is_replaced_node
            }
            tt::TokenTree::Subtree(_) => true,
        })
        .flat_map(|tt| match tt {
            tt::TokenTree::Subtree(mut tt) => {
                if tt.delimiter.close.anchor.ast_id == FIXUP_DUMMY_AST_ID
                    || tt.delimiter.open.anchor.ast_id == FIXUP_DUMMY_AST_ID
                {
                    // Even though fixup never creates subtrees with fixup spans, the old proc-macro server
                    // might copy them if the proc-macro asks for it, so we need to filter those out
                    // here as well.
                    return SmallVec::new_const();
                }
                reverse_fixups_(&mut tt, undo_info);
                SmallVec::from_const([tt.into()])
            }
            tt::TokenTree::Leaf(leaf) => {
                if leaf.span().anchor.ast_id == FIXUP_DUMMY_AST_ID {
                    // we have a fake node here, we need to replace it again with the original
                    let original = undo_info[u32::from(leaf.span().range.start()) as usize].clone();
                    if original.delimiter.kind == tt::DelimiterKind::Invisible {
                        SmallVec::from(original.token_trees.into_vec())
                    } else {
                        SmallVec::from_const([original.into()])
                    }
                } else {
                    // just a normal leaf
                    SmallVec::from_const([leaf.into()])
                }
            }
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use base_db::FileId;
    use expect_test::{expect, Expect};
    use syntax::TextRange;
    use triomphe::Arc;

    use crate::{
        fixup::reverse_fixups,
        span_map::{RealSpanMap, SpanMap},
        tt,
    };

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
            && a.token_trees.iter().zip(b.token_trees.iter()).all(|(a, b)| check_tt_eq(a, b))
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
        let span_map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(FileId::from_raw(0))));
        let fixups = super::fixup_syntax(
            span_map.as_ref(),
            &parsed.syntax_node(),
            span_map.span_for_range(TextRange::empty(0.into())),
        );
        let mut tt = mbe::syntax_node_to_token_tree_modified(
            &parsed.syntax_node(),
            span_map.as_ref(),
            fixups.append,
            fixups.remove,
            span_map.span_for_range(TextRange::empty(0.into())),
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

        reverse_fixups(&mut tt, &fixups.undo_info);

        // the fixed-up + reversed version should be equivalent to the original input
        // modulo token IDs and `Punct`s' spacing.
        let original_as_tt = mbe::syntax_node_to_token_tree(
            &parsed.syntax_node(),
            span_map.as_ref(),
            span_map.span_for_range(TextRange::empty(0.into())),
        );
        assert!(
            check_subtree_eq(&tt, &original_as_tt),
            "different token tree:\n{tt:?}\n\n{original_as_tt:?}"
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
fn foo () {for _ in __ra_fixup { }}
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
fn foo () {for bar in qux { }}
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
fn foo () {match __ra_fixup { }}
"#]],
        )
    }

    #[test]
    fn match_expr_no_arms() {
        check(
            r#"
fn foo() {
    match it {

    }
}
"#,
            expect![[r#"
fn foo () {match it {}}
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
fn foo () {match __ra_fixup { }}
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
    let it = a
}
"#,
            expect![[r#"
fn foo () {let it = a ;}
"#]],
        )
    }

    #[test]
    fn incomplete_field_expr_in_let() {
        check(
            r#"
fn foo() {
    let it = a.
}
"#,
            expect![[r#"
fn foo () {let it = a . __ra_fixup ;}
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
fn foo () {if a { }}
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
fn foo () {if __ra_fixup { }}
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
fn foo () {if __ra_fixup {} { }}
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
fn foo () {while __ra_fixup { }}
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
fn foo () {while foo { }}
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
fn foo () {loop { }}
"#]],
        )
    }
}
