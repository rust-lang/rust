//! To make attribute macros work reliably when typing, we need to take care to
//! fix up syntax errors in the code we're passing to them.

use intern::sym;
use rustc_hash::{FxHashMap, FxHashSet};
use span::{
    ErasedFileAstId, FIXUP_ERASED_FILE_AST_ID_MARKER, ROOT_ERASED_FILE_AST_ID, Span, SpanAnchor,
    SyntaxContext,
};
use stdx::never;
use syntax::{
    SyntaxElement, SyntaxKind, SyntaxNode, TextRange, TextSize,
    ast::{self, AstNode, HasLoopBody},
    match_ast,
};
use syntax_bridge::DocCommentDesugarMode;
use triomphe::Arc;
use tt::Spacing;

use crate::{
    span_map::SpanMapRef,
    tt::{self, Ident, Leaf, Punct, TopSubtree},
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
    original: Option<Arc<Box<[TopSubtree]>>>,
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
    mode: DocCommentDesugarMode,
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
            let original_tree =
                syntax_bridge::syntax_node_to_token_tree(&node, span_map, call_site, mode);
            let idx = original.len() as u32;
            original.push(original_tree);
            let span = span_map.span_for_range(node_range);
            let replacement = Leaf::Ident(Ident {
                sym: sym::__ra_fixup,
                span: Span {
                    range: TextRange::new(TextSize::new(idx), FIXUP_DUMMY_RANGE_END),
                    anchor: SpanAnchor { ast_id: FIXUP_DUMMY_AST_ID, ..span.anchor },
                    ctx: span.ctx,
                },
                is_raw: tt::IdentIsRaw::No,
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
                                sym: sym::__ra_fixup,
                                span: fake_span(node_range),
                                is_raw: tt::IdentIsRaw::No
                            }),
                        ]);
                    }
                },
                ast::ExprStmt(it) => {
                    let needs_semi = it.semicolon_token().is_none() && it.expr().is_some_and(|e| e.syntax().kind() != SyntaxKind::BLOCK_EXPR);
                    if needs_semi {
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
                                sym: sym::__ra_fixup,
                                span: fake_span(node_range),
                                is_raw: tt::IdentIsRaw::No
                            }),
                        ]);
                    }
                    if it.then_branch().is_none() {
                        append.insert(node.clone().into(), vec![
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
                                sym: sym::__ra_fixup,
                                span: fake_span(node_range),
                                is_raw: tt::IdentIsRaw::No
                            }),
                        ]);
                    }
                    if it.loop_body().is_none() {
                        append.insert(node.clone().into(), vec![
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
                                sym: sym::__ra_fixup,
                                span: fake_span(node_range),
                                is_raw: tt::IdentIsRaw::No
                            }),
                        ]);
                    }
                    if it.match_arm_list().is_none() {
                        // No match arms
                        append.insert(node.clone().into(), vec![
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
                         sym::underscore,
                         sym::in_,
                         sym::__ra_fixup,
                    ].map(|sym|
                        Leaf::Ident(Ident {
                            sym,
                            span: fake_span(node_range),
                            is_raw: tt::IdentIsRaw::No
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
                ast::RecordExprField(it) => {
                    if let Some(colon) = it.colon_token()
                        && it.name_ref().is_some() && it.expr().is_none() {
                            append.insert(colon.into(), vec![
                                Leaf::Ident(Ident {
                                    sym: sym::__ra_fixup,
                                    span: fake_span(node_range),
                                    is_raw: tt::IdentIsRaw::No
                                })
                            ]);
                        }
                },
                ast::Path(it) => {
                    if let Some(colon) = it.coloncolon_token()
                        && it.segment().is_none() {
                            append.insert(colon.into(), vec![
                                Leaf::Ident(Ident {
                                    sym: sym::__ra_fixup,
                                    span: fake_span(node_range),
                                    is_raw: tt::IdentIsRaw::No
                                })
                            ]);
                        }
                },
                ast::ClosureExpr(it) => {
                    if it.body().is_none() {
                        append.insert(node.into(), vec![
                            Leaf::Ident(Ident {
                                sym: sym::__ra_fixup,
                                span: fake_span(node_range),
                                is_raw: tt::IdentIsRaw::No
                            })
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

pub(crate) fn reverse_fixups(tt: &mut TopSubtree, undo_info: &SyntaxFixupUndoInfo) {
    let Some(undo_info) = undo_info.original.as_deref() else { return };
    let undo_info = &**undo_info;
    let delimiter = tt.top_subtree_delimiter_mut();
    #[allow(deprecated)]
    if never!(
        delimiter.close.anchor.ast_id == FIXUP_DUMMY_AST_ID
            || delimiter.open.anchor.ast_id == FIXUP_DUMMY_AST_ID
    ) {
        let span = |file_id| Span {
            range: TextRange::empty(TextSize::new(0)),
            anchor: SpanAnchor { file_id, ast_id: ROOT_ERASED_FILE_AST_ID },
            ctx: SyntaxContext::root(span::Edition::Edition2015),
        };
        delimiter.open = span(delimiter.open.anchor.file_id);
        delimiter.close = span(delimiter.close.anchor.file_id);
    }
    reverse_fixups_(tt, undo_info);
}

#[derive(Debug)]
enum TransformTtAction<'a> {
    Keep,
    ReplaceWith(tt::TokenTreesView<'a>),
}

impl TransformTtAction<'_> {
    fn remove() -> Self {
        Self::ReplaceWith(tt::TokenTreesView::new(&[]))
    }
}

/// This function takes a token tree, and calls `callback` with each token tree in it.
/// Then it does what the callback says: keeps the tt or replaces it with a (possibly empty)
/// tts view.
fn transform_tt<'a, 'b>(
    tt: &'a mut Vec<tt::TokenTree>,
    mut callback: impl FnMut(&mut tt::TokenTree) -> TransformTtAction<'b>,
) {
    // We need to keep a stack of the currently open subtrees, because we need to update
    // them if we change the number of items in them.
    let mut subtrees_stack = Vec::new();
    let mut i = 0;
    while i < tt.len() {
        'pop_finished_subtrees: while let Some(&subtree_idx) = subtrees_stack.last() {
            let tt::TokenTree::Subtree(subtree) = &tt[subtree_idx] else {
                unreachable!("non-subtree on subtrees stack");
            };
            if i >= subtree_idx + 1 + subtree.usize_len() {
                subtrees_stack.pop();
            } else {
                break 'pop_finished_subtrees;
            }
        }

        let action = callback(&mut tt[i]);
        match action {
            TransformTtAction::Keep => {
                // This cannot be shared with the replaced case, because then we may push the same subtree
                // twice, and will update it twice which will lead to errors.
                if let tt::TokenTree::Subtree(_) = &tt[i] {
                    subtrees_stack.push(i);
                }

                i += 1;
            }
            TransformTtAction::ReplaceWith(replacement) => {
                let old_len = 1 + match &tt[i] {
                    tt::TokenTree::Leaf(_) => 0,
                    tt::TokenTree::Subtree(subtree) => subtree.usize_len(),
                };
                let len_diff = replacement.len() as i64 - old_len as i64;
                tt.splice(i..i + old_len, replacement.flat_tokens().iter().cloned());
                // Skip the newly inserted replacement, we don't want to visit it.
                i += replacement.len();

                for &subtree_idx in &subtrees_stack {
                    let tt::TokenTree::Subtree(subtree) = &mut tt[subtree_idx] else {
                        unreachable!("non-subtree on subtrees stack");
                    };
                    subtree.len = (i64::from(subtree.len) + len_diff).try_into().unwrap();
                }
            }
        }
    }
}

fn reverse_fixups_(tt: &mut TopSubtree, undo_info: &[TopSubtree]) {
    let mut tts = std::mem::take(&mut tt.0).into_vec();
    transform_tt(&mut tts, |tt| match tt {
        tt::TokenTree::Leaf(leaf) => {
            let span = leaf.span();
            let is_real_leaf = span.anchor.ast_id != FIXUP_DUMMY_AST_ID;
            let is_replaced_node = span.range.end() == FIXUP_DUMMY_RANGE_END;
            if !is_real_leaf && !is_replaced_node {
                return TransformTtAction::remove();
            }

            if !is_real_leaf {
                // we have a fake node here, we need to replace it again with the original
                let original = &undo_info[u32::from(leaf.span().range.start()) as usize];
                TransformTtAction::ReplaceWith(original.view().strip_invisible())
            } else {
                // just a normal leaf
                TransformTtAction::Keep
            }
        }
        tt::TokenTree::Subtree(tt) => {
            // fixup should only create matching delimiters, but proc macros
            // could just copy the span to one of the delimiters. We don't want
            // to leak the dummy ID, so we remove both.
            if tt.delimiter.close.anchor.ast_id == FIXUP_DUMMY_AST_ID
                || tt.delimiter.open.anchor.ast_id == FIXUP_DUMMY_AST_ID
            {
                return TransformTtAction::remove();
            }
            TransformTtAction::Keep
        }
    });
    tt.0 = tts.into_boxed_slice();
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use span::{Edition, EditionedFileId, FileId};
    use syntax::TextRange;
    use syntax_bridge::DocCommentDesugarMode;
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
            (tt::Leaf::Literal(a), tt::Leaf::Literal(b)) => a.symbol == b.symbol,
            (tt::Leaf::Punct(a), tt::Leaf::Punct(b)) => a.char == b.char,
            (tt::Leaf::Ident(a), tt::Leaf::Ident(b)) => a.sym == b.sym,
            _ => false,
        }
    }

    fn check_subtree_eq(a: &tt::TopSubtree, b: &tt::TopSubtree) -> bool {
        let a = a.view().as_token_trees().flat_tokens();
        let b = b.view().as_token_trees().flat_tokens();
        a.len() == b.len() && std::iter::zip(a, b).all(|(a, b)| check_tt_eq(a, b))
    }

    fn check_tt_eq(a: &tt::TokenTree, b: &tt::TokenTree) -> bool {
        match (a, b) {
            (tt::TokenTree::Leaf(a), tt::TokenTree::Leaf(b)) => check_leaf_eq(a, b),
            (tt::TokenTree::Subtree(a), tt::TokenTree::Subtree(b)) => {
                a.delimiter.kind == b.delimiter.kind
            }
            _ => false,
        }
    }

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, mut expect: Expect) {
        let parsed = syntax::SourceFile::parse(ra_fixture, span::Edition::CURRENT);
        let span_map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(EditionedFileId::new(
            FileId::from_raw(0),
            Edition::CURRENT,
        ))));
        let fixups = super::fixup_syntax(
            span_map.as_ref(),
            &parsed.syntax_node(),
            span_map.span_for_range(TextRange::empty(0.into())),
            DocCommentDesugarMode::Mbe,
        );
        let mut tt = syntax_bridge::syntax_node_to_token_tree_modified(
            &parsed.syntax_node(),
            span_map.as_ref(),
            fixups.append,
            fixups.remove,
            span_map.span_for_range(TextRange::empty(0.into())),
            DocCommentDesugarMode::Mbe,
        );

        let actual = format!("{tt}\n");

        expect.indent(false);
        expect.assert_eq(&actual);

        // the fixed-up tree should be syntactically valid
        let (parse, _) = syntax_bridge::token_tree_to_syntax_node(
            &tt,
            syntax_bridge::TopEntryPoint::MacroItems,
            &mut |_| parser::Edition::CURRENT,
            parser::Edition::CURRENT,
        );
        assert!(
            parse.errors().is_empty(),
            "parse has syntax errors. parse tree:\n{:#?}",
            parse.syntax_node()
        );

        // the fixed-up tree should not contain braces as punct
        // FIXME: should probably instead check that it's a valid punctuation character
        for x in tt.token_trees().flat_tokens() {
            match x {
                ::tt::TokenTree::Leaf(::tt::Leaf::Punct(punct)) => {
                    assert!(!matches!(punct.char, '{' | '}' | '(' | ')' | '[' | ']'))
                }
                _ => (),
            }
        }

        reverse_fixups(&mut tt, &fixups.undo_info);

        // the fixed-up + reversed version should be equivalent to the original input
        // modulo token IDs and `Punct`s' spacing.
        let original_as_tt = syntax_bridge::syntax_node_to_token_tree(
            &parsed.syntax_node(),
            span_map.as_ref(),
            span_map.span_for_range(TextRange::empty(0.into())),
            DocCommentDesugarMode::Mbe,
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

    #[test]
    fn fixup_path() {
        check(
            r#"
fn foo() {
    path::
}
"#,
            expect![[r#"
fn foo () {path :: __ra_fixup}
"#]],
        )
    }

    #[test]
    fn fixup_record_ctor_field() {
        check(
            r#"
fn foo() {
    R { f: }
}
"#,
            expect![[r#"
fn foo () {R {f : __ra_fixup}}
"#]],
        )
    }

    #[test]
    fn no_fixup_record_ctor_field() {
        check(
            r#"
fn foo() {
    R { f: a }
}
"#,
            expect![[r#"
fn foo () {R {f : a}}
"#]],
        )
    }

    #[test]
    fn fixup_arg_list() {
        check(
            r#"
fn foo() {
    foo(a
}
"#,
            expect![[r#"
fn foo () {foo (a)}
"#]],
        );
        check(
            r#"
fn foo() {
    bar.foo(a
}
"#,
            expect![[r#"
fn foo () {bar . foo (a)}
"#]],
        );
    }

    #[test]
    fn fixup_closure() {
        check(
            r#"
fn foo() {
    ||
}
"#,
            expect![[r#"
fn foo () {|| __ra_fixup}
"#]],
        );
    }

    #[test]
    fn fixup_regression_() {
        check(
            r#"
fn foo() {
    {}
    {}
}
"#,
            expect![[r#"
fn foo () {{} {}}
"#]],
        );
    }
}
