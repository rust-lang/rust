use std::ops::ControlFlow;
use std::sync::Arc;

use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_lint_allowed;
use clippy_utils::source::walk_span_to_context;
use clippy_utils::visitors::{Descend, for_each_expr};
use hir::HirId;
use rustc_hir as hir;
use rustc_hir::{Block, BlockCheckMode, ItemKind, Node, UnsafeSource};
use rustc_lexer::{TokenKind, tokenize};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Pos, RelativeBytePos, Span, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `unsafe` blocks and impls without a `// SAFETY: ` comment
    /// explaining why the unsafe operations performed inside
    /// the block are safe.
    ///
    /// Note the comment must appear on the line(s) preceding the unsafe block
    /// with nothing appearing in between. The following is ok:
    /// ```ignore
    /// foo(
    ///     // SAFETY:
    ///     // This is a valid safety comment
    ///     unsafe { *x }
    /// )
    /// ```
    /// But neither of these are:
    /// ```ignore
    /// // SAFETY:
    /// // This is not a valid safety comment
    /// foo(
    ///     /* SAFETY: Neither is this */ unsafe { *x },
    /// );
    /// ```
    ///
    /// ### Why restrict this?
    /// Undocumented unsafe blocks and impls can make it difficult to read and maintain code.
    /// Writing out the safety justification may help in discovering unsoundness or bugs.
    ///
    /// ### Example
    /// ```no_run
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// let ptr = unsafe { NonNull::new_unchecked(a) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// // SAFETY: references are guaranteed to be non-null.
    /// let ptr = unsafe { NonNull::new_unchecked(a) };
    /// ```
    #[clippy::version = "1.58.0"]
    pub UNDOCUMENTED_UNSAFE_BLOCKS,
    restriction,
    "creating an unsafe block without explaining why it is safe"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for `// SAFETY: ` comments on safe code.
    ///
    /// ### Why restrict this?
    /// Safe code has no safety requirements, so there is no need to
    /// describe safety invariants.
    ///
    /// ### Example
    /// ```no_run
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// // SAFETY: references are guaranteed to be non-null.
    /// let ptr = NonNull::new(a).unwrap();
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// let ptr = NonNull::new(a).unwrap();
    /// ```
    #[clippy::version = "1.67.0"]
    pub UNNECESSARY_SAFETY_COMMENT,
    restriction,
    "annotating safe code with a safety comment"
}

pub struct UndocumentedUnsafeBlocks {
    accept_comment_above_statement: bool,
    accept_comment_above_attributes: bool,
}

impl UndocumentedUnsafeBlocks {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            accept_comment_above_statement: conf.accept_comment_above_statement,
            accept_comment_above_attributes: conf.accept_comment_above_attributes,
        }
    }
}

impl_lint_pass!(UndocumentedUnsafeBlocks => [UNDOCUMENTED_UNSAFE_BLOCKS, UNNECESSARY_SAFETY_COMMENT]);

impl<'tcx> LateLintPass<'tcx> for UndocumentedUnsafeBlocks {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && !block.span.in_external_macro(cx.tcx.sess.source_map())
            && !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, block.hir_id)
            && !is_unsafe_from_proc_macro(cx, block.span)
            && !block_has_safety_comment(cx, block.span)
            && !block_parents_have_safety_comment(
                self.accept_comment_above_statement,
                self.accept_comment_above_attributes,
                cx,
                block.hir_id,
            )
        {
            let source_map = cx.tcx.sess.source_map();
            let span = if source_map.is_multiline(block.span) {
                source_map.span_until_char(block.span, '\n')
            } else {
                block.span
            };

            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                UNDOCUMENTED_UNSAFE_BLOCKS,
                span,
                "unsafe block missing a safety comment",
                |diag| {
                    diag.help("consider adding a safety comment on the preceding line");
                },
            );
        }

        if let Some(tail) = block.expr
            && !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, tail.hir_id)
            && !tail.span.in_external_macro(cx.tcx.sess.source_map())
            && let HasSafetyComment::Yes(pos) = stmt_has_safety_comment(cx, tail.span, tail.hir_id)
            && let Some(help_span) = expr_has_unnecessary_safety_comment(cx, tail, pos)
        {
            span_lint_and_then(
                cx,
                UNNECESSARY_SAFETY_COMMENT,
                tail.span,
                "expression has unnecessary safety comment",
                |diag| {
                    diag.span_help(help_span, "consider removing the safety comment");
                },
            );
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &hir::Stmt<'tcx>) {
        let (hir::StmtKind::Let(&hir::LetStmt { init: Some(expr), .. })
        | hir::StmtKind::Expr(expr)
        | hir::StmtKind::Semi(expr)) = stmt.kind
        else {
            return;
        };
        if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, stmt.hir_id)
            && !stmt.span.in_external_macro(cx.tcx.sess.source_map())
            && let HasSafetyComment::Yes(pos) = stmt_has_safety_comment(cx, stmt.span, stmt.hir_id)
            && let Some(help_span) = expr_has_unnecessary_safety_comment(cx, expr, pos)
        {
            span_lint_and_then(
                cx,
                UNNECESSARY_SAFETY_COMMENT,
                stmt.span,
                "statement has unnecessary safety comment",
                |diag| {
                    diag.span_help(help_span, "consider removing the safety comment");
                },
            );
        }
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if item.span.in_external_macro(cx.tcx.sess.source_map()) {
            return;
        }

        let mk_spans = |pos: BytePos| {
            let source_map = cx.tcx.sess.source_map();
            let span = Span::new(pos, pos, SyntaxContext::root(), None);
            let help_span = source_map.span_extend_to_next_char(span, '\n', true);
            let span = if source_map.is_multiline(item.span) {
                source_map.span_until_char(item.span, '\n')
            } else {
                item.span
            };
            (span, help_span)
        };

        let item_has_safety_comment = item_has_safety_comment(cx, item);
        match (&item.kind, item_has_safety_comment) {
            // lint unsafe impl without safety comment
            (ItemKind::Impl(impl_), HasSafetyComment::No) if impl_.safety.is_unsafe() => {
                if !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, item.hir_id())
                    && !is_unsafe_from_proc_macro(cx, item.span)
                {
                    let source_map = cx.tcx.sess.source_map();
                    let span = if source_map.is_multiline(item.span) {
                        source_map.span_until_char(item.span, '\n')
                    } else {
                        item.span
                    };

                    #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
                    span_lint_and_then(
                        cx,
                        UNDOCUMENTED_UNSAFE_BLOCKS,
                        span,
                        "unsafe impl missing a safety comment",
                        |diag| {
                            diag.help("consider adding a safety comment on the preceding line");
                        },
                    );
                }
            },
            // lint safe impl with unnecessary safety comment
            (ItemKind::Impl(impl_), HasSafetyComment::Yes(pos)) if impl_.safety.is_safe() => {
                if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, item.hir_id()) {
                    let (span, help_span) = mk_spans(pos);

                    span_lint_and_then(
                        cx,
                        UNNECESSARY_SAFETY_COMMENT,
                        span,
                        "impl has unnecessary safety comment",
                        |diag| {
                            diag.span_help(help_span, "consider removing the safety comment");
                        },
                    );
                }
            },
            (ItemKind::Impl(_), _) => {},
            // const and static items only need a safety comment if their body is an unsafe block, lint otherwise
            (&ItemKind::Const(.., body) | &ItemKind::Static(.., body), HasSafetyComment::Yes(pos)) => {
                if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, body.hir_id) {
                    let body = cx.tcx.hir_body(body);
                    if !matches!(
                        body.value.kind, hir::ExprKind::Block(block, _)
                        if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
                    ) {
                        let (span, help_span) = mk_spans(pos);

                        span_lint_and_then(
                            cx,
                            UNNECESSARY_SAFETY_COMMENT,
                            span,
                            format!(
                                "{} has unnecessary safety comment",
                                cx.tcx.def_descr(item.owner_id.to_def_id()),
                            ),
                            |diag| {
                                diag.span_help(help_span, "consider removing the safety comment");
                            },
                        );
                    }
                }
            },
            // Aside from unsafe impls and consts/statics with an unsafe block, items in general
            // do not have safety invariants that need to be documented, so lint those.
            (_, HasSafetyComment::Yes(pos)) => {
                if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, item.hir_id()) {
                    let (span, help_span) = mk_spans(pos);

                    span_lint_and_then(
                        cx,
                        UNNECESSARY_SAFETY_COMMENT,
                        span,
                        format!(
                            "{} has unnecessary safety comment",
                            cx.tcx.def_descr(item.owner_id.to_def_id()),
                        ),
                        |diag| {
                            diag.span_help(help_span, "consider removing the safety comment");
                        },
                    );
                }
            },
            _ => (),
        }
    }
}

fn expr_has_unnecessary_safety_comment<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    comment_pos: BytePos,
) -> Option<Span> {
    if cx.tcx.hir_parent_iter(expr.hir_id).any(|(_, ref node)| {
        matches!(
            node,
            Node::Block(Block {
                rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                ..
            }),
        )
    }) {
        return None;
    }

    // this should roughly be the reverse of `block_parents_have_safety_comment`
    if for_each_expr(cx, expr, |expr| match expr.kind {
        hir::ExprKind::Block(
            Block {
                rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                ..
            },
            _,
        ) => ControlFlow::Break(()),
        // `_ = foo()` is desugared to `{ let _ = foo(); }`
        hir::ExprKind::Block(
            Block {
                rules: BlockCheckMode::DefaultBlock,
                stmts:
                    [
                        hir::Stmt {
                            kind:
                                hir::StmtKind::Let(hir::LetStmt {
                                    source: hir::LocalSource::AssignDesugar(_),
                                    ..
                                }),
                            ..
                        },
                    ],
                ..
            },
            _,
        ) => ControlFlow::Continue(Descend::Yes),
        // statements will be handled by check_stmt itself again
        hir::ExprKind::Block(..) => ControlFlow::Continue(Descend::No),
        _ => ControlFlow::Continue(Descend::Yes),
    })
    .is_some()
    {
        return None;
    }

    let source_map = cx.tcx.sess.source_map();
    let span = Span::new(comment_pos, comment_pos, SyntaxContext::root(), None);
    let help_span = source_map.span_extend_to_next_char(span, '\n', true);

    Some(help_span)
}

fn is_unsafe_from_proc_macro(cx: &LateContext<'_>, span: Span) -> bool {
    let source_map = cx.sess().source_map();
    let file_pos = source_map.lookup_byte_offset(span.lo());
    file_pos
        .sf
        .src
        .as_deref()
        .and_then(|src| src.get(file_pos.pos.to_usize()..))
        .is_none_or(|src| !src.starts_with("unsafe"))
}

fn find_unsafe_block_parent_in_expr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<(Span, HirId)> {
    match cx.tcx.parent_hir_node(expr.hir_id) {
        Node::LetStmt(hir::LetStmt { span, hir_id, .. })
        | Node::Expr(hir::Expr {
            hir_id,
            kind: hir::ExprKind::Assign(_, _, span),
            ..
        }) => Some((*span, *hir_id)),
        Node::Expr(expr) => find_unsafe_block_parent_in_expr(cx, expr),
        node if let Some((span, hir_id)) = span_and_hid_of_item_alike_node(&node)
            && is_const_or_static(&node) =>
        {
            Some((span, hir_id))
        },

        _ => {
            if is_branchy(expr) {
                return None;
            }
            Some((expr.span, expr.hir_id))
        },
    }
}

// Checks if any parent {expression, statement, block, local, const, static}
// has a safety comment
fn block_parents_have_safety_comment(
    accept_comment_above_statement: bool,
    accept_comment_above_attributes: bool,
    cx: &LateContext<'_>,
    id: HirId,
) -> bool {
    let (span, hir_id) = match cx.tcx.parent_hir_node(id) {
        Node::Expr(expr) if let Some(inner) = find_unsafe_block_parent_in_expr(cx, expr) => inner,
        Node::Stmt(hir::Stmt {
            kind:
                hir::StmtKind::Let(hir::LetStmt { span, hir_id, .. })
                | hir::StmtKind::Expr(hir::Expr { span, hir_id, .. })
                | hir::StmtKind::Semi(hir::Expr { span, hir_id, .. }),
            ..
        })
        | Node::LetStmt(hir::LetStmt { span, hir_id, .. }) => (*span, *hir_id),

        node if let Some((span, hir_id)) = span_and_hid_of_item_alike_node(&node)
            && is_const_or_static(&node) =>
        {
            (span, hir_id)
        },

        _ => return false,
    };
    // if unsafe block is part of a let/const/static statement,
    // and accept_comment_above_statement is set to true
    // we accept the safety comment in the line the precedes this statement.
    accept_comment_above_statement
        && span_with_attrs_has_safety_comment(cx, span, hir_id, accept_comment_above_attributes)
}

/// Extends `span` to also include its attributes, then checks if that span has a safety comment.
fn span_with_attrs_has_safety_comment(
    cx: &LateContext<'_>,
    span: Span,
    hir_id: HirId,
    accept_comment_above_attributes: bool,
) -> bool {
    let span = if accept_comment_above_attributes {
        include_attrs_in_span(cx, hir_id, span)
    } else {
        span
    };

    span_has_safety_comment(cx, span)
}

/// Checks if an expression is "branchy", e.g. loop, match/if/etc.
fn is_branchy(expr: &hir::Expr<'_>) -> bool {
    matches!(
        expr.kind,
        hir::ExprKind::If(..) | hir::ExprKind::Loop(..) | hir::ExprKind::Match(..)
    )
}

/// Checks if the lines immediately preceding the block contain a safety comment.
fn block_has_safety_comment(cx: &LateContext<'_>, span: Span) -> bool {
    // This intentionally ignores text before the start of a function so something like:
    // ```
    //     // SAFETY: reason
    //     fn foo() { unsafe { .. } }
    // ```
    // won't work. This is to avoid dealing with where such a comment should be place relative to
    // attributes and doc comments.

    matches!(
        span_from_macro_expansion_has_safety_comment(cx, span),
        HasSafetyComment::Yes(_)
    ) || span_has_safety_comment(cx, span)
}

fn include_attrs_in_span(cx: &LateContext<'_>, hir_id: HirId, span: Span) -> Span {
    span.to(cx.tcx.hir_attrs(hir_id).iter().fold(span, |acc, attr| {
        if attr.is_doc_comment() {
            return acc;
        }
        acc.to(attr.span())
    }))
}

enum HasSafetyComment {
    Yes(BytePos),
    No,
    Maybe,
}

/// Checks if the lines immediately preceding the item contain a safety comment.
#[allow(clippy::collapsible_match)]
fn item_has_safety_comment(cx: &LateContext<'_>, item: &hir::Item<'_>) -> HasSafetyComment {
    match span_from_macro_expansion_has_safety_comment(cx, item.span) {
        HasSafetyComment::Maybe => (),
        has_safety_comment => return has_safety_comment,
    }

    if item.span.ctxt() != SyntaxContext::root() {
        return HasSafetyComment::No;
    }
    let comment_start = match cx.tcx.parent_hir_node(item.hir_id()) {
        Node::Crate(parent_mod) => comment_start_before_item_in_mod(cx, parent_mod, parent_mod.spans.inner_span, item),
        Node::Item(parent_item) => {
            if let ItemKind::Mod(_, parent_mod) = &parent_item.kind {
                comment_start_before_item_in_mod(cx, parent_mod, parent_item.span, item)
            } else {
                // Doesn't support impls in this position. Pretend a comment was found.
                return HasSafetyComment::Maybe;
            }
        },
        Node::Stmt(stmt) => {
            if let Node::Block(block) = cx.tcx.parent_hir_node(stmt.hir_id) {
                walk_span_to_context(block.span, SyntaxContext::root()).map(Span::lo)
            } else {
                // Problem getting the parent node. Pretend a comment was found.
                return HasSafetyComment::Maybe;
            }
        },
        _ => {
            // Doesn't support impls in this position. Pretend a comment was found.
            return HasSafetyComment::Maybe;
        },
    };

    let source_map = cx.sess().source_map();
    if let Some(comment_start) = comment_start
        && let Ok(unsafe_line) = source_map.lookup_line(item.span.lo())
        && let Ok(comment_start_line) = source_map.lookup_line(comment_start)
        && Arc::ptr_eq(&unsafe_line.sf, &comment_start_line.sf)
        && let Some(src) = unsafe_line.sf.src.as_deref()
    {
        return if comment_start_line.line >= unsafe_line.line {
            HasSafetyComment::No
        } else {
            match text_has_safety_comment(
                src,
                &unsafe_line.sf.lines()[comment_start_line.line + 1..=unsafe_line.line],
                unsafe_line.sf.start_pos,
            ) {
                Some(b) => HasSafetyComment::Yes(b),
                None => HasSafetyComment::No,
            }
        };
    }
    HasSafetyComment::Maybe
}

/// Checks if the lines immediately preceding the item contain a safety comment.
#[allow(clippy::collapsible_match)]
fn stmt_has_safety_comment(cx: &LateContext<'_>, span: Span, hir_id: HirId) -> HasSafetyComment {
    match span_from_macro_expansion_has_safety_comment(cx, span) {
        HasSafetyComment::Maybe => (),
        has_safety_comment => return has_safety_comment,
    }

    if span.ctxt() != SyntaxContext::root() {
        return HasSafetyComment::No;
    }

    let comment_start = match cx.tcx.parent_hir_node(hir_id) {
        Node::Block(block) => walk_span_to_context(block.span, SyntaxContext::root()).map(Span::lo),
        _ => return HasSafetyComment::Maybe,
    };

    let source_map = cx.sess().source_map();
    if let Some(comment_start) = comment_start
        && let Ok(unsafe_line) = source_map.lookup_line(span.lo())
        && let Ok(comment_start_line) = source_map.lookup_line(comment_start)
        && Arc::ptr_eq(&unsafe_line.sf, &comment_start_line.sf)
        && let Some(src) = unsafe_line.sf.src.as_deref()
    {
        return if comment_start_line.line >= unsafe_line.line {
            HasSafetyComment::No
        } else {
            match text_has_safety_comment(
                src,
                &unsafe_line.sf.lines()[comment_start_line.line + 1..=unsafe_line.line],
                unsafe_line.sf.start_pos,
            ) {
                Some(b) => HasSafetyComment::Yes(b),
                None => HasSafetyComment::No,
            }
        };
    }
    HasSafetyComment::Maybe
}

fn comment_start_before_item_in_mod(
    cx: &LateContext<'_>,
    parent_mod: &hir::Mod<'_>,
    parent_mod_span: Span,
    item: &hir::Item<'_>,
) -> Option<BytePos> {
    parent_mod.item_ids.iter().enumerate().find_map(|(idx, item_id)| {
        if *item_id == item.item_id() {
            if idx == 0 {
                // mod A { /* comment */ unsafe impl T {} ... }
                // ^------------------------------------------^ returns the start of this span
                // ^---------------------^ finally checks comments in this range
                if let Some(sp) = walk_span_to_context(parent_mod_span, SyntaxContext::root()) {
                    return Some(sp.lo());
                }
            } else {
                // some_item /* comment */ unsafe impl T {}
                // ^-------^ returns the end of this span
                //         ^---------------^ finally checks comments in this range
                let prev_item = cx.tcx.hir_item(parent_mod.item_ids[idx - 1]);
                if let Some(sp) = walk_span_to_context(prev_item.span, SyntaxContext::root()) {
                    return Some(sp.hi());
                }
            }
        }
        None
    })
}

fn span_from_macro_expansion_has_safety_comment(cx: &LateContext<'_>, span: Span) -> HasSafetyComment {
    let source_map = cx.sess().source_map();
    let ctxt = span.ctxt();
    if ctxt == SyntaxContext::root() {
        HasSafetyComment::Maybe
    }
    // From a macro expansion. Get the text from the start of the macro declaration to start of the
    // unsafe block.
    //     macro_rules! foo { () => { stuff }; (x) => { unsafe { stuff } }; }
    //     ^--------------------------------------------^
    else if let Ok(unsafe_line) = source_map.lookup_line(span.lo())
        && let Ok(macro_line) = source_map.lookup_line(ctxt.outer_expn_data().def_site.lo())
        && Arc::ptr_eq(&unsafe_line.sf, &macro_line.sf)
        && let Some(src) = unsafe_line.sf.src.as_deref()
    {
        if macro_line.line < unsafe_line.line {
            match text_has_safety_comment(
                src,
                &unsafe_line.sf.lines()[macro_line.line + 1..=unsafe_line.line],
                unsafe_line.sf.start_pos,
            ) {
                Some(b) => HasSafetyComment::Yes(b),
                None => HasSafetyComment::No,
            }
        } else {
            HasSafetyComment::No
        }
    } else {
        // Problem getting source text. Pretend a comment was found.
        HasSafetyComment::Maybe
    }
}

fn get_body_search_span(cx: &LateContext<'_>) -> Option<Span> {
    let body = cx.enclosing_body?;
    let mut maybe_mod_item = None;

    for (_, parent_node) in cx.tcx.hir_parent_iter(body.hir_id) {
        match parent_node {
            Node::Crate(mod_) => return Some(mod_.spans.inner_span),
            Node::Item(hir::Item {
                kind: ItemKind::Mod(_, mod_),
                span,
                ..
            }) => {
                return maybe_mod_item
                    .and_then(|item| comment_start_before_item_in_mod(cx, mod_, *span, &item))
                    .map(|comment_start| mod_.spans.inner_span.with_lo(comment_start))
                    .or(Some(*span));
            },
            node if let Some((span, _)) = span_and_hid_of_item_alike_node(&node)
                && !is_const_or_static(&node) =>
            {
                return Some(span);
            },
            Node::Item(item) => {
                maybe_mod_item = Some(*item);
            },
            _ => {
                maybe_mod_item = None;
            },
        }
    }
    None
}

fn span_has_safety_comment(cx: &LateContext<'_>, span: Span) -> bool {
    let source_map = cx.sess().source_map();
    let ctxt = span.ctxt();
    if ctxt.is_root()
        && let Some(search_span) = get_body_search_span(cx)
    {
        if let Ok(unsafe_line) = source_map.lookup_line(span.lo())
            && let Some(body_span) = walk_span_to_context(search_span, SyntaxContext::root())
            && let Ok(body_line) = source_map.lookup_line(body_span.lo())
            && Arc::ptr_eq(&unsafe_line.sf, &body_line.sf)
            && let Some(src) = unsafe_line.sf.src.as_deref()
        {
            // Get the text from the start of function body to the unsafe block.
            //     fn foo() { some_stuff; unsafe { stuff }; other_stuff; }
            //              ^-------------^
            body_line.line < unsafe_line.line
                && text_has_safety_comment(
                    src,
                    &unsafe_line.sf.lines()[body_line.line + 1..=unsafe_line.line],
                    unsafe_line.sf.start_pos,
                )
                .is_some()
        } else {
            // Problem getting source text. Pretend a comment was found.
            true
        }
    } else {
        false
    }
}

/// Checks if the given text has a safety comment for the immediately proceeding line.
fn text_has_safety_comment(src: &str, line_starts: &[RelativeBytePos], start_pos: BytePos) -> Option<BytePos> {
    let mut lines = line_starts
        .array_windows::<2>()
        .rev()
        .map_while(|[start, end]| {
            let start = start.to_usize();
            let end = end.to_usize();
            let text = src.get(start..end)?;
            let trimmed = text.trim_start();
            Some((start + (text.len() - trimmed.len()), trimmed))
        })
        .filter(|(_, text)| !text.is_empty());

    let (line_start, line) = lines.next()?;
    let mut in_codeblock = false;
    // Check for a sequence of line comments.
    if line.starts_with("//") {
        let (mut line, mut line_start) = (line, line_start);
        loop {
            // Don't lint if the safety comment is part of a codeblock in a doc comment.
            // It may or may not be required, and we can't very easily check it (and we shouldn't, since
            // the safety comment isn't referring to the node we're currently checking)
            if line.trim_start_matches("///").trim_start().starts_with("```") {
                in_codeblock = !in_codeblock;
            }

            if line.to_ascii_uppercase().contains("SAFETY:") && !in_codeblock {
                return Some(start_pos + BytePos(u32::try_from(line_start).unwrap()));
            }
            match lines.next() {
                Some((s, x)) if x.starts_with("//") => (line, line_start) = (x, s),
                _ => return None,
            }
        }
    }
    // No line comments; look for the start of a block comment.
    // This will only find them if they are at the start of a line.
    let (mut line_start, mut line) = (line_start, line);
    loop {
        if line.starts_with("/*") {
            let src = &src[line_start..line_starts.last().unwrap().to_usize()];
            let mut tokens = tokenize(src);
            return (src[..tokens.next().unwrap().len as usize]
                .to_ascii_uppercase()
                .contains("SAFETY:")
                && tokens.all(|t| t.kind == TokenKind::Whitespace))
            .then_some(start_pos + BytePos(u32::try_from(line_start).unwrap()));
        }
        match lines.next() {
            Some(x) => (line_start, line) = x,
            None => return None,
        }
    }
}

fn span_and_hid_of_item_alike_node(node: &Node<'_>) -> Option<(Span, HirId)> {
    match node {
        Node::Item(item) => Some((item.span, item.owner_id.into())),
        Node::TraitItem(ti) => Some((ti.span, ti.owner_id.into())),
        Node::ImplItem(ii) => Some((ii.span, ii.owner_id.into())),
        _ => None,
    }
}

fn is_const_or_static(node: &Node<'_>) -> bool {
    matches!(
        node,
        Node::Item(hir::Item {
            kind: ItemKind::Const(..) | ItemKind::Static(..),
            ..
        }) | Node::ImplItem(hir::ImplItem {
            kind: hir::ImplItemKind::Const(..),
            ..
        }) | Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Const(..),
            ..
        })
    )
}
