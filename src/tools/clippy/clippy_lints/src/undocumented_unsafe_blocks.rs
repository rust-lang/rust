use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::walk_span_to_context;
use clippy_utils::visitors::{for_each_expr_with_closures, Descend};
use clippy_utils::{get_parent_node, is_lint_allowed};
use hir::HirId;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::{Block, BlockCheckMode, ItemKind, Node, UnsafeSource};
use rustc_lexer::{tokenize, TokenKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{BytePos, Pos, Span, SyntaxContext};

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
    /// ### Why is this bad?
    /// Undocumented unsafe blocks and impls can make it difficult to
    /// read and maintain code, as well as uncover unsoundness
    /// and bugs.
    ///
    /// ### Example
    /// ```rust
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// let ptr = unsafe { NonNull::new_unchecked(a) };
    /// ```
    /// Use instead:
    /// ```rust
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
    /// ### Why is this bad?
    /// Safe code has no safety requirements, so there is no need to
    /// describe safety invariants.
    ///
    /// ### Example
    /// ```rust
    /// use std::ptr::NonNull;
    /// let a = &mut 42;
    ///
    /// // SAFETY: references are guaranteed to be non-null.
    /// let ptr = NonNull::new(a).unwrap();
    /// ```
    /// Use instead:
    /// ```rust
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

#[derive(Copy, Clone)]
pub struct UndocumentedUnsafeBlocks {
    accept_comment_above_statement: bool,
    accept_comment_above_attributes: bool,
}

impl UndocumentedUnsafeBlocks {
    pub fn new(accept_comment_above_statement: bool, accept_comment_above_attributes: bool) -> Self {
        Self {
            accept_comment_above_statement,
            accept_comment_above_attributes,
        }
    }
}

impl_lint_pass!(UndocumentedUnsafeBlocks => [UNDOCUMENTED_UNSAFE_BLOCKS, UNNECESSARY_SAFETY_COMMENT]);

impl<'tcx> LateLintPass<'tcx> for UndocumentedUnsafeBlocks {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && !in_external_macro(cx.tcx.sess, block.span)
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

            span_lint_and_help(
                cx,
                UNDOCUMENTED_UNSAFE_BLOCKS,
                span,
                "unsafe block missing a safety comment",
                None,
                "consider adding a safety comment on the preceding line",
            );
        }

        if let Some(tail) = block.expr
            && !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, tail.hir_id)
            && !in_external_macro(cx.tcx.sess, tail.span)
            && let HasSafetyComment::Yes(pos) = stmt_has_safety_comment(cx, tail.span, tail.hir_id)
            && let Some(help_span) = expr_has_unnecessary_safety_comment(cx, tail, pos)
        {
            span_lint_and_help(
                cx,
                UNNECESSARY_SAFETY_COMMENT,
                tail.span,
                "expression has unnecessary safety comment",
                Some(help_span),
                "consider removing the safety comment",
            );
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &hir::Stmt<'tcx>) {
        let (
            hir::StmtKind::Local(&hir::Local { init: Some(expr), .. })
            | hir::StmtKind::Expr(expr)
            | hir::StmtKind::Semi(expr)
        ) = stmt.kind else { return };
        if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, stmt.hir_id)
            && !in_external_macro(cx.tcx.sess, stmt.span)
            && let HasSafetyComment::Yes(pos) = stmt_has_safety_comment(cx, stmt.span, stmt.hir_id)
            && let Some(help_span) = expr_has_unnecessary_safety_comment(cx, expr, pos)
        {
            span_lint_and_help(
                cx,
                UNNECESSARY_SAFETY_COMMENT,
                stmt.span,
                "statement has unnecessary safety comment",
                Some(help_span),
                "consider removing the safety comment",
            );
        }
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if in_external_macro(cx.tcx.sess, item.span) {
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
            (hir::ItemKind::Impl(impl_), HasSafetyComment::No) if impl_.unsafety == hir::Unsafety::Unsafe => {
                if !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, item.hir_id())
                    && !is_unsafe_from_proc_macro(cx, item.span)
                {
                    let source_map = cx.tcx.sess.source_map();
                    let span = if source_map.is_multiline(item.span) {
                        source_map.span_until_char(item.span, '\n')
                    } else {
                        item.span
                    };

                    span_lint_and_help(
                        cx,
                        UNDOCUMENTED_UNSAFE_BLOCKS,
                        span,
                        "unsafe impl missing a safety comment",
                        None,
                        "consider adding a safety comment on the preceding line",
                    );
                }
            },
            // lint safe impl with unnecessary safety comment
            (hir::ItemKind::Impl(impl_), HasSafetyComment::Yes(pos)) if impl_.unsafety == hir::Unsafety::Normal => {
                if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, item.hir_id()) {
                    let (span, help_span) = mk_spans(pos);

                    span_lint_and_help(
                        cx,
                        UNNECESSARY_SAFETY_COMMENT,
                        span,
                        "impl has unnecessary safety comment",
                        Some(help_span),
                        "consider removing the safety comment",
                    );
                }
            },
            (hir::ItemKind::Impl(_), _) => {},
            // const and static items only need a safety comment if their body is an unsafe block, lint otherwise
            (&hir::ItemKind::Const(.., body) | &hir::ItemKind::Static(.., body), HasSafetyComment::Yes(pos)) => {
                if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, body.hir_id) {
                    let body = cx.tcx.hir().body(body);
                    if !matches!(
                        body.value.kind, hir::ExprKind::Block(block, _)
                        if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
                    ) {
                        let (span, help_span) = mk_spans(pos);

                        span_lint_and_help(
                            cx,
                            UNNECESSARY_SAFETY_COMMENT,
                            span,
                            &format!("{} has unnecessary safety comment", item.kind.descr()),
                            Some(help_span),
                            "consider removing the safety comment",
                        );
                    }
                }
            },
            // Aside from unsafe impls and consts/statics with an unsafe block, items in general
            // do not have safety invariants that need to be documented, so lint those.
            (_, HasSafetyComment::Yes(pos)) => {
                if !is_lint_allowed(cx, UNNECESSARY_SAFETY_COMMENT, item.hir_id()) {
                    let (span, help_span) = mk_spans(pos);

                    span_lint_and_help(
                        cx,
                        UNNECESSARY_SAFETY_COMMENT,
                        span,
                        &format!("{} has unnecessary safety comment", item.kind.descr()),
                        Some(help_span),
                        "consider removing the safety comment",
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
    if cx.tcx.hir().parent_iter(expr.hir_id).any(|(_, ref node)| {
        matches!(
            node,
            Node::Block(&Block {
                rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                ..
            }),
        )
    }) {
        return None;
    }

    // this should roughly be the reverse of `block_parents_have_safety_comment`
    if for_each_expr_with_closures(cx, expr, |expr| match expr.kind {
        hir::ExprKind::Block(
            Block {
                rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                ..
            },
            _,
        ) => ControlFlow::Break(()),
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
        .map_or(true, |src| !src.starts_with("unsafe"))
}

// Checks if any parent {expression, statement, block, local, const, static}
// has a safety comment
fn block_parents_have_safety_comment(
    accept_comment_above_statement: bool,
    accept_comment_above_attributes: bool,
    cx: &LateContext<'_>,
    id: hir::HirId,
) -> bool {
    if let Some(node) = get_parent_node(cx.tcx, id) {
        return match node {
            Node::Expr(expr) => {
                if let Some(
                    Node::Local(hir::Local { span, .. })
                    | Node::Item(hir::Item {
                        kind: hir::ItemKind::Const(..) | ItemKind::Static(..),
                        span,
                        ..
                    }),
                ) = get_parent_node(cx.tcx, expr.hir_id)
                {
                    let hir_id = match get_parent_node(cx.tcx, expr.hir_id) {
                        Some(Node::Local(hir::Local { hir_id, .. })) => *hir_id,
                        Some(Node::Item(hir::Item { owner_id, .. })) => {
                            cx.tcx.hir().local_def_id_to_hir_id(owner_id.def_id)
                        },
                        _ => unreachable!(),
                    };

                    // if unsafe block is part of a let/const/static statement,
                    // and accept_comment_above_statement is set to true
                    // we accept the safety comment in the line the precedes this statement.
                    accept_comment_above_statement
                        && span_with_attrs_in_body_has_safety_comment(
                            cx,
                            *span,
                            hir_id,
                            accept_comment_above_attributes,
                        )
                } else {
                    !is_branchy(expr)
                        && span_with_attrs_in_body_has_safety_comment(
                            cx,
                            expr.span,
                            expr.hir_id,
                            accept_comment_above_attributes,
                        )
                }
            },
            Node::Stmt(hir::Stmt {
                kind:
                    hir::StmtKind::Local(hir::Local { span, hir_id, .. })
                    | hir::StmtKind::Expr(hir::Expr { span, hir_id, .. })
                    | hir::StmtKind::Semi(hir::Expr { span, hir_id, .. }),
                ..
            })
            | Node::Local(hir::Local { span, hir_id, .. }) => {
                span_with_attrs_in_body_has_safety_comment(cx, *span, *hir_id, accept_comment_above_attributes)
            },
            Node::Item(hir::Item {
                kind: hir::ItemKind::Const(..) | ItemKind::Static(..),
                span,
                owner_id,
                ..
            }) => span_with_attrs_in_body_has_safety_comment(
                cx,
                *span,
                cx.tcx.hir().local_def_id_to_hir_id(owner_id.def_id),
                accept_comment_above_attributes,
            ),
            _ => false,
        };
    }
    false
}

/// Extends `span` to also include its attributes, then checks if that span has a safety comment.
fn span_with_attrs_in_body_has_safety_comment(
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

    span_in_body_has_safety_comment(cx, span)
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
    ) || span_in_body_has_safety_comment(cx, span)
}

fn include_attrs_in_span(cx: &LateContext<'_>, hir_id: HirId, span: Span) -> Span {
    span.to(cx
        .tcx
        .hir()
        .attrs(hir_id)
        .iter()
        .fold(span, |acc, attr| acc.to(attr.span)))
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
    if let Some(parent_node) = get_parent_node(cx.tcx, item.hir_id()) {
        let comment_start = match parent_node {
            Node::Crate(parent_mod) => {
                comment_start_before_item_in_mod(cx, parent_mod, parent_mod.spans.inner_span, item)
            },
            Node::Item(parent_item) => {
                if let ItemKind::Mod(parent_mod) = &parent_item.kind {
                    comment_start_before_item_in_mod(cx, parent_mod, parent_item.span, item)
                } else {
                    // Doesn't support impls in this position. Pretend a comment was found.
                    return HasSafetyComment::Maybe;
                }
            },
            Node::Stmt(stmt) => {
                if let Some(Node::Block(block)) = get_parent_node(cx.tcx, stmt.hir_id) {
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
            && Lrc::ptr_eq(&unsafe_line.sf, &comment_start_line.sf)
            && let Some(src) = unsafe_line.sf.src.as_deref()
        {
            return unsafe_line.sf.lines(|lines| {
                if comment_start_line.line >= unsafe_line.line {
                    HasSafetyComment::No
                } else {
                    match text_has_safety_comment(
                        src,
                        &lines[comment_start_line.line + 1..=unsafe_line.line],
                        unsafe_line.sf.start_pos.to_usize(),
                    ) {
                        Some(b) => HasSafetyComment::Yes(b),
                        None => HasSafetyComment::No,
                    }
                }
            });
        }
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

    if let Some(parent_node) = get_parent_node(cx.tcx, hir_id) {
        let comment_start = match parent_node {
            Node::Block(block) => walk_span_to_context(block.span, SyntaxContext::root()).map(Span::lo),
            _ => return HasSafetyComment::Maybe,
        };

        let source_map = cx.sess().source_map();
        if let Some(comment_start) = comment_start
            && let Ok(unsafe_line) = source_map.lookup_line(span.lo())
            && let Ok(comment_start_line) = source_map.lookup_line(comment_start)
            && Lrc::ptr_eq(&unsafe_line.sf, &comment_start_line.sf)
            && let Some(src) = unsafe_line.sf.src.as_deref()
        {
            return unsafe_line.sf.lines(|lines| {
                if comment_start_line.line >= unsafe_line.line {
                    HasSafetyComment::No
                } else {
                    match text_has_safety_comment(
                        src,
                        &lines[comment_start_line.line + 1..=unsafe_line.line],
                        unsafe_line.sf.start_pos.to_usize(),
                    ) {
                        Some(b) => HasSafetyComment::Yes(b),
                        None => HasSafetyComment::No,
                    }
                }
            });
        }
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
                let prev_item = cx.tcx.hir().item(parent_mod.item_ids[idx - 1]);
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
    } else {
        // From a macro expansion. Get the text from the start of the macro declaration to start of the
        // unsafe block.
        //     macro_rules! foo { () => { stuff }; (x) => { unsafe { stuff } }; }
        //     ^--------------------------------------------^
        if let Ok(unsafe_line) = source_map.lookup_line(span.lo())
            && let Ok(macro_line) = source_map.lookup_line(ctxt.outer_expn_data().def_site.lo())
            && Lrc::ptr_eq(&unsafe_line.sf, &macro_line.sf)
            && let Some(src) = unsafe_line.sf.src.as_deref()
        {
            unsafe_line.sf.lines(|lines| {
                if macro_line.line < unsafe_line.line {
                    match text_has_safety_comment(
                        src,
                        &lines[macro_line.line + 1..=unsafe_line.line],
                        unsafe_line.sf.start_pos.to_usize(),
                    ) {
                        Some(b) => HasSafetyComment::Yes(b),
                        None => HasSafetyComment::No,
                    }
                } else {
                    HasSafetyComment::No
                }
            })
        } else {
            // Problem getting source text. Pretend a comment was found.
            HasSafetyComment::Maybe
        }
    }
}

fn get_body_search_span(cx: &LateContext<'_>) -> Option<Span> {
    let body = cx.enclosing_body?;
    let map = cx.tcx.hir();
    let mut span = map.body(body).value.span;
    for (_, node) in map.parent_iter(body.hir_id) {
        match node {
            Node::Expr(e) => span = e.span,
            Node::Block(_)
            | Node::Arm(_)
            | Node::Stmt(_)
            | Node::Local(_)
            | Node::Item(hir::Item {
                kind: hir::ItemKind::Const(..) | ItemKind::Static(..),
                ..
            }) => (),
            _ => break,
        }
    }
    Some(span)
}

fn span_in_body_has_safety_comment(cx: &LateContext<'_>, span: Span) -> bool {
    let source_map = cx.sess().source_map();
    let ctxt = span.ctxt();
    if ctxt == SyntaxContext::root()
        && let Some(search_span) = get_body_search_span(cx)
    {
        if let Ok(unsafe_line) = source_map.lookup_line(span.lo())
            && let Some(body_span) = walk_span_to_context(search_span, SyntaxContext::root())
            && let Ok(body_line) = source_map.lookup_line(body_span.lo())
            && Lrc::ptr_eq(&unsafe_line.sf, &body_line.sf)
            && let Some(src) = unsafe_line.sf.src.as_deref()
        {
            // Get the text from the start of function body to the unsafe block.
            //     fn foo() { some_stuff; unsafe { stuff }; other_stuff; }
            //              ^-------------^
            unsafe_line.sf.lines(|lines| {
                body_line.line < unsafe_line.line && text_has_safety_comment(
                    src,
                    &lines[body_line.line + 1..=unsafe_line.line],
                    unsafe_line.sf.start_pos.to_usize(),
                ).is_some()
            })
        } else {
            // Problem getting source text. Pretend a comment was found.
            true
        }
    } else {
        false
    }
}

/// Checks if the given text has a safety comment for the immediately proceeding line.
fn text_has_safety_comment(src: &str, line_starts: &[BytePos], offset: usize) -> Option<BytePos> {
    let mut lines = line_starts
        .array_windows::<2>()
        .rev()
        .map_while(|[start, end]| {
            let start = start.to_usize() - offset;
            let end = end.to_usize() - offset;
            let text = src.get(start..end)?;
            let trimmed = text.trim_start();
            Some((start + (text.len() - trimmed.len()), trimmed))
        })
        .filter(|(_, text)| !text.is_empty());

    let Some((line_start, line)) = lines.next() else {
        return None;
    };
    // Check for a sequence of line comments.
    if line.starts_with("//") {
        let (mut line, mut line_start) = (line, line_start);
        loop {
            if line.to_ascii_uppercase().contains("SAFETY:") {
                return Some(BytePos(
                    u32::try_from(line_start).unwrap() + u32::try_from(offset).unwrap(),
                ));
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
            let src = &src[line_start..line_starts.last().unwrap().to_usize() - offset];
            let mut tokens = tokenize(src);
            return (src[..tokens.next().unwrap().len as usize]
                .to_ascii_uppercase()
                .contains("SAFETY:")
                && tokens.all(|t| t.kind == TokenKind::Whitespace))
            .then_some(BytePos(
                u32::try_from(line_start).unwrap() + u32::try_from(offset).unwrap(),
            ));
        }
        match lines.next() {
            Some(x) => (line_start, line) = x,
            None => return None,
        }
    }
}
