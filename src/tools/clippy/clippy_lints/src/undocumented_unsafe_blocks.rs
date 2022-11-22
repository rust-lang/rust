use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::walk_span_to_context;
use clippy_utils::{get_parent_node, is_lint_allowed};
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::{Block, BlockCheckMode, ItemKind, Node, UnsafeSource};
use rustc_lexer::{tokenize, TokenKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
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

declare_lint_pass!(UndocumentedUnsafeBlocks => [UNDOCUMENTED_UNSAFE_BLOCKS]);

impl LateLintPass<'_> for UndocumentedUnsafeBlocks {
    fn check_block(&mut self, cx: &LateContext<'_>, block: &'_ Block<'_>) {
        if block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && !in_external_macro(cx.tcx.sess, block.span)
            && !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, block.hir_id)
            && !is_unsafe_from_proc_macro(cx, block.span)
            && !block_has_safety_comment(cx, block.span)
            && !block_parents_have_safety_comment(cx, block.hir_id)
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
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &hir::Item<'_>) {
        if let hir::ItemKind::Impl(imple) = item.kind
            && imple.unsafety == hir::Unsafety::Unsafe
            && !in_external_macro(cx.tcx.sess, item.span)
            && !is_lint_allowed(cx, UNDOCUMENTED_UNSAFE_BLOCKS, item.hir_id())
            && !is_unsafe_from_proc_macro(cx, item.span)
            && !item_has_safety_comment(cx, item)
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
    }
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
fn block_parents_have_safety_comment(cx: &LateContext<'_>, id: hir::HirId) -> bool {
    if let Some(node) = get_parent_node(cx.tcx, id) {
        return match node {
            Node::Expr(expr) => !is_branchy(expr) && span_in_body_has_safety_comment(cx, expr.span),
            Node::Stmt(hir::Stmt {
                kind:
                    hir::StmtKind::Local(hir::Local { span, .. })
                    | hir::StmtKind::Expr(hir::Expr { span, .. })
                    | hir::StmtKind::Semi(hir::Expr { span, .. }),
                ..
            })
            | Node::Local(hir::Local { span, .. })
            | Node::Item(hir::Item {
                kind: hir::ItemKind::Const(..) | ItemKind::Static(..),
                span,
                ..
            }) => span_in_body_has_safety_comment(cx, *span),
            _ => false,
        };
    }
    false
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

    span_from_macro_expansion_has_safety_comment(cx, span) || span_in_body_has_safety_comment(cx, span)
}

/// Checks if the lines immediately preceding the item contain a safety comment.
#[allow(clippy::collapsible_match)]
fn item_has_safety_comment(cx: &LateContext<'_>, item: &hir::Item<'_>) -> bool {
    if span_from_macro_expansion_has_safety_comment(cx, item.span) {
        return true;
    }

    if item.span.ctxt() == SyntaxContext::root() {
        if let Some(parent_node) = get_parent_node(cx.tcx, item.hir_id()) {
            let comment_start = match parent_node {
                Node::Crate(parent_mod) => {
                    comment_start_before_impl_in_mod(cx, parent_mod, parent_mod.spans.inner_span, item)
                },
                Node::Item(parent_item) => {
                    if let ItemKind::Mod(parent_mod) = &parent_item.kind {
                        comment_start_before_impl_in_mod(cx, parent_mod, parent_item.span, item)
                    } else {
                        // Doesn't support impls in this position. Pretend a comment was found.
                        return true;
                    }
                },
                Node::Stmt(stmt) => {
                    if let Some(stmt_parent) = get_parent_node(cx.tcx, stmt.hir_id) {
                        match stmt_parent {
                            Node::Block(block) => walk_span_to_context(block.span, SyntaxContext::root()).map(Span::lo),
                            _ => {
                                // Doesn't support impls in this position. Pretend a comment was found.
                                return true;
                            },
                        }
                    } else {
                        // Problem getting the parent node. Pretend a comment was found.
                        return true;
                    }
                },
                _ => {
                    // Doesn't support impls in this position. Pretend a comment was found.
                    return true;
                },
            };

            let source_map = cx.sess().source_map();
            if let Some(comment_start) = comment_start
                && let Ok(unsafe_line) = source_map.lookup_line(item.span.lo())
                && let Ok(comment_start_line) = source_map.lookup_line(comment_start)
                && Lrc::ptr_eq(&unsafe_line.sf, &comment_start_line.sf)
                && let Some(src) = unsafe_line.sf.src.as_deref()
            {
                unsafe_line.sf.lines(|lines| {
                    comment_start_line.line < unsafe_line.line && text_has_safety_comment(
                        src,
                        &lines[comment_start_line.line + 1..=unsafe_line.line],
                        unsafe_line.sf.start_pos.to_usize(),
                    )
                })
            } else {
                // Problem getting source text. Pretend a comment was found.
                true
            }
        } else {
            // No parent node. Pretend a comment was found.
            true
        }
    } else {
        false
    }
}

fn comment_start_before_impl_in_mod(
    cx: &LateContext<'_>,
    parent_mod: &hir::Mod<'_>,
    parent_mod_span: Span,
    imple: &hir::Item<'_>,
) -> Option<BytePos> {
    parent_mod.item_ids.iter().enumerate().find_map(|(idx, item_id)| {
        if *item_id == imple.item_id() {
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

fn span_from_macro_expansion_has_safety_comment(cx: &LateContext<'_>, span: Span) -> bool {
    let source_map = cx.sess().source_map();
    let ctxt = span.ctxt();
    if ctxt == SyntaxContext::root() {
        false
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
                macro_line.line < unsafe_line.line && text_has_safety_comment(
                    src,
                    &lines[macro_line.line + 1..=unsafe_line.line],
                    unsafe_line.sf.start_pos.to_usize(),
                )
            })
        } else {
            // Problem getting source text. Pretend a comment was found.
            true
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
            Node::Block(_) | Node::Arm(_) | Node::Stmt(_) | Node::Local(_) => (),
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
                )
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
fn text_has_safety_comment(src: &str, line_starts: &[BytePos], offset: usize) -> bool {
    let mut lines = line_starts
        .array_windows::<2>()
        .rev()
        .map_while(|[start, end]| {
            let start = start.to_usize() - offset;
            let end = end.to_usize() - offset;
            src.get(start..end).map(|text| (start, text.trim_start()))
        })
        .filter(|(_, text)| !text.is_empty());

    let Some((line_start, line)) = lines.next() else {
        return false;
    };
    // Check for a sequence of line comments.
    if line.starts_with("//") {
        let mut line = line;
        loop {
            if line.to_ascii_uppercase().contains("SAFETY:") {
                return true;
            }
            match lines.next() {
                Some((_, x)) if x.starts_with("//") => line = x,
                _ => return false,
            }
        }
    }
    // No line comments; look for the start of a block comment.
    // This will only find them if they are at the start of a line.
    let (mut line_start, mut line) = (line_start, line);
    loop {
        if line.starts_with("/*") {
            let src = src[line_start..line_starts.last().unwrap().to_usize() - offset].trim_start();
            let mut tokens = tokenize(src);
            return src[..tokens.next().unwrap().len as usize]
                .to_ascii_uppercase()
                .contains("SAFETY:")
                && tokens.all(|t| t.kind == TokenKind::Whitespace);
        }
        match lines.next() {
            Some(x) => (line_start, line) = x,
            None => return false,
        }
    }
}
