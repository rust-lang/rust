use std::borrow::Cow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, span_contains_cfg, span_contains_comment};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for hard to read slices of byte characters, that could be more easily expressed as a
    /// byte string.
    ///
    /// ### Why is this bad?
    ///
    /// Potentially makes the string harder to read.
    ///
    /// ### Example
    /// ```ignore
    /// &[b'H', b'e', b'l', b'l', b'o'];
    /// ```
    /// Use instead:
    /// ```ignore
    /// b"Hello"
    /// ```
    #[clippy::version = "1.81.0"]
    pub BYTE_CHAR_SLICES,
    style,
    "hard to read byte char slice"
}

declare_lint_pass!(ByteCharSlice => [BYTE_CHAR_SLICES]);

impl<'tcx> LateLintPass<'tcx> for ByteCharSlice {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !expr.span.from_expansion()
            && let Some((has_ref, slice)) = is_byte_char_slices(cx, expr)
        {
            span_lint_and_then(
                cx,
                BYTE_CHAR_SLICES,
                expr.span,
                "can be more succinctly written as a byte str",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let mut sugg = Sugg::hir_from_snippet(cx, expr, |_| {
                        let mut slice = slice.iter().fold("b\"".to_owned(), |mut acc, span| {
                            let snippet = snippet_with_applicability(cx, *span, "b'?'", &mut app);
                            acc.push_str(match &snippet[2..snippet.len() - 1] {
                                "\"" => "\\\"",
                                "\\'" => "'",
                                other => other,
                            });
                            acc
                        });
                        slice.push('"');
                        Cow::Owned(slice)
                    });
                    if !has_ref && !cx.typeck_results().expr_ty_adjusted(expr).is_array_slice() {
                        sugg = sugg.deref();
                    }

                    diag.span_suggestion(expr.span, "try", sugg, app);
                },
            );
        }
    }
}

/// Checks whether the slice is that of byte chars, and if so, builds a byte-string out of it
fn is_byte_char_slices<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<(bool, Vec<Span>)> {
    let (has_ref, expr) = if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) = expr.kind {
        (true, inner)
    } else if let Some(parent) = get_parent_expr(cx, expr) // Already checked by the parent expr.
        && let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) = parent.kind
    {
        return None;
    } else {
        (false, expr)
    };

    if let ExprKind::Array(members) = expr.kind
        && !members.is_empty()
        && !span_contains_comment(cx, expr.span)
        && !span_contains_cfg(cx, expr.span)
    {
        return members
            .iter()
            .try_fold(Vec::new(), |mut acc, member| {
                if let ExprKind::Lit(lit) = member.kind
                    && let LitKind::Byte(_) = lit.node
                    && expr.span.eq_ctxt(member.span)
                {
                    acc.push(lit.span);
                    return Some(acc);
                }
                None
            })
            .map(|s| (has_ref, s));
    }

    None
}
