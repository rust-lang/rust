use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, MatchSource};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `match vec[idx]` or `match vec[n..m]`.
    ///
    /// ### Why is this bad?
    /// This can panic at runtime.
    ///
    /// ### Example
    /// ```rust, no_run
    /// let arr = vec![0, 1, 2, 3];
    /// let idx = 1;
    ///
    /// // Bad
    /// match arr[idx] {
    ///     0 => println!("{}", 0),
    ///     1 => println!("{}", 3),
    ///     _ => {},
    /// }
    /// ```
    /// Use instead:
    /// ```rust, no_run
    /// let arr = vec![0, 1, 2, 3];
    /// let idx = 1;
    ///
    /// // Good
    /// match arr.get(idx) {
    ///     Some(0) => println!("{}", 0),
    ///     Some(1) => println!("{}", 3),
    ///     _ => {},
    /// }
    /// ```
    pub MATCH_ON_VEC_ITEMS,
    pedantic,
    "matching on vector elements can panic"
}

declare_lint_pass!(MatchOnVecItems => [MATCH_ON_VEC_ITEMS]);

impl<'tcx> LateLintPass<'tcx> for MatchOnVecItems {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if_chain! {
            if !in_external_macro(cx.sess(), expr.span);
            if let ExprKind::Match(match_expr, _, MatchSource::Normal) = expr.kind;
            if let Some(idx_expr) = is_vec_indexing(cx, match_expr);
            if let ExprKind::Index(vec, idx) = idx_expr.kind;

            then {
                // FIXME: could be improved to suggest surrounding every pattern with Some(_),
                // but only when `or_patterns` are stabilized.
                span_lint_and_sugg(
                    cx,
                    MATCH_ON_VEC_ITEMS,
                    match_expr.span,
                    "indexing into a vector may panic",
                    "try this",
                    format!(
                        "{}.get({})",
                        snippet(cx, vec.span, ".."),
                        snippet(cx, idx.span, "..")
                    ),
                    Applicability::MaybeIncorrect
                );
            }
        }
    }
}

fn is_vec_indexing<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if_chain! {
        if let ExprKind::Index(array, index) = expr.kind;
        if is_vector(cx, array);
        if !is_full_range(cx, index);

        then {
            return Some(expr);
        }
    }

    None
}

fn is_vector(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    let ty = ty.peel_refs();
    is_type_diagnostic_item(cx, ty, sym::Vec)
}

fn is_full_range(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    let ty = ty.peel_refs();
    is_type_lang_item(cx, ty, LangItem::RangeFull)
}
