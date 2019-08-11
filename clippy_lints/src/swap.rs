use crate::utils::sugg::Sugg;
use crate::utils::{
    differing_macro_contexts, match_type, paths, snippet, span_lint_and_then, walk_ptrs_ty, SpanlessEq,
};
use if_chain::if_chain;
use matches::matches;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;

declare_clippy_lint! {
    /// **What it does:** Checks for manual swapping.
    ///
    /// **Why is this bad?** The `std::mem::swap` function exposes the intent better
    /// without deinitializing or copying either variable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let mut a = 42;
    /// let mut b = 1337;
    ///
    /// let t = b;
    /// b = a;
    /// a = t;
    /// ```
    /// Use std::mem::swap():
    /// ```rust
    /// let mut a = 1;
    /// let mut b = 2;
    /// std::mem::swap(&mut a, &mut b);
    /// ```
    pub MANUAL_SWAP,
    complexity,
    "manual swap of two variables"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `foo = bar; bar = foo` sequences.
    ///
    /// **Why is this bad?** This looks like a failed attempt to swap.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let mut a = 1;
    /// # let mut b = 2;
    /// a = b;
    /// b = a;
    /// ```
    pub ALMOST_SWAPPED,
    correctness,
    "`foo = bar; bar = foo` sequence"
}

declare_lint_pass!(Swap => [MANUAL_SWAP, ALMOST_SWAPPED]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Swap {
    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx Block) {
        check_manual_swap(cx, block);
        check_suspicious_swap(cx, block);
    }
}

/// Implementation of the `MANUAL_SWAP` lint.
fn check_manual_swap(cx: &LateContext<'_, '_>, block: &Block) {
    for w in block.stmts.windows(3) {
        if_chain! {
            // let t = foo();
            if let StmtKind::Local(ref tmp) = w[0].node;
            if let Some(ref tmp_init) = tmp.init;
            if let PatKind::Binding(.., ident, None) = tmp.pat.node;

            // foo() = bar();
            if let StmtKind::Semi(ref first) = w[1].node;
            if let ExprKind::Assign(ref lhs1, ref rhs1) = first.node;

            // bar() = t;
            if let StmtKind::Semi(ref second) = w[2].node;
            if let ExprKind::Assign(ref lhs2, ref rhs2) = second.node;
            if let ExprKind::Path(QPath::Resolved(None, ref rhs2)) = rhs2.node;
            if rhs2.segments.len() == 1;

            if ident.as_str() == rhs2.segments[0].ident.as_str();
            if SpanlessEq::new(cx).ignore_fn().eq_expr(tmp_init, lhs1);
            if SpanlessEq::new(cx).ignore_fn().eq_expr(rhs1, lhs2);
            then {
                fn check_for_slice<'a>(
                    cx: &LateContext<'_, '_>,
                    lhs1: &'a Expr,
                    lhs2: &'a Expr,
                ) -> Option<(&'a Expr, &'a Expr, &'a Expr)> {
                    if let ExprKind::Index(ref lhs1, ref idx1) = lhs1.node {
                        if let ExprKind::Index(ref lhs2, ref idx2) = lhs2.node {
                            if SpanlessEq::new(cx).ignore_fn().eq_expr(lhs1, lhs2) {
                                let ty = walk_ptrs_ty(cx.tables.expr_ty(lhs1));

                                if matches!(ty.sty, ty::Slice(_)) ||
                                    matches!(ty.sty, ty::Array(_, _)) ||
                                    match_type(cx, ty, &paths::VEC) ||
                                    match_type(cx, ty, &paths::VEC_DEQUE) {
                                        return Some((lhs1, idx1, idx2));
                                }
                            }
                        }
                    }

                    None
                }

                let (replace, what, sugg) = if let Some((slice, idx1, idx2)) = check_for_slice(cx, lhs1, lhs2) {
                    if let Some(slice) = Sugg::hir_opt(cx, slice) {
                        (false,
                         format!(" elements of `{}`", slice),
                         format!("{}.swap({}, {})",
                                 slice.maybe_par(),
                                 snippet(cx, idx1.span, ".."),
                                 snippet(cx, idx2.span, "..")))
                    } else {
                        (false, String::new(), String::new())
                    }
                } else if let (Some(first), Some(second)) = (Sugg::hir_opt(cx, lhs1), Sugg::hir_opt(cx, rhs1)) {
                    (true, format!(" `{}` and `{}`", first, second),
                        format!("std::mem::swap({}, {})", first.mut_addr(), second.mut_addr()))
                } else {
                    (true, String::new(), String::new())
                };

                let span = w[0].span.to(second.span);

                span_lint_and_then(cx,
                                   MANUAL_SWAP,
                                   span,
                                   &format!("this looks like you are swapping{} manually", what),
                                   |db| {
                                       if !sugg.is_empty() {
                                           db.span_suggestion(
                                               span,
                                               "try",
                                               sugg,
                                               Applicability::Unspecified,
                                           );

                                           if replace {
                                               db.note("or maybe you should use `std::mem::replace`?");
                                           }
                                       }
                                   });
            }
        }
    }
}

/// Implementation of the `ALMOST_SWAPPED` lint.
fn check_suspicious_swap(cx: &LateContext<'_, '_>, block: &Block) {
    for w in block.stmts.windows(2) {
        if_chain! {
            if let StmtKind::Semi(ref first) = w[0].node;
            if let StmtKind::Semi(ref second) = w[1].node;
            if !differing_macro_contexts(first.span, second.span);
            if let ExprKind::Assign(ref lhs0, ref rhs0) = first.node;
            if let ExprKind::Assign(ref lhs1, ref rhs1) = second.node;
            if SpanlessEq::new(cx).ignore_fn().eq_expr(lhs0, rhs1);
            if SpanlessEq::new(cx).ignore_fn().eq_expr(lhs1, rhs0);
            then {
                let lhs0 = Sugg::hir_opt(cx, lhs0);
                let rhs0 = Sugg::hir_opt(cx, rhs0);
                let (what, lhs, rhs) = if let (Some(first), Some(second)) = (lhs0, rhs0) {
                    (
                        format!(" `{}` and `{}`", first, second),
                        first.mut_addr().to_string(),
                        second.mut_addr().to_string(),
                    )
                } else {
                    (String::new(), String::new(), String::new())
                };

                let span = first.span.to(second.span);

                span_lint_and_then(cx,
                                   ALMOST_SWAPPED,
                                   span,
                                   &format!("this looks like you are trying to swap{}", what),
                                   |db| {
                                       if !what.is_empty() {
                                           db.span_suggestion(
                                               span,
                                               "try",
                                               format!(
                                                   "std::mem::swap({}, {})",
                                                   lhs,
                                                   rhs,
                                               ),
                                               Applicability::MaybeIncorrect,
                                           );
                                           db.note("or maybe you should use `std::mem::replace`?");
                                       }
                                   });
            }
        }
    }
}
