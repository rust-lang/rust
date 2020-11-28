use crate::utils::{
    get_trait_def_id, if_sequence, implements_trait, parent_node_is_if_expr, paths, span_lint_and_help, SpanlessEq,
};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks comparison chains written with `if` that can be
    /// rewritten with `match` and `cmp`.
    ///
    /// **Why is this bad?** `if` is not guaranteed to be exhaustive and conditionals can get
    /// repetitive
    ///
    /// **Known problems:** The match statement may be slower due to the compiler
    /// not inlining the call to cmp. See issue #5354
    ///
    /// **Example:**
    /// ```rust,ignore
    /// # fn a() {}
    /// # fn b() {}
    /// # fn c() {}
    /// fn f(x: u8, y: u8) {
    ///     if x > y {
    ///         a()
    ///     } else if x < y {
    ///         b()
    ///     } else {
    ///         c()
    ///     }
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust,ignore
    /// use std::cmp::Ordering;
    /// # fn a() {}
    /// # fn b() {}
    /// # fn c() {}
    /// fn f(x: u8, y: u8) {
    ///      match x.cmp(&y) {
    ///          Ordering::Greater => a(),
    ///          Ordering::Less => b(),
    ///          Ordering::Equal => c()
    ///      }
    /// }
    /// ```
    pub COMPARISON_CHAIN,
    style,
    "`if`s that can be rewritten with `match` and `cmp`"
}

declare_lint_pass!(ComparisonChain => [COMPARISON_CHAIN]);

impl<'tcx> LateLintPass<'tcx> for ComparisonChain {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        // We only care about the top-most `if` in the chain
        if parent_node_is_if_expr(expr, cx) {
            return;
        }

        // Check that there exists at least one explicit else condition
        let (conds, _) = if_sequence(expr);
        if conds.len() < 2 {
            return;
        }

        for cond in conds.windows(2) {
            if let (
                &ExprKind::Binary(ref kind1, ref lhs1, ref rhs1),
                &ExprKind::Binary(ref kind2, ref lhs2, ref rhs2),
            ) = (&cond[0].kind, &cond[1].kind)
            {
                if !kind_is_cmp(kind1.node) || !kind_is_cmp(kind2.node) {
                    return;
                }

                // Check that both sets of operands are equal
                let mut spanless_eq = SpanlessEq::new(cx);
                let same_fixed_operands = spanless_eq.eq_expr(lhs1, lhs2) && spanless_eq.eq_expr(rhs1, rhs2);
                let same_transposed_operands = spanless_eq.eq_expr(lhs1, rhs2) && spanless_eq.eq_expr(rhs1, lhs2);

                if !same_fixed_operands && !same_transposed_operands {
                    return;
                }

                // Check that if the operation is the same, either it's not `==` or the operands are transposed
                if kind1.node == kind2.node {
                    if kind1.node == BinOpKind::Eq {
                        return;
                    }
                    if !same_transposed_operands {
                        return;
                    }
                }

                // Check that the type being compared implements `core::cmp::Ord`
                let ty = cx.typeck_results().expr_ty(lhs1);
                let is_ord = get_trait_def_id(cx, &paths::ORD).map_or(false, |id| implements_trait(cx, ty, id, &[]));

                if !is_ord {
                    return;
                }
            } else {
                // We only care about comparison chains
                return;
            }
        }
        span_lint_and_help(
            cx,
            COMPARISON_CHAIN,
            expr.span,
            "`if` chain can be rewritten with `match`",
            None,
            "Consider rewriting the `if` chain to use `cmp` and `match`.",
        )
    }
}

fn kind_is_cmp(kind: BinOpKind) -> bool {
    matches!(kind, BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Eq)
}
