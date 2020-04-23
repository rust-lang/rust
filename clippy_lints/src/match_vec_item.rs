use crate::utils::{is_wild, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir::{Arm, Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, AdtDef};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `match vec[idx]` or `match vec[n..m]`.
    ///
    /// **Why is this bad?** This can panic at runtime.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
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
    pub MATCH_VEC_ITEM,
    style,
    "match vector by indexing can panic"
}

declare_lint_pass!(MatchVecItem => [MATCH_VEC_ITEM]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MatchVecItem {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'tcx>) {
        if_chain! {
            if !in_external_macro(cx.sess(), expr.span);
            if let ExprKind::Match(ref ex, ref arms, MatchSource::Normal) = expr.kind;
            if contains_wild_arm(arms);
            if is_vec_indexing(cx, ex);

            then {
                span_lint_and_help(
                    cx,
                    MATCH_VEC_ITEM,
                    expr.span,
                    "indexing vector may panic",
                    None,
                    "consider using `get(..)` instead.",
                );
            }
        }
    }
}

fn contains_wild_arm(arms: &[Arm<'_>]) -> bool {
    arms.iter().any(|arm| is_wild(&arm.pat) && is_unit_expr(arm.body))
}

fn is_vec_indexing<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if_chain! {
        if let ExprKind::Index(ref array, _) = expr.kind;
        let ty = cx.tables.expr_ty(array);
        if let ty::Adt(def, _) = ty.kind;
        if is_vec(cx, def);

        then {
            return true;
        }
    }

    false
}

fn is_vec<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, def: &'a AdtDef) -> bool {
    if_chain! {
        let def_path = cx.tcx.def_path(def.did);
        if def_path.data.len() == 2;
        if let Some(module) = def_path.data.get(0);
        if module.data.as_symbol() == sym!(vec);
        if let Some(name) = def_path.data.get(1);
        if name.data.as_symbol() == sym!(Vec);

        then {
            return true;
        }
    }

    false
}

fn is_unit_expr(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Tup(ref v) if v.is_empty() => true,
        ExprKind::Block(ref b, _) if b.stmts.is_empty() && b.expr.is_none() => true,
        _ => false,
    }
}
