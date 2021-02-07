use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{Expr, ExprKind, HirId, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, FloatTy, IntTy};
use rustc_session::{declare_tool_lint, impl_lint_pass};

use if_chain::if_chain;

use crate::utils::span_lint_and_help;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of unconstrained numeric literals which may cause default numeric fallback in type
    /// inference.
    ///
    /// Default numeric fallback means that if numeric types have not yet been bound to concrete
    /// types at the end of type inference, then integer type is bound to `i32`, and similarly
    /// floating type is bound to `f64`.
    ///
    /// See [RFC0212](https://github.com/rust-lang/rfcs/blob/master/text/0212-restore-int-fallback.md) for more information about the fallback.
    ///
    /// **Why is this bad?** For those who are very careful about types, default numeric fallback
    /// can be a pitfall that cause unexpected runtime behavior.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let i = 10;
    /// let f = 1.23;
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let i = 10i32;
    /// let f: f64 = 1.23;
    /// ```
    pub DEFAULT_NUMERIC_FALLBACK,
    restriction,
    "usage of unconstrained numeric literals which may cause default numeric fallback."
}

#[derive(Default)]
pub struct DefaultNumericFallback {
    /// Hold `init` in `Local` if `Local` has a type annotation.
    bounded_inits: FxHashSet<HirId>,
}

impl_lint_pass!(DefaultNumericFallback => [DEFAULT_NUMERIC_FALLBACK]);

impl LateLintPass<'_> for DefaultNumericFallback {
    fn check_stmt(&mut self, _: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if_chain! {
            if let StmtKind::Local(local) = stmt.kind;
            if local.ty.is_some();
            if let Some(init) = local.init;
            then {
                self.bounded_inits.insert(init.hir_id);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let expr_ty = cx.typeck_results().expr_ty(expr);
        let hir_id = expr.hir_id;
        if_chain! {
            if let ExprKind::Lit(ref lit) = expr.kind;
            if matches!(lit.node,
                        LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed));
            if matches!(expr_ty.kind(), ty::Int(IntTy::I32) | ty::Float(FloatTy::F64));
            if !self.bounded_inits.contains(&hir_id);
            if !cx.tcx.hir().parent_iter(hir_id).any(|(ref hir_id, _)| self.bounded_inits.contains(hir_id));
            then {
                 span_lint_and_help(
                    cx,
                    DEFAULT_NUMERIC_FALLBACK,
                    lit.span,
                    "default numeric fallback might occur",
                    None,
                    "consider adding suffix to avoid default numeric fallback",
                 )
            }
        }
    }
}
