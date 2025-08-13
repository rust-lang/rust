use rustc_hir::{Expr, ExprKind, Stmt, StmtKind};
use rustc_middle::ty::{self};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;

use crate::lints::MappingToUnit;
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `map_unit_fn` lint checks for `Iterator::map` receive
    /// a callable that returns `()`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(items: &mut Vec<u8>) {
    ///     items.sort();
    /// }
    ///
    /// fn main() {
    ///     let mut x: Vec<Vec<u8>> = vec![
    ///         vec![0, 2, 1],
    ///         vec![5, 4, 3],
    ///     ];
    ///     x.iter_mut().map(foo);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Mapping to `()` is almost always a mistake.
    pub MAP_UNIT_FN,
    Warn,
    "`Iterator::map` call that discard the iterator's values"
}

declare_lint_pass!(MapUnitFn => [MAP_UNIT_FN]);

impl<'tcx> LateLintPass<'tcx> for MapUnitFn {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &Stmt<'_>) {
        let StmtKind::Semi(expr) = stmt.kind else {
            return;
        };
        let ExprKind::MethodCall(path, receiver, [arg], span) = expr.kind else {
            return;
        };
        if path.ident.name != sym::map
            || stmt.span.from_expansion()
            || receiver.span.from_expansion()
            || arg.span.from_expansion()
            || !is_impl_slice(cx, receiver)
            || !cx
                .typeck_results()
                .type_dependent_def_id(expr.hir_id)
                .is_some_and(|id| cx.tcx.is_diagnostic_item(sym::IteratorMap, id))
        {
            return;
        }
        let (id, sig) = match *cx.typeck_results().expr_ty(arg).kind() {
            ty::Closure(id, subs) => (id, subs.as_closure().sig()),
            ty::FnDef(id, _) => (id, cx.tcx.fn_sig(id).skip_binder()),
            _ => return,
        };
        let ret_ty = sig.output().skip_binder();
        if !(ret_ty.is_unit() || ret_ty.is_never()) {
            return;
        }
        cx.emit_span_lint(
            MAP_UNIT_FN,
            span,
            MappingToUnit {
                function_label: cx.tcx.span_of_impl(id).unwrap_or(arg.span),
                argument_label: arg.span,
                map_label: span,
                suggestion: path.ident.span,
            },
        );
    }
}

fn is_impl_slice(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
    {
        return cx.tcx.type_of(impl_id).skip_binder().is_slice();
    }
    false
}
