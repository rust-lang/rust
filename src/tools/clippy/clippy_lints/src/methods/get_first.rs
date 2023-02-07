use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_slice_of_primitives;
use clippy_utils::source::snippet_with_applicability;
use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::GET_FIRST;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    arg: &'tcx hir::Expr<'_>,
) {
    if_chain! {
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id);
        if cx.tcx.bound_type_of(impl_id).subst_identity().is_slice();
        if let Some(_) = is_slice_of_primitives(cx, recv);
        if let hir::ExprKind::Lit(Spanned { node: LitKind::Int(0, _), .. }) = arg.kind;
        then {
            let mut app = Applicability::MachineApplicable;
            let slice_name = snippet_with_applicability(cx, recv.span, "..", &mut app);
            span_lint_and_sugg(
                cx,
                GET_FIRST,
                expr.span,
                &format!("accessing first element with `{slice_name}.get(0)`"),
                "try",
                format!("{slice_name}.first()"),
                app,
            );
        }
    }
}
