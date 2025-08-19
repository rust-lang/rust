use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;
use rustc_span::sym;

use super::GET_FIRST;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    arg: &'tcx hir::Expr<'_>,
) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
        && let identity = cx.tcx.type_of(impl_id).instantiate_identity()
        && let hir::ExprKind::Lit(Spanned {
            node: LitKind::Int(Pu128(0), _),
            ..
        }) = arg.kind
    {
        if identity.is_slice() {
            let mut app = Applicability::MachineApplicable;
            let slice_name = snippet_with_applicability(cx, recv.span, "..", &mut app);
            span_lint_and_sugg(
                cx,
                GET_FIRST,
                expr.span,
                format!("accessing first element with `{slice_name}.get(0)`"),
                "try",
                format!("{slice_name}.first()"),
                app,
            );
        } else if is_type_diagnostic_item(cx, identity, sym::VecDeque) {
            let mut app = Applicability::MachineApplicable;
            let slice_name = snippet_with_applicability(cx, recv.span, "..", &mut app);
            span_lint_and_sugg(
                cx,
                GET_FIRST,
                expr.span,
                format!("accessing first element with `{slice_name}.get(0)`"),
                "try",
                format!("{slice_name}.front()"),
                app,
            );
        }
    }
}
