use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{is_expr_path_def_path, is_trait_method, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::{source_map::Span, sym};

use super::FLAT_MAP_IDENTITY;

/// lint use of `flat_map` for `Iterators` where `flatten` would be sufficient
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    flat_map_arg: &'tcx hir::Expr<'_>,
    flat_map_span: Span,
) {
    if is_trait_method(cx, expr, sym::Iterator) {
        let apply_lint = |message: &str| {
            span_lint_and_sugg(
                cx,
                FLAT_MAP_IDENTITY,
                flat_map_span.with_hi(expr.span.hi()),
                message,
                "try",
                "flatten()".to_string(),
                Applicability::MachineApplicable,
            );
        };

        if_chain! {
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = flat_map_arg.kind;
            let body = cx.tcx.hir().body(body_id);

            if let hir::PatKind::Binding(_, _, binding_ident, _) = body.params[0].pat.kind;
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = body.value.kind;

            if path.segments.len() == 1;
            if path.segments[0].ident.name == binding_ident.name;

            then {
                apply_lint("called `flat_map(|x| x)` on an `Iterator`");
            }
        }

        if is_expr_path_def_path(cx, flat_map_arg, &paths::CONVERT_IDENTITY) {
            apply_lint("called `flat_map(std::convert::identity)` on an `Iterator`");
        }
    }
}
