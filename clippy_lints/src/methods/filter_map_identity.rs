use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{is_trait_method, match_qpath, path_to_local_id, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::{source_map::Span, sym};

use super::FILTER_MAP_IDENTITY;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    filter_map_args: &[hir::Expr<'_>],
    filter_map_span: Span,
) {
    if is_trait_method(cx, expr, sym::Iterator) {
        let arg_node = &filter_map_args[1].kind;

        let apply_lint = |message: &str| {
            span_lint_and_sugg(
                cx,
                FILTER_MAP_IDENTITY,
                filter_map_span.with_hi(expr.span.hi()),
                message,
                "try",
                "flatten()".to_string(),
                Applicability::MachineApplicable,
            );
        };

        if_chain! {
            if let hir::ExprKind::Closure(_, _, body_id, _, _) = arg_node;
            let body = cx.tcx.hir().body(*body_id);

            if let hir::PatKind::Binding(_, binding_id, ..) = body.params[0].pat.kind;
            if path_to_local_id(&body.value, binding_id);
            then {
                apply_lint("called `filter_map(|x| x)` on an `Iterator`");
            }
        }

        if_chain! {
            if let hir::ExprKind::Path(ref qpath) = arg_node;

            if match_qpath(qpath, &paths::STD_CONVERT_IDENTITY);

            then {
                apply_lint("called `filter_map(std::convert::identity)` on an `Iterator`");
            }
        }
    }
}
