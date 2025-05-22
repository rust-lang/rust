use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::usage::local_used_after_expr;
use clippy_utils::{path_res, sym};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_hir::def::Res;
use rustc_lint::LateContext;
use rustc_span::Symbol;

use super::NEEDLESS_OPTION_AS_DEREF;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, name: Symbol) {
    let typeck = cx.typeck_results();
    let outer_ty = typeck.expr_ty(expr);

    if is_type_diagnostic_item(cx, outer_ty, sym::Option) && outer_ty == typeck.expr_ty(recv) {
        if name == sym::as_deref_mut && recv.is_syntactic_place_expr() {
            let Res::Local(binding_id) = path_res(cx, recv) else {
                return;
            };

            if local_used_after_expr(cx, binding_id, recv) {
                return;
            }
        }

        span_lint_and_sugg(
            cx,
            NEEDLESS_OPTION_AS_DEREF,
            expr.span,
            "derefed type is same as origin",
            "try",
            recv.span.get_source_text(cx).unwrap().to_owned(),
            Applicability::MachineApplicable,
        );
    }
}
