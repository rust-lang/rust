use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_lang_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, LangItem};
use rustc_lint::LateContext;

use super::REPEAT_ONCE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    recv: &'tcx Expr<'_>,
    repeat_arg: &'tcx Expr<'_>,
) {
    if ConstEvalCtxt::new(cx).eval(repeat_arg) == Some(Constant::Int(1)) {
        let ty = cx.typeck_results().expr_ty(recv).peel_refs();
        if ty.is_str() {
            span_lint_and_sugg(
                cx,
                REPEAT_ONCE,
                expr.span,
                "calling `repeat(1)` on str",
                "consider using `.to_string()` instead",
                format!("{}.to_string()", snippet(cx, recv.span, r#""...""#)),
                Applicability::MachineApplicable,
            );
        } else if ty.builtin_index().is_some() {
            span_lint_and_sugg(
                cx,
                REPEAT_ONCE,
                expr.span,
                "calling `repeat(1)` on slice",
                "consider using `.to_vec()` instead",
                format!("{}.to_vec()", snippet(cx, recv.span, r#""...""#)),
                Applicability::MachineApplicable,
            );
        } else if is_type_lang_item(cx, ty, LangItem::String) {
            span_lint_and_sugg(
                cx,
                REPEAT_ONCE,
                expr.span,
                "calling `repeat(1)` on a string literal",
                "consider using `.clone()` instead",
                format!("{}.clone()", snippet(cx, recv.span, r#""...""#)),
                Applicability::MachineApplicable,
            );
        }
    }
}
