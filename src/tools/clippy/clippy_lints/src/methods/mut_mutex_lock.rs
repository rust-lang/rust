use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{sym, Span};

use super::MUT_MUTEX_LOCK;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, ex: &'tcx Expr<'tcx>, recv: &'tcx Expr<'tcx>, name_span: Span) {
    if_chain! {
        if let ty::Ref(_, _, Mutability::Mut) = cx.typeck_results().expr_ty(recv).kind();
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(ex.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id);
        if is_type_diagnostic_item(cx, cx.tcx.type_of(impl_id), sym::Mutex);
        then {
            span_lint_and_sugg(
                cx,
                MUT_MUTEX_LOCK,
                name_span,
                "calling `&mut Mutex::lock` unnecessarily locks an exclusive (mutable) reference",
                "change this to",
                "get_mut".to_owned(),
                Applicability::MaybeIncorrect,
            );
        }
    }
}
