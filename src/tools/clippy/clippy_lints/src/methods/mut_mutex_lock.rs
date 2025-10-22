use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::expr_custom_deref_adjustment;
use clippy_utils::res::MaybeDef;
use clippy_utils::ty::peel_and_count_ty_refs;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::MUT_MUTEX_LOCK;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, ex: &'tcx Expr<'tcx>, recv: &'tcx Expr<'tcx>, name_span: Span) {
    if matches!(expr_custom_deref_adjustment(cx, recv), None | Some(Mutability::Mut))
        && let (_, _, Some(Mutability::Mut)) = peel_and_count_ty_refs(cx.typeck_results().expr_ty(recv))
        && let Some(method_id) = cx.typeck_results().type_dependent_def_id(ex.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
        && cx.tcx.type_of(impl_id).is_diag_item(cx, sym::Mutex)
    {
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
