use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::ty::{get_iterator_item_ty, is_copy};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{sym, Span};

use super::CLONED_INSTEAD_OF_COPIED;

pub fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, msrv: &Msrv) {
    let recv_ty = cx.typeck_results().expr_ty_adjusted(recv);
    let inner_ty = match recv_ty.kind() {
        // `Option<T>` -> `T`
        ty::Adt(adt, subst)
            if cx.tcx.is_diagnostic_item(sym::Option, adt.did()) && msrv.meets(msrvs::OPTION_COPIED) =>
        {
            subst.type_at(0)
        },
        _ if is_trait_method(cx, expr, sym::Iterator) && msrv.meets(msrvs::ITERATOR_COPIED) => {
            match get_iterator_item_ty(cx, recv_ty) {
                // <T as Iterator>::Item
                Some(ty) => ty,
                _ => return,
            }
        },
        _ => return,
    };
    match inner_ty.kind() {
        // &T where T: Copy
        ty::Ref(_, ty, _) if is_copy(cx, *ty) => {},
        _ => return,
    };
    span_lint_and_sugg(
        cx,
        CLONED_INSTEAD_OF_COPIED,
        span,
        "used `cloned` where `copied` could be used instead",
        "try",
        "copied".into(),
        Applicability::MachineApplicable,
    );
}
