use crate::methods::TYPE_ID_ON_BOX;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_middle::ty::{self, ExistentialPredicate, Ty};
use rustc_span::{sym, Span};

fn is_dyn_any(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    if let ty::Dynamic(preds, ..) = ty.kind() {
        preds.iter().any(|p| match p.skip_binder() {
            ExistentialPredicate::Trait(tr) => cx.tcx.is_diagnostic_item(sym::Any, tr.def_id),
            _ => false,
        })
    } else {
        false
    }
}

pub(super) fn check(cx: &LateContext<'_>, receiver: &Expr<'_>, call_span: Span) {
    let recv_adjusts = cx.typeck_results().expr_adjustments(receiver);

    if let Some(Adjustment { target: recv_ty, .. }) = recv_adjusts.last()
        && let ty::Ref(_, ty, _) = recv_ty.kind()
        && let ty::Adt(adt, substs) = ty.kind()
        && adt.is_box()
        && is_dyn_any(cx, substs.type_at(0))
    {
        span_lint_and_then(
            cx,
            TYPE_ID_ON_BOX,
            call_span,
            "calling `.type_id()` on a `Box<dyn Any>`",
            |diag| {
                let derefs = recv_adjusts
                    .iter()
                    .filter(|adj| matches!(adj.kind, Adjust::Deref(None)))
                    .count();

                let mut sugg = "*".repeat(derefs + 1);
                sugg += &snippet(cx, receiver.span, "<expr>");

                diag.note(
                    "this returns the type id of the literal type `Box<dyn Any>` instead of the \
                    type id of the boxed value, which is most likely not what you want"
                )
                .note(
                    "if this is intentional, use `TypeId::of::<Box<dyn Any>>()` instead, \
                    which makes it more clear"
                )
                .span_suggestion(
                    receiver.span,
                    "consider dereferencing first",
                    format!("({sugg})"),
                    Applicability::MaybeIncorrect,
                );
            },
        );
    }
}
