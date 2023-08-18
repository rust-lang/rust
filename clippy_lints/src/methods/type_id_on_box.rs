use std::borrow::Cow;

use crate::methods::TYPE_ID_ON_BOX;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::{self, ExistentialPredicate, Ty};
use rustc_span::{sym, Span};

/// Checks if a [`Ty`] is a `dyn Any` or a `dyn Trait` where `Trait: Any`
/// and returns the name of the trait object.
fn is_dyn_any(cx: &LateContext<'_>, ty: Ty<'_>) -> Option<Cow<'static, str>> {
    if let ty::Dynamic(preds, ..) = ty.kind() {
        preds.iter().find_map(|p| match p.skip_binder() {
            ExistentialPredicate::Trait(tr) => {
                if cx.tcx.is_diagnostic_item(sym::Any, tr.def_id) {
                    Some(Cow::Borrowed("Any"))
                } else if cx
                    .tcx
                    .super_predicates_of(tr.def_id)
                    .predicates
                    .iter()
                    .any(|(clause, _)| {
                        matches!(clause.kind().skip_binder(), ty::ClauseKind::Trait(super_tr)
                            if cx.tcx.is_diagnostic_item(sym::Any, super_tr.def_id()))
                    })
                {
                    Some(Cow::Owned(with_forced_trimmed_paths!(cx.tcx.def_path_str(tr.def_id))))
                } else {
                    None
                }
            },
            _ => None,
        })
    } else {
        None
    }
}

pub(super) fn check(cx: &LateContext<'_>, receiver: &Expr<'_>, call_span: Span) {
    let recv_adjusts = cx.typeck_results().expr_adjustments(receiver);

    if let Some(Adjustment { target: recv_ty, .. }) = recv_adjusts.last()
        && let ty::Ref(_, ty, _) = recv_ty.kind()
        && let ty::Adt(adt, args) = ty.kind()
        && adt.is_box()
        && let Some(trait_path) = is_dyn_any(cx, args.type_at(0))
    {
        span_lint_and_then(
            cx,
            TYPE_ID_ON_BOX,
            call_span,
            &format!("calling `.type_id()` on `Box<dyn {trait_path}>`"),
            |diag| {
                let derefs = recv_adjusts
                    .iter()
                    .filter(|adj| matches!(adj.kind, Adjust::Deref(None)))
                    .count();

                let mut sugg = "*".repeat(derefs + 1);
                sugg += &snippet(cx, receiver.span, "<expr>");

                diag.note(
                    "this returns the type id of the literal type `Box<_>` instead of the \
                    type id of the boxed value, which is most likely not what you want",
                )
                .note(format!(
                    "if this is intentional, use `TypeId::of::<Box<dyn {trait_path}>>()` instead, \
                    which makes it more clear"
                ))
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
