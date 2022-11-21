use std::borrow::Cow;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, TypeAndMut};

use super::PTR_AS_PTR;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, msrv: &Msrv) {
    if !msrv.meets(msrvs::POINTER_CAST) {
        return;
    }

    if_chain! {
        if let ExprKind::Cast(cast_expr, cast_to_hir_ty) = expr.kind;
        let (cast_from, cast_to) = (cx.typeck_results().expr_ty(cast_expr), cx.typeck_results().expr_ty(expr));
        if let ty::RawPtr(TypeAndMut { mutbl: from_mutbl, .. }) = cast_from.kind();
        if let ty::RawPtr(TypeAndMut { ty: to_pointee_ty, mutbl: to_mutbl }) = cast_to.kind();
        if matches!((from_mutbl, to_mutbl),
            (Mutability::Not, Mutability::Not) | (Mutability::Mut, Mutability::Mut));
        // The `U` in `pointer::cast` have to be `Sized`
        // as explained here: https://github.com/rust-lang/rust/issues/60602.
        if to_pointee_ty.is_sized(cx.tcx.at(expr.span), cx.param_env);
        then {
            let mut applicability = Applicability::MachineApplicable;
            let cast_expr_sugg = Sugg::hir_with_applicability(cx, cast_expr, "_", &mut applicability);
            let turbofish = match &cast_to_hir_ty.kind {
                    TyKind::Infer => Cow::Borrowed(""),
                    TyKind::Ptr(mut_ty) if matches!(mut_ty.ty.kind, TyKind::Infer) => Cow::Borrowed(""),
                    _ => Cow::Owned(format!("::<{to_pointee_ty}>")),
                };
            span_lint_and_sugg(
                cx,
                PTR_AS_PTR,
                expr.span,
                "`as` casting between raw pointers without changing its mutability",
                "try `pointer::cast`, a safer alternative",
                format!("{}.cast{turbofish}()", cast_expr_sugg.maybe_par()),
                applicability,
            );
        }
    }
}
