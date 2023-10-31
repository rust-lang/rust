use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{Msrv, POINTER_CAST_CONSTNESS};
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TypeAndMut};

use super::PTR_CAST_CONSTNESS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
    msrv: &Msrv,
) {
    if_chain! {
        if msrv.meets(POINTER_CAST_CONSTNESS);
        if let ty::RawPtr(TypeAndMut { mutbl: from_mutbl, ty: from_ty }) = cast_from.kind();
        if let ty::RawPtr(TypeAndMut { mutbl: to_mutbl, ty: to_ty }) = cast_to.kind();
        if matches!((from_mutbl, to_mutbl),
            (Mutability::Not, Mutability::Mut) | (Mutability::Mut, Mutability::Not));
        if from_ty == to_ty;
        then {
            let sugg = Sugg::hir(cx, cast_expr, "_");
            let constness = match *to_mutbl {
                Mutability::Not => "const",
                Mutability::Mut => "mut",
            };

            span_lint_and_sugg(
                cx,
                PTR_CAST_CONSTNESS,
                expr.span,
                "`as` casting between raw pointers while changing only its constness",
                &format!("try `pointer::cast_{constness}`, a safer alternative"),
                format!("{}.cast_{constness}()", sugg.maybe_par()),
                Applicability::MachineApplicable,
            );
        }
    }
}
