use clippy_utils::msrvs::POINTER_CAST_CONSTNESS;
use clippy_utils::sugg::Sugg;
use clippy_utils::{diagnostics::span_lint_and_sugg, msrvs::Msrv};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TypeAndMut};

use super::PTR_CAST_CONSTNESS;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
    msrv: &Msrv,
) {
    if_chain! {
        if msrv.meets(POINTER_CAST_CONSTNESS);
        if let ty::RawPtr(TypeAndMut { mutbl: from_mutbl, .. }) = cast_from.kind();
        if let ty::RawPtr(TypeAndMut { mutbl: to_mutbl, .. }) = cast_to.kind();
        if matches!((from_mutbl, to_mutbl),
            (Mutability::Not, Mutability::Mut) | (Mutability::Mut, Mutability::Not));
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
                "`as` casting between raw pointers while changing its constness",
                &format!("try `pointer::cast_{constness}`, a safer alternative"),
                format!("{}.cast_{constness}()", sugg.maybe_par()),
                Applicability::MachineApplicable,
            );
        }
    }
}
