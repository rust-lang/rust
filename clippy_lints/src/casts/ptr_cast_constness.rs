use clippy_config::msrvs::{self, Msrv};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::PTR_CAST_CONSTNESS;

pub(super) fn check<'tcx>(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_expr: &Expr<'_>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
    msrv: &Msrv,
) {
    if msrv.meets(msrvs::POINTER_CAST_CONSTNESS)
        && let ty::RawPtr(from_ty, from_mutbl) = cast_from.kind()
        && let ty::RawPtr(to_ty, to_mutbl) = cast_to.kind()
        && matches!(
            (from_mutbl, to_mutbl),
            (Mutability::Not, Mutability::Mut) | (Mutability::Mut, Mutability::Not)
        )
        && from_ty == to_ty
    {
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
