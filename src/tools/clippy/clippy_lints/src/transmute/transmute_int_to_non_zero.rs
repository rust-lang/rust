use super::TRANSMUTE_INT_TO_NON_ZERO;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;

/// Checks for `transmute_int_to_non_zero` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
) -> bool {
    let tcx = cx.tcx;

    let (ty::Int(_) | ty::Uint(_), ty::Adt(adt, substs)) = (&from_ty.kind(), to_ty.kind()) else {
        return false;
    };

    if !tcx.is_diagnostic_item(sym::NonZero, adt.did()) {
        return false;
    };

    // FIXME: This can be simplified once `NonZero<T>` is stable.
    let coercable_types = [
        ("NonZeroU8", tcx.types.u8),
        ("NonZeroU16", tcx.types.u16),
        ("NonZeroU32", tcx.types.u32),
        ("NonZeroU64", tcx.types.u64),
        ("NonZeroU128", tcx.types.u128),
        ("NonZeroUsize", tcx.types.usize),
        ("NonZeroI8", tcx.types.i8),
        ("NonZeroI16", tcx.types.i16),
        ("NonZeroI32", tcx.types.i32),
        ("NonZeroI64", tcx.types.i64),
        ("NonZeroI128", tcx.types.i128),
        ("NonZeroIsize", tcx.types.isize),
    ];

    let int_type = substs.type_at(0);

    let Some(nonzero_alias) = coercable_types.iter().find_map(|(nonzero_alias, t)| {
        if *t == int_type && *t == from_ty {
            Some(nonzero_alias)
        } else {
            None
        }
    }) else {
        return false;
    };

    span_lint_and_then(
        cx,
        TRANSMUTE_INT_TO_NON_ZERO,
        e.span,
        &format!("transmute from a `{from_ty}` to a `{nonzero_alias}`"),
        |diag| {
            let arg = sugg::Sugg::hir(cx, arg, "..");
            diag.span_suggestion(
                e.span,
                "consider using",
                format!("{nonzero_alias}::{}({arg})", sym::new_unchecked),
                Applicability::Unspecified,
            );
        },
    );
    true
}
