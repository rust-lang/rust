use super::{TRANSMUTE_BYTES_TO_STR, TRANSMUTE_PTR_TO_PTR};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::Msrv;
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_in_const_context, msrvs};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::{LateContext, std_or_core};
use rustc_middle::ty::{self, Ty};

/// Checks for `transmute_bytes_to_str` and `transmute_ptr_to_ptr` lints.
/// Returns `true` if either one triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    msrv: Msrv,
) -> bool {
    let arg_sugg = || Sugg::hir_with_context(cx, arg, e.span.ctxt(), "..", &mut Applicability::Unspecified);
    if let ty::Ref(_, ty_from, from_mutbl) = *from_ty.kind()
        && let ty::Ref(_, ty_to, to_mutbl) = *to_ty.kind()
    {
        if let ty::Slice(slice_ty) = *ty_from.kind()
            && ty_to.is_str()
            && let ty::Uint(ty::UintTy::U8) = slice_ty.kind()
            && from_mutbl == to_mutbl
            && let Some(top_crate) = std_or_core(cx)
            && let Some(sugg) = transmute_bytes_to_str_is_possible(cx, msrv, from_mutbl, &arg_sugg(), top_crate)
        {
            span_lint_and_sugg(
                cx,
                TRANSMUTE_BYTES_TO_STR,
                e.span,
                format!("transmute from a `{from_ty}` to a `{to_ty}`"),
                "consider using",
                sugg,
                Applicability::MaybeIncorrect,
            );
            return true;
        }

        if (cx.tcx.erase_and_anonymize_regions(from_ty) != cx.tcx.erase_and_anonymize_regions(to_ty))
            && !is_in_const_context(cx)
        {
            span_lint_and_then(
                cx,
                TRANSMUTE_PTR_TO_PTR,
                e.span,
                "transmute from a reference to a reference",
                |diag| {
                    let sugg_paren = arg_sugg()
                        .as_ty(Ty::new_ptr(cx.tcx, ty_from, from_mutbl))
                        .as_ty(Ty::new_ptr(cx.tcx, ty_to, to_mutbl));
                    let sugg = if to_mutbl == Mutability::Mut {
                        sugg_paren.mut_addr_deref()
                    } else {
                        sugg_paren.addr_deref()
                    };
                    diag.span_suggestion(e.span, "try", sugg, Applicability::Unspecified);
                },
            );

            return true;
        }
    }

    false
}

/// Produce the suggestion for transmuting bytes to string if possible
fn transmute_bytes_to_str_is_possible(
    cx: &LateContext<'_>,
    msrv: Msrv,
    mutability: Mutability,
    args: &Sugg<'_>,
    top_crate: &str,
) -> Option<String> {
    if is_in_const_context(cx) {
        // stable in const contexts:
        // - `{top_crate}::str::from_utf8_unchecked` after 1.55
        // - `{top_crate}::mem::transmute` after 1.56
        // - `{top_crate}::str::from_utf8_unchecked_mut` after 1.83
        // - `str::from_utf8_unchecked` after 1.87
        // - `str::from_utf8_unchecked_mut` after 1.87
        match mutability {
            Mutability::Mut if msrv.meets(cx, msrvs::CONST_STR_FROM_UTF8_UNCHECKED_MUT) => {
                Some(format!("str::from_utf8_unchecked_mut({args})"))
            },
            Mutability::Mut if msrv.meets(cx, msrvs::CONST_STD_STR_FROM_UTF8_UNCHECKED_MUT) => {
                Some(format!("{top_crate}::str::from_utf8_unchecked_mut({args})"))
            },
            Mutability::Not if msrv.meets(cx, msrvs::CONST_STR_FROM_UTF8_UNCHECKED) => {
                Some(format!("str::from_utf8_unchecked({args})"))
            },
            Mutability::Not if msrv.meets(cx, msrvs::CONST_STD_STR_FROM_UTF8_UNCHECKED) => {
                Some(format!("{top_crate}::str::from_utf8_unchecked({args})"))
            },
            _ => None,
        }
    } else {
        // stable in regular context:
        // - `{top_crate}::mem::transmute` after 1.0
        // - `std::str::from_utf8` after 1.0
        // - `core::str::from_utf8` after 1.6
        // - `{top_crate}::str::from_utf8_mut` after 1.20
        // - `str::from_utf8_mut` after 1.87
        // - `str::from_utf8` after 1.87
        match mutability {
            Mutability::Mut if msrv.meets(cx, msrvs::STR_FROM_UTF8_MUT) => {
                Some(format!("str::from_utf8_mut({args}).unwrap()"))
            },
            Mutability::Mut if msrv.meets(cx, msrvs::STD_STR_FROM_UTF8_MUT) => {
                Some(format!("{top_crate}::str::from_utf8_mut({args}).unwrap()"))
            },
            Mutability::Not if msrv.meets(cx, msrvs::STR_FROM_UTF8) => Some(format!("str::from_utf8({args}).unwrap()")),
            Mutability::Not if msrv.meets(cx, msrvs::CORE_STR_FROM_UTF8) => {
                Some(format!("{top_crate}::str::from_utf8({args}).unwrap()"))
            },
            Mutability::Not if top_crate == "std" => Some(format!("std::str::from_utf8({args}).unwrap()")),
            _ => None,
        }
    }
}
