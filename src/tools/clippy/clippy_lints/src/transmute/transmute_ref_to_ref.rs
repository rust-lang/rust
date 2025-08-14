use super::{TRANSMUTE_BYTES_TO_STR, TRANSMUTE_PTR_TO_PTR};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::{std_or_core, sugg};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `transmute_bytes_to_str` and `transmute_ptr_to_ptr` lints.
/// Returns `true` if either one triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    const_context: bool,
) -> bool {
    let mut triggered = false;

    if let (ty::Ref(_, ty_from, from_mutbl), ty::Ref(_, ty_to, to_mutbl)) = (*from_ty.kind(), *to_ty.kind()) {
        if let ty::Slice(slice_ty) = *ty_from.kind()
            && ty_to.is_str()
            && let ty::Uint(ty::UintTy::U8) = slice_ty.kind()
            && from_mutbl == to_mutbl
        {
            let Some(top_crate) = std_or_core(cx) else { return true };

            let postfix = if from_mutbl == Mutability::Mut { "_mut" } else { "" };

            let snippet = snippet(cx, arg.span, "..");

            span_lint_and_sugg(
                cx,
                TRANSMUTE_BYTES_TO_STR,
                e.span,
                format!("transmute from a `{from_ty}` to a `{to_ty}`"),
                "consider using",
                if const_context {
                    format!("{top_crate}::str::from_utf8_unchecked{postfix}({snippet})")
                } else {
                    format!("{top_crate}::str::from_utf8{postfix}({snippet}).unwrap()")
                },
                Applicability::MaybeIncorrect,
            );
            triggered = true;
        } else if (cx.tcx.erase_regions(from_ty) != cx.tcx.erase_regions(to_ty)) && !const_context {
            span_lint_and_then(
                cx,
                TRANSMUTE_PTR_TO_PTR,
                e.span,
                "transmute from a reference to a reference",
                |diag| {
                    if let Some(arg) = sugg::Sugg::hir_opt(cx, arg) {
                        let sugg_paren = arg
                            .as_ty(Ty::new_ptr(cx.tcx, ty_from, from_mutbl))
                            .as_ty(Ty::new_ptr(cx.tcx, ty_to, to_mutbl));
                        let sugg = if to_mutbl == Mutability::Mut {
                            sugg_paren.mut_addr_deref()
                        } else {
                            sugg_paren.addr_deref()
                        };
                        diag.span_suggestion(e.span, "try", sugg, Applicability::Unspecified);
                    }
                },
            );

            triggered = true;
        }
    }

    triggered
}
