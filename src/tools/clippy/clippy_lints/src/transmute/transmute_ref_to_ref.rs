use super::{TRANSMUTE_BYTES_TO_STR, TRANSMUTE_PTR_TO_PTR};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet;
use clippy_utils::sugg;
use if_chain::if_chain;
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

    if let (ty::Ref(_, ty_from, from_mutbl), ty::Ref(_, ty_to, to_mutbl)) = (&from_ty.kind(), &to_ty.kind()) {
        if_chain! {
            if let (&ty::Slice(slice_ty), &ty::Str) = (&ty_from.kind(), &ty_to.kind());
            if let ty::Uint(ty::UintTy::U8) = slice_ty.kind();
            if from_mutbl == to_mutbl;
            then {
                let postfix = if *from_mutbl == Mutability::Mut {
                    "_mut"
                } else {
                    ""
                };

                span_lint_and_sugg(
                    cx,
                    TRANSMUTE_BYTES_TO_STR,
                    e.span,
                    &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                    "consider using",
                    format!(
                        "std::str::from_utf8{}({}).unwrap()",
                        postfix,
                        snippet(cx, arg.span, ".."),
                    ),
                    Applicability::Unspecified,
                );
                triggered = true;
            } else {
                if (cx.tcx.erase_regions(from_ty) != cx.tcx.erase_regions(to_ty))
                    && !const_context {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_PTR_TO_PTR,
                        e.span,
                        "transmute from a reference to a reference",
                        |diag| if let Some(arg) = sugg::Sugg::hir_opt(cx, arg) {
                            let ty_from_and_mut = ty::TypeAndMut {
                                ty: *ty_from,
                                mutbl: *from_mutbl
                            };
                            let ty_to_and_mut = ty::TypeAndMut { ty: *ty_to, mutbl: *to_mutbl };
                            let sugg_paren = arg
                                .as_ty(cx.tcx.mk_ptr(ty_from_and_mut))
                                .as_ty(cx.tcx.mk_ptr(ty_to_and_mut));
                            let sugg = if *to_mutbl == Mutability::Mut {
                                sugg_paren.mut_addr_deref()
                            } else {
                                sugg_paren.addr_deref()
                            };
                            diag.span_suggestion(
                                e.span,
                                "try",
                                sugg.to_string(),
                                Applicability::Unspecified,
                            );
                        },
                    );

                    triggered = true;
                }
            }
        }
    }

    triggered
}
