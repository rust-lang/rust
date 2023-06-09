use super::TRANSMUTE_PTR_TO_REF;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, GenericArg, Mutability, Path, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

/// Checks for `transmute_ptr_to_ref` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    path: &'tcx Path<'_>,
    msrv: &Msrv,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::RawPtr(from_ptr_ty), ty::Ref(_, to_ref_ty, mutbl)) => {
            span_lint_and_then(
                cx,
                TRANSMUTE_PTR_TO_REF,
                e.span,
                &format!("transmute from a pointer type (`{from_ty}`) to a reference type (`{to_ty}`)"),
                |diag| {
                    let arg = sugg::Sugg::hir(cx, arg, "..");
                    let (deref, cast) = if *mutbl == Mutability::Mut {
                        ("&mut *", "*mut")
                    } else {
                        ("&*", "*const")
                    };
                    let mut app = Applicability::MachineApplicable;

                    let sugg = if let Some(ty) = get_explicit_type(path) {
                        let ty_snip = snippet_with_applicability(cx, ty.span, "..", &mut app);
                        if msrv.meets(msrvs::POINTER_CAST) {
                            format!("{deref}{}.cast::<{ty_snip}>()", arg.maybe_par())
                        } else if from_ptr_ty.has_erased_regions() {
                            sugg::make_unop(deref, arg.as_ty(format!("{cast} () as {cast} {ty_snip}"))).to_string()
                        } else {
                            sugg::make_unop(deref, arg.as_ty(format!("{cast} {ty_snip}"))).to_string()
                        }
                    } else if from_ptr_ty.ty == *to_ref_ty {
                        if from_ptr_ty.has_erased_regions() {
                            if msrv.meets(msrvs::POINTER_CAST) {
                                format!("{deref}{}.cast::<{to_ref_ty}>()", arg.maybe_par())
                            } else {
                                sugg::make_unop(deref, arg.as_ty(format!("{cast} () as {cast} {to_ref_ty}")))
                                    .to_string()
                            }
                        } else {
                            sugg::make_unop(deref, arg).to_string()
                        }
                    } else {
                        sugg::make_unop(deref, arg.as_ty(format!("{cast} {to_ref_ty}"))).to_string()
                    };

                    diag.span_suggestion(e.span, "try", sugg, app);
                },
            );
            true
        },
        _ => false,
    }
}

/// Gets the type `Bar` in `…::transmute<Foo, &Bar>`.
fn get_explicit_type<'tcx>(path: &'tcx Path<'tcx>) -> Option<&'tcx hir::Ty<'tcx>> {
    if let GenericArg::Type(ty) = path.segments.last()?.args?.args.get(1)?
        && let TyKind::Ref(_, ty) = &ty.kind
    {
        Some(ty.ty)
    } else {
        None
    }
}
