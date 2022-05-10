use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::{path_def_id, qpath_generic_tys};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, def, def_id::DefId, PrimTy, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::{utils, REDUNDANT_ALLOCATION};

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    let outer_sym = if Some(def_id) == cx.tcx.lang_items().owned_box() {
        "Box"
    } else if cx.tcx.is_diagnostic_item(sym::Rc, def_id) {
        "Rc"
    } else if cx.tcx.is_diagnostic_item(sym::Arc, def_id) {
        "Arc"
    } else {
        return false;
    };

    if let Some(span) = utils::match_borrows_parameter(cx, qpath) {
        let mut applicability = Applicability::MaybeIncorrect;
        let generic_snippet = snippet_with_applicability(cx, span, "..", &mut applicability);
        span_lint_and_then(
            cx,
            REDUNDANT_ALLOCATION,
            hir_ty.span,
            &format!("usage of `{}<{}>`", outer_sym, generic_snippet),
            |diag| {
                diag.span_suggestion(hir_ty.span, "try", format!("{}", generic_snippet), applicability);
                diag.note(&format!(
                    "`{generic}` is already a pointer, `{outer}<{generic}>` allocates a pointer on the heap",
                    outer = outer_sym,
                    generic = generic_snippet
                ));
            },
        );
        return true;
    }

    let Some(ty) = qpath_generic_tys(qpath).next() else { return false };
    let Some(id) = path_def_id(cx, ty) else { return false };
    let (inner_sym, ty) = match cx.tcx.get_diagnostic_name(id) {
        Some(sym::Arc) => ("Arc", ty),
        Some(sym::Rc) => ("Rc", ty),
        _ if Some(id) == cx.tcx.lang_items().owned_box() => ("Box", ty),
        _ => return false,
    };

    let inner_qpath = match &ty.kind {
        TyKind::Path(inner_qpath) => inner_qpath,
        _ => return false,
    };
    let inner_span = match qpath_generic_tys(inner_qpath).next() {
        Some(ty) => {
            if alloc_makes_pointer_thin(cx, ty) {
                return false;
            }
            ty.span
        },
        None => return false,
    };
    if inner_sym == outer_sym {
        let mut applicability = Applicability::MaybeIncorrect;
        let generic_snippet = snippet_with_applicability(cx, inner_span, "..", &mut applicability);
        span_lint_and_then(
            cx,
            REDUNDANT_ALLOCATION,
            hir_ty.span,
            &format!("usage of `{}<{}<{}>>`", outer_sym, inner_sym, generic_snippet),
            |diag| {
                diag.span_suggestion(
                    hir_ty.span,
                    "try",
                    format!("{}<{}>", outer_sym, generic_snippet),
                    applicability,
                );
                diag.note(&format!(
                    "`{inner}<{generic}>` is already on the heap, `{outer}<{inner}<{generic}>>` makes an extra allocation",
                    outer = outer_sym,
                    inner = inner_sym,
                    generic = generic_snippet
                ));
            },
        );
    } else {
        let generic_snippet = snippet(cx, inner_span, "..");
        span_lint_and_then(
            cx,
            REDUNDANT_ALLOCATION,
            hir_ty.span,
            &format!("usage of `{}<{}<{}>>`", outer_sym, inner_sym, generic_snippet),
            |diag| {
                diag.note(&format!(
                    "`{inner}<{generic}>` is already on the heap, `{outer}<{inner}<{generic}>>` makes an extra allocation",
                    outer = outer_sym,
                    inner = inner_sym,
                    generic = generic_snippet
                ));
                diag.help(&format!(
                    "consider using just `{outer}<{generic}>` or `{inner}<{generic}>`",
                    outer = outer_sym,
                    inner = inner_sym,
                    generic = generic_snippet
                ));
            },
        );
    }
    true
}

/// Returns `true` if the allocation would make `hir_ty` go from fat to thin.
fn alloc_makes_pointer_thin(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>) -> bool {
    match &hir_ty.kind {
        TyKind::TraitObject(..) => true,
        TyKind::Path(ty_qpath) => {
            let ty_res = cx.qpath_res(ty_qpath, hir_ty.hir_id);
            if let def::Res::PrimTy(prim_ty) = ty_res {
                if matches!(prim_ty, PrimTy::Str) {
                    return true;
                }
            }
            false
        },
        _ => false,
    }
}
