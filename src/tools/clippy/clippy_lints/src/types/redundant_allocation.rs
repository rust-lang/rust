use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::{path_def_id, qpath_generic_tys};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, QPath, TyKind};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::LateContext;
use rustc_middle::ty::TypeVisitableExt;
use rustc_span::symbol::sym;

use super::{utils, REDUNDANT_ALLOCATION};

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    let mut applicability = Applicability::MaybeIncorrect;
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
        let generic_snippet = snippet_with_applicability(cx, span, "..", &mut applicability);
        span_lint_and_then(
            cx,
            REDUNDANT_ALLOCATION,
            hir_ty.span,
            &format!("usage of `{outer_sym}<{generic_snippet}>`"),
            |diag| {
                diag.span_suggestion(hir_ty.span, "try", format!("{generic_snippet}"), applicability);
                diag.note(format!(
                    "`{generic_snippet}` is already a pointer, `{outer_sym}<{generic_snippet}>` allocates a pointer on the heap"
                ));
            },
        );
        return true;
    }

    let Some(ty) = qpath_generic_tys(qpath).next() else {
        return false;
    };
    let Some(id) = path_def_id(cx, ty) else { return false };
    let (inner_sym, ty) = match cx.tcx.get_diagnostic_name(id) {
        Some(sym::Arc) => ("Arc", ty),
        Some(sym::Rc) => ("Rc", ty),
        _ if Some(id) == cx.tcx.lang_items().owned_box() => ("Box", ty),
        _ => return false,
    };

    let TyKind::Path(inner_qpath) = &ty.kind else {
        return false;
    };
    let inner_span = match qpath_generic_tys(inner_qpath).next() {
        Some(hir_ty) => {
            // Reallocation of a fat pointer causes it to become thin. `hir_ty_to_ty` is safe to use
            // here because `mod.rs` guarantees this lint is only run on types outside of bodies and
            // is not run on locals.
            let ty = hir_ty_to_ty(cx.tcx, hir_ty);
            if ty.has_escaping_bound_vars() || !ty.is_sized(cx.tcx, cx.param_env) {
                return false;
            }
            hir_ty.span
        },
        None => return false,
    };
    if inner_sym == outer_sym {
        let generic_snippet = snippet_with_applicability(cx, inner_span, "..", &mut applicability);
        span_lint_and_then(
            cx,
            REDUNDANT_ALLOCATION,
            hir_ty.span,
            &format!("usage of `{outer_sym}<{inner_sym}<{generic_snippet}>>`"),
            |diag| {
                diag.span_suggestion(
                    hir_ty.span,
                    "try",
                    format!("{outer_sym}<{generic_snippet}>"),
                    applicability,
                );
                diag.note(format!(
                    "`{inner_sym}<{generic_snippet}>` is already on the heap, `{outer_sym}<{inner_sym}<{generic_snippet}>>` makes an extra allocation"
                ));
            },
        );
    } else {
        let generic_snippet = snippet(cx, inner_span, "..");
        span_lint_and_then(
            cx,
            REDUNDANT_ALLOCATION,
            hir_ty.span,
            &format!("usage of `{outer_sym}<{inner_sym}<{generic_snippet}>>`"),
            |diag| {
                diag.note(format!(
                    "`{inner_sym}<{generic_snippet}>` is already on the heap, `{outer_sym}<{inner_sym}<{generic_snippet}>>` makes an extra allocation"
                ));
                diag.help(format!(
                    "consider using just `{outer_sym}<{generic_snippet}>` or `{inner_sym}<{generic_snippet}>`"
                ));
            },
        );
    }
    true
}
