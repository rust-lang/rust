use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::{get_qpath_generic_tys, is_ty_param_diagnostic_item, is_ty_param_lang_item};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, def_id::DefId, LangItem, QPath, TyKind};
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

    let (inner_sym, ty) = if let Some(ty) = is_ty_param_lang_item(cx, qpath, LangItem::OwnedBox) {
        ("Box", ty)
    } else if let Some(ty) = is_ty_param_diagnostic_item(cx, qpath, sym::Rc) {
        ("Rc", ty)
    } else if let Some(ty) = is_ty_param_diagnostic_item(cx, qpath, sym::Arc) {
        ("Arc", ty)
    } else {
        return false;
    };

    let inner_qpath = match &ty.kind {
        TyKind::Path(inner_qpath) => inner_qpath,
        _ => return false,
    };
    let inner_span = match get_qpath_generic_tys(inner_qpath).next() {
        Some(ty) => {
            // Box<Box<dyn T>> is smaller than Box<dyn T> because of wide pointers
            if matches!(ty.kind, TyKind::TraitObject(..)) {
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
