use rustc_errors::Applicability;
use rustc_hir::{self as hir, def_id::DefId, LangItem, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use crate::utils::{
    get_qpath_generic_tys, is_ty_param_diagnostic_item, is_ty_param_lang_item, snippet_with_applicability,
    span_lint_and_sugg,
};

use super::{utils, REDUNDANT_ALLOCATION};

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) {
    if Some(def_id) == cx.tcx.lang_items().owned_box() {
        if let Some(span) = utils::match_borrows_parameter(cx, qpath) {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                REDUNDANT_ALLOCATION,
                hir_ty.span,
                "usage of `Box<&T>`",
                "try",
                snippet_with_applicability(cx, span, "..", &mut applicability).to_string(),
                applicability,
            );
            return;
        }
    }

    if cx.tcx.is_diagnostic_item(sym::Rc, def_id) {
        if let Some(ty) = is_ty_param_diagnostic_item(cx, qpath, sym::Rc) {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                REDUNDANT_ALLOCATION,
                hir_ty.span,
                "usage of `Rc<Rc<T>>`",
                "try",
                snippet_with_applicability(cx, ty.span, "..", &mut applicability).to_string(),
                applicability,
            );
        } else if let Some(ty) = is_ty_param_lang_item(cx, qpath, LangItem::OwnedBox) {
            let qpath = match &ty.kind {
                TyKind::Path(qpath) => qpath,
                _ => return,
            };
            let inner_span = match get_qpath_generic_tys(qpath).next() {
                Some(ty) => ty.span,
                None => return,
            };
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                REDUNDANT_ALLOCATION,
                hir_ty.span,
                "usage of `Rc<Box<T>>`",
                "try",
                format!(
                    "Rc<{}>",
                    snippet_with_applicability(cx, inner_span, "..", &mut applicability)
                ),
                applicability,
            );
        } else if let Some(span) = utils::match_borrows_parameter(cx, qpath) {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                REDUNDANT_ALLOCATION,
                hir_ty.span,
                "usage of `Rc<&T>`",
                "try",
                snippet_with_applicability(cx, span, "..", &mut applicability).to_string(),
                applicability,
            );
            return; // don't recurse into the type
        }
    }
}
