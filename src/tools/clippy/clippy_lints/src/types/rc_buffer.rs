use rustc_errors::Applicability;
use rustc_hir::{self as hir, def_id::DefId, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use crate::utils::{
    get_qpath_generic_tys, is_ty_param_diagnostic_item, snippet_with_applicability, span_lint_and_sugg,
};

use super::RC_BUFFER;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if cx.tcx.is_diagnostic_item(sym::Rc, def_id) {
        if let Some(alternate) = match_buffer_type(cx, qpath) {
            span_lint_and_sugg(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Rc<T>` when T is a buffer type",
                "try",
                format!("Rc<{}>", alternate),
                Applicability::MachineApplicable,
            );
        } else if let Some(ty) = is_ty_param_diagnostic_item(cx, qpath, sym::vec_type) {
            let qpath = match &ty.kind {
                TyKind::Path(qpath) => qpath,
                _ => return false,
            };
            let inner_span = match get_qpath_generic_tys(qpath).next() {
                Some(ty) => ty.span,
                None => return false,
            };
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Rc<T>` when T is a buffer type",
                "try",
                format!(
                    "Rc<[{}]>",
                    snippet_with_applicability(cx, inner_span, "..", &mut applicability)
                ),
                Applicability::MachineApplicable,
            );
            return true;
        }
    } else if cx.tcx.is_diagnostic_item(sym::Arc, def_id) {
        if let Some(alternate) = match_buffer_type(cx, qpath) {
            span_lint_and_sugg(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Arc<T>` when T is a buffer type",
                "try",
                format!("Arc<{}>", alternate),
                Applicability::MachineApplicable,
            );
        } else if let Some(ty) = is_ty_param_diagnostic_item(cx, qpath, sym::vec_type) {
            let qpath = match &ty.kind {
                TyKind::Path(qpath) => qpath,
                _ => return false,
            };
            let inner_span = match get_qpath_generic_tys(qpath).next() {
                Some(ty) => ty.span,
                None => return false,
            };
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Arc<T>` when T is a buffer type",
                "try",
                format!(
                    "Arc<[{}]>",
                    snippet_with_applicability(cx, inner_span, "..", &mut applicability)
                ),
                Applicability::MachineApplicable,
            );
            return true;
        }
    }

    false
}

fn match_buffer_type(cx: &LateContext<'_>, qpath: &QPath<'_>) -> Option<&'static str> {
    if is_ty_param_diagnostic_item(cx, qpath, sym::string_type).is_some() {
        Some("str")
    } else if is_ty_param_diagnostic_item(cx, qpath, sym::OsString).is_some() {
        Some("std::ffi::OsStr")
    } else if is_ty_param_diagnostic_item(cx, qpath, sym::PathBuf).is_some() {
        Some("std::path::Path")
    } else {
        None
    }
}
