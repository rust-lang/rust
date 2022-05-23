use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{path_def_id, qpath_generic_tys};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, def_id::DefId, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

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
        } else {
            let Some(ty) = qpath_generic_tys(qpath).next() else { return false };
            let Some(id) = path_def_id(cx, ty) else { return false };
            if !cx.tcx.is_diagnostic_item(sym::Vec, id) {
                return false;
            }
            let qpath = match &ty.kind {
                TyKind::Path(qpath) => qpath,
                _ => return false,
            };
            let inner_span = match qpath_generic_tys(qpath).next() {
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
        } else if let Some(ty) = qpath_generic_tys(qpath).next() {
            let Some(id) = path_def_id(cx, ty) else { return false };
            if !cx.tcx.is_diagnostic_item(sym::Vec, id) {
                return false;
            }
            let qpath = match &ty.kind {
                TyKind::Path(qpath) => qpath,
                _ => return false,
            };
            let inner_span = match qpath_generic_tys(qpath).next() {
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
    let ty = qpath_generic_tys(qpath).next()?;
    let id = path_def_id(cx, ty)?;
    let path = match cx.tcx.get_diagnostic_name(id)? {
        sym::String => "str",
        sym::OsString => "std::ffi::OsStr",
        sym::PathBuf => "std::path::Path",
        _ => return None,
    };
    Some(path)
}
