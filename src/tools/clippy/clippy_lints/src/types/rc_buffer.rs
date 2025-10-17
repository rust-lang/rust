use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{path_def_id, qpath_generic_tys};
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::RC_BUFFER;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    let app = Applicability::Unspecified;
    let name = cx.tcx.get_diagnostic_name(def_id);
    if name == Some(sym::Rc) {
        if let Some(alternate) = match_buffer_type(cx, qpath) {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Rc<T>` when T is a buffer type",
                |diag| {
                    diag.span_suggestion(hir_ty.span, "try", format!("Rc<{alternate}>"), app);
                },
            );
        } else {
            let Some(ty) = qpath_generic_tys(qpath).next() else {
                return false;
            };
            let Some(id) = path_def_id(cx, ty) else { return false };
            if !cx.tcx.is_diagnostic_item(sym::Vec, id) {
                return false;
            }
            let TyKind::Path(qpath) = &ty.kind else { return false };
            let inner_span = match qpath_generic_tys(qpath).next() {
                Some(ty) => ty.span,
                None => return false,
            };
            span_lint_and_then(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Rc<T>` when T is a buffer type",
                |diag| {
                    let mut applicability = app;
                    diag.span_suggestion(
                        hir_ty.span,
                        "try",
                        format!(
                            "Rc<[{}]>",
                            snippet_with_applicability(cx, inner_span, "..", &mut applicability)
                        ),
                        app,
                    );
                },
            );
            return true;
        }
    } else if name == Some(sym::Arc) {
        if let Some(alternate) = match_buffer_type(cx, qpath) {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Arc<T>` when T is a buffer type",
                |diag| {
                    diag.span_suggestion(hir_ty.span, "try", format!("Arc<{alternate}>"), app);
                },
            );
        } else if let Some(ty) = qpath_generic_tys(qpath).next() {
            let Some(id) = path_def_id(cx, ty) else { return false };
            if !cx.tcx.is_diagnostic_item(sym::Vec, id) {
                return false;
            }
            let TyKind::Path(qpath) = &ty.kind else { return false };
            let inner_span = match qpath_generic_tys(qpath).next() {
                Some(ty) => ty.span,
                None => return false,
            };
            span_lint_and_then(
                cx,
                RC_BUFFER,
                hir_ty.span,
                "usage of `Arc<T>` when T is a buffer type",
                |diag| {
                    let mut applicability = app;
                    diag.span_suggestion(
                        hir_ty.span,
                        "try",
                        format!(
                            "Arc<[{}]>",
                            snippet_with_applicability(cx, inner_span, "..", &mut applicability)
                        ),
                        app,
                    );
                },
            );
            return true;
        }
    }

    false
}

fn match_buffer_type(cx: &LateContext<'_>, qpath: &QPath<'_>) -> Option<&'static str> {
    let ty = qpath_generic_tys(qpath).next()?;
    let id = path_def_id(cx, ty)?;
    let path = match cx.tcx.get_diagnostic_name(id) {
        Some(sym::OsString) => "std::ffi::OsStr",
        Some(sym::PathBuf) => "std::path::Path",
        _ if Some(id) == cx.tcx.lang_items().string() => "str",
        _ => return None,
    };
    Some(path)
}
