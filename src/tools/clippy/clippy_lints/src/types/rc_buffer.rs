use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::qpath_generic_tys;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{QPath, Ty, TyKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use std::borrow::Cow;

use super::RC_BUFFER;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    let mut app = Applicability::Unspecified;
    let rc = match cx.tcx.get_diagnostic_name(def_id) {
        Some(sym::Rc) => "Rc",
        Some(sym::Arc) => "Arc",
        _ => return false,
    };
    if let Some(ty) = qpath_generic_tys(qpath).next()
        && let Some(alternate) = match_buffer_type(cx, ty, &mut app)
    {
        span_lint_and_then(
            cx,
            RC_BUFFER,
            hir_ty.span,
            format!("usage of `{rc}<T>` when `T` is a buffer type"),
            |diag| {
                diag.span_suggestion_verbose(ty.span, "try", alternate, app);
            },
        );
        true
    } else {
        false
    }
}

fn match_buffer_type(
    cx: &LateContext<'_>,
    ty: &Ty<'_>,
    applicability: &mut Applicability,
) -> Option<Cow<'static, str>> {
    let id = ty.basic_res().opt_def_id()?;
    let path = match cx.tcx.get_diagnostic_name(id) {
        Some(sym::OsString) => "std::ffi::OsStr".into(),
        Some(sym::PathBuf) => "std::path::Path".into(),
        Some(sym::Vec) => {
            let TyKind::Path(vec_qpath) = &ty.kind else {
                return None;
            };
            let vec_generic_ty = qpath_generic_tys(vec_qpath).next()?;
            let snippet = snippet_with_applicability(cx, vec_generic_ty.span, "_", applicability);
            format!("[{snippet}]").into()
        },
        _ if Some(id) == cx.tcx.lang_items().string() => "str".into(),
        _ => return None,
    };
    Some(path)
}
