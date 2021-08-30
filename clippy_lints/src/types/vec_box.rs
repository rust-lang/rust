use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::last_path_segment;
use clippy_utils::source::snippet;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, def_id::DefId, GenericArg, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::TypeFoldable;
use rustc_span::symbol::sym;
use rustc_typeck::hir_ty_to_ty;

use super::VEC_BOX;

pub(super) fn check(
    cx: &LateContext<'_>,
    hir_ty: &hir::Ty<'_>,
    qpath: &QPath<'_>,
    def_id: DefId,
    box_size_threshold: u64,
) -> bool {
    if cx.tcx.is_diagnostic_item(sym::vec_type, def_id) {
        if_chain! {
            // Get the _ part of Vec<_>
            if let Some(last) = last_path_segment(qpath).args;
            if let Some(ty) = last.args.iter().find_map(|arg| match arg {
                GenericArg::Type(ty) => Some(ty),
                _ => None,
            });
            // ty is now _ at this point
            if let TyKind::Path(ref ty_qpath) = ty.kind;
            let res = cx.qpath_res(ty_qpath, ty.hir_id);
            if let Some(def_id) = res.opt_def_id();
            if Some(def_id) == cx.tcx.lang_items().owned_box();
            // At this point, we know ty is Box<T>, now get T
            if let Some(last) = last_path_segment(ty_qpath).args;
            if let Some(boxed_ty) = last.args.iter().find_map(|arg| match arg {
                GenericArg::Type(ty) => Some(ty),
                _ => None,
            });
            let ty_ty = hir_ty_to_ty(cx.tcx, boxed_ty);
            if !ty_ty.has_escaping_bound_vars();
            if ty_ty.is_sized(cx.tcx.at(ty.span), cx.param_env);
            if let Ok(ty_ty_size) = cx.layout_of(ty_ty).map(|l| l.size.bytes());
            if ty_ty_size <= box_size_threshold;
            then {
                span_lint_and_sugg(
                    cx,
                    VEC_BOX,
                    hir_ty.span,
                    "`Vec<T>` is already on the heap, the boxing is unnecessary",
                    "try",
                    format!("Vec<{}>", snippet(cx, boxed_ty.span, "..")),
                    Applicability::MachineApplicable,
                );
                true
            } else {
                false
            }
        }
    } else {
        false
    }
}
