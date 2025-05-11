use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::last_path_segment;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, GenericArg, LangItem, QPath, TyKind};
use rustc_hir_analysis::lower_ty;
use rustc_lint::LateContext;
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::layout::LayoutOf;
use rustc_span::symbol::sym;

use super::VEC_BOX;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    hir_ty: &hir::Ty<'_>,
    qpath: &QPath<'tcx>,
    def_id: DefId,
    box_size_threshold: u64,
) -> bool {
    if cx.tcx.is_diagnostic_item(sym::Vec, def_id)
        && let Some(last) = last_path_segment(qpath).args
        // Get the _ part of Vec<_>
        && let Some(GenericArg::Type(ty)) = last.args.first()
        // extract allocator from the Vec for later
        && let vec_alloc_ty = last.args.get(1)
        // ty is now _ at this point
        && let TyKind::Path(ref ty_qpath) = ty.kind
        && let res = cx.qpath_res(ty_qpath, ty.hir_id)
        && let Some(def_id) = res.opt_def_id()
        && Some(def_id) == cx.tcx.lang_items().owned_box()
        // At this point, we know ty is Box<T>, now get T
        && let Some(last) = last_path_segment(ty_qpath).args
        && let Some(GenericArg::Type(boxed_ty)) = last.args.first()
        // extract allocator from the Box for later
        && let boxed_alloc_ty = last.args.get(1)
        // we don't expect to encounter `_` here so ignore `GenericArg::Infer` is okay
        && let ty_ty = lower_ty(cx.tcx, boxed_ty.as_unambig_ty())
        && !ty_ty.has_escaping_bound_vars()
        && ty_ty.is_sized(cx.tcx, cx.typing_env())
        && let Ok(ty_ty_size) = cx.layout_of(ty_ty).map(|l| l.size.bytes())
        && ty_ty_size < box_size_threshold
        // https://github.com/rust-lang/rust-clippy/issues/7114
        && match (vec_alloc_ty, boxed_alloc_ty) {
            (None, None) => true,
            // this is in the event that we have something like
            // Vec<_, Global>, in which case is equivalent to
            // Vec<_>
            (None, Some(GenericArg::Type(inner))) | (Some(GenericArg::Type(inner)), None) => {
                if let TyKind::Path(path) = inner.kind
                    && let Some(did) = cx.qpath_res(&path, inner.hir_id).opt_def_id() {
                    cx.tcx.lang_items().get(LangItem::GlobalAlloc) == Some(did)
                } else {
                    false
                }
            },
            (Some(GenericArg::Type(l)), Some(GenericArg::Type(r))) =>
                // we don't expect to encounter `_` here so ignore `GenericArg::Infer` is okay
                lower_ty(cx.tcx, l.as_unambig_ty()) == lower_ty(cx.tcx, r.as_unambig_ty()),
            _ => false
        }
    {
        span_lint_and_sugg(
            cx,
            VEC_BOX,
            hir_ty.span,
            "`Vec<T>` is already on the heap, the boxing is unnecessary",
            "try",
            format!("Vec<{}>", snippet(cx, boxed_ty.span, "..")),
            Applicability::Unspecified,
        );
        true
    } else {
        false
    }
}
