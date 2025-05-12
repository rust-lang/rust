use super::UNSOUND_COLLECTION_TRANSMUTE;
use super::utils::is_layout_incompatible;
use clippy_utils::diagnostics::span_lint;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;

/// Checks for `unsound_collection_transmute` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, from_ty: Ty<'tcx>, to_ty: Ty<'tcx>) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Adt(from_adt, from_args), ty::Adt(to_adt, to_args)) => {
            if from_adt.did() != to_adt.did() {
                return false;
            }
            if !matches!(
                cx.tcx.get_diagnostic_name(to_adt.did()),
                Some(
                    sym::BTreeMap
                        | sym::BTreeSet
                        | sym::BinaryHeap
                        | sym::HashMap
                        | sym::HashSet
                        | sym::Vec
                        | sym::VecDeque
                )
            ) {
                return false;
            }
            if from_args
                .types()
                .zip(to_args.types())
                .any(|(from_ty, to_ty)| is_layout_incompatible(cx, from_ty, to_ty))
            {
                span_lint(
                    cx,
                    UNSOUND_COLLECTION_TRANSMUTE,
                    e.span,
                    format!("transmute from `{from_ty}` to `{to_ty}` with mismatched layout is unsound"),
                );
                true
            } else {
                false
            }
        },
        _ => false,
    }
}
