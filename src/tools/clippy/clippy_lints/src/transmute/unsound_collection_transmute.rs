use super::utils::is_layout_incompatible;
use super::UNSOUND_COLLECTION_TRANSMUTE;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::match_any_diagnostic_items;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::{sym, Symbol};

// used to check for UNSOUND_COLLECTION_TRANSMUTE
static COLLECTIONS: &[Symbol] = &[
    sym::Vec,
    sym::VecDeque,
    sym::BinaryHeap,
    sym::BTreeSet,
    sym::BTreeMap,
    sym::HashSet,
    sym::HashMap,
];

/// Checks for `unsound_collection_transmute` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, from_ty: Ty<'tcx>, to_ty: Ty<'tcx>) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Adt(from_adt, from_substs), ty::Adt(to_adt, to_substs)) => {
            if from_adt.did != to_adt.did || match_any_diagnostic_items(cx, to_adt.did, COLLECTIONS).is_none() {
                return false;
            }
            if from_substs
                .types()
                .zip(to_substs.types())
                .any(|(from_ty, to_ty)| is_layout_incompatible(cx, from_ty, to_ty))
            {
                span_lint(
                    cx,
                    UNSOUND_COLLECTION_TRANSMUTE,
                    e.span,
                    &format!(
                        "transmute from `{}` to `{}` with mismatched layout is unsound",
                        from_ty, to_ty
                    ),
                );
                true
            } else {
                false
            }
        },
        _ => false,
    }
}
