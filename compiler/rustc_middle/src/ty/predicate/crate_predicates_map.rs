use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_span::Span;

use crate::ty::Clause;

/// The crate outlives map is computed during typeck and contains the
/// outlives of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.inferred_outlives_of()` to get the outlives for a *particular*
/// item.
#[derive(HashStable, Debug)]
pub struct CratePredicatesMap<'tcx> {
    /// For each struct with outlive bounds, maps to a vector of the
    /// predicate of its outlive bounds. If an item has no outlives
    /// bounds, it will have no entry.
    pub predicates: FxHashMap<DefId, &'tcx [(Clause<'tcx>, Span)]>,
}
