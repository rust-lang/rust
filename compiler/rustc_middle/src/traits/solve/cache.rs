use super::{inspect, CanonicalInput, QueryResult};
use crate::ty::TyCtxt;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lock;
use rustc_query_system::cache::WithDepNode;
use rustc_query_system::dep_graph::DepNodeIndex;
use rustc_session::Limit;
use rustc_type_ir::solve::CacheData;

/// The trait solver cache used by `-Znext-solver`.
///
/// FIXME(@lcnr): link to some official documentation of how
/// this works.
#[derive(Default)]
pub struct EvaluationCache<'tcx> {
    map: Lock<FxHashMap<CanonicalInput<'tcx>, CacheEntry<'tcx>>>,
}

impl<'tcx> rustc_type_ir::inherent::EvaluationCache<TyCtxt<'tcx>> for &'tcx EvaluationCache<'tcx> {
    /// Insert a final result into the global cache.
    fn insert(
        &self,
        tcx: TyCtxt<'tcx>,
        key: CanonicalInput<'tcx>,
        proof_tree: Option<&'tcx inspect::CanonicalGoalEvaluationStep<TyCtxt<'tcx>>>,
        additional_depth: usize,
        encountered_overflow: bool,
        cycle_participants: FxHashSet<CanonicalInput<'tcx>>,
        dep_node: DepNodeIndex,
        result: QueryResult<'tcx>,
    ) {
        let mut map = self.map.borrow_mut();
        let entry = map.entry(key).or_default();
        let data = WithDepNode::new(dep_node, QueryData { result, proof_tree });
        entry.cycle_participants.extend(cycle_participants);
        if encountered_overflow {
            entry.with_overflow.insert(additional_depth, data);
        } else {
            entry.success = Some(Success { data, additional_depth });
        }

        if cfg!(debug_assertions) {
            drop(map);
            let expected = CacheData { result, proof_tree, additional_depth, encountered_overflow };
            let actual = self.get(tcx, key, [], additional_depth);
            if !actual.as_ref().is_some_and(|actual| expected == *actual) {
                bug!("failed to lookup inserted element for {key:?}: {expected:?} != {actual:?}");
            }
        }
    }

    /// Try to fetch a cached result, checking the recursion limit
    /// and handling root goals of coinductive cycles.
    ///
    /// If this returns `Some` the cache result can be used.
    fn get(
        &self,
        tcx: TyCtxt<'tcx>,
        key: CanonicalInput<'tcx>,
        stack_entries: impl IntoIterator<Item = CanonicalInput<'tcx>>,
        available_depth: usize,
    ) -> Option<CacheData<TyCtxt<'tcx>>> {
        let map = self.map.borrow();
        let entry = map.get(&key)?;

        for stack_entry in stack_entries {
            if entry.cycle_participants.contains(&stack_entry) {
                return None;
            }
        }

        if let Some(ref success) = entry.success {
            if Limit(available_depth).value_within_limit(success.additional_depth) {
                let QueryData { result, proof_tree } = success.data.get(tcx);
                return Some(CacheData {
                    result,
                    proof_tree,
                    additional_depth: success.additional_depth,
                    encountered_overflow: false,
                });
            }
        }

        entry.with_overflow.get(&available_depth).map(|e| {
            let QueryData { result, proof_tree } = e.get(tcx);
            CacheData {
                result,
                proof_tree,
                additional_depth: available_depth,
                encountered_overflow: true,
            }
        })
    }
}

struct Success<'tcx> {
    data: WithDepNode<QueryData<'tcx>>,
    additional_depth: usize,
}

#[derive(Clone, Copy)]
pub struct QueryData<'tcx> {
    pub result: QueryResult<'tcx>,
    pub proof_tree: Option<&'tcx inspect::CanonicalGoalEvaluationStep<TyCtxt<'tcx>>>,
}

/// The cache entry for a goal `CanonicalInput`.
///
/// This contains results whose computation never hit the
/// recursion limit in `success`, and all results which hit
/// the recursion limit in `with_overflow`.
#[derive(Default)]
struct CacheEntry<'tcx> {
    success: Option<Success<'tcx>>,
    /// We have to be careful when caching roots of cycles.
    ///
    /// See the doc comment of `StackEntry::cycle_participants` for more
    /// details.
    cycle_participants: FxHashSet<CanonicalInput<'tcx>>,
    with_overflow: FxHashMap<usize, WithDepNode<QueryData<'tcx>>>,
}
