use super::{inspect, CanonicalInput, QueryResult};
use crate::ty::TyCtxt;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lock;
use rustc_query_system::cache::WithDepNode;
use rustc_query_system::dep_graph::DepNodeIndex;
use rustc_session::Limit;
/// The trait solver cache used by `-Ztrait-solver=next`.
///
/// FIXME(@lcnr): link to some official documentation of how
/// this works.
#[derive(Default)]
pub struct EvaluationCache<'tcx> {
    map: Lock<FxHashMap<CanonicalInput<'tcx>, CacheEntry<'tcx>>>,
}

#[derive(PartialEq, Eq)]
pub struct CacheData<'tcx> {
    pub result: QueryResult<'tcx>,
    pub proof_tree: Option<&'tcx [inspect::GoalEvaluationStep<'tcx>]>,
    pub reached_depth: usize,
    pub encountered_overflow: bool,
}

impl<'tcx> EvaluationCache<'tcx> {
    /// Insert a final result into the global cache.
    pub fn insert(
        &self,
        tcx: TyCtxt<'tcx>,
        key: CanonicalInput<'tcx>,
        proof_tree: Option<&'tcx [inspect::GoalEvaluationStep<'tcx>]>,
        reached_depth: usize,
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
            entry.with_overflow.insert(reached_depth, data);
        } else {
            entry.success = Some(Success { data, reached_depth });
        }

        if cfg!(debug_assertions) {
            drop(map);
            if Some(CacheData { result, proof_tree, reached_depth, encountered_overflow })
                != self.get(tcx, key, |_| false, Limit(reached_depth))
            {
                bug!("unable to retrieve inserted element from cache: {key:?}");
            }
        }
    }

    /// Try to fetch a cached result, checking the recursion limit
    /// and handling root goals of coinductive cycles.
    ///
    /// If this returns `Some` the cache result can be used.
    pub fn get(
        &self,
        tcx: TyCtxt<'tcx>,
        key: CanonicalInput<'tcx>,
        cycle_participant_in_stack: impl FnOnce(&FxHashSet<CanonicalInput<'tcx>>) -> bool,
        available_depth: Limit,
    ) -> Option<CacheData<'tcx>> {
        let map = self.map.borrow();
        let entry = map.get(&key)?;

        if cycle_participant_in_stack(&entry.cycle_participants) {
            return None;
        }

        if let Some(ref success) = entry.success {
            if available_depth.value_within_limit(success.reached_depth) {
                let QueryData { result, proof_tree } = success.data.get(tcx);
                return Some(CacheData {
                    result,
                    proof_tree,
                    reached_depth: success.reached_depth,
                    encountered_overflow: false,
                });
            }
        }

        entry.with_overflow.get(&available_depth.0).map(|e| {
            let QueryData { result, proof_tree } = e.get(tcx);
            CacheData {
                result,
                proof_tree,
                reached_depth: available_depth.0,
                encountered_overflow: true,
            }
        })
    }
}

struct Success<'tcx> {
    data: WithDepNode<QueryData<'tcx>>,
    reached_depth: usize,
}

#[derive(Clone, Copy)]
pub struct QueryData<'tcx> {
    pub result: QueryResult<'tcx>,
    pub proof_tree: Option<&'tcx [inspect::GoalEvaluationStep<'tcx>]>,
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
