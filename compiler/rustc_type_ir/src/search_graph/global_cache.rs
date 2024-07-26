use derive_where::derive_where;
use rustc_index::IndexVec;

use super::{AvailableDepth, Cx, StackDepth, StackEntry};
use crate::data_structures::{HashMap, HashSet};

#[derive_where(Debug, Clone, Copy; X: Cx)]
struct QueryData<X: Cx> {
    result: X::Result,
    proof_tree: X::ProofTree,
}

struct Success<X: Cx> {
    data: X::Tracked<QueryData<X>>,
    additional_depth: usize,
}

/// The cache entry for a given input.
///
/// This contains results whose computation never hit the
/// recursion limit in `success`, and all results which hit
/// the recursion limit in `with_overflow`.
#[derive_where(Default; X: Cx)]
struct CacheEntry<X: Cx> {
    success: Option<Success<X>>,
    /// We have to be careful when caching roots of cycles.
    ///
    /// See the doc comment of `StackEntry::cycle_participants` for more
    /// details.
    nested_goals: HashSet<X::Input>,
    with_overflow: HashMap<usize, X::Tracked<QueryData<X>>>,
}

#[derive_where(Debug; X: Cx)]
pub(super) struct CacheData<'a, X: Cx> {
    pub(super) result: X::Result,
    pub(super) proof_tree: X::ProofTree,
    pub(super) additional_depth: usize,
    pub(super) encountered_overflow: bool,
    // FIXME: This is currently unused, but impacts the design
    // by requiring a closure for `Cx::with_global_cache`.
    #[allow(dead_code)]
    pub(super) nested_goals: &'a HashSet<X::Input>,
}
#[derive_where(Default; X: Cx)]
pub struct GlobalCache<X: Cx> {
    map: HashMap<X::Input, CacheEntry<X>>,
}

impl<X: Cx> GlobalCache<X> {
    /// Insert a final result into the global cache.
    pub(super) fn insert(
        &mut self,
        cx: X,
        input: X::Input,

        result: X::Result,
        proof_tree: X::ProofTree,
        dep_node: X::DepNodeIndex,

        additional_depth: usize,
        encountered_overflow: bool,
        nested_goals: &HashSet<X::Input>,
    ) {
        let data = cx.mk_tracked(QueryData { result, proof_tree }, dep_node);
        let entry = self.map.entry(input).or_default();
        entry.nested_goals.extend(nested_goals);
        if encountered_overflow {
            entry.with_overflow.insert(additional_depth, data);
        } else {
            entry.success = Some(Success { data, additional_depth });
        }
    }

    /// Try to fetch a cached result, checking the recursion limit
    /// and handling root goals of coinductive cycles.
    ///
    /// If this returns `Some` the cache result can be used.
    pub(super) fn get<'a>(
        &'a self,
        cx: X,
        input: X::Input,
        stack: &IndexVec<StackDepth, StackEntry<X>>,
        available_depth: AvailableDepth,
    ) -> Option<CacheData<'a, X>> {
        let entry = self.map.get(&input)?;
        if stack.iter().any(|e| entry.nested_goals.contains(&e.input)) {
            return None;
        }

        if let Some(ref success) = entry.success {
            if available_depth.cache_entry_is_applicable(success.additional_depth) {
                let QueryData { result, proof_tree } = cx.get_tracked(&success.data);
                return Some(CacheData {
                    result,
                    proof_tree,
                    additional_depth: success.additional_depth,
                    encountered_overflow: false,
                    nested_goals: &entry.nested_goals,
                });
            }
        }

        entry.with_overflow.get(&available_depth.0).map(|e| {
            let QueryData { result, proof_tree } = cx.get_tracked(e);
            CacheData {
                result,
                proof_tree,
                additional_depth: available_depth.0,
                encountered_overflow: true,
                nested_goals: &entry.nested_goals,
            }
        })
    }
}
