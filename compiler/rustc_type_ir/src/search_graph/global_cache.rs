use derive_where::derive_where;

use super::{AvailableDepth, Cx, NestedGoals};
use crate::data_structures::HashMap;
use crate::search_graph::EvaluationResult;

struct Success<X: Cx> {
    required_depth: usize,
    nested_goals: NestedGoals<X>,
    result: X::Tracked<X::Result>,
}

struct WithOverflow<X: Cx> {
    nested_goals: NestedGoals<X>,
    result: X::Tracked<X::Result>,
}

/// The cache entry for a given input.
///
/// This contains results whose computation never hit the
/// recursion limit in `success`, and all results which hit
/// the recursion limit in `with_overflow`.
#[derive_where(Default; X: Cx)]
struct CacheEntry<X: Cx> {
    success: Option<Success<X>>,
    with_overflow: HashMap<usize, WithOverflow<X>>,
}

#[derive_where(Debug; X: Cx)]
pub(super) struct CacheData<'a, X: Cx> {
    pub(super) result: X::Result,
    pub(super) required_depth: usize,
    pub(super) encountered_overflow: bool,
    pub(super) nested_goals: &'a NestedGoals<X>,
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
        evaluation_result: EvaluationResult<X>,
        dep_node: X::DepNodeIndex,
    ) {
        let EvaluationResult { encountered_overflow, required_depth, heads, nested_goals, result } =
            evaluation_result;
        debug_assert!(heads.is_empty());
        let result = cx.mk_tracked(result, dep_node);
        let entry = self.map.entry(input).or_default();
        if encountered_overflow {
            let with_overflow = WithOverflow { nested_goals, result };
            let prev = entry.with_overflow.insert(required_depth, with_overflow);
            if let Some(prev) = &prev {
                assert!(cx.evaluation_is_concurrent());
                assert_eq!(cx.get_tracked(&prev.result), evaluation_result.result);
            }
        } else {
            let prev = entry.success.replace(Success { required_depth, nested_goals, result });
            if let Some(prev) = &prev {
                assert!(cx.evaluation_is_concurrent());
                assert_eq!(cx.get_tracked(&prev.result), evaluation_result.result);
            }
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
        available_depth: AvailableDepth,
        mut candidate_is_applicable: impl FnMut(&NestedGoals<X>) -> bool,
    ) -> Option<CacheData<'a, X>> {
        let entry = self.map.get(&input)?;
        if let Some(Success { required_depth, ref nested_goals, ref result }) = entry.success {
            if available_depth.cache_entry_is_applicable(required_depth)
                && candidate_is_applicable(nested_goals)
            {
                return Some(CacheData {
                    result: cx.get_tracked(&result),
                    required_depth,
                    encountered_overflow: false,
                    nested_goals,
                });
            }
        }

        let additional_depth = available_depth.0;
        if let Some(WithOverflow { nested_goals, result }) =
            entry.with_overflow.get(&additional_depth)
        {
            if candidate_is_applicable(nested_goals) {
                return Some(CacheData {
                    result: cx.get_tracked(result),
                    required_depth: additional_depth,
                    encountered_overflow: true,
                    nested_goals,
                });
            }
        }

        None
    }
}
