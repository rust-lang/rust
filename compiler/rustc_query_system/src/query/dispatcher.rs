use std::fmt::Debug;
use std::hash::Hash;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_span::ErrorGuaranteed;

use super::QueryStackFrameExtra;
use crate::dep_graph::{DepKind, DepNode, DepNodeParams, HasDepContext, SerializedDepNodeIndex};
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::{CycleError, CycleErrorHandling, DepNodeIndex, QueryContext, QueryState};

pub type HashResult<V> = Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>;

/// Unambiguous shorthand for `<This::Qcx as HasDepContext>::DepContext`.
#[expect(type_alias_bounds)]
type DepContextOf<This: QueryDispatcher> =
    <<This as QueryDispatcher>::Qcx as HasDepContext>::DepContext;

/// Trait that can be used as a vtable for a single query, providing operations
/// and metadata for that query.
///
/// Implemented by `rustc_query_impl::SemiDynamicQueryDispatcher`, which
/// mostly delegates to `rustc_middle::query::plumbing::QueryVTable`.
/// Those types are not visible from this `rustc_query_system` crate.
///
/// "Dispatcher" should be understood as a near-synonym of "vtable".
pub trait QueryDispatcher: Copy {
    fn name(self) -> &'static str;

    /// Query context used by this dispatcher, i.e. `rustc_query_impl::QueryCtxt`.
    type Qcx: QueryContext;

    // `Key` and `Value` are `Copy` instead of `Clone` to ensure copying them stays cheap,
    // but it isn't necessary.
    type Key: DepNodeParams<DepContextOf<Self>> + Eq + Hash + Copy + Debug;
    type Value: Copy;

    type Cache: QueryCache<Key = Self::Key, Value = Self::Value>;

    fn format_value(self) -> fn(&Self::Value) -> String;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(
        self,
        tcx: Self::Qcx,
    ) -> &'a QueryState<Self::Key, <Self::Qcx as QueryContext>::QueryInfo>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(self, tcx: Self::Qcx) -> &'a Self::Cache;

    fn cache_on_disk(self, tcx: DepContextOf<Self>, key: &Self::Key) -> bool;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn execute_query(self, tcx: DepContextOf<Self>, k: Self::Key) -> Self::Value;

    fn compute(self, tcx: Self::Qcx, key: Self::Key) -> Self::Value;

    fn try_load_from_disk(
        self,
        tcx: Self::Qcx,
        key: &Self::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<Self::Value>;

    fn loadable_from_disk(
        self,
        qcx: Self::Qcx,
        key: &Self::Key,
        idx: SerializedDepNodeIndex,
    ) -> bool;

    /// Synthesize an error value to let compilation continue after a cycle.
    fn value_from_cycle_error(
        self,
        tcx: DepContextOf<Self>,
        cycle_error: &CycleError<QueryStackFrameExtra>,
        guar: ErrorGuaranteed,
    ) -> Self::Value;

    fn anon(self) -> bool;
    fn eval_always(self) -> bool;
    fn depth_limit(self) -> bool;
    fn feedable(self) -> bool;

    fn dep_kind(self) -> DepKind;
    fn cycle_error_handling(self) -> CycleErrorHandling;
    fn hash_result(self) -> HashResult<Self::Value>;

    // Just here for convenience and checking that the key matches the kind, don't override this.
    fn construct_dep_node(self, tcx: DepContextOf<Self>, key: &Self::Key) -> DepNode {
        DepNode::construct(tcx, self.dep_kind(), key)
    }
}
