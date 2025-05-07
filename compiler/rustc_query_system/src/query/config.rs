//! Query configuration and description traits.

use std::fmt::Debug;
use std::hash::Hash;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_span::ErrorGuaranteed;

use super::QueryStackFrameExtra;
use crate::dep_graph::{DepKind, DepNode, DepNodeParams, SerializedDepNodeIndex};
use crate::error::HandleCycleError;
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::{CycleError, DepNodeIndex, QueryContext, QueryState};

pub type HashResult<V> = Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>;

pub trait QueryConfig<Qcx: QueryContext>: Copy {
    fn name(self) -> &'static str;

    // `Key` and `Value` are `Copy` instead of `Clone` to ensure copying them stays cheap,
    // but it isn't necessary.
    type Key: DepNodeParams<Qcx::DepContext> + Eq + Hash + Copy + Debug;
    type Value: Copy;

    type Cache: QueryCache<Key = Self::Key, Value = Self::Value>;

    fn format_value(self) -> fn(&Self::Value) -> String;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(self, tcx: Qcx) -> &'a QueryState<Self::Key, Qcx::QueryInfo>
    where
        Qcx: 'a;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(self, tcx: Qcx) -> &'a Self::Cache
    where
        Qcx: 'a;

    fn cache_on_disk(self, tcx: Qcx::DepContext, key: &Self::Key) -> bool;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn execute_query(self, tcx: Qcx::DepContext, k: Self::Key) -> Self::Value;

    fn compute(self, tcx: Qcx, key: Self::Key) -> Self::Value;

    fn try_load_from_disk(
        self,
        tcx: Qcx,
        key: &Self::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<Self::Value>;

    fn loadable_from_disk(self, qcx: Qcx, key: &Self::Key, idx: SerializedDepNodeIndex) -> bool;

    /// Synthesize an error value to let compilation continue after a cycle.
    fn value_from_cycle_error(
        self,
        tcx: Qcx::DepContext,
        cycle_error: &CycleError<QueryStackFrameExtra>,
        guar: ErrorGuaranteed,
    ) -> Self::Value;

    fn anon(self) -> bool;
    fn eval_always(self) -> bool;
    fn depth_limit(self) -> bool;
    fn feedable(self) -> bool;

    fn dep_kind(self) -> DepKind;
    fn handle_cycle_error(self) -> HandleCycleError;
    fn hash_result(self) -> HashResult<Self::Value>;

    // Just here for convenience and checking that the key matches the kind, don't override this.
    fn construct_dep_node(self, tcx: Qcx::DepContext, key: &Self::Key) -> DepNode {
        DepNode::construct(tcx, self.dep_kind(), key)
    }
}
