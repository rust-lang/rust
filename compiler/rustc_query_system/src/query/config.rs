//! Query configuration and description traits.

use crate::dep_graph::{DepNode, DepNodeParams, SerializedDepNodeIndex};
use crate::error::HandleCycleError;
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::{QueryContext, QueryState};

use rustc_data_structures::fingerprint::Fingerprint;
use std::fmt::Debug;
use std::hash::Hash;

pub type HashResult<V> = Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>;

pub type TryLoadFromDisk<Qcx, V> = Option<fn(Qcx, SerializedDepNodeIndex) -> Option<V>>;

pub trait QueryConfig<Qcx: QueryContext>: Copy {
    fn name(self) -> &'static str;

    // `Key` and `Value` are `Copy` instead of `Clone` to ensure copying them stays cheap,
    // but it isn't necessary.
    type Key: DepNodeParams<Qcx::DepContext> + Eq + Hash + Copy + Debug;
    type Value: Debug + Copy;

    type Cache: QueryCache<Key = Self::Key, Value = Self::Value>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(self, tcx: Qcx) -> &'a QueryState<Self::Key, Qcx::DepKind>
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

    fn try_load_from_disk(self, qcx: Qcx, idx: &Self::Key) -> TryLoadFromDisk<Qcx, Self::Value>;

    fn anon(self) -> bool;
    fn eval_always(self) -> bool;
    fn depth_limit(self) -> bool;
    fn feedable(self) -> bool;

    fn dep_kind(self) -> Qcx::DepKind;
    fn handle_cycle_error(self) -> HandleCycleError;
    fn hash_result(self) -> HashResult<Self::Value>;

    // Just here for convernience and checking that the key matches the kind, don't override this.
    fn construct_dep_node(self, tcx: Qcx::DepContext, key: &Self::Key) -> DepNode<Qcx::DepKind> {
        DepNode::construct(tcx, self.dep_kind(), key)
    }
}
