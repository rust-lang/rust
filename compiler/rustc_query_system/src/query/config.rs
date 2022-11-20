//! Query configuration and description traits.

use crate::dep_graph::DepNode;
use crate::dep_graph::SerializedDepNodeIndex;
use crate::error::HandleCycleError;
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::{QueryContext, QueryState};

use rustc_data_structures::fingerprint::Fingerprint;
use std::fmt::Debug;
use std::hash::Hash;

pub trait QueryConfig<Qcx: QueryContext> {
    const NAME: &'static str;

    type Key: Eq + Hash + Clone + Debug;
    type Value;
    type Stored: Clone;

    type Cache: QueryCache<Key = Self::Key, Stored = Self::Stored, Value = Self::Value>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: Qcx) -> &'a QueryState<Self::Key>
    where
        Qcx: 'a;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: Qcx) -> &'a Self::Cache
    where
        Qcx: 'a;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn make_vtable(tcx: Qcx, key: &Self::Key) -> QueryVTable<Qcx, Self::Key, Self::Value>;

    fn cache_on_disk(tcx: Qcx::DepContext, key: &Self::Key) -> bool;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn execute_query(tcx: Qcx::DepContext, k: Self::Key) -> Self::Stored;
}

#[derive(Copy, Clone)]
pub struct QueryVTable<Qcx: QueryContext, K, V> {
    pub anon: bool,
    pub dep_kind: Qcx::DepKind,
    pub eval_always: bool,
    pub depth_limit: bool,

    pub compute: fn(Qcx::DepContext, K) -> V,
    pub hash_result: Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>,
    pub handle_cycle_error: HandleCycleError,
    // NOTE: this is also `None` if `cache_on_disk()` returns false, not just if it's unsupported by the query
    pub try_load_from_disk: Option<fn(Qcx, SerializedDepNodeIndex) -> Option<V>>,
}

impl<Qcx: QueryContext, K, V> QueryVTable<Qcx, K, V> {
    pub(crate) fn to_dep_node(&self, tcx: Qcx::DepContext, key: &K) -> DepNode<Qcx::DepKind>
    where
        K: crate::dep_graph::DepNodeParams<Qcx::DepContext>,
    {
        DepNode::construct(tcx, self.dep_kind, key)
    }

    pub(crate) fn compute(&self, tcx: Qcx::DepContext, key: K) -> V {
        (self.compute)(tcx, key)
    }
}
