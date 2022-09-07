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

pub trait QueryConfig {
    const NAME: &'static str;

    type Key: Eq + Hash + Clone + Debug;
    type Value;
    type Stored: Clone;
}

#[derive(Copy, Clone)]
pub struct QueryVTable<CTX: QueryContext, K, V> {
    pub anon: bool,
    pub dep_kind: CTX::DepKind,
    pub eval_always: bool,
    pub depth_limit: bool,
    pub cache_on_disk: bool,

    pub compute: fn(CTX::DepContext, K) -> V,
    pub hash_result: Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>,
    pub handle_cycle_error: HandleCycleError,
    pub try_load_from_disk: Option<fn(CTX, SerializedDepNodeIndex) -> Option<V>>,
}

impl<CTX: QueryContext, K, V> QueryVTable<CTX, K, V> {
    pub(crate) fn to_dep_node(&self, tcx: CTX::DepContext, key: &K) -> DepNode<CTX::DepKind>
    where
        K: crate::dep_graph::DepNodeParams<CTX::DepContext>,
    {
        DepNode::construct(tcx, self.dep_kind, key)
    }

    pub(crate) fn compute(&self, tcx: CTX::DepContext, key: K) -> V {
        (self.compute)(tcx, key)
    }

    pub(crate) fn try_load_from_disk(&self, tcx: CTX, index: SerializedDepNodeIndex) -> Option<V> {
        self.try_load_from_disk
            .expect("QueryDescription::load_from_disk() called for an unsupported query.")(
            tcx, index,
        )
    }
}

pub trait QueryDescription<CTX: QueryContext>: QueryConfig {
    const TRY_LOAD_FROM_DISK: Option<fn(CTX, SerializedDepNodeIndex) -> Option<Self::Value>>;

    type Cache: QueryCache<Key = Self::Key, Stored = Self::Stored, Value = Self::Value>;

    fn describe(tcx: CTX, key: Self::Key) -> String;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: CTX) -> &'a QueryState<Self::Key>
    where
        CTX: 'a;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: CTX) -> &'a Self::Cache
    where
        CTX: 'a;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn make_vtable(tcx: CTX, key: &Self::Key) -> QueryVTable<CTX, Self::Key, Self::Value>;

    fn cache_on_disk(tcx: CTX::DepContext, key: &Self::Key) -> bool;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn execute_query(tcx: CTX::DepContext, k: Self::Key) -> Self::Stored;
}
