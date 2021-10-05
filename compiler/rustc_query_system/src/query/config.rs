//! Query configuration and description traits.

use crate::dep_graph::DepNode;
use crate::dep_graph::SerializedDepNodeIndex;
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::{QueryCacheStore, QueryContext, QueryState};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_errors::DiagnosticBuilder;
use std::fmt::Debug;
use std::hash::Hash;

pub trait QueryConfig {
    const NAME: &'static str;

    type Key: Eq + Hash + Clone + Debug;
    type Value;
    type Stored: Clone;
}

pub(crate) struct QueryVtable<CTX: QueryContext, K, V> {
    pub anon: bool,
    pub dep_kind: CTX::DepKind,
    pub eval_always: bool,

    pub hash_result: fn(&mut StableHashingContext<'_>, &V) -> Option<Fingerprint>,
    pub handle_cycle_error: fn(CTX, DiagnosticBuilder<'_>) -> V,
    pub cache_on_disk: fn(CTX, &K, Option<&V>) -> bool,
    pub try_load_from_disk: fn(CTX, SerializedDepNodeIndex) -> Option<V>,
}

impl<CTX: QueryContext, K, V> QueryVtable<CTX, K, V> {
    pub(crate) fn to_dep_node(&self, tcx: CTX::DepContext, key: &K) -> DepNode<CTX::DepKind>
    where
        K: crate::dep_graph::DepNodeParams<CTX::DepContext>,
    {
        DepNode::construct(tcx, self.dep_kind, key)
    }

    pub(crate) fn hash_result(
        &self,
        hcx: &mut StableHashingContext<'_>,
        value: &V,
    ) -> Option<Fingerprint> {
        (self.hash_result)(hcx, value)
    }

    pub(crate) fn cache_on_disk(&self, tcx: CTX, key: &K, value: Option<&V>) -> bool {
        (self.cache_on_disk)(tcx, key, value)
    }

    pub(crate) fn try_load_from_disk(&self, tcx: CTX, index: SerializedDepNodeIndex) -> Option<V> {
        (self.try_load_from_disk)(tcx, index)
    }
}

pub trait QueryAccessors<CTX: QueryContext>: QueryConfig {
    const ANON: bool;
    const EVAL_ALWAYS: bool;
    const DEP_KIND: CTX::DepKind;

    type Cache: QueryCache<Key = Self::Key, Stored = Self::Stored, Value = Self::Value>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: CTX) -> &'a QueryState<CTX::DepKind, Self::Key>
    where
        CTX: 'a;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: CTX) -> &'a QueryCacheStore<Self::Cache>
    where
        CTX: 'a;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute_fn(tcx: CTX, key: &Self::Key) -> fn(CTX::DepContext, Self::Key) -> Self::Value;

    fn hash_result(hcx: &mut StableHashingContext<'_>, result: &Self::Value)
    -> Option<Fingerprint>;

    fn handle_cycle_error(tcx: CTX, diag: DiagnosticBuilder<'_>) -> Self::Value;
}

pub trait QueryDescription<CTX: QueryContext>: QueryAccessors<CTX> {
    fn describe(tcx: CTX, key: Self::Key) -> String;

    #[inline]
    fn cache_on_disk(_: CTX, _: &Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    fn try_load_from_disk(_: CTX, _: SerializedDepNodeIndex) -> Option<Self::Value> {
        panic!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

pub(crate) trait QueryVtableExt<CTX: QueryContext, K, V> {
    const VTABLE: QueryVtable<CTX, K, V>;
}

impl<CTX, Q> QueryVtableExt<CTX, Q::Key, Q::Value> for Q
where
    CTX: QueryContext,
    Q: QueryDescription<CTX>,
{
    const VTABLE: QueryVtable<CTX, Q::Key, Q::Value> = QueryVtable {
        anon: Q::ANON,
        dep_kind: Q::DEP_KIND,
        eval_always: Q::EVAL_ALWAYS,
        hash_result: Q::hash_result,
        handle_cycle_error: Q::handle_cycle_error,
        cache_on_disk: Q::cache_on_disk,
        try_load_from_disk: Q::try_load_from_disk,
    };
}
