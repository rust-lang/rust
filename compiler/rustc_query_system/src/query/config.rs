//! Query configuration and description traits.

use crate::dep_graph::DepNode;
use crate::dep_graph::SerializedDepNodeIndex;
use crate::query::caches::QueryCache;
use crate::query::plumbing::CycleError;
use crate::query::{QueryContext, QueryState};

use rustc_data_structures::fingerprint::Fingerprint;
use std::borrow::Cow;
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

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    pub compute: fn(CTX, K) -> V,

    pub hash_result: fn(&mut CTX::StableHashingContext, &V) -> Option<Fingerprint>,
    pub handle_cycle_error: fn(CTX, CycleError<CTX::Query>) -> V,
    pub cache_on_disk: fn(CTX, &K, Option<&V>) -> bool,
    pub try_load_from_disk: fn(CTX, SerializedDepNodeIndex) -> Option<V>,
}

impl<CTX: QueryContext, K, V> QueryVtable<CTX, K, V> {
    pub(crate) fn to_dep_node(&self, tcx: CTX, key: &K) -> DepNode<CTX::DepKind>
    where
        K: crate::dep_graph::DepNodeParams<CTX>,
    {
        DepNode::construct(tcx, self.dep_kind, key)
    }

    pub(crate) fn compute(&self, tcx: CTX, key: K) -> V {
        (self.compute)(tcx, key)
    }

    pub(crate) fn hash_result(
        &self,
        hcx: &mut CTX::StableHashingContext,
        value: &V,
    ) -> Option<Fingerprint> {
        (self.hash_result)(hcx, value)
    }

    pub(crate) fn handle_cycle_error(&self, tcx: CTX, error: CycleError<CTX::Query>) -> V {
        (self.handle_cycle_error)(tcx, error)
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
    fn query_state<'a>(tcx: CTX) -> &'a QueryState<CTX::DepKind, CTX::Query, Self::Cache>;

    fn to_dep_node(tcx: CTX, key: &Self::Key) -> DepNode<CTX::DepKind>
    where
        Self::Key: crate::dep_graph::DepNodeParams<CTX>,
    {
        DepNode::construct(tcx, Self::DEP_KIND, key)
    }

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute(tcx: CTX, key: Self::Key) -> Self::Value;

    fn hash_result(
        hcx: &mut CTX::StableHashingContext,
        result: &Self::Value,
    ) -> Option<Fingerprint>;

    fn handle_cycle_error(tcx: CTX, error: CycleError<CTX::Query>) -> Self::Value;
}

pub trait QueryDescription<CTX: QueryContext>: QueryAccessors<CTX> {
    fn describe(tcx: CTX, key: Self::Key) -> Cow<'static, str>;

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
        compute: Q::compute,
        hash_result: Q::hash_result,
        handle_cycle_error: Q::handle_cycle_error,
        cache_on_disk: Q::cache_on_disk,
        try_load_from_disk: Q::try_load_from_disk,
    };
}
