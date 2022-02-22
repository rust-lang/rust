//! Query configuration and description traits.

use crate::query::caches::QueryCache;
use crate::query::{QueryCacheStore, QueryContext, QueryState};

use rustc_errors::DiagnosticBuilder;
use std::fmt::Debug;
use std::hash::Hash;

pub trait QueryConfig {
    const NAME: &'static str;

    type Key: Eq + Hash + Clone + Debug;
    type Value;
    type Stored: Clone;
}

pub trait QueryDescription<CTX: QueryContext>: QueryConfig {
    type Cache: QueryCache<Key = Self::Key, Stored = Self::Stored, Value = Self::Value>;

    fn describe(tcx: CTX, key: Self::Key) -> String;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: CTX) -> &'a QueryState<Self::Key>
    where
        CTX: 'a;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: CTX) -> &'a QueryCacheStore<Self::Cache>
    where
        CTX: 'a;

    fn compute(tcx: CTX, key: &Self::Key) -> fn(CTX::DepContext, Self::Key) -> Self::Value;
    fn handle_cycle_error(tcx: CTX, diag: DiagnosticBuilder<'_>) -> Self::Value;
}
