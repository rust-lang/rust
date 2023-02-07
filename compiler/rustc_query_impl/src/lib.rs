//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
// this shouldn't be necessary, but the check for `&mut _` is too naive and denies returning a function pointer that takes a mut ref
#![feature(const_mut_refs)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(rustc_attrs)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_middle;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::AtomicU64;
use rustc_erase::{erase, restore, Erase};
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKindStruct};
use rustc_middle::query::Key;
use rustc_middle::ty::query::QueryEngine;
use rustc_middle::ty::query::{
    query_keys, query_provided, query_provided_to_value, query_storage, query_values,
};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::DepNodeParams;
use rustc_query_system::ich::StableHashingContext;
use rustc_span::Span;
use std::fmt;

#[macro_use]
mod plumbing;
pub use plumbing::QueryCtxt;
use rustc_query_system::query::*;
#[cfg(parallel_compiler)]
pub use rustc_query_system::query::{deadlock, QueryContext};

pub use rustc_query_system::query::QueryConfig;
use rustc_query_system::HandleCycleError;

mod on_disk_cache;
pub use on_disk_cache::OnDiskCache;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

struct DynamicQuery<C: QueryCache + RemapQueryCache> {
    name: &'static str,
    query_state: for<'a, 'tcx> fn(
        tcx: &'a QueryCtxt<'tcx>,
    ) -> &'a QueryState<
        <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key,
        rustc_middle::dep_graph::DepKind,
    >,
    query_cache:
        for<'a, 'tcx> fn(tcx: &'a QueryCtxt<'tcx>) -> &'a <C as RemapQueryCache>::Remap<'tcx>,
    cache_on_disk: for<'tcx> fn(
        tcx: TyCtxt<'tcx>,
        key: &<<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key,
    ) -> bool,
    execute_query: for<'tcx> fn(
        tcx: TyCtxt<'tcx>,
        k: <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key,
    ) -> <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Value,
    compute: for<'tcx> fn(
        tcx: TyCtxt<'tcx>,
        key: <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key,
    ) -> <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Value,
    try_load_from_disk: for<'tcx> fn(
        qcx: QueryCtxt<'tcx>,
        idx: &<<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key,
    ) -> TryLoadFromDisk<
        QueryCtxt<'tcx>,
        <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Value,
    >,
    anon: bool,
    eval_always: bool,
    depth_limit: bool,
    feedable: bool,
    dep_kind: rustc_middle::dep_graph::DepKind,
    handle_cycle_error: HandleCycleError,
    hash_result: Option<
        fn(
            &mut StableHashingContext<'_>,
            &<<C as RemapQueryCache>::Remap<'_> as QueryCache>::Value,
        ) -> Fingerprint,
    >,
}

struct ErasedQuery<C: QueryCache + RemapQueryCache + 'static> {
    dynamic: &'static DynamicQuery<C>,
}

impl<C: QueryCache + RemapQueryCache + 'static> Copy for ErasedQuery<C> {}
impl<C: QueryCache + RemapQueryCache + 'static> Clone for ErasedQuery<C> {
    fn clone(&self) -> Self {
        ErasedQuery { dynamic: self.dynamic }
    }
}

impl<C: QueryCache + RemapQueryCache + 'static> fmt::Debug for ErasedQuery<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ErasedQuery<{}>", self.dynamic.name)
    }
}

impl<'tcx, C: QueryCache + RemapQueryCache + 'static> QueryConfig<QueryCtxt<'tcx>>
    for ErasedQuery<C>
where
    for<'a> <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key:
        HashStable<StableHashingContext<'a>>,
{
    type Key = <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key;
    type Value = <<C as RemapQueryCache>::Remap<'tcx> as QueryCache>::Value;
    type Cache = <C as RemapQueryCache>::Remap<'tcx>;

    #[inline(always)]
    fn name(&self) -> &'static str {
        self.dynamic.name
    }

    #[inline]
    fn cache_on_disk(&self, tcx: TyCtxt<'tcx>, key: &Self::Key) -> bool {
        (self.dynamic.cache_on_disk)(tcx, key)
    }

    #[inline(always)]
    fn query_state<'a>(
        &self,
        cx: &'a QueryCtxt<'tcx>,
    ) -> &'a QueryState<Self::Key, crate::dep_graph::DepKind> {
        (self.dynamic.query_state)(&cx)
    }

    #[inline(always)]
    fn query_cache<'a>(&self, tcx: &'a QueryCtxt<'tcx>) -> &'a Self::Cache {
        (self.dynamic.query_cache)(&tcx)
    }

    fn execute_query(&self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.dynamic.execute_query)(tcx, key)
    }

    #[inline(always)]
    fn compute(&self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.dynamic.compute)(tcx, key)
    }

    #[inline]
    fn try_load_from_disk(
        &self,
        qcx: QueryCtxt<'tcx>,
        key: &Self::Key,
    ) -> rustc_query_system::query::TryLoadFromDisk<QueryCtxt<'tcx>, Self::Value> {
        (self.dynamic.try_load_from_disk)(qcx, key)
    }

    #[inline(always)]
    fn anon(&self) -> bool {
        self.dynamic.anon
    }

    #[inline(always)]
    fn eval_always(&self) -> bool {
        self.dynamic.eval_always
    }

    #[inline(always)]
    fn depth_limit(&self) -> bool {
        self.dynamic.depth_limit
    }

    #[inline(always)]
    fn feedable(&self) -> bool {
        self.dynamic.feedable
    }

    #[inline(always)]
    fn dep_kind(&self) -> rustc_middle::dep_graph::DepKind {
        self.dynamic.dep_kind
    }

    #[inline(always)]
    fn handle_cycle_error(&self) -> rustc_query_system::HandleCycleError {
        self.dynamic.handle_cycle_error
    }

    #[inline(always)]
    fn hash_result(&self) -> rustc_query_system::query::HashResult<Self::Value> {
        self.dynamic.hash_result
    }
}

trait QueryErasable<'tcx>
where
    for<'a> <<Self::Cache as RemapQueryCache>::Remap<'tcx> as QueryCache>::Key:
        HashStable<StableHashingContext<'a>>,
{
    type Cache: QueryCache + RemapQueryCache + 'static;
    type Key: DepNodeParams<TyCtxt<'tcx>>;
    type Value;

    fn erase() -> ErasedQuery<Self::Cache>;
    fn restore(
        value: <<Self::Cache as RemapQueryCache>::Remap<'tcx> as QueryCache>::Value,
    ) -> Self::Value;
}

rustc_query_append! { define_queries! }

impl<'tcx> Queries<'tcx> {
    // Force codegen in the dyn-trait transformation in this crate.
    pub fn as_dyn(&'tcx self) -> &'tcx dyn QueryEngine<'tcx> {
        self
    }
}
