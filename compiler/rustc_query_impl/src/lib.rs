//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
// this shouldn't be necessary, but the check for `&mut _` is too naive and denies returning a function pointer that takes a mut ref
#![feature(const_mut_refs)]
#![feature(const_refs_to_cell)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(rustc_attrs)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability, unused_parens)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_middle;

use memoffset::offset_of;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::HashStable;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKindStruct};
use rustc_middle::query::Key;
use rustc_middle::ty::query::QueryCaches;
use rustc_middle::ty::query::{
    query_keys, query_provided, query_provided_to_value, query_storage, query_values,
};
use rustc_middle::ty::query::{Providers, QueryEngine};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::DepNodeParams;
use rustc_query_system::ich::StableHashingContext;
use rustc_span::Span;
use std::fmt;
use std::marker::PhantomData;

#[macro_use]
mod plumbing;
pub use plumbing::{Queries, QueryCtxt};
use rustc_query_system::query::*;
#[cfg(parallel_compiler)]
pub use rustc_query_system::query::{deadlock, QueryContext};

pub use rustc_query_system::query::QueryConfig;
use rustc_query_system::HandleCycleError;

mod on_disk_cache;
pub use on_disk_cache::OnDiskCache;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

trait Bool {
    fn value() -> bool;
}

struct True;

impl Bool for True {
    fn value() -> bool {
        true
    }
}
struct False;

impl Bool for False {
    fn value() -> bool {
        false
    }
}
struct DynamicQuery<'tcx, C: QueryCache> {
    name: &'static str,
    cache_on_disk: fn(tcx: TyCtxt<'tcx>, key: &C::Key) -> bool,
    execute_query: fn(tcx: TyCtxt<'tcx>, k: C::Key) -> C::Value,
    compute: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value,
    try_load_from_disk:
        fn(qcx: QueryCtxt<'tcx>, idx: &C::Key) -> TryLoadFromDisk<QueryCtxt<'tcx>, C::Value>,
    query_state: usize,
    query_cache: usize,
    eval_always: bool,
    dep_kind: rustc_middle::dep_graph::DepKind,
    handle_cycle_error: HandleCycleError,
    hash_result: Option<fn(&mut StableHashingContext<'_>, &C::Value) -> Fingerprint>,
}

struct DynamicConfig<'tcx, C: QueryCache, Anon, DepthLimit, Feedable> {
    dynamic: &'tcx DynamicQuery<'tcx, C>,
    data: PhantomData<(Anon, DepthLimit, Feedable)>,
}

impl<'tcx, C: QueryCache, Anon, DepthLimit, Feedable> Copy
    for DynamicConfig<'tcx, C, Anon, DepthLimit, Feedable>
{
}
impl<'tcx, C: QueryCache, Anon, DepthLimit, Feedable> Clone
    for DynamicConfig<'tcx, C, Anon, DepthLimit, Feedable>
{
    fn clone(&self) -> Self {
        DynamicConfig { dynamic: self.dynamic, data: PhantomData }
    }
}

impl<'tcx, C: QueryCache, Anon, DepthLimit, Feedable> fmt::Debug
    for DynamicConfig<'tcx, C, Anon, DepthLimit, Feedable>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DynamicConfig<{}>", self.dynamic.name)
    }
}

impl<'tcx, C: QueryCache, Anon: Bool, DepthLimit: Bool, Feedable: Bool> QueryConfig<QueryCtxt<'tcx>>
    for DynamicConfig<'tcx, C, Anon, DepthLimit, Feedable>
where
    for<'a> C::Key: HashStable<StableHashingContext<'a>>,
{
    type Key = C::Key;
    type Value = C::Value;
    type Cache = C;

    #[inline(always)]
    fn name(self) -> &'static str {
        self.dynamic.name
    }

    #[inline(always)]
    fn cache_on_disk(self, tcx: TyCtxt<'tcx>, key: &Self::Key) -> bool {
        (self.dynamic.cache_on_disk)(tcx, key)
    }

    #[inline(always)]
    fn query_state<'a>(
        self,
        tcx: QueryCtxt<'tcx>,
    ) -> &'a QueryState<Self::Key, crate::dep_graph::DepKind>
    where
        QueryCtxt<'tcx>: 'a,
    {
        unsafe {
            &*((&tcx.queries.query_states as *const QueryStates<'tcx> as *const u8)
                .offset(self.dynamic.query_state as isize) as *const _)
        }
    }

    #[inline(always)]
    fn query_cache<'a>(self, tcx: QueryCtxt<'tcx>) -> &'a Self::Cache
    where
        'tcx: 'a,
    {
        unsafe {
            &*((&tcx.query_system.caches as *const QueryCaches<'tcx> as *const u8)
                .offset(self.dynamic.query_cache as isize) as *const _)
        }
    }

    #[inline(always)]
    fn execute_query(self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.dynamic.execute_query)(tcx, key)
    }

    #[inline(always)]
    fn compute(self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.dynamic.compute)(tcx, key)
    }

    #[inline(always)]
    fn try_load_from_disk(
        self,
        qcx: QueryCtxt<'tcx>,
        key: &Self::Key,
    ) -> rustc_query_system::query::TryLoadFromDisk<QueryCtxt<'tcx>, Self::Value> {
        (self.dynamic.try_load_from_disk)(qcx, key)
    }

    #[inline(always)]
    fn anon(self) -> bool {
        Anon::value()
    }

    #[inline(always)]
    fn eval_always(self) -> bool {
        self.dynamic.eval_always
    }

    #[inline(always)]
    fn depth_limit(self) -> bool {
        DepthLimit::value()
    }

    #[inline(always)]
    fn feedable(self) -> bool {
        Feedable::value()
    }

    #[inline(always)]
    fn dep_kind(self) -> rustc_middle::dep_graph::DepKind {
        self.dynamic.dep_kind
    }

    #[inline(always)]
    fn handle_cycle_error(self) -> rustc_query_system::HandleCycleError {
        self.dynamic.handle_cycle_error
    }

    #[inline(always)]
    fn hash_result(self) -> rustc_query_system::query::HashResult<Self::Value> {
        self.dynamic.hash_result
    }
}

trait QueryToConfig<'tcx>: 'tcx
where
    for<'a> <Self::Cache as QueryCache>::Key: HashStable<StableHashingContext<'a>>,
{
    type Cache: QueryCache;
    type Key: DepNodeParams<TyCtxt<'tcx>>;

    type Anon: Bool;
    type DepthLimit: Bool;
    type Feedable: Bool;

    fn config(
        qcx: QueryCtxt<'tcx>,
    ) -> DynamicConfig<'tcx, Self::Cache, Self::Anon, Self::DepthLimit, Self::Feedable>;
}

rustc_query_append! { define_queries! }
