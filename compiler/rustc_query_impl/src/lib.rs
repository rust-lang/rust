//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![feature(generic_nonzero)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![allow(rustc::potential_query_instability, unused_parens)]
#![allow(internal_features)]

#[macro_use]
extern crate rustc_middle;

use crate::plumbing::{__rust_begin_short_backtrace, encode_all_query_results, try_mark_green};
use field_offset::offset_of;
use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::AtomicU64;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::DepNodeIndex;
use rustc_middle::dep_graph::{self, DepKind, DepKindStruct};
use rustc_middle::query::erase::{erase, restore, Erase};
use rustc_middle::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use rustc_middle::query::plumbing::{
    DynamicQuery, QueryKeyStringCache, QuerySystem, QuerySystemFns,
};
use rustc_middle::query::AsLocalKey;
use rustc_middle::query::{
    queries, DynamicQueries, ExternProviders, Providers, QueryCaches, QueryEngine, QueryStates,
};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::SerializedDepNodeIndex;
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{
    get_query_incr, get_query_non_incr, CycleError, HashResult, QueryCache, QueryConfig, QueryMap,
    QueryMode, QueryState,
};
use rustc_query_system::HandleCycleError;
use rustc_query_system::Value;
use rustc_span::{ErrorGuaranteed, Span};

#[macro_use]
mod plumbing;
pub use crate::plumbing::QueryCtxt;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

struct DynamicConfig<
    'tcx,
    C: QueryCache,
    const ANON: bool,
    const DEPTH_LIMIT: bool,
    const FEEDABLE: bool,
> {
    dynamic: &'tcx DynamicQuery<'tcx, C>,
}

impl<'tcx, C: QueryCache, const ANON: bool, const DEPTH_LIMIT: bool, const FEEDABLE: bool> Copy
    for DynamicConfig<'tcx, C, ANON, DEPTH_LIMIT, FEEDABLE>
{
}
impl<'tcx, C: QueryCache, const ANON: bool, const DEPTH_LIMIT: bool, const FEEDABLE: bool> Clone
    for DynamicConfig<'tcx, C, ANON, DEPTH_LIMIT, FEEDABLE>
{
    fn clone(&self) -> Self {
        DynamicConfig { dynamic: self.dynamic }
    }
}

impl<'tcx, C: QueryCache, const ANON: bool, const DEPTH_LIMIT: bool, const FEEDABLE: bool>
    QueryConfig<QueryCtxt<'tcx>> for DynamicConfig<'tcx, C, ANON, DEPTH_LIMIT, FEEDABLE>
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
    fn query_state<'a>(self, qcx: QueryCtxt<'tcx>) -> &'a QueryState<Self::Key>
    where
        QueryCtxt<'tcx>: 'a,
    {
        self.dynamic.query_state.apply(&qcx.tcx.query_system.states)
    }

    #[inline(always)]
    fn query_cache<'a>(self, qcx: QueryCtxt<'tcx>) -> &'a Self::Cache
    where
        'tcx: 'a,
    {
        self.dynamic.query_cache.apply(&qcx.tcx.query_system.caches)
    }

    #[inline(always)]
    fn execute_query(self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.dynamic.execute_query)(tcx, key)
    }

    #[inline(always)]
    fn compute(self, qcx: QueryCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.dynamic.compute)(qcx.tcx, key)
    }

    #[inline(always)]
    fn try_load_from_disk(
        self,
        qcx: QueryCtxt<'tcx>,
        key: &Self::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<Self::Value> {
        if self.dynamic.can_load_from_disk {
            (self.dynamic.try_load_from_disk)(qcx.tcx, key, prev_index, index)
        } else {
            None
        }
    }

    #[inline]
    fn loadable_from_disk(
        self,
        qcx: QueryCtxt<'tcx>,
        key: &Self::Key,
        index: SerializedDepNodeIndex,
    ) -> bool {
        (self.dynamic.loadable_from_disk)(qcx.tcx, key, index)
    }

    fn value_from_cycle_error(
        self,
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self::Value {
        (self.dynamic.value_from_cycle_error)(tcx, cycle_error, guar)
    }

    #[inline(always)]
    fn format_value(self) -> fn(&Self::Value) -> String {
        self.dynamic.format_value
    }

    #[inline(always)]
    fn anon(self) -> bool {
        ANON
    }

    #[inline(always)]
    fn eval_always(self) -> bool {
        self.dynamic.eval_always
    }

    #[inline(always)]
    fn depth_limit(self) -> bool {
        DEPTH_LIMIT
    }

    #[inline(always)]
    fn feedable(self) -> bool {
        FEEDABLE
    }

    #[inline(always)]
    fn dep_kind(self) -> DepKind {
        self.dynamic.dep_kind
    }

    #[inline(always)]
    fn handle_cycle_error(self) -> HandleCycleError {
        self.dynamic.handle_cycle_error
    }

    #[inline(always)]
    fn hash_result(self) -> HashResult<Self::Value> {
        self.dynamic.hash_result
    }
}

/// This is implemented per query. It allows restoring query values from their erased state
/// and constructing a QueryConfig.
trait QueryConfigRestored<'tcx> {
    type RestoredValue;
    type Config: QueryConfig<QueryCtxt<'tcx>>;

    const NAME: &'static &'static str;

    fn config(tcx: TyCtxt<'tcx>) -> Self::Config;
    fn restore(value: <Self::Config as QueryConfig<QueryCtxt<'tcx>>>::Value)
    -> Self::RestoredValue;
}

pub fn query_system<'tcx>(
    local_providers: Providers,
    extern_providers: ExternProviders,
    on_disk_cache: Option<OnDiskCache<'tcx>>,
    incremental: bool,
) -> QuerySystem<'tcx> {
    QuerySystem {
        states: Default::default(),
        arenas: Default::default(),
        caches: Default::default(),
        dynamic_queries: dynamic_queries(),
        on_disk_cache,
        fns: QuerySystemFns {
            engine: engine(incremental),
            local_providers,
            extern_providers,
            encode_query_results: encode_all_query_results,
            try_mark_green: try_mark_green,
        },
        jobs: AtomicU64::new(1),
    }
}

rustc_query_append! { define_queries! }
