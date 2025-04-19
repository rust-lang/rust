//! Support for serializing the dep-graph and reloading it.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(unused_parens)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::AtomicU64;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKind, DepKindStruct, DepNodeIndex};
use rustc_middle::query::erase::{Erase, erase, restore};
use rustc_middle::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use rustc_middle::query::plumbing::{DynamicQuery, QuerySystem, QuerySystemFns};
use rustc_middle::query::{
    AsLocalKey, DynamicQueries, ExternProviders, Providers, QueryCaches, QueryEngine, QueryStates,
    queries,
};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::SerializedDepNodeIndex;
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{
    CycleError, HashResult, QueryCache, QueryConfig, QueryMap, QueryMode, QueryStackDeferred,
    QueryState, get_query_incr, get_query_non_incr,
};
use rustc_query_system::{HandleCycleError, Value};
use rustc_span::{ErrorGuaranteed, Span};

use crate::plumbing::{__rust_begin_short_backtrace, encode_all_query_results, try_mark_green};
use crate::profiling_support::QueryKeyStringCache;

#[macro_use]
mod plumbing;
pub use crate::plumbing::{QueryCtxt, query_key_hash_verify_all};

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
    fn query_state<'a>(
        self,
        qcx: QueryCtxt<'tcx>,
    ) -> &'a QueryState<Self::Key, QueryStackDeferred<'tcx>>
    where
        QueryCtxt<'tcx>: 'a,
    {
        // Safety:
        // This is just manually doing the subfield referencing through pointer math.
        unsafe {
            &*(&qcx.tcx.query_system.states as *const QueryStates<'tcx>)
                .byte_add(self.dynamic.query_state)
                .cast::<QueryState<Self::Key, QueryStackDeferred<'tcx>>>()
        }
    }

    #[inline(always)]
    fn query_cache<'a>(self, qcx: QueryCtxt<'tcx>) -> &'a Self::Cache
    where
        'tcx: 'a,
    {
        // Safety:
        // This is just manually doing the subfield referencing through pointer math.
        unsafe {
            &*(&qcx.tcx.query_system.caches as *const QueryCaches<'tcx>)
                .byte_add(self.dynamic.query_cache)
                .cast::<Self::Cache>()
        }
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

pub fn query_system<'a>(
    local_providers: Providers,
    extern_providers: ExternProviders,
    on_disk_cache: Option<OnDiskCache>,
    incremental: bool,
) -> QuerySystem<'a> {
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
            try_mark_green,
        },
        jobs: AtomicU64::new(1),
    }
}

rustc_middle::rustc_query_append! { define_queries! }

pub fn provide(providers: &mut rustc_middle::util::Providers) {
    providers.hooks.alloc_self_profile_query_strings = alloc_self_profile_query_strings;
    providers.hooks.query_key_hash_verify_all = query_key_hash_verify_all;
}
