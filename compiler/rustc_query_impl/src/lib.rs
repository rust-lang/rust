//! Support for serializing the dep-graph and reloading it.

// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
// tidy-alphabetical-end

use rustc_data_structures::stable_hasher::HashStable;
use rustc_data_structures::sync::AtomicU64;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKind, DepKindVTable, DepNodeIndex};
use rustc_middle::query::erase::{Erase, erase, restore};
use rustc_middle::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use rustc_middle::query::plumbing::{QuerySystem, QuerySystemFns, QueryVTable};
use rustc_middle::query::{
    AsLocalKey, ExternProviders, Providers, QueryCaches, QueryEngine, QueryStates, queries,
};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::Value;
use rustc_query_system::dep_graph::SerializedDepNodeIndex;
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{
    CycleError, CycleErrorHandling, HashResult, QueryCache, QueryDispatcher, QueryMap, QueryMode,
    QueryState, get_query_incr, get_query_non_incr,
};
use rustc_span::{ErrorGuaranteed, Span};

use crate::plumbing::{encode_all_query_results, try_mark_green};
use crate::profiling_support::QueryKeyStringCache;

#[macro_use]
mod plumbing;
pub use crate::plumbing::{QueryCtxt, query_key_hash_verify_all};

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

/// Combines a [`QueryVTable`] with some additional compile-time booleans
/// to implement [`QueryDispatcher`], for use by code in [`rustc_query_system`].
///
/// Baking these boolean flags into the type gives a modest but measurable
/// improvement to compiler perf and compiler code size; see
/// <https://github.com/rust-lang/rust/pull/151633>.
struct SemiDynamicQueryDispatcher<
    'tcx,
    C: QueryCache,
    const ANON: bool,
    const DEPTH_LIMIT: bool,
    const FEEDABLE: bool,
> {
    vtable: &'tcx QueryVTable<'tcx, C>,
}

// Manually implement Copy/Clone, because deriving would put trait bounds on the cache type.
impl<'tcx, C: QueryCache, const ANON: bool, const DEPTH_LIMIT: bool, const FEEDABLE: bool> Copy
    for SemiDynamicQueryDispatcher<'tcx, C, ANON, DEPTH_LIMIT, FEEDABLE>
{
}
impl<'tcx, C: QueryCache, const ANON: bool, const DEPTH_LIMIT: bool, const FEEDABLE: bool> Clone
    for SemiDynamicQueryDispatcher<'tcx, C, ANON, DEPTH_LIMIT, FEEDABLE>
{
    fn clone(&self) -> Self {
        *self
    }
}

// This is `impl QueryDispatcher for SemiDynamicQueryDispatcher`.
impl<'tcx, C: QueryCache, const ANON: bool, const DEPTH_LIMIT: bool, const FEEDABLE: bool>
    QueryDispatcher<QueryCtxt<'tcx>>
    for SemiDynamicQueryDispatcher<'tcx, C, ANON, DEPTH_LIMIT, FEEDABLE>
where
    for<'a> C::Key: HashStable<StableHashingContext<'a>>,
{
    type Key = C::Key;
    type Value = C::Value;
    type Cache = C;

    #[inline(always)]
    fn name(self) -> &'static str {
        self.vtable.name
    }

    #[inline(always)]
    fn cache_on_disk(self, tcx: TyCtxt<'tcx>, key: &Self::Key) -> bool {
        (self.vtable.cache_on_disk)(tcx, key)
    }

    #[inline(always)]
    fn query_state<'a>(self, qcx: QueryCtxt<'tcx>) -> &'a QueryState<Self::Key>
    where
        QueryCtxt<'tcx>: 'a,
    {
        // Safety:
        // This is just manually doing the subfield referencing through pointer math.
        unsafe {
            &*(&qcx.tcx.query_system.states as *const QueryStates<'tcx>)
                .byte_add(self.vtable.query_state)
                .cast::<QueryState<Self::Key>>()
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
                .byte_add(self.vtable.query_cache)
                .cast::<Self::Cache>()
        }
    }

    #[inline(always)]
    fn execute_query(self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.vtable.execute_query)(tcx, key)
    }

    #[inline(always)]
    fn compute(self, qcx: QueryCtxt<'tcx>, key: Self::Key) -> Self::Value {
        (self.vtable.compute)(qcx.tcx, key)
    }

    #[inline(always)]
    fn try_load_from_disk(
        self,
        qcx: QueryCtxt<'tcx>,
        key: &Self::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<Self::Value> {
        if self.vtable.can_load_from_disk {
            (self.vtable.try_load_from_disk)(qcx.tcx, key, prev_index, index)
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
        (self.vtable.loadable_from_disk)(qcx.tcx, key, index)
    }

    fn value_from_cycle_error(
        self,
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> Self::Value {
        (self.vtable.value_from_cycle_error)(tcx, cycle_error, guar)
    }

    #[inline(always)]
    fn format_value(self) -> fn(&Self::Value) -> String {
        self.vtable.format_value
    }

    #[inline(always)]
    fn anon(self) -> bool {
        ANON
    }

    #[inline(always)]
    fn eval_always(self) -> bool {
        self.vtable.eval_always
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
        self.vtable.dep_kind
    }

    #[inline(always)]
    fn cycle_error_handling(self) -> CycleErrorHandling {
        self.vtable.cycle_error_handling
    }

    #[inline(always)]
    fn hash_result(self) -> HashResult<Self::Value> {
        self.vtable.hash_result
    }
}

/// Provides access to vtable-like operations for a query
/// (by creating a [`QueryDispatcher`]),
/// but also keeps track of the "unerased" value type of the query
/// (i.e. the actual result type in the query declaration).
///
/// This trait allows some per-query code to be defined in generic functions
/// with a trait bound, instead of having to be defined inline within a macro
/// expansion.
///
/// There is one macro-generated implementation of this trait for each query,
/// on the type `rustc_query_impl::query_impl::$name::QueryType`.
trait QueryDispatcherUnerased<'tcx> {
    type UnerasedValue;
    type Dispatcher: QueryDispatcher<QueryCtxt<'tcx>>;

    const NAME: &'static &'static str;

    fn query_dispatcher(tcx: TyCtxt<'tcx>) -> Self::Dispatcher;

    fn restore_val(
        value: <Self::Dispatcher as QueryDispatcher<QueryCtxt<'tcx>>>::Value,
    ) -> Self::UnerasedValue;
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
        query_vtables: make_query_vtables(),
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

rustc_middle::rustc_with_all_queries! { define_queries! }

pub fn provide(providers: &mut rustc_middle::util::Providers) {
    providers.hooks.alloc_self_profile_query_strings = alloc_self_profile_query_strings;
    providers.hooks.query_key_hash_verify_all = query_key_hash_verify_all;
}
