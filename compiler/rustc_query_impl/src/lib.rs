//! Support for serializing the dep-graph and reloading it.

// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(adt_const_params)]
#![feature(core_intrinsics)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use std::fmt;
use std::marker::ConstParamTy;

use rustc_data_structures::sync::AtomicU64;
use rustc_middle::dep_graph::{self, DepKind, DepNode, DepNodeIndex, SerializedDepNodeIndex};
use rustc_middle::queries::{
    self, ExternProviders, Providers, QueryCaches, QueryEngine, QueryStates,
};
use rustc_middle::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use rustc_middle::query::plumbing::{
    HashResult, QueryState, QuerySystem, QuerySystemFns, QueryVTable,
};
use rustc_middle::query::{AsLocalKey, CycleError, CycleErrorHandling, QueryCache, QueryMode};
use rustc_middle::ty::TyCtxt;
use rustc_span::{ErrorGuaranteed, Span};

pub use crate::dep_kind_vtables::make_dep_kind_vtables;
pub use crate::job::{QueryJobMap, break_query_cycles, print_query_stack};
pub use crate::plumbing::{collect_active_jobs_from_all_queries, query_key_hash_verify_all};
use crate::plumbing::{encode_all_query_results, try_mark_green};
use crate::profiling_support::QueryKeyStringCache;
pub use crate::profiling_support::alloc_self_profile_query_strings;
use crate::values::Value;

#[macro_use]
mod plumbing;

mod dep_kind_vtables;
mod error;
mod execution;
mod job;
mod profiling_support;
mod values;

#[derive(ConstParamTy)] // Allow this struct to be used for const-generic values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueryFlags {
    /// True if this query has the `anon` modifier.
    is_anon: bool,
    /// True if this query has the `depth_limit` modifier.
    is_depth_limit: bool,
    /// True if this query has the `feedable` modifier.
    is_feedable: bool,
}

/// Combines a [`QueryVTable`] with some additional compile-time booleans.
/// "Dispatcher" should be understood as a near-synonym of "vtable".
///
/// Baking these boolean flags into the type gives a modest but measurable
/// improvement to compiler perf and compiler code size; see
/// <https://github.com/rust-lang/rust/pull/151633>.
struct SemiDynamicQueryDispatcher<'tcx, C: QueryCache, const FLAGS: QueryFlags> {
    vtable: &'tcx QueryVTable<'tcx, C>,
}

// Manually implement Copy/Clone, because deriving would put trait bounds on the cache type.
impl<'tcx, C: QueryCache, const FLAGS: QueryFlags> Copy
    for SemiDynamicQueryDispatcher<'tcx, C, FLAGS>
{
}
impl<'tcx, C: QueryCache, const FLAGS: QueryFlags> Clone
    for SemiDynamicQueryDispatcher<'tcx, C, FLAGS>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'tcx, C: QueryCache, const FLAGS: QueryFlags> fmt::Debug
    for SemiDynamicQueryDispatcher<'tcx, C, FLAGS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // When debug-printing a query dispatcher (e.g. for ICE or tracing),
        // just print the query name to know what query we're dealing with.
        // The other fields and flags are probably just unhelpful noise.
        //
        // If there is need for a more detailed dump of all flags and fields,
        // consider writing a separate dump method and calling it explicitly.
        f.write_str(self.name())
    }
}

impl<'tcx, C: QueryCache, const FLAGS: QueryFlags> SemiDynamicQueryDispatcher<'tcx, C, FLAGS> {
    #[inline(always)]
    fn name(self) -> &'static str {
        self.vtable.name
    }

    #[inline(always)]
    fn will_cache_on_disk_for_key(self, tcx: TyCtxt<'tcx>, key: &C::Key) -> bool {
        self.vtable.will_cache_on_disk_for_key_fn.map_or(false, |f| f(tcx, key))
    }

    // Don't use this method to access query results, instead use the methods on TyCtxt.
    #[inline(always)]
    fn query_state(self, tcx: TyCtxt<'tcx>) -> &'tcx QueryState<'tcx, C::Key> {
        // Safety:
        // This is just manually doing the subfield referencing through pointer math.
        unsafe {
            &*(&tcx.query_system.states as *const QueryStates<'tcx>)
                .byte_add(self.vtable.query_state)
                .cast::<QueryState<'tcx, C::Key>>()
        }
    }

    // Don't use this method to access query results, instead use the methods on TyCtxt.
    #[inline(always)]
    fn query_cache(self, tcx: TyCtxt<'tcx>) -> &'tcx C {
        // Safety:
        // This is just manually doing the subfield referencing through pointer math.
        unsafe {
            &*(&tcx.query_system.caches as *const QueryCaches<'tcx>)
                .byte_add(self.vtable.query_cache)
                .cast::<C>()
        }
    }

    /// Calls `tcx.$query(key)` for this query, and discards the returned value.
    /// See [`QueryVTable::call_query_method_fn`] for details of this strange operation.
    #[inline(always)]
    fn call_query_method(self, tcx: TyCtxt<'tcx>, key: C::Key) {
        (self.vtable.call_query_method_fn)(tcx, key)
    }

    /// Calls the actual provider function for this query.
    /// See [`QueryVTable::invoke_provider_fn`] for more details.
    #[inline(always)]
    fn invoke_provider(self, tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value {
        (self.vtable.invoke_provider_fn)(tcx, key)
    }

    #[inline(always)]
    fn try_load_from_disk(
        self,
        tcx: TyCtxt<'tcx>,
        key: &C::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<C::Value> {
        // `?` will return None immediately for queries that never cache to disk.
        self.vtable.try_load_from_disk_fn?(tcx, key, prev_index, index)
    }

    #[inline]
    fn is_loadable_from_disk(
        self,
        tcx: TyCtxt<'tcx>,
        key: &C::Key,
        index: SerializedDepNodeIndex,
    ) -> bool {
        self.vtable.is_loadable_from_disk_fn.map_or(false, |f| f(tcx, key, index))
    }

    /// Synthesize an error value to let compilation continue after a cycle.
    fn value_from_cycle_error(
        self,
        tcx: TyCtxt<'tcx>,
        cycle_error: &CycleError,
        guar: ErrorGuaranteed,
    ) -> C::Value {
        (self.vtable.value_from_cycle_error)(tcx, cycle_error, guar)
    }

    #[inline(always)]
    fn format_value(self) -> fn(&C::Value) -> String {
        self.vtable.format_value
    }

    #[inline(always)]
    fn anon(self) -> bool {
        FLAGS.is_anon
    }

    #[inline(always)]
    fn eval_always(self) -> bool {
        self.vtable.eval_always
    }

    #[inline(always)]
    fn depth_limit(self) -> bool {
        FLAGS.is_depth_limit
    }

    #[inline(always)]
    fn feedable(self) -> bool {
        FLAGS.is_feedable
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
    fn hash_result(self) -> HashResult<C::Value> {
        self.vtable.hash_result
    }

    fn construct_dep_node(self, tcx: TyCtxt<'tcx>, key: &C::Key) -> DepNode {
        DepNode::construct(tcx, self.dep_kind(), key)
    }
}

/// Provides access to vtable-like operations for a query
/// (by creating a [`SemiDynamicQueryDispatcher`]),
/// but also keeps track of the "unerased" value type of the query
/// (i.e. the actual result type in the query declaration).
///
/// This trait allows some per-query code to be defined in generic functions
/// with a trait bound, instead of having to be defined inline within a macro
/// expansion.
///
/// There is one macro-generated implementation of this trait for each query,
/// on the type `rustc_query_impl::query_impl::$name::QueryType`.
trait QueryDispatcherUnerased<'tcx, C: QueryCache, const FLAGS: QueryFlags> {
    type UnerasedValue;

    const NAME: &'static &'static str;

    fn query_dispatcher(tcx: TyCtxt<'tcx>) -> SemiDynamicQueryDispatcher<'tcx, C, FLAGS>;

    fn restore_val(value: C::Value) -> Self::UnerasedValue;
}

pub fn query_system<'tcx>(
    local_providers: Providers,
    extern_providers: ExternProviders,
    on_disk_cache: Option<OnDiskCache>,
    incremental: bool,
) -> QuerySystem<'tcx> {
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
