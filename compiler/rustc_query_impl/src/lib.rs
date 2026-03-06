//! Support for serializing the dep-graph and reloading it.

// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use rustc_data_structures::sync::AtomicU64;
use rustc_middle::dep_graph;
use rustc_middle::queries::{self, ExternProviders, Providers, TaggedQueryKey};
use rustc_middle::query::on_disk_cache::OnDiskCache;
use rustc_middle::query::plumbing::{QuerySystem, QueryVTable};
use rustc_middle::query::{AsLocalQueryKey, QueryCache, QueryMode};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

pub use crate::dep_kind_vtables::make_dep_kind_vtables;
pub use crate::execution::collect_active_jobs_from_all_queries;
pub use crate::job::{QueryJobMap, break_query_cycles, print_query_stack};

#[macro_use]
mod plumbing;

mod dep_kind_vtables;
mod error;
mod execution;
mod from_cycle_error;
mod job;
mod profiling_support;

/// Trait that knows how to look up the [`QueryVTable`] for a particular query.
///
/// This trait allows some per-query code to be defined in generic functions
/// with a trait bound, instead of having to be defined inline within a macro
/// expansion.
///
/// There is one macro-generated implementation of this trait for each query,
/// on the type `rustc_query_impl::query_impl::$name::VTableGetter`.
trait GetQueryVTable<'tcx> {
    type Cache: QueryCache + 'tcx;

    fn query_vtable(tcx: TyCtxt<'tcx>) -> &'tcx QueryVTable<'tcx, Self::Cache>;
}

pub fn query_system<'tcx>(
    local_providers: Providers,
    extern_providers: ExternProviders,
    on_disk_cache: Option<OnDiskCache>,
    incremental: bool,
) -> QuerySystem<'tcx> {
    let mut query_vtables = make_query_vtables(incremental);
    from_cycle_error::specialize_query_vtables(&mut query_vtables);
    QuerySystem {
        arenas: Default::default(),
        query_vtables,
        on_disk_cache,
        local_providers,
        extern_providers,
        jobs: AtomicU64::new(1),
    }
}

rustc_middle::rustc_with_all_queries! { define_queries! }

pub fn provide(providers: &mut rustc_middle::util::Providers) {
    providers.hooks.alloc_self_profile_query_strings =
        profiling_support::alloc_self_profile_query_strings;
    providers.hooks.query_key_hash_verify_all = plumbing::query_key_hash_verify_all;
    providers.hooks.encode_all_query_results = plumbing::encode_all_query_results;
}
