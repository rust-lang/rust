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
use rustc_middle::queries::{ExternProviders, Providers};
use rustc_middle::query::on_disk_cache::OnDiskCache;
use rustc_middle::query::{QueryCache, QueryHelper, QuerySystem, QueryVTable};
use rustc_middle::ty::TyCtxt;

pub use crate::dep_kind_vtables::make_dep_kind_vtables;
pub use crate::execution::{CollectActiveJobsKind, collect_active_query_jobs};
pub use crate::job::{QueryJobMap, break_query_cycle, print_query_stack};

mod dep_kind_vtables;
mod error;
mod execution;
mod handle_cycle_error;
mod job;
mod plumbing;
mod profiling_support;
mod query_impl;

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
    type Helper: QueryHelper<'tcx, <Self::Cache as QueryCache>::Key, <Self::Cache as QueryCache>::Value>;

    fn query_vtable(tcx: TyCtxt<'tcx>) -> &'tcx QueryVTable<'tcx, Self::Cache, Self::Helper>;
}

pub fn query_system<'tcx>(
    local_providers: Providers,
    extern_providers: ExternProviders,
    on_disk_cache: Option<OnDiskCache>,
    incremental: bool,
) -> QuerySystem<'tcx> {
    QuerySystem {
        arenas: Default::default(),
        query_vtables: query_impl::make_query_vtables(incremental),
        side_effects: Default::default(),
        on_disk_cache,
        local_providers,
        extern_providers,
        jobs: AtomicU64::new(1),
    }
}

pub fn provide(providers: &mut rustc_middle::util::Providers) {
    providers.hooks.alloc_self_profile_query_strings =
        profiling_support::alloc_self_profile_query_strings;
    providers.hooks.verify_query_key_hashes = plumbing::verify_query_key_hashes;
    providers.hooks.encode_query_values = plumbing::encode_query_values;
}
