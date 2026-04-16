//! Support for serializing the dep-graph and reloading it.

// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use rustc_data_structures::sync::{AtomicU64, Lock};
use rustc_middle::dep_graph;
use rustc_middle::queries::{ExternProviders, Providers};
use rustc_middle::query::on_disk_cache::OnDiskCache;
use rustc_middle::query::{QueryCache, QuerySystem, QueryVTable};
use rustc_middle::ty::TyCtxt;

pub use crate::dep_kind_vtables::make_dep_kind_vtables;
pub use crate::execution::{CollectActiveJobsKind, collect_active_query_jobs};
pub use crate::job::{QueryJobMap, break_query_cycle, print_query_stack};
use crate::query_impl::for_each_query_vtable;
mod callfront;
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

    fn query_vtable(tcx: TyCtxt<'tcx>) -> &'tcx QueryVTable<'tcx, Self::Cache>;
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
        cycle_handler_nesting: Lock::new(0),
    }
}

fn enter_sandbox<'tcx, C: QueryCache>(query: &'tcx QueryVTable<'tcx, C>) {
    query.save_keys();
}

fn leave_sandbox<'tcx, C: QueryCache>(query: &'tcx QueryVTable<'tcx, C>) {
    if query.name == "lower_delayed_owner"
        || query.name == "delayed_owner"
        || query.name == "def_kind"
    {
        return;
    }

    query.invalidate_saved_keys();
}

pub fn provide(providers: &mut rustc_middle::util::Providers) {
    providers.hooks.alloc_self_profile_query_strings =
        profiling_support::alloc_self_profile_query_strings;
    providers.hooks.verify_query_key_hashes = plumbing::verify_query_key_hashes;
    providers.hooks.encode_query_values = plumbing::encode_query_values;
    providers.hooks.enter_query_sandbox = |tcx| {
        for_each_query_vtable!(ALL, tcx, |query| {
            enter_sandbox(query);
        });
    };
    providers.hooks.leave_query_sandbox = |tcx| {
        for_each_query_vtable!(ALL, tcx, |query| {
            leave_sandbox(query);
        });
    };
}
