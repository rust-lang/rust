// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use rustc_data_structures::sync::{AtomicU64, Lock};
use rustc_middle::arena::Arena;
use rustc_middle::queries::{ExternProviders, Providers};
use rustc_middle::query::QuerySystem;
use rustc_middle::query::on_disk_cache::OnDiskCache;

pub use crate::job::{
    CollectActiveJobsKind, QueryJobMap, break_query_cycle, collect_active_query_jobs,
    print_query_stack,
};

mod dep_kind_vtables;
mod diagnostics;
mod execution;
mod handle_cycle_error;
mod incremental;
mod job;
mod query_vtables;
mod self_profile;

pub fn query_system<'tcx>(
    arena: &'tcx Arena<'tcx>,
    local_providers: Providers,
    extern_providers: ExternProviders,
    on_disk_cache: Option<OnDiskCache>,
    incremental: bool,
) -> QuerySystem<'tcx> {
    QuerySystem {
        arenas: Default::default(),
        dep_kind_vtables: dep_kind_vtables::make_dep_kind_vtables(arena),
        query_vtables: query_vtables::make_query_vtables(incremental),
        side_effects: Default::default(),
        used_features: Default::default(),
        on_disk_cache,
        local_providers,
        extern_providers,
        jobs: AtomicU64::new(1),
        cycle_handler_nesting: Lock::new(0),
    }
}

pub fn provide(providers: &mut rustc_middle::util::Providers) {
    providers.hooks.alloc_self_profile_query_strings =
        self_profile::alloc_self_profile_query_strings;
    providers.hooks.verify_query_key_hashes = incremental::verify_query_key_hashes;
    providers.hooks.encode_query_values = incremental::encode_query_values;
}
