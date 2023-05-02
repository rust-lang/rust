//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
// this shouldn't be necessary, but the check for `&mut _` is too naive and denies returning a function pointer that takes a mut ref
#![feature(const_mut_refs)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_middle;

use crate::plumbing::{encode_all_query_results, try_mark_green};
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKind, DepKindStruct};
use rustc_middle::query::erase::{erase, restore, Erase};
use rustc_middle::query::AsLocalKey;
use rustc_middle::ty::query::{
    query_keys, query_provided, query_provided_to_value, query_storage, query_values,
};
use rustc_middle::ty::query::{ExternProviders, Providers, QueryEngine, QuerySystemFns};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::SerializedDepNodeIndex;
use rustc_query_system::Value;
use rustc_span::Span;

#[macro_use]
mod plumbing;
pub use crate::plumbing::QueryCtxt;

pub use rustc_query_system::query::QueryConfig;
use rustc_query_system::query::*;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

/// This is implemented per query and restoring query values from their erased state.
trait QueryConfigRestored<'tcx>: QueryConfig<QueryCtxt<'tcx>> + Default {
    type RestoredValue;

    fn restore(value: <Self as QueryConfig<QueryCtxt<'tcx>>>::Value) -> Self::RestoredValue;
}

rustc_query_append! { define_queries! }

pub fn query_system_fns<'tcx>(
    local_providers: Providers,
    extern_providers: ExternProviders,
) -> QuerySystemFns<'tcx> {
    QuerySystemFns {
        engine: engine(),
        local_providers,
        extern_providers,
        query_structs: make_dep_kind_array!(query_structs).to_vec(),
        encode_query_results: encode_all_query_results,
        try_mark_green: try_mark_green,
    }
}
