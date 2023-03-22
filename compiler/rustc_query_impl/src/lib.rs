//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
// this shouldn't be necessary, but the check for `&mut _` is too naive and denies returning a function pointer that takes a mut ref
#![feature(const_mut_refs)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(rustc_attrs)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_middle;

use rustc_data_structures::sync::AtomicU64;
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKindStruct};
use rustc_middle::query::AsLocalKey;
use rustc_middle::ty::query::{
    query_keys, query_provided, query_provided_to_value, query_storage, query_values,
};
use rustc_middle::ty::query::{ExternProviders, Providers, QueryEngine};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

#[macro_use]
mod plumbing;
pub use plumbing::QueryCtxt;
use rustc_query_system::dep_graph::SerializedDepNodeIndex;
use rustc_query_system::query::*;
#[cfg(parallel_compiler)]
pub use rustc_query_system::query::{deadlock, QueryContext};

pub use rustc_query_system::query::QueryConfig;

mod on_disk_cache;
pub use on_disk_cache::OnDiskCache;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

rustc_query_append! { define_queries! }

impl<'tcx> Queries<'tcx> {
    // Force codegen in the dyn-trait transformation in this crate.
    pub fn as_dyn(&'tcx self) -> &'tcx dyn QueryEngine<'tcx> {
        self
    }
}
