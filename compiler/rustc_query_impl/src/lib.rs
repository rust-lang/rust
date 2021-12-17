//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(crate_visibility_modifier)]
#![feature(nll)]
#![feature(min_specialization)]
#![feature(once_cell)]
#![feature(rustc_attrs)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate rustc_middle;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_middle::arena::Arena;
use rustc_middle::dep_graph::{self, DepKindStruct, SerializedDepNodeIndex};
use rustc_middle::ty::query::{query_keys, query_storage, query_stored, query_values};
use rustc_middle::ty::query::{ExternProviders, Providers, QueryEngine};
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;

#[macro_use]
mod plumbing;
pub use plumbing::QueryCtxt;
use rustc_query_system::query::*;

mod stats;
pub use self::stats::print_stats;

mod keys;
use keys::Key;

mod values;
use self::values::Value;

pub use rustc_query_system::query::QueryConfig;
pub(crate) use rustc_query_system::query::{QueryDescription, QueryVtable};

mod on_disk_cache;
pub use on_disk_cache::OnDiskCache;

mod profiling_support;
pub use self::profiling_support::alloc_self_profile_query_strings;

mod util;

fn describe_as_module(def_id: LocalDefId, tcx: TyCtxt<'_>) -> String {
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id.to_def_id()))
    }
}

rustc_query_append! { [define_queries!][<'tcx>] }

impl<'tcx> Queries<'tcx> {
    // Force codegen in the dyn-trait transformation in this crate.
    pub fn as_dyn(&'tcx self) -> &'tcx dyn QueryEngine<'tcx> {
        self
    }
}
