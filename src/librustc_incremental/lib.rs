//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(specialization)]

#![recursion_limit="256"]

#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

#[macro_use] extern crate rustc;
#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize; // used by deriving

#[macro_use] extern crate log;

mod assert_dep_graph;
pub mod assert_module_sources;
mod persist;

pub use assert_dep_graph::assert_dep_graph;
pub use persist::dep_graph_tcx_init;
pub use persist::{DepGraphFuture, load_dep_graph};
pub use persist::load_query_result_cache;
pub use persist::LoadResult;
pub use persist::copy_cgu_workproducts_to_incr_comp_cache_dir;
pub use persist::save_dep_graph;
pub use persist::save_work_product_index;
pub use persist::in_incr_comp_dir;
pub use persist::in_incr_comp_dir_sess;
pub use persist::prepare_session_directory;
pub use persist::finalize_session_directory;
pub use persist::delete_workproduct_files;
pub use persist::garbage_collect_session_directories;
