//! Support for serializing the dep-graph and reloading it.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

mod assert_dep_graph;
pub mod assert_module_sources;
mod persist;

pub use assert_dep_graph::assert_dep_graph;
pub use persist::copy_cgu_workproduct_to_incr_comp_cache_dir;
pub use persist::delete_workproduct_files;
pub use persist::finalize_session_directory;
pub use persist::garbage_collect_session_directories;
pub use persist::in_incr_comp_dir;
pub use persist::in_incr_comp_dir_sess;
pub use persist::load_query_result_cache;
pub use persist::prepare_session_directory;
pub use persist::save_dep_graph;
pub use persist::save_work_product_index;
pub use persist::LoadResult;
pub use persist::{load_dep_graph, DepGraphFuture};
