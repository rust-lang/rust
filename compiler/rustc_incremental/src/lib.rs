//! Support for serializing the dep-graph and reloading it.

// tidy-alphabetical-start
#![allow(internal_features)]
#![deny(missing_docs)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(file_buffered)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

mod assert_dep_graph;
mod errors;
mod persist;

pub use persist::{
    LoadResult, copy_cgu_workproduct_to_incr_comp_cache_dir, finalize_session_directory,
    in_incr_comp_dir, in_incr_comp_dir_sess, load_query_result_cache, save_work_product_index,
    setup_dep_graph,
};
use rustc_middle::util::Providers;

#[allow(missing_docs)]
pub fn provide(providers: &mut Providers) {
    providers.hooks.save_dep_graph =
        |tcx| tcx.sess.time("serialize_dep_graph", || persist::save_dep_graph(tcx));
}

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
