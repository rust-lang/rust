//! Support for serializing the dep-graph and reloading it.

#![deny(missing_docs)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(never_type)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_middle;
#[macro_use]
extern crate tracing;

mod assert_dep_graph;
pub mod assert_module_sources;
mod errors;
mod persist;

use assert_dep_graph::assert_dep_graph;
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
pub use persist::{build_dep_graph, load_dep_graph, DepGraphFuture};

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;

fluent_messages! { "../messages.ftl" }
