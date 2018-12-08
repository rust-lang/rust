//! When in incremental mode, this pass dumps out the dependency graph
//! into the given directory. At the same time, it also hashes the
//! various HIR nodes.

mod data;
mod dirty_clean;
mod fs;
mod load;
mod save;
mod work_product;
mod file_format;

pub use fs::finalize_session_directory;
pub use fs::garbage_collect_session_directories;
pub use fs::in_incr_comp_dir;
pub use fs::in_incr_comp_dir_sess;
pub use fs::prepare_session_directory;
pub use load::dep_graph_tcx_init;
pub use load::{DepGraphFuture, load_dep_graph};
pub use load::load_query_result_cache;
pub use load::LoadResult;
pub use save::save_dep_graph;
pub use save::save_work_product_index;
pub use work_product::copy_cgu_workproducts_to_incr_comp_cache_dir;
pub use work_product::delete_workproduct_files;
