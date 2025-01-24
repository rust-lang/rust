//! When in incremental mode, this pass dumps out the dependency graph
//! into the given directory. At the same time, it also hashes the
//! various HIR nodes.

mod data;
mod dirty_clean;
mod file_format;
mod fs;
mod load;
mod save;
mod work_product;

pub use fs::{finalize_session_directory, in_incr_comp_dir, in_incr_comp_dir_sess};
pub use load::{LoadResult, load_query_result_cache, setup_dep_graph};
pub(crate) use save::save_dep_graph;
pub use save::save_work_product_index;
pub use work_product::copy_cgu_workproduct_to_incr_comp_cache_dir;
