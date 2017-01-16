// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! When in incremental mode, this pass dumps out the dependency graph
//! into the given directory. At the same time, it also hashes the
//! various HIR nodes.

mod data;
mod directory;
mod dirty_clean;
mod fs;
mod hash;
mod load;
mod preds;
mod save;
mod work_product;
mod file_format;

pub use self::fs::finalize_session_directory;
pub use self::fs::in_incr_comp_dir;
pub use self::load::load_dep_graph;
pub use self::save::save_dep_graph;
pub use self::save::save_work_products;
pub use self::work_product::save_trans_partition;
pub use self::work_product::delete_workproduct_files;
