// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains files for saving intermediate work-products.

use persist::util::*;
use rustc::dep_graph::{WorkProduct, WorkProductId};
use rustc::session::Session;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub fn save_trans_partition(sess: &Session,
                            partition_name: &str,
                            partition_hash: u64,
                            path_to_obj_file: &Path) {
    debug!("save_trans_partition({:?},{},{})",
           partition_name,
           partition_hash,
           path_to_obj_file.display());
    if sess.opts.incremental.is_none() {
        return;
    }
    let id = Arc::new(WorkProductId::PartitionObjectFile(partition_name.to_string()));
    let file_name = format!("cgu-{}", partition_name);
    let path_in_incr_dir = in_incr_comp_dir(sess, &file_name).unwrap();

    // try to delete the file if it already exists
    //
    // FIXME(#34955) we can be smarter here -- if we are re-using, no need to do anything
    if path_in_incr_dir.exists() {
        let _ = fs::remove_file(&path_in_incr_dir);
    }

    match
        fs::hard_link(path_to_obj_file, &path_in_incr_dir)
        .or_else(|_| fs::copy(path_to_obj_file, &path_in_incr_dir).map(|_| ()))
    {
        Ok(_) => {
            let work_product = WorkProduct {
                input_hash: partition_hash,
                file_name: file_name,
            };
            sess.dep_graph.insert_work_product(&id, work_product);
        }
        Err(err) => {
            sess.warn(&format!("error copying object file `{}` \
                                to incremental directory as `{}`: {}",
                               path_to_obj_file.display(),
                               path_in_incr_dir.display(),
                               err));
        }
    }
}
