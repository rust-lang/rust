//! This module contains files for saving intermediate work-products.

use crate::persist::fs::*;
use rustc_fs_util::link_or_copy;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::Session;
use std::fs as std_fs;
use std::path::PathBuf;

pub fn copy_cgu_workproducts_to_incr_comp_cache_dir(
    sess: &Session,
    cgu_name: &str,
    files: &[PathBuf],
) -> Option<(WorkProductId, WorkProduct)> {
    debug!("copy_cgu_workproducts_to_incr_comp_cache_dir({:?},{:?})", cgu_name, files);
    sess.opts.incremental.as_ref()?;

    let saved_files = files
        .iter()
        .map(|path| {
            let file_name = format!("{}.o", cgu_name);
            let path_in_incr_dir = in_incr_comp_dir_sess(sess, &file_name);
            match link_or_copy(path, &path_in_incr_dir) {
                Ok(_) => Some(file_name),
                Err(err) => {
                    sess.warn(&format!(
                        "error copying object file `{}` \
                                             to incremental directory as `{}`: {}",
                        path.display(),
                        path_in_incr_dir.display(),
                        err
                    ));
                    None
                }
            }
        })
        .collect::<Option<Vec<_>>>()?;

    let work_product = WorkProduct { cgu_name: cgu_name.to_string(), saved_files };

    let work_product_id = WorkProductId::from_cgu_name(cgu_name);
    Some((work_product_id, work_product))
}

pub fn delete_workproduct_files(sess: &Session, work_product: &WorkProduct) {
    for file_name in &work_product.saved_files {
        let path = in_incr_comp_dir_sess(sess, file_name);
        match std_fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) => {
                sess.warn(&format!(
                    "file-system error deleting outdated file `{}`: {}",
                    path.display(),
                    err
                ));
            }
        }
    }
}
