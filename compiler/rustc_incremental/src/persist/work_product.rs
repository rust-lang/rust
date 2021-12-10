//! Functions for saving and removing intermediate [work products].
//!
//! [work products]: WorkProduct

use crate::persist::fs::*;
use rustc_fs_util::link_or_copy;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::Session;
use std::fs as std_fs;
use std::path::PathBuf;

/// Copies a CGU work product to the incremental compilation directory, so next compilation can find and reuse it.
pub fn copy_cgu_workproduct_to_incr_comp_cache_dir(
    sess: &Session,
    cgu_name: &str,
    path: &Option<PathBuf>,
) -> Option<(WorkProductId, WorkProduct)> {
    debug!("copy_cgu_workproduct_to_incr_comp_cache_dir({:?},{:?})", cgu_name, path);
    sess.opts.incremental.as_ref()?;

    let saved_file = if let Some(path) = path {
        let file_name = format!("{}.o", cgu_name);
        let path_in_incr_dir = in_incr_comp_dir_sess(sess, &file_name);
        match link_or_copy(path, &path_in_incr_dir) {
            Ok(_) => Some(file_name),
            Err(err) => {
                sess.warn(&format!(
                    "error copying object file `{}` to incremental directory as `{}`: {}",
                    path.display(),
                    path_in_incr_dir.display(),
                    err
                ));
                return None;
            }
        }
    } else {
        None
    };

    let work_product = WorkProduct { cgu_name: cgu_name.to_string(), saved_file };

    let work_product_id = WorkProductId::from_cgu_name(cgu_name);
    Some((work_product_id, work_product))
}

/// Removes files for a given work product.
pub fn delete_workproduct_files(sess: &Session, work_product: &WorkProduct) {
    if let Some(ref file_name) = work_product.saved_file {
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
