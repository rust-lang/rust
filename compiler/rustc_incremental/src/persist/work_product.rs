//! Functions for saving and removing intermediate [work products].
//!
//! [work products]: WorkProduct

use crate::errors;
use crate::persist::fs::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_fs_util::link_or_copy;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::Session;
use std::fs as std_fs;
use std::path::Path;

/// Copies a CGU work product to the incremental compilation directory, so next compilation can find and reuse it.
pub fn copy_cgu_workproduct_to_incr_comp_cache_dir(
    sess: &Session,
    cgu_name: &str,
    files: &[(&'static str, &Path)],
) -> Option<(WorkProductId, WorkProduct)> {
    debug!(?cgu_name, ?files);
    sess.opts.incremental.as_ref()?;

    let mut saved_files = FxHashMap::default();
    for (ext, path) in files {
        let file_name = format!("{cgu_name}.{ext}");
        let path_in_incr_dir = in_incr_comp_dir_sess(sess, &file_name);
        match link_or_copy(path, &path_in_incr_dir) {
            Ok(_) => {
                let _ = saved_files.insert(ext.to_string(), file_name);
            }
            Err(err) => {
                sess.emit_warning(errors::CopyWorkProductToCache {
                    from: &path,
                    to: &path_in_incr_dir,
                    err,
                });
            }
        }
    }

    let work_product = WorkProduct { cgu_name: cgu_name.to_string(), saved_files };
    debug!(?work_product);
    let work_product_id = WorkProductId::from_cgu_name(cgu_name);
    Some((work_product_id, work_product))
}

/// Removes files for a given work product.
pub fn delete_workproduct_files(sess: &Session, work_product: &WorkProduct) {
    for (_, path) in &work_product.saved_files {
        let path = in_incr_comp_dir_sess(sess, path);
        if let Err(err) = std_fs::remove_file(&path) {
            sess.emit_warning(errors::DeleteWorkProduct { path: &path, err });
        }
    }
}
