//! Functions for saving and removing intermediate [work products].
//!
//! [work products]: WorkProduct

use std::fs as std_fs;
use std::path::{Path, PathBuf};

use rustc_data_structures::unord::UnordMap;
use rustc_fs_util::link_or_copy;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::Session;
use tracing::debug;

use crate::errors;
use crate::persist::fs::*;

/// Copies a CGU work product to the incremental compilation directory, so next compilation can
/// find and reuse it.
pub fn copy_cgu_workproduct_to_incr_comp_cache_dir(
    sess: &Session,
    cgu_name: &str,
    files: &[(&'static str, &Path)],
    known_links: &[PathBuf],
) -> Option<(WorkProductId, WorkProduct)> {
    debug!(?cgu_name, ?files);
    sess.opts.incremental.as_ref()?;

    let mut saved_files = UnordMap::default();
    for (ext, path) in files {
        let file_name = format!("{cgu_name}.{ext}");
        let path_in_incr_dir = in_incr_comp_dir_sess(sess, &file_name);
        if known_links.contains(&path_in_incr_dir) {
            let _ = saved_files.insert(ext.to_string(), file_name);
            continue;
        }
        match link_or_copy(path, &path_in_incr_dir) {
            Ok(_) => {
                let _ = saved_files.insert(ext.to_string(), file_name);
            }
            Err(err) => {
                sess.dcx().emit_warn(errors::CopyWorkProductToCache {
                    from: path,
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
pub(crate) fn delete_workproduct_files(sess: &Session, work_product: &WorkProduct) {
    for (_, path) in work_product.saved_files.items().into_sorted_stable_ord() {
        let path = in_incr_comp_dir_sess(sess, path);
        if let Err(err) = std_fs::remove_file(&path) {
            sess.dcx().emit_warn(errors::DeleteWorkProduct { path: &path, err });
        }
    }
}
