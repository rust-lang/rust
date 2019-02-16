//! This module contains files for saving intermediate work-products.

use crate::persist::fs::*;
use rustc::dep_graph::{WorkProduct, WorkProductId, WorkProductFileKind};
use rustc::session::Session;
use rustc_fs_util::link_or_copy;
use std::path::PathBuf;
use std::fs as std_fs;

pub fn copy_cgu_workproducts_to_incr_comp_cache_dir(
    sess: &Session,
    cgu_name: &str,
    files: &[(WorkProductFileKind, PathBuf)]
) -> Option<(WorkProductId, WorkProduct)> {
    debug!("copy_cgu_workproducts_to_incr_comp_cache_dir({:?},{:?})",
           cgu_name,
           files);
    if sess.opts.incremental.is_none() {
        return None
    }

    let saved_files =
        files.iter()
             .map(|&(kind, ref path)| {
                 let extension = match kind {
                     WorkProductFileKind::Object => "o",
                     WorkProductFileKind::Bytecode => "bc",
                     WorkProductFileKind::BytecodeCompressed => "bc.z",
                 };
                 let file_name = format!("{}.{}", cgu_name, extension);
                 let path_in_incr_dir = in_incr_comp_dir_sess(sess, &file_name);
                 match link_or_copy(path, &path_in_incr_dir) {
                     Ok(_) => Some((kind, file_name)),
                     Err(err) => {
                         sess.warn(&format!("error copying object file `{}` \
                                             to incremental directory as `{}`: {}",
                                            path.display(),
                                            path_in_incr_dir.display(),
                                            err));
                         None
                     }
                 }
             })
             .collect::<Option<Vec<_>>>()?;

    let work_product = WorkProduct {
        cgu_name: cgu_name.to_string(),
        saved_files,
    };

    let work_product_id = WorkProductId::from_cgu_name(cgu_name);
    Some((work_product_id, work_product))
}

pub fn delete_workproduct_files(sess: &Session, work_product: &WorkProduct) {
    for &(_, ref file_name) in &work_product.saved_files {
        let path = in_incr_comp_dir_sess(sess, file_name);
        match std_fs::remove_file(&path) {
            Ok(()) => { }
            Err(err) => {
                sess.warn(
                    &format!("file-system error deleting outdated file `{}`: {}",
                             path.display(), err));
            }
        }
    }
}
