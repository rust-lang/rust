//! Functions for saving and removing intermediate [work products].
//!
//! [work products]: WorkProduct

use std::path::{Path, PathBuf};
use std::{fs as std_fs, io};

use rustc_data_structures::unord::UnordMap;
use rustc_fs_util::link_or_copy;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_session::config::{DWARF_OBJECT_EXT, OutputFilenames};
use rustc_session::{IncrCompSession, Session};
use tracing::debug;

use crate::diagnostics;
use crate::persist::fs::*;

/// Copies a CGU work product to the incremental compilation directory, so next compilation can
/// find and reuse it.
///
/// Panics when incr comp is disabled.
pub fn copy_cgu_workproduct_to_incr_comp_cache_dir(
    sess: &Session,
    incr_comp_session: &IncrCompSession,
    cgu_name: &str,
    files: &[(&'static str, &Path)],
    known_links: &[PathBuf],
    invocation_temp: Option<&str>,
    preserved_debuginfo_extensions: &[&str],
) -> (WorkProductId, WorkProduct) {
    debug!(?cgu_name, ?files);
    assert!(sess.opts.incremental.is_some());

    let mut saved_files = UnordMap::default();
    for (ext, path) in files {
        let file_name = format!("{cgu_name}.{ext}");
        let path_in_incr_dir = in_incr_comp_dir_sess(incr_comp_session, &file_name);
        if known_links.contains(&path_in_incr_dir) {
            let _ = saved_files.insert(ext.to_string(), file_name);
            continue;
        }
        match link_or_copy(path, &path_in_incr_dir) {
            Ok(_) => {
                let _ = saved_files.insert(ext.to_string(), file_name);
            }
            Err(err) => {
                sess.dcx().emit_warn(diagnostics::CopyWorkProductToCache {
                    from: path,
                    to: &path_in_incr_dir,
                    err,
                });
            }
        }
    }

    let work_product = WorkProduct {
        cgu_name: cgu_name.to_string(),
        saved_files,
        invocation_temp: invocation_temp.map(String::from),
        preserved_debuginfo_extensions: preserved_debuginfo_extensions
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    debug!(?work_product);
    let work_product_id = WorkProductId::from_cgu_name(cgu_name);
    (work_product_id, work_product)
}

/// Removes the temporaries a previous session's work product left in the output directory for
/// debuginfo. Their paths are derived from this session's output settings, so only the
/// directories this session writes to are touched. Files that are already gone are not an error.
pub(crate) fn delete_preserved_debuginfo_files(
    sess: &Session,
    output_filenames: &OutputFilenames,
    work_product: &WorkProduct,
) {
    // Files with this session's own invocation string are its own temporaries.
    if output_filenames.invocation_temp == work_product.invocation_temp {
        return;
    }

    let mut previous_output_filenames = output_filenames.clone();
    previous_output_filenames.invocation_temp = work_product.invocation_temp.clone();
    for ext in &work_product.preserved_debuginfo_extensions {
        let path = if ext == DWARF_OBJECT_EXT {
            previous_output_filenames.temp_path_dwo_for_cgu(&work_product.cgu_name)
        } else {
            previous_output_filenames.temp_path_ext_for_cgu(ext, &work_product.cgu_name)
        };
        if let Err(err) = std_fs::remove_file(&path)
            && err.kind() != io::ErrorKind::NotFound
        {
            sess.dcx().emit_warn(diagnostics::DeleteWorkProduct { path: &path, err });
        }
    }
}

/// Removes files for a given work product.
pub(crate) fn delete_workproduct_files(
    sess: &Session,
    incr_comp_session: &IncrCompSession,
    work_product: &WorkProduct,
) {
    for (_, path) in work_product.saved_files.items().into_sorted_stable_ord() {
        let path = in_incr_comp_dir_sess(incr_comp_session, path);
        if let Err(err) = std_fs::remove_file(&path) {
            sess.dcx().emit_warn(diagnostics::DeleteWorkProduct { path: &path, err });
        }
    }
}
