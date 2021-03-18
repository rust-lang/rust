use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::join;
use rustc_middle::dep_graph::{DepGraph, WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use rustc_serialize::Encodable as RustcEncodable;
use rustc_session::Session;
use std::fs;
use std::io;
use std::path::PathBuf;

use super::data::*;
use super::dirty_clean;
use super::file_format;
use super::fs::*;
use super::work_product;

pub fn save_dep_graph(tcx: TyCtxt<'_>) {
    debug!("save_dep_graph()");
    tcx.dep_graph.with_ignore(|| {
        let sess = tcx.sess;
        if sess.opts.incremental.is_none() {
            return;
        }
        // This is going to be deleted in finalize_session_directory, so let's not create it
        if sess.has_errors_or_delayed_span_bugs() {
            return;
        }

        let query_cache_path = query_cache_path(sess);
        let dep_graph_path = dep_graph_path(sess);

        join(
            move || {
                sess.time("incr_comp_persist_result_cache", || {
                    save_in(sess, query_cache_path, "query cache", |e| encode_query_cache(tcx, e));
                });
            },
            || {
                sess.time("incr_comp_persist_dep_graph", || {
                    save_in(sess, dep_graph_path, "dependency graph", |e| {
                        sess.time("incr_comp_encode_dep_graph", || encode_dep_graph(tcx, e))
                    });
                });
            },
        );

        dirty_clean::check_dirty_clean_annotations(tcx);
    })
}

pub fn save_work_product_index(
    sess: &Session,
    dep_graph: &DepGraph,
    new_work_products: FxHashMap<WorkProductId, WorkProduct>,
) {
    if sess.opts.incremental.is_none() {
        return;
    }
    // This is going to be deleted in finalize_session_directory, so let's not create it
    if sess.has_errors_or_delayed_span_bugs() {
        return;
    }

    debug!("save_work_product_index()");
    dep_graph.assert_ignored();
    let path = work_products_path(sess);
    save_in(sess, path, "work product index", |e| encode_work_product_index(&new_work_products, e));

    // We also need to clean out old work-products, as not all of them are
    // deleted during invalidation. Some object files don't change their
    // content, they are just not needed anymore.
    let previous_work_products = dep_graph.previous_work_products();
    for (id, wp) in previous_work_products.iter() {
        if !new_work_products.contains_key(id) {
            work_product::delete_workproduct_files(sess, wp);
            debug_assert!(
                wp.saved_file.as_ref().map_or(true, |file_name| {
                    !in_incr_comp_dir_sess(sess, &file_name).exists()
                })
            );
        }
    }

    // Check that we did not delete one of the current work-products:
    debug_assert!({
        new_work_products
            .iter()
            .flat_map(|(_, wp)| wp.saved_file.iter())
            .map(|name| in_incr_comp_dir_sess(sess, name))
            .all(|path| path.exists())
    });
}

fn save_in<F>(sess: &Session, path_buf: PathBuf, name: &str, encode: F)
where
    F: FnOnce(&mut FileEncoder) -> FileEncodeResult,
{
    debug!("save: storing data in {}", path_buf.display());

    // Delete the old file, if any.
    // Note: It's important that we actually delete the old file and not just
    // truncate and overwrite it, since it might be a shared hard-link, the
    // underlying data of which we don't want to modify
    match fs::remove_file(&path_buf) {
        Ok(()) => {
            debug!("save: remove old file");
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => (),
        Err(err) => {
            sess.err(&format!(
                "unable to delete old {} at `{}`: {}",
                name,
                path_buf.display(),
                err
            ));
            return;
        }
    }

    let mut encoder = match FileEncoder::new(&path_buf) {
        Ok(encoder) => encoder,
        Err(err) => {
            sess.err(&format!("failed to create {} at `{}`: {}", name, path_buf.display(), err));
            return;
        }
    };

    if let Err(err) = file_format::write_file_header(&mut encoder, sess.is_nightly_build()) {
        sess.err(&format!("failed to write {} header to `{}`: {}", name, path_buf.display(), err));
        return;
    }

    if let Err(err) = encode(&mut encoder) {
        sess.err(&format!("failed to write {} to `{}`: {}", name, path_buf.display(), err));
        return;
    }

    if let Err(err) = encoder.flush() {
        sess.err(&format!("failed to flush {} to `{}`: {}", name, path_buf.display(), err));
        return;
    }

    debug!("save: data written to disk successfully");
}

fn encode_dep_graph(tcx: TyCtxt<'_>, encoder: &mut FileEncoder) -> FileEncodeResult {
    // First encode the commandline arguments hash
    tcx.sess.opts.dep_tracking_hash().encode(encoder)?;

    if tcx.sess.opts.debugging_opts.incremental_info {
        tcx.dep_graph.print_incremental_info();
    }

    // There is a tiny window between printing the incremental info above and encoding the dep
    // graph below in which the dep graph could change, thus making the printed incremental info
    // slightly out of date. If this matters to you, please feel free to submit a patch. :)

    tcx.sess.time("incr_comp_encode_serialized_dep_graph", || tcx.dep_graph.encode(encoder))
}

fn encode_work_product_index(
    work_products: &FxHashMap<WorkProductId, WorkProduct>,
    encoder: &mut FileEncoder,
) -> FileEncodeResult {
    let serialized_products: Vec<_> = work_products
        .iter()
        .map(|(id, work_product)| SerializedWorkProduct {
            id: *id,
            work_product: work_product.clone(),
        })
        .collect();

    serialized_products.encode(encoder)
}

fn encode_query_cache(tcx: TyCtxt<'_>, encoder: &mut FileEncoder) -> FileEncodeResult {
    tcx.sess.time("incr_comp_serialize_result_cache", || tcx.serialize_query_result_cache(encoder))
}
