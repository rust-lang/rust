use std::fs;
use std::sync::Arc;

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::join;
use rustc_middle::dep_graph::{
    DepGraph, SerializedDepGraph, WorkProduct, WorkProductId, WorkProductMap,
};
use rustc_middle::ty::TyCtxt;
use rustc_serialize::Encodable as RustcEncodable;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use rustc_session::Session;
use tracing::debug;

use super::data::*;
use super::fs::*;
use super::{dirty_clean, file_format, work_product};
use crate::assert_dep_graph::assert_dep_graph;
use crate::errors;

/// Saves and writes the [`DepGraph`] to the file system.
///
/// This function saves both the dep-graph and the query result cache,
/// and drops the result cache.
///
/// This function should only run after all queries have completed.
/// Trying to execute a query afterwards would attempt to read the result cache we just dropped.
pub(crate) fn save_dep_graph(tcx: TyCtxt<'_>) {
    debug!("save_dep_graph()");
    tcx.dep_graph.with_ignore(|| {
        let sess = tcx.sess;
        if sess.opts.incremental.is_none() {
            return;
        }
        // This is going to be deleted in finalize_session_directory, so let's not create it.
        if sess.dcx().has_errors_or_delayed_bugs().is_some() {
            return;
        }

        let query_cache_path = query_cache_path(sess);
        let dep_graph_path = dep_graph_path(sess);
        let staging_dep_graph_path = staging_dep_graph_path(sess);

        sess.time("assert_dep_graph", || assert_dep_graph(tcx));
        sess.time("check_dirty_clean", || dirty_clean::check_dirty_clean_annotations(tcx));

        join(
            move || {
                sess.time("incr_comp_persist_dep_graph", || {
                    if let Err(err) = fs::rename(&staging_dep_graph_path, &dep_graph_path) {
                        sess.dcx().emit_err(errors::MoveDepGraph {
                            from: &staging_dep_graph_path,
                            to: &dep_graph_path,
                            err,
                        });
                    }
                });
            },
            move || {
                // We execute this after `incr_comp_persist_dep_graph` for the serial compiler
                // to catch any potential query execution writing to the dep graph.
                sess.time("incr_comp_persist_result_cache", || {
                    // Drop the memory map so that we can remove the file and write to it.
                    if let Some(odc) = &tcx.query_system.on_disk_cache {
                        odc.drop_serialized_data(tcx);
                    }

                    file_format::save_in(sess, query_cache_path, "query cache", |e| {
                        encode_query_cache(tcx, e)
                    });
                });
            },
        );
    })
}

/// Saves the work product index.
pub fn save_work_product_index(
    sess: &Session,
    dep_graph: &DepGraph,
    new_work_products: FxIndexMap<WorkProductId, WorkProduct>,
) {
    if sess.opts.incremental.is_none() {
        return;
    }
    // This is going to be deleted in finalize_session_directory, so let's not create it
    if sess.dcx().has_errors().is_some() {
        return;
    }

    debug!("save_work_product_index()");
    dep_graph.assert_ignored();
    let path = work_products_path(sess);
    file_format::save_in(sess, path, "work product index", |mut e| {
        encode_work_product_index(&new_work_products, &mut e);
        e.finish()
    });

    // We also need to clean out old work-products, as not all of them are
    // deleted during invalidation. Some object files don't change their
    // content, they are just not needed anymore.
    let previous_work_products = dep_graph.previous_work_products();
    for (id, wp) in previous_work_products.to_sorted_stable_ord() {
        if !new_work_products.contains_key(id) {
            work_product::delete_workproduct_files(sess, wp);
            debug_assert!(
                !wp.saved_files.items().all(|(_, path)| in_incr_comp_dir_sess(sess, path).exists())
            );
        }
    }

    // Check that we did not delete one of the current work-products:
    debug_assert!({
        new_work_products.iter().all(|(_, wp)| {
            wp.saved_files.items().all(|(_, path)| in_incr_comp_dir_sess(sess, path).exists())
        })
    });
}

fn encode_work_product_index(
    work_products: &FxIndexMap<WorkProductId, WorkProduct>,
    encoder: &mut FileEncoder,
) {
    let serialized_products: Vec<_> = work_products
        .iter()
        .map(|(id, work_product)| SerializedWorkProduct {
            id: *id,
            work_product: work_product.clone(),
        })
        .collect();

    serialized_products.encode(encoder)
}

fn encode_query_cache(tcx: TyCtxt<'_>, encoder: FileEncoder) -> FileEncodeResult {
    tcx.sess.time("incr_comp_serialize_result_cache", || tcx.serialize_query_result_cache(encoder))
}

/// Builds the dependency graph.
///
/// This function creates the *staging dep-graph*. When the dep-graph is modified by a query
/// execution, the new dependency information is not kept in memory but directly
/// output to this file. `save_dep_graph` then finalizes the staging dep-graph
/// and moves it to the permanent dep-graph path
pub(crate) fn build_dep_graph(
    sess: &Session,
    prev_graph: Arc<SerializedDepGraph>,
    prev_work_products: WorkProductMap,
) -> Option<DepGraph> {
    if sess.opts.incremental.is_none() {
        // No incremental compilation.
        return None;
    }

    // Stream the dep-graph to an alternate file, to avoid overwriting anything in case of errors.
    let path_buf = staging_dep_graph_path(sess);

    let mut encoder = match FileEncoder::new(&path_buf) {
        Ok(encoder) => encoder,
        Err(err) => {
            sess.dcx().emit_err(errors::CreateDepGraph { path: &path_buf, err });
            return None;
        }
    };

    file_format::write_file_header(&mut encoder, sess);

    // First encode the commandline arguments hash
    sess.opts.dep_tracking_hash(false).encode(&mut encoder);

    Some(DepGraph::new(sess, prev_graph, prev_work_products, encoder))
}
