use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::join;
use rustc_middle::dep_graph::{DepGraph, DepKind, WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_serialize::opaque::Encoder;
use rustc_serialize::Encodable as RustcEncodable;
use rustc_session::Session;
use std::fs;
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
                    save_in(sess, query_cache_path, |e| encode_query_cache(tcx, e));
                });
            },
            || {
                sess.time("incr_comp_persist_dep_graph", || {
                    save_in(sess, dep_graph_path, |e| {
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
    save_in(sess, path, |e| encode_work_product_index(&new_work_products, e));

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

fn save_in<F>(sess: &Session, path_buf: PathBuf, encode: F)
where
    F: FnOnce(&mut Encoder),
{
    debug!("save: storing data in {}", path_buf.display());

    // delete the old dep-graph, if any
    // Note: It's important that we actually delete the old file and not just
    // truncate and overwrite it, since it might be a shared hard-link, the
    // underlying data of which we don't want to modify
    if path_buf.exists() {
        match fs::remove_file(&path_buf) {
            Ok(()) => {
                debug!("save: remove old file");
            }
            Err(err) => {
                sess.err(&format!(
                    "unable to delete old dep-graph at `{}`: {}",
                    path_buf.display(),
                    err
                ));
                return;
            }
        }
    }

    // generate the data in a memory buffer
    let mut encoder = Encoder::new(Vec::new());
    file_format::write_file_header(&mut encoder);
    encode(&mut encoder);

    // write the data out
    let data = encoder.into_inner();
    match fs::write(&path_buf, data) {
        Ok(_) => {
            debug!("save: data written to disk successfully");
        }
        Err(err) => {
            sess.err(&format!("failed to write dep-graph to `{}`: {}", path_buf.display(), err));
        }
    }
}

fn encode_dep_graph(tcx: TyCtxt<'_>, encoder: &mut Encoder) {
    // First encode the commandline arguments hash
    tcx.sess.opts.dep_tracking_hash().encode(encoder).unwrap();

    // Encode the graph data.
    let serialized_graph =
        tcx.sess.time("incr_comp_serialize_dep_graph", || tcx.dep_graph.serialize());

    if tcx.sess.opts.debugging_opts.incremental_info {
        #[derive(Clone)]
        struct Stat {
            kind: DepKind,
            node_counter: u64,
            edge_counter: u64,
        }

        let total_node_count = serialized_graph.nodes.len();
        let total_edge_count = serialized_graph.edge_list_data.len();

        let mut counts: FxHashMap<_, Stat> =
            FxHashMap::with_capacity_and_hasher(total_node_count, Default::default());

        for (i, &node) in serialized_graph.nodes.iter_enumerated() {
            let stat = counts.entry(node.kind).or_insert(Stat {
                kind: node.kind,
                node_counter: 0,
                edge_counter: 0,
            });

            stat.node_counter += 1;
            let (edge_start, edge_end) = serialized_graph.edge_list_indices[i];
            stat.edge_counter += (edge_end - edge_start) as u64;
        }

        let mut counts: Vec<_> = counts.values().cloned().collect();
        counts.sort_by_key(|s| -(s.node_counter as i64));

        println!("[incremental]");
        println!("[incremental] DepGraph Statistics");

        const SEPARATOR: &str = "[incremental] --------------------------------\
                                 ----------------------------------------------\
                                 ------------";

        println!("{}", SEPARATOR);
        println!("[incremental]");
        println!("[incremental] Total Node Count: {}", total_node_count);
        println!("[incremental] Total Edge Count: {}", total_edge_count);
        if let Some((total_edge_reads, total_duplicate_edge_reads)) =
            tcx.dep_graph.edge_deduplication_data()
        {
            println!("[incremental] Total Edge Reads: {}", total_edge_reads);
            println!("[incremental] Total Duplicate Edge Reads: {}", total_duplicate_edge_reads);
        }
        println!("[incremental]");
        println!(
            "[incremental]  {:<36}| {:<17}| {:<12}| {:<17}|",
            "Node Kind", "Node Frequency", "Node Count", "Avg. Edge Count"
        );
        println!(
            "[incremental] -------------------------------------\
                  |------------------\
                  |-------------\
                  |------------------|"
        );

        for stat in counts.iter() {
            println!(
                "[incremental]  {:<36}|{:>16.1}% |{:>12} |{:>17.1} |",
                format!("{:?}", stat.kind),
                (100.0 * (stat.node_counter as f64)) / (total_node_count as f64), // percentage of all nodes
                stat.node_counter,
                (stat.edge_counter as f64) / (stat.node_counter as f64), // average edges per kind
            );
        }

        println!("{}", SEPARATOR);
        println!("[incremental]");
    }

    tcx.sess.time("incr_comp_encode_serialized_dep_graph", || {
        serialized_graph.encode(encoder).unwrap();
    });
}

fn encode_work_product_index(
    work_products: &FxHashMap<WorkProductId, WorkProduct>,
    encoder: &mut Encoder,
) {
    let serialized_products: Vec<_> = work_products
        .iter()
        .map(|(id, work_product)| SerializedWorkProduct {
            id: *id,
            work_product: work_product.clone(),
        })
        .collect();

    serialized_products.encode(encoder).unwrap();
}

fn encode_query_cache(tcx: TyCtxt<'_>, encoder: &mut Encoder) {
    tcx.sess.time("incr_comp_serialize_result_cache", || {
        tcx.serialize_query_result_cache(encoder).unwrap();
    })
}
