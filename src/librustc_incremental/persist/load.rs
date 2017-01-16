// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code to save/load the dep-graph from files.

use rustc::dep_graph::DepNode;
use rustc::hir::def_id::DefId;
use rustc::hir::svh::Svh;
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_serialize::Decodable as RustcDecodable;
use rustc_serialize::opaque::Decoder;
use std::path::{Path};

use IncrementalHashesMap;
use ich::Fingerprint;
use super::data::*;
use super::directory::*;
use super::dirty_clean;
use super::hash::*;
use super::fs::*;
use super::file_format;
use super::work_product;

pub type DirtyNodes = FxHashSet<DepNode<DefPathIndex>>;

/// If we are in incremental mode, and a previous dep-graph exists,
/// then load up those nodes/edges that are still valid into the
/// dep-graph for this session. (This is assumed to be running very
/// early in compilation, before we've really done any work, but
/// actually it doesn't matter all that much.) See `README.md` for
/// more general overview.
pub fn load_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                incremental_hashes_map: &IncrementalHashesMap) {
    if tcx.sess.opts.incremental.is_none() {
        return;
    }

    match prepare_session_directory(tcx) {
        Ok(true) => {
            // We successfully allocated a session directory and there is
            // something in it to load, so continue
        }
        Ok(false) => {
            // We successfully allocated a session directory, but there is no
            // dep-graph data in it to load (because this is the first
            // compilation session with this incr. comp. dir.)
            return
        }
        Err(()) => {
            // Something went wrong while trying to allocate the session
            // directory. Don't try to use it any further.
            return
        }
    }

    let _ignore = tcx.dep_graph.in_ignore();
    load_dep_graph_if_exists(tcx, incremental_hashes_map);
}

fn load_dep_graph_if_exists<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      incremental_hashes_map: &IncrementalHashesMap) {
    let dep_graph_path = dep_graph_path(tcx.sess);
    let dep_graph_data = match load_data(tcx.sess, &dep_graph_path) {
        Some(p) => p,
        None => return // no file
    };

    let work_products_path = work_products_path(tcx.sess);
    let work_products_data = match load_data(tcx.sess, &work_products_path) {
        Some(p) => p,
        None => return // no file
    };

    match decode_dep_graph(tcx, incremental_hashes_map, &dep_graph_data, &work_products_data) {
        Ok(dirty_nodes) => dirty_nodes,
        Err(err) => {
            tcx.sess.warn(
                &format!("decoding error in dep-graph from `{}` and `{}`: {}",
                         dep_graph_path.display(),
                         work_products_path.display(),
                         err));
        }
    }
}

fn load_data(sess: &Session, path: &Path) -> Option<Vec<u8>> {
    match file_format::read_file(sess, path) {
        Ok(Some(data)) => return Some(data),
        Ok(None) => {
            // The file either didn't exist or was produced by an incompatible
            // compiler version. Neither is an error.
        }
        Err(err) => {
            sess.err(
                &format!("could not load dep-graph from `{}`: {}",
                         path.display(), err));
        }
    }

    if let Err(err) = delete_all_session_dir_contents(sess) {
        sess.err(&format!("could not clear incompatible incremental \
                           compilation session directory `{}`: {}",
                          path.display(), err));
    }

    None
}

/// Decode the dep graph and load the edges/nodes that are still clean
/// into `tcx.dep_graph`.
pub fn decode_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  incremental_hashes_map: &IncrementalHashesMap,
                                  dep_graph_data: &[u8],
                                  work_products_data: &[u8])
                                  -> Result<(), String>
{
    // Decode the list of work_products
    let mut work_product_decoder = Decoder::new(work_products_data, 0);
    let work_products = <Vec<SerializedWorkProduct>>::decode(&mut work_product_decoder)?;

    // Deserialize the directory and dep-graph.
    let mut dep_graph_decoder = Decoder::new(dep_graph_data, 0);
    let prev_commandline_args_hash = u64::decode(&mut dep_graph_decoder)?;

    if prev_commandline_args_hash != tcx.sess.opts.dep_tracking_hash() {
        if tcx.sess.opts.debugging_opts.incremental_info {
            println!("incremental: completely ignoring cache because of \
                      differing commandline arguments");
        }
        // We can't reuse the cache, purge it.
        debug!("decode_dep_graph: differing commandline arg hashes");
        for swp in work_products {
            delete_dirty_work_product(tcx, swp);
        }

        // No need to do any further work
        return Ok(());
    }

    let directory = DefIdDirectory::decode(&mut dep_graph_decoder)?;
    let serialized_dep_graph = SerializedDepGraph::decode(&mut dep_graph_decoder)?;

    // Retrace the paths in the directory to find their current location (if any).
    let retraced = directory.retrace(tcx);

    // Compute the set of Hir nodes whose data has changed or which
    // have been removed.  These are "raw" source nodes, which means
    // that they still use the original `DefPathIndex` values from the
    // encoding, rather than having been retraced to a `DefId`. The
    // reason for this is that this way we can include nodes that have
    // been removed (which no longer have a `DefId` in the current
    // compilation).
    let dirty_raw_source_nodes = dirty_nodes(tcx,
                                             incremental_hashes_map,
                                             &serialized_dep_graph.hashes,
                                             &retraced);

    // Create a list of (raw-source-node ->
    // retracted-target-node) edges. In the process of retracing the
    // target nodes, we may discover some of them def-paths no longer exist,
    // in which case there is no need to mark the corresopnding nodes as dirty
    // (they are just not present). So this list may be smaller than the original.
    //
    // Note though that in the common case the target nodes are
    // `DepNode::WorkProduct` instances, and those don't have a
    // def-id, so they will never be considered to not exist. Instead,
    // we do a secondary hashing step (later, in trans) when we know
    // the set of symbols that go into a work-product: if any symbols
    // have been removed (or added) the hash will be different and
    // we'll ignore the work-product then.
    let retraced_edges: Vec<_> =
        serialized_dep_graph.edges.iter()
                                  .filter_map(|&(ref raw_source_node, ref raw_target_node)| {
                                      retraced.map(raw_target_node)
                                              .map(|target_node| (raw_source_node, target_node))
                                  })
                                  .collect();

    // Compute which work-products have an input that has changed or
    // been removed. Put the dirty ones into a set.
    let mut dirty_target_nodes = FxHashSet();
    for &(raw_source_node, ref target_node) in &retraced_edges {
        if dirty_raw_source_nodes.contains(raw_source_node) {
            if !dirty_target_nodes.contains(target_node) {
                dirty_target_nodes.insert(target_node.clone());

                if tcx.sess.opts.debugging_opts.incremental_info {
                    // It'd be nice to pretty-print these paths better than just
                    // using the `Debug` impls, but wev.
                    println!("incremental: module {:?} is dirty because {:?} \
                              changed or was removed",
                             target_node,
                             raw_source_node.map_def(|&index| {
                                 Some(directory.def_path_string(tcx, index))
                             }).unwrap());
                }
            }
        }
    }

    // For work-products that are still clean, add their deps into the
    // graph. This is needed because later we will have to save this
    // back out again!
    let dep_graph = tcx.dep_graph.clone();
    for (raw_source_node, target_node) in retraced_edges {
        if dirty_target_nodes.contains(&target_node) {
            continue;
        }

        let source_node = retraced.map(raw_source_node).unwrap();

        debug!("decode_dep_graph: clean edge: {:?} -> {:?}", source_node, target_node);

        let _task = dep_graph.in_task(target_node);
        dep_graph.read(source_node);
    }

    // Add in work-products that are still clean, and delete those that are
    // dirty.
    reconcile_work_products(tcx, work_products, &dirty_target_nodes);

    dirty_clean::check_dirty_clean_annotations(tcx, &dirty_raw_source_nodes, &retraced);

    load_prev_metadata_hashes(tcx,
                              &retraced,
                              &mut *incremental_hashes_map.prev_metadata_hashes.borrow_mut());
    Ok(())
}

/// Computes which of the original set of def-ids are dirty. Stored in
/// a bit vector where the index is the DefPathIndex.
fn dirty_nodes<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         incremental_hashes_map: &IncrementalHashesMap,
                         serialized_hashes: &[SerializedHash],
                         retraced: &RetracedDefIdDirectory)
                         -> DirtyNodes {
    let mut hcx = HashContext::new(tcx, incremental_hashes_map);
    let mut dirty_nodes = FxHashSet();

    for hash in serialized_hashes {
        if let Some(dep_node) = retraced.map(&hash.dep_node) {
            let current_hash = hcx.hash(&dep_node).unwrap();
            if current_hash == hash.hash {
                debug!("initial_dirty_nodes: {:?} is clean (hash={:?})",
                   dep_node.map_def(|&def_id| Some(tcx.def_path(def_id))).unwrap(),
                   current_hash);
                continue;
            }

            if tcx.sess.opts.debugging_opts.incremental_dump_hash {
                println!("node {:?} is dirty as hash is {:?} was {:?}",
                         dep_node.map_def(|&def_id| Some(tcx.def_path(def_id))).unwrap(),
                         current_hash,
                         hash.hash);
            }

            debug!("initial_dirty_nodes: {:?} is dirty as hash is {:?}, was {:?}",
                   dep_node.map_def(|&def_id| Some(tcx.def_path(def_id))).unwrap(),
                   current_hash,
                   hash.hash);
        } else {
            if tcx.sess.opts.debugging_opts.incremental_dump_hash {
                println!("node {:?} is dirty as it was removed",
                         hash.dep_node);
            }

            debug!("initial_dirty_nodes: {:?} is dirty as it was removed",
                   hash.dep_node);
        }

        dirty_nodes.insert(hash.dep_node.clone());
    }

    dirty_nodes
}

/// Go through the list of work-products produced in the previous run.
/// Delete any whose nodes have been found to be dirty or which are
/// otherwise no longer applicable.
fn reconcile_work_products<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     work_products: Vec<SerializedWorkProduct>,
                                     dirty_target_nodes: &FxHashSet<DepNode<DefId>>) {
    debug!("reconcile_work_products({:?})", work_products);
    for swp in work_products {
        if dirty_target_nodes.contains(&DepNode::WorkProduct(swp.id.clone())) {
            debug!("reconcile_work_products: dep-node for {:?} is dirty", swp);
            delete_dirty_work_product(tcx, swp);
        } else {
            let mut all_files_exist = true;
            for &(_, ref file_name) in swp.work_product.saved_files.iter() {
                let path = in_incr_comp_dir_sess(tcx.sess, file_name);
                if !path.exists() {
                    all_files_exist = false;

                    if tcx.sess.opts.debugging_opts.incremental_info {
                        println!("incremental: could not find file for up-to-date work product: {}",
                                 path.display());
                    }
                }
            }

            if all_files_exist {
                debug!("reconcile_work_products: all files for {:?} exist", swp);
                tcx.dep_graph.insert_previous_work_product(&swp.id, swp.work_product);
            } else {
                debug!("reconcile_work_products: some file for {:?} does not exist", swp);
                delete_dirty_work_product(tcx, swp);
            }
        }
    }
}

fn delete_dirty_work_product(tcx: TyCtxt,
                             swp: SerializedWorkProduct) {
    debug!("delete_dirty_work_product({:?})", swp);
    work_product::delete_workproduct_files(tcx.sess, &swp.work_product);
}

fn load_prev_metadata_hashes(tcx: TyCtxt,
                             retraced: &RetracedDefIdDirectory,
                             output: &mut FxHashMap<DefId, Fingerprint>) {
    if !tcx.sess.opts.debugging_opts.query_dep_graph {
        return
    }

    debug!("load_prev_metadata_hashes() - Loading previous metadata hashes");

    let file_path = metadata_hash_export_path(tcx.sess);

    if !file_path.exists() {
        debug!("load_prev_metadata_hashes() - Couldn't find file containing \
                hashes at `{}`", file_path.display());
        return
    }

    debug!("load_prev_metadata_hashes() - File: {}", file_path.display());

    let data = match file_format::read_file(tcx.sess, &file_path) {
        Ok(Some(data)) => data,
        Ok(None) => {
            debug!("load_prev_metadata_hashes() - File produced by incompatible \
                    compiler version: {}", file_path.display());
            return
        }
        Err(err) => {
            debug!("load_prev_metadata_hashes() - Error reading file `{}`: {}",
                   file_path.display(), err);
            return
        }
    };

    debug!("load_prev_metadata_hashes() - Decoding hashes");
    let mut decoder = Decoder::new(&data, 0);
    let _ = Svh::decode(&mut decoder).unwrap();
    let serialized_hashes = SerializedMetadataHashes::decode(&mut decoder).unwrap();

    debug!("load_prev_metadata_hashes() - Mapping DefIds");

    assert_eq!(serialized_hashes.index_map.len(), serialized_hashes.hashes.len());
    for serialized_hash in serialized_hashes.hashes {
        let def_path_index = serialized_hashes.index_map[&serialized_hash.def_index];
        if let Some(def_id) = retraced.def_id(def_path_index) {
            let old = output.insert(def_id, serialized_hash.hash);
            assert!(old.is_none(), "already have hash for {:?}", def_id);
        }
    }

    debug!("load_prev_metadata_hashes() - successfully loaded {} hashes",
           serialized_hashes.index_map.len());
}

