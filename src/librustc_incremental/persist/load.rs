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

use rustc::dep_graph::{DepNode, WorkProductId, DepKind};
use rustc::hir::def_id::DefId;
use rustc::hir::svh::Svh;
use rustc::ich::Fingerprint;
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_serialize::Decodable as RustcDecodable;
use rustc_serialize::opaque::Decoder;
use std::path::{Path};

use IncrementalHashesMap;
use super::data::*;
use super::dirty_clean;
use super::hash::*;
use super::fs::*;
use super::file_format;
use super::work_product;

// The key is a dirty node. The value is **some** base-input that we
// can blame it on.
pub type DirtyNodes = FxHashMap<DepNodeIndex, DepNodeIndex>;

/// If we are in incremental mode, and a previous dep-graph exists,
/// then load up those nodes/edges that are still valid into the
/// dep-graph for this session. (This is assumed to be running very
/// early in compilation, before we've really done any work, but
/// actually it doesn't matter all that much.) See `README.md` for
/// more general overview.
pub fn load_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                incremental_hashes_map: &IncrementalHashesMap) {
    if tcx.sess.incr_session_load_dep_graph() {
        let _ignore = tcx.dep_graph.in_ignore();
        load_dep_graph_if_exists(tcx, incremental_hashes_map);
    }
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

/// Check if a DepNode from the previous dep-graph refers to something that
/// still exists in the current compilation session. Only works for DepNode
/// variants that represent inputs (HIR and imported Metadata).
fn does_still_exist(tcx: TyCtxt, dep_node: &DepNode) -> bool {
    match dep_node.kind {
        DepKind::Hir |
        DepKind::HirBody |
        DepKind::InScopeTraits |
        DepKind::MetaData => {
            dep_node.extract_def_id(tcx).is_some()
        }
        _ => {
            bug!("unexpected Input DepNode: {:?}", dep_node)
        }
    }
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
            eprintln!("incremental: completely ignoring cache because of \
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

    let serialized_dep_graph = SerializedDepGraph::decode(&mut dep_graph_decoder)?;

    // Compute the set of nodes from the old graph where some input
    // has changed or been removed.
    let dirty_raw_nodes = initial_dirty_nodes(tcx,
                                              incremental_hashes_map,
                                              &serialized_dep_graph.nodes,
                                              &serialized_dep_graph.hashes);
    let dirty_raw_nodes = transitive_dirty_nodes(&serialized_dep_graph,
                                                 dirty_raw_nodes);

    // Recreate the edges in the graph that are still clean.
    let mut clean_work_products = FxHashSet();
    let mut dirty_work_products = FxHashSet(); // incomplete; just used to suppress debug output
    for (source, targets) in serialized_dep_graph.edge_list_indices.iter_enumerated() {
        let target_begin = targets.0 as usize;
        let target_end = targets.1 as usize;

        for &target in &serialized_dep_graph.edge_list_data[target_begin .. target_end] {
            process_edge(tcx,
                         source,
                         target,
                         &serialized_dep_graph.nodes,
                         &dirty_raw_nodes,
                         &mut clean_work_products,
                         &mut dirty_work_products,
                         &work_products);
        }
    }

    // Recreate bootstrap outputs, which are outputs that have no incoming edges
    // (and hence cannot be dirty).
    for bootstrap_output in &serialized_dep_graph.bootstrap_outputs {
        if let DepKind::WorkProduct = bootstrap_output.kind {
            let wp_id = WorkProductId::from_fingerprint(bootstrap_output.hash);
            clean_work_products.insert(wp_id);
        }

        tcx.dep_graph.add_node_directly(*bootstrap_output);
    }

    // Add in work-products that are still clean, and delete those that are
    // dirty.
    reconcile_work_products(tcx, work_products, &clean_work_products);

    dirty_clean::check_dirty_clean_annotations(tcx,
                                               &serialized_dep_graph.nodes,
                                               &dirty_raw_nodes);

    load_prev_metadata_hashes(tcx,
                              &mut *incremental_hashes_map.prev_metadata_hashes.borrow_mut());
    Ok(())
}

/// Computes which of the original set of def-ids are dirty. Stored in
/// a bit vector where the index is the DefPathIndex.
fn initial_dirty_nodes<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 incremental_hashes_map: &IncrementalHashesMap,
                                 nodes: &IndexVec<DepNodeIndex, DepNode>,
                                 serialized_hashes: &[(DepNodeIndex, Fingerprint)])
                                 -> DirtyNodes {
    let mut hcx = HashContext::new(tcx, incremental_hashes_map);
    let mut dirty_nodes = FxHashMap();

    for &(dep_node_index, prev_hash) in serialized_hashes {
        let dep_node = nodes[dep_node_index];
        if does_still_exist(tcx, &dep_node) {
            let current_hash = hcx.hash(&dep_node).unwrap_or_else(|| {
                bug!("Cannot find current ICH for input that still exists?")
            });

            if current_hash == prev_hash {
                debug!("initial_dirty_nodes: {:?} is clean (hash={:?})",
                       dep_node,
                       current_hash);
                continue;
            }

            if tcx.sess.opts.debugging_opts.incremental_dump_hash {
                println!("node {:?} is dirty as hash is {:?}, was {:?}",
                         dep_node,
                         current_hash,
                         prev_hash);
            }

            debug!("initial_dirty_nodes: {:?} is dirty as hash is {:?}, was {:?}",
                   dep_node,
                   current_hash,
                   prev_hash);
        } else {
            if tcx.sess.opts.debugging_opts.incremental_dump_hash {
                println!("node {:?} is dirty as it was removed", dep_node);
            }

            debug!("initial_dirty_nodes: {:?} is dirty as it was removed", dep_node);
        }
        dirty_nodes.insert(dep_node_index, dep_node_index);
    }

    dirty_nodes
}

fn transitive_dirty_nodes(serialized_dep_graph: &SerializedDepGraph,
                          mut dirty_nodes: DirtyNodes)
                          -> DirtyNodes
{
    let mut stack: Vec<(DepNodeIndex, DepNodeIndex)> = vec![];
    stack.extend(dirty_nodes.iter().map(|(&s, &b)| (s, b)));
    while let Some((source, blame)) = stack.pop() {
        // we know the source is dirty (because of the node `blame`)...
        debug_assert!(dirty_nodes.contains_key(&source));

        // ...so we dirty all the targets (with the same blame)
        for &target in serialized_dep_graph.edge_targets_from(source) {
            if !dirty_nodes.contains_key(&target) {
                dirty_nodes.insert(target, blame);
                stack.push((target, blame));
            }
        }
    }
    dirty_nodes
}

/// Go through the list of work-products produced in the previous run.
/// Delete any whose nodes have been found to be dirty or which are
/// otherwise no longer applicable.
fn reconcile_work_products<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     work_products: Vec<SerializedWorkProduct>,
                                     clean_work_products: &FxHashSet<WorkProductId>) {
    debug!("reconcile_work_products({:?})", work_products);
    for swp in work_products {
        if !clean_work_products.contains(&swp.id) {
            debug!("reconcile_work_products: dep-node for {:?} is dirty", swp);
            delete_dirty_work_product(tcx, swp);
        } else {
            let mut all_files_exist = true;
            for &(_, ref file_name) in swp.work_product.saved_files.iter() {
                let path = in_incr_comp_dir_sess(tcx.sess, file_name);
                if !path.exists() {
                    all_files_exist = false;

                    if tcx.sess.opts.debugging_opts.incremental_info {
                        eprintln!("incremental: could not find file for \
                                   up-to-date work product: {}", path.display());
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
                             output: &mut FxHashMap<DefId, Fingerprint>) {
    if !tcx.sess.opts.debugging_opts.query_dep_graph {
        // Previous metadata hashes are only needed for testing.
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

    assert_eq!(serialized_hashes.index_map.len(), serialized_hashes.entry_hashes.len());
    let def_path_hash_to_def_id = tcx.def_path_hash_to_def_id.as_ref().unwrap();

    for serialized_hash in serialized_hashes.entry_hashes {
        let def_path_hash = serialized_hashes.index_map[&serialized_hash.def_index];
        if let Some(&def_id) = def_path_hash_to_def_id.get(&def_path_hash) {
            let old = output.insert(def_id, serialized_hash.hash);
            assert!(old.is_none(), "already have hash for {:?}", def_id);
        }
    }

    debug!("load_prev_metadata_hashes() - successfully loaded {} hashes",
           serialized_hashes.index_map.len());
}

fn process_edge<'a, 'tcx, 'edges>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    source: DepNodeIndex,
    target: DepNodeIndex,
    nodes: &IndexVec<DepNodeIndex, DepNode>,
    dirty_raw_nodes: &DirtyNodes,
    clean_work_products: &mut FxHashSet<WorkProductId>,
    dirty_work_products: &mut FxHashSet<WorkProductId>,
    work_products: &[SerializedWorkProduct])
{
    // If the target is dirty, skip the edge. If this is an edge
    // that targets a work-product, we can print the blame
    // information now.
    if let Some(&blame) = dirty_raw_nodes.get(&target) {
        let target = nodes[target];
        if let DepKind::WorkProduct = target.kind {
            if tcx.sess.opts.debugging_opts.incremental_info {
                let wp_id = WorkProductId::from_fingerprint(target.hash);

                if dirty_work_products.insert(wp_id) {
                    // Try to reconstruct the human-readable version of the
                    // DepNode. This cannot be done for things that where
                    // removed.
                    let blame = nodes[blame];
                    let blame_str = if let Some(def_id) = blame.extract_def_id(tcx) {
                        format!("{:?}({})",
                                blame.kind,
                                tcx.def_path(def_id).to_string(tcx))
                    } else {
                        format!("{:?}", blame)
                    };

                    let wp = work_products.iter().find(|swp| swp.id == wp_id).unwrap();

                    eprintln!("incremental: module {:?} is dirty because \
                              {:?} changed or was removed",
                              wp.work_product.cgu_name,
                              blame_str);
                }
            }
        }
        return;
    }

    // At this point we have asserted that the target is clean -- otherwise, we
    // would have hit the return above. We can do some further consistency
    // checks based on this fact:

    // We should never have an edge where the target is clean but the source
    // was dirty. Otherwise something was wrong with the dirtying pass above:
    debug_assert!(!dirty_raw_nodes.contains_key(&source));

    // We also never should encounter an edge going from a removed input to a
    // clean target because removing the input would have dirtied the input
    // node and transitively dirtied the target.
    debug_assert!(match nodes[source].kind {
        DepKind::Hir | DepKind::HirBody | DepKind::MetaData => {
            does_still_exist(tcx, &nodes[source])
        }
        _ => true,
    });

    if !dirty_raw_nodes.contains_key(&target) {
        let target = nodes[target];
        let source = nodes[source];
        tcx.dep_graph.add_edge_directly(source, target);

        if let DepKind::WorkProduct = target.kind {
            let wp_id = WorkProductId::from_fingerprint(target.hash);
            clean_work_products.insert(wp_id);
        }
    }
}
