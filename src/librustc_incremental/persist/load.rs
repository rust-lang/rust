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

use rbml::Error;
use rbml::opaque::Decoder;
use rustc::dep_graph::DepNode;
use rustc::hir::def_id::DefId;
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc_data_structures::fnv::FnvHashSet;
use rustc_serialize::Decodable as RustcDecodable;
use std::io::Read;
use std::fs::{self, File};
use std::path::{Path};

use super::data::*;
use super::directory::*;
use super::dirty_clean;
use super::hash::*;
use super::util::*;

type DirtyNodes = FnvHashSet<DepNode<DefId>>;

type CleanEdges = Vec<(DepNode<DefId>, DepNode<DefId>)>;

/// If we are in incremental mode, and a previous dep-graph exists,
/// then load up those nodes/edges that are still valid into the
/// dep-graph for this session. (This is assumed to be running very
/// early in compilation, before we've really done any work, but
/// actually it doesn't matter all that much.) See `README.md` for
/// more general overview.
pub fn load_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    if tcx.sess.opts.incremental.is_none() {
        return;
    }

    let _ignore = tcx.dep_graph.in_ignore();
    load_dep_graph_if_exists(tcx);
    dirty_clean::check_dirty_clean_annotations(tcx);
}

fn load_dep_graph_if_exists<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let dep_graph_path = dep_graph_path(tcx).unwrap();
    let dep_graph_data = match load_data(tcx.sess, &dep_graph_path) {
        Some(p) => p,
        None => return // no file
    };

    let work_products_path = tcx_work_products_path(tcx).unwrap();
    let work_products_data = match load_data(tcx.sess, &work_products_path) {
        Some(p) => p,
        None => return // no file
    };

    match decode_dep_graph(tcx, &dep_graph_data, &work_products_data) {
        Ok(()) => return,
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
    if !path.exists() {
        return None;
    }

    let mut data = vec![];
    match
        File::open(path)
        .and_then(|mut file| file.read_to_end(&mut data))
    {
        Ok(_) => {
            Some(data)
        }
        Err(err) => {
            sess.err(
                &format!("could not load dep-graph from `{}`: {}",
                         path.display(), err));
            None
        }
    }

}

/// Decode the dep graph and load the edges/nodes that are still clean
/// into `tcx.dep_graph`.
pub fn decode_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  dep_graph_data: &[u8],
                                  work_products_data: &[u8])
                                  -> Result<(), Error>
{
    // Deserialize the directory and dep-graph.
    let mut dep_graph_decoder = Decoder::new(dep_graph_data, 0);
    let directory = try!(DefIdDirectory::decode(&mut dep_graph_decoder));
    let serialized_dep_graph = try!(SerializedDepGraph::decode(&mut dep_graph_decoder));

    debug!("decode_dep_graph: directory = {:#?}", directory);
    debug!("decode_dep_graph: serialized_dep_graph = {:#?}", serialized_dep_graph);

    // Retrace the paths in the directory to find their current location (if any).
    let retraced = directory.retrace(tcx);

    debug!("decode_dep_graph: retraced = {:#?}", retraced);

    // Compute the set of Hir nodes whose data has changed.
    let mut dirty_nodes =
        initial_dirty_nodes(tcx, &serialized_dep_graph.hashes, &retraced);

    debug!("decode_dep_graph: initial dirty_nodes = {:#?}", dirty_nodes);

    // Find all DepNodes reachable from that core set. This loop
    // iterates repeatedly over the list of edges whose source is not
    // known to be dirty (`clean_edges`). If it finds an edge whose
    // source is dirty, it removes it from that list and adds the
    // target to `dirty_nodes`. It stops when it reaches a fixed
    // point.
    let clean_edges = compute_clean_edges(&serialized_dep_graph.edges,
                                          &retraced,
                                          &mut dirty_nodes);

    // Add synthetic `foo->foo` edges for each clean node `foo` that
    // we had before. This is sort of a hack to create clean nodes in
    // the graph, since the existence of a node is a signal that the
    // work it represents need not be repeated.
    let clean_nodes =
        serialized_dep_graph.nodes
                            .iter()
                            .filter_map(|node| retraced.map(node))
                            .filter(|node| !dirty_nodes.contains(node))
                            .map(|node| (node.clone(), node));

    // Add nodes and edges that are not dirty into our main graph.
    let dep_graph = tcx.dep_graph.clone();
    for (source, target) in clean_edges.into_iter().chain(clean_nodes) {
        debug!("decode_dep_graph: clean edge: {:?} -> {:?}", source, target);

        let _task = dep_graph.in_task(target);
        dep_graph.read(source);
    }

    // Add in work-products that are still clean, and delete those that are
    // dirty.
    let mut work_product_decoder = Decoder::new(work_products_data, 0);
    let work_products = try!(<Vec<SerializedWorkProduct>>::decode(&mut work_product_decoder));
    reconcile_work_products(tcx, work_products, &dirty_nodes);

    Ok(())
}

fn initial_dirty_nodes<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 hashes: &[SerializedHash],
                                 retraced: &RetracedDefIdDirectory)
                                 -> DirtyNodes {
    let mut hcx = HashContext::new(tcx);
    let mut items_removed = false;
    let mut dirty_nodes = FnvHashSet();
    for hash in hashes {
        match hash.node.map_def(|&i| retraced.def_id(i)) {
            Some(dep_node) => {
                let current_hash = hcx.hash(&dep_node).unwrap();
                if current_hash != hash.hash {
                    debug!("initial_dirty_nodes: {:?} is dirty as hash is {:?}, was {:?}",
                           dep_node, current_hash, hash.hash);
                    dirty_nodes.insert(dep_node);
                }
            }
            None => {
                items_removed = true;
            }
        }
    }

    // If any of the items in the krate have changed, then we consider
    // the meta-node `Krate` to be dirty, since that means something
    // which (potentially) read the contents of every single item.
    if items_removed || !dirty_nodes.is_empty() {
        dirty_nodes.insert(DepNode::Krate);
    }

    dirty_nodes
}

fn compute_clean_edges(serialized_edges: &[(SerializedEdge)],
                       retraced: &RetracedDefIdDirectory,
                       dirty_nodes: &mut DirtyNodes)
                       -> CleanEdges {
    // Build up an initial list of edges. Include an edge (source,
    // target) if neither node has been removed. If the source has
    // been removed, add target to the list of dirty nodes.
    let mut clean_edges = Vec::with_capacity(serialized_edges.len());
    for &(ref serialized_source, ref serialized_target) in serialized_edges {
        if let Some(target) = retraced.map(serialized_target) {
            if let Some(source) = retraced.map(serialized_source) {
                clean_edges.push((source, target))
            } else {
                // source removed, target must be dirty
                debug!("compute_clean_edges: {:?} dirty because {:?} no longer exists",
                       target, serialized_source);
                dirty_nodes.insert(target);
            }
        } else {
            // target removed, ignore the edge
        }
    }

    debug!("compute_clean_edges: dirty_nodes={:#?}", dirty_nodes);

    // Propagate dirty marks by iterating repeatedly over
    // `clean_edges`. If we find an edge `(source, target)` where
    // `source` is dirty, add `target` to the list of dirty nodes and
    // remove it. Keep doing this until we find no more dirty nodes.
    let mut previous_size = 0;
    while dirty_nodes.len() > previous_size {
        debug!("compute_clean_edges: previous_size={}", previous_size);
        previous_size = dirty_nodes.len();
        let mut i = 0;
        while i < clean_edges.len() {
            if dirty_nodes.contains(&clean_edges[i].0) {
                let (source, target) = clean_edges.swap_remove(i);
                debug!("compute_clean_edges: dirty source {:?} -> {:?}",
                       source, target);
                dirty_nodes.insert(target);
            } else if dirty_nodes.contains(&clean_edges[i].1) {
                let (source, target) = clean_edges.swap_remove(i);
                debug!("compute_clean_edges: dirty target {:?} -> {:?}",
                       source, target);
            } else {
                i += 1;
            }
        }
    }

    clean_edges
}

/// Go through the list of work-products produced in the previous run.
/// Delete any whose nodes have been found to be dirty or which are
/// otherwise no longer applicable.
fn reconcile_work_products<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     work_products: Vec<SerializedWorkProduct>,
                                     dirty_nodes: &DirtyNodes) {
    debug!("reconcile_work_products({:?})", work_products);
    for swp in work_products {
        let dep_node = DepNode::WorkProduct(swp.id.clone());
        if dirty_nodes.contains(&dep_node) {
            debug!("reconcile_work_products: dep-node for {:?} is dirty", swp);
            delete_dirty_work_product(tcx, swp);
        } else {
            let all_files_exist =
                swp.work_product
                   .saved_files
                   .iter()
                   .all(|&(_, ref file_name)| {
                       let path = in_incr_comp_dir(tcx.sess, &file_name).unwrap();
                       path.exists()
                   });
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
    for &(_, ref file_name) in &swp.work_product.saved_files {
        let path = in_incr_comp_dir(tcx.sess, file_name).unwrap();
        match fs::remove_file(&path) {
            Ok(()) => { }
            Err(err) => {
                tcx.sess.warn(
                    &format!("file-system error deleting outdated file `{}`: {}",
                             path.display(), err));
            }
        }
    }
}
