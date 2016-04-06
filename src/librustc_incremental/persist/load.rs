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

use calculate_svh::SvhCalculate;
use rbml::Error;
use rbml::opaque::Decoder;
use rustc::dep_graph::DepNode;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc_data_structures::fnv::FnvHashSet;
use rustc_serialize::Decodable as RustcDecodable;
use std::io::Read;
use std::fs::File;
use std::path::Path;

use super::data::*;
use super::directory::*;
use super::dirty_clean;
use super::util::*;

type DirtyNodes = FnvHashSet<DepNode<DefId>>;

type CleanEdges = Vec<(DepNode<DefId>, DepNode<DefId>)>;

/// If we are in incremental mode, and a previous dep-graph exists,
/// then load up those nodes/edges that are still valid into the
/// dep-graph for this session. (This is assumed to be running very
/// early in compilation, before we've really done any work, but
/// actually it doesn't matter all that much.) See `README.md` for
/// more general overview.
pub fn load_dep_graph<'tcx>(tcx: &ty::TyCtxt<'tcx>) {
    let _ignore = tcx.dep_graph.in_ignore();

    if let Some(dep_graph) = dep_graph_path(tcx) {
        // FIXME(#32754) lock file?
        load_dep_graph_if_exists(tcx, &dep_graph);
        dirty_clean::check_dirty_clean_annotations(tcx);
    }
}

pub fn load_dep_graph_if_exists<'tcx>(tcx: &ty::TyCtxt<'tcx>, path: &Path) {
    if !path.exists() {
        return;
    }

    let mut data = vec![];
    match
        File::open(path)
        .and_then(|mut file| file.read_to_end(&mut data))
    {
        Ok(_) => { }
        Err(err) => {
            tcx.sess.err(
                &format!("could not load dep-graph from `{}`: {}",
                         path.display(), err));
            return;
        }
    }

    match decode_dep_graph(tcx, &data) {
        Ok(dirty) => dirty,
        Err(err) => {
            bug!("decoding error in dep-graph from `{}`: {}", path.display(), err);
        }
    }
}

pub fn decode_dep_graph<'tcx>(tcx: &ty::TyCtxt<'tcx>, data: &[u8])
                              -> Result<(), Error>
{
    // Deserialize the directory and dep-graph.
    let mut decoder = Decoder::new(data, 0);
    let directory = try!(DefIdDirectory::decode(&mut decoder));
    let serialized_dep_graph = try!(SerializedDepGraph::decode(&mut decoder));

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
                            .filter_map(|&node| retraced.map(node))
                            .filter(|node| !dirty_nodes.contains(node))
                            .map(|node| (node, node));

    // Add nodes and edges that are not dirty into our main graph.
    let dep_graph = tcx.dep_graph.clone();
    for (source, target) in clean_edges.into_iter().chain(clean_nodes) {
        let _task = dep_graph.in_task(target);
        dep_graph.read(source);

        debug!("decode_dep_graph: clean edge: {:?} -> {:?}", source, target);
    }

    Ok(())
}

fn initial_dirty_nodes<'tcx>(tcx: &ty::TyCtxt<'tcx>,
                             hashed_items: &[SerializedHash],
                             retraced: &RetracedDefIdDirectory)
                             -> DirtyNodes {
    let mut items_removed = false;
    let mut dirty_nodes = FnvHashSet();
    for hashed_item in hashed_items {
        match retraced.def_id(hashed_item.index) {
            Some(def_id) => {
                // FIXME(#32753) -- should we use a distinct hash here
                let current_hash = tcx.calculate_item_hash(def_id);
                debug!("initial_dirty_nodes: hash of {:?} is {:?}, was {:?}",
                       def_id, current_hash, hashed_item.hash);
                if current_hash != hashed_item.hash {
                    dirty_nodes.insert(DepNode::Hir(def_id));
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
    for &(serialized_source, serialized_target) in serialized_edges {
        if let Some(target) = retraced.map(serialized_target) {
            if let Some(source) = retraced.map(serialized_source) {
                clean_edges.push((source, target))
            } else {
                // source removed, target must be dirty
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
