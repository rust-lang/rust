// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use calculate_svh::SvhCalculate;
use rbml::opaque::Encoder;
use rustc::dep_graph::DepNode;
use rustc::ty;
use rustc_serialize::{Encodable as RustcEncodable};
use std::io::{self, Cursor, Write};
use std::fs::{self, File};

use super::data::*;
use super::directory::*;
use super::util::*;

pub fn save_dep_graph<'tcx>(tcx: &ty::TyCtxt<'tcx>) {
    let _ignore = tcx.dep_graph.in_ignore();

    if let Some(dep_graph) = dep_graph_path(tcx) {
        // FIXME(#32754) lock file?

        // delete the old dep-graph, if any
        if dep_graph.exists() {
            match fs::remove_file(&dep_graph) {
                Ok(()) => { }
                Err(err) => {
                    tcx.sess.err(
                        &format!("unable to delete old dep-graph at `{}`: {}",
                                 dep_graph.display(), err));
                    return;
                }
            }
        }

        // generate the data in a memory buffer
        let mut wr = Cursor::new(Vec::new());
        match encode_dep_graph(tcx, &mut Encoder::new(&mut wr)) {
            Ok(()) => { }
            Err(err) => {
                tcx.sess.err(
                    &format!("could not encode dep-graph to `{}`: {}",
                             dep_graph.display(), err));
                return;
            }
        }

        // write the data out
        let data = wr.into_inner();
        match
            File::create(&dep_graph)
            .and_then(|mut file| file.write_all(&data))
        {
            Ok(_) => { }
            Err(err) => {
                tcx.sess.err(
                    &format!("failed to write dep-graph to `{}`: {}",
                             dep_graph.display(), err));
                return;
            }
        }
    }
}

pub fn encode_dep_graph<'tcx>(tcx: &ty::TyCtxt<'tcx>,
                              encoder: &mut Encoder)
                              -> io::Result<()>
{
    // Here we take advantage of how RBML allows us to skip around
    // and encode the depgraph as a two-part structure:
    //
    // ```
    // <dep-graph>[SerializedDepGraph]</dep-graph> // tag 0
    // <directory>[DefIdDirectory]</directory>     // tag 1
    // ```
    //
    // Then later we can load the directory by skipping to find tag 1.

    let query = tcx.dep_graph.query();

    let mut builder = DefIdDirectoryBuilder::new(tcx);

    // Create hashes for things we can persist.
    let hashes =
        query.nodes()
             .into_iter()
             .filter_map(|dep_node| match dep_node {
                 DepNode::Hir(def_id) => {
                     assert!(def_id.is_local());
                     builder.add(def_id)
                            .map(|index| {
                                // FIXME(#32753) -- should we use a distinct hash here
                                let hash = tcx.calculate_item_hash(def_id);
                                SerializedHash { index: index, hash: hash }
                            })
                 }
                 _ => None
             })
             .collect();

    // Create the serialized dep-graph, dropping nodes that are
    // from other crates or from inlined items.
    //
    // FIXME(#32015) fix handling of other crates
    let graph = SerializedDepGraph {
        nodes: query.nodes().into_iter()
                            .flat_map(|node| builder.map(node))
                            .collect(),
        edges: query.edges().into_iter()
                            .flat_map(|(source_node, target_node)| {
                                builder.map(source_node)
                                       .and_then(|source| {
                                           builder.map(target_node)
                                                  .map(|target| (source, target))
                                       })
                            })
                            .collect(),
        hashes: hashes,
    };

    debug!("graph = {:#?}", graph);

    // Encode the directory and then the graph data.
    let directory = builder.into_directory();
    try!(directory.encode(encoder));
    try!(graph.encode(encoder));

    Ok(())
}

