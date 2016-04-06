// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rbml::opaque::Encoder;
use rustc::dep_graph::DepNode;
use rustc::middle::cstore::LOCAL_CRATE;
use rustc::ty::TyCtxt;
use rustc_serialize::{Encodable as RustcEncodable};
use std::hash::{Hasher, SipHasher};
use std::io::{self, Cursor, Write};
use std::fs::{self, File};
use std::path::PathBuf;

use super::data::*;
use super::directory::*;
use super::util::*;

pub fn save_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let _ignore = tcx.dep_graph.in_ignore();

    save_in(tcx, dep_graph_path(tcx), encode_dep_graph);
    save_in(tcx, metadata_hash_path(tcx, LOCAL_CRATE), encode_metadata_hashes);
}

fn save_in<'a,'tcx,F>(tcx: TyCtxt<'a, 'tcx, 'tcx>, opt_path_buf: Option<PathBuf>, encode: F)
    where F: FnOnce(TyCtxt<'a, 'tcx, 'tcx>, &mut Encoder) -> io::Result<()>
{
    let path_buf = match opt_path_buf {
        Some(p) => p,
        None => return
    };

    // FIXME(#32754) lock file?

    // delete the old dep-graph, if any
    if path_buf.exists() {
        match fs::remove_file(&path_buf) {
            Ok(()) => { }
            Err(err) => {
                tcx.sess.err(
                    &format!("unable to delete old dep-graph at `{}`: {}",
                             path_buf.display(), err));
                return;
            }
        }
    }

    // generate the data in a memory buffer
    let mut wr = Cursor::new(Vec::new());
    match encode(tcx, &mut Encoder::new(&mut wr)) {
        Ok(()) => { }
        Err(err) => {
            tcx.sess.err(
                &format!("could not encode dep-graph to `{}`: {}",
                         path_buf.display(), err));
            return;
        }
    }

    // write the data out
    let data = wr.into_inner();
    match
        File::create(&path_buf)
        .and_then(|mut file| file.write_all(&data))
    {
        Ok(_) => { }
        Err(err) => {
            tcx.sess.err(
                &format!("failed to write dep-graph to `{}`: {}",
                         path_buf.display(), err));
            return;
        }
    }
}

pub fn encode_dep_graph<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  encoder: &mut Encoder)
                                  -> io::Result<()> {
    let query = tcx.dep_graph.query();

    let mut builder = DefIdDirectoryBuilder::new(tcx);

    // Create hashes for inputs.
    let hashes =
        query.nodes()
             .into_iter()
             .filter_map(|dep_node| {
                 dep_node.hash(tcx)
                         .map(|hash| {
                             let node = builder.map(dep_node).unwrap();
                             SerializedHash { node: node, hash: hash }
                         })
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

pub fn encode_metadata_hashes<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        encoder: &mut Encoder)
                                        -> io::Result<()>
{
    let query = tcx.dep_graph.query();

    let mut builder = DefIdDirectoryBuilder::new(tcx);

    let serialized_hashes = {
        // Identify the `MetaData(X)` nodes where `X` is local. These are
        // the metadata items we export. Downstream crates will want to
        // see a hash that tells them whether we might have changed the
        // metadata for a given item since they last compiled.
        let meta_data_def_ids =
            query.nodes()
                 .into_iter()
                 .filter_map(|dep_node| match dep_node {
                     DepNode::MetaData(def_id) if def_id.is_local() => Some(def_id),
                     _ => None,
                 });

        // To create the hash for each item `X`, we don't hash the raw
        // bytes of the metadata (though in principle we could). Instead,
        // we walk the predecessors of `MetaData(X)` from the
        // dep-graph. This corresponds to all the inputs that were read to
        // construct the metadata. To create the hash for the metadata, we
        // hash (the hash of) all of those inputs.
        let hashes =
            meta_data_def_ids
            .map(|def_id| {
                let mut state = SipHasher::new();
                for node in query.transitive_predecessors(DepNode::MetaData(def_id)) {
                    if let Some(hash) = node.hash(tcx) {
                        state.write_u64(hash.to_le());
                    }
                }
                (def_id, state.finish())
            });

        // Now create the `SerializedHash` data structures that others
        // will load later.
        let hashes =
            hashes
            .map(|(def_id, hash)| {
                let index = builder.add(def_id).unwrap();
                SerializedHash {
                    node: DepNode::MetaData(index),
                    hash: hash
                }
            });

        // Collect these up into a vector.
        SerializedMetadataHashes {
            hashes: hashes.collect()
        }
    };

    // Encode everything.
    let directory = builder.into_directory();
    try!(directory.encode(encoder));
    try!(serialized_hashes.encode(encoder));

    Ok(())
}
