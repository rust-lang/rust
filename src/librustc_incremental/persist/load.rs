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

use rustc::dep_graph::{PreviousDepGraph, SerializedDepGraph};
use rustc::hir::svh::Svh;
use rustc::ich::Fingerprint;
use rustc::session::Session;
use rustc::ty::TyCtxt;
use rustc::ty::maps::OnDiskCache;
use rustc::util::nodemap::DefIdMap;
use rustc_serialize::Decodable as RustcDecodable;
use rustc_serialize::opaque::Decoder;
use std::path::Path;

use super::data::*;
use super::fs::*;
use super::file_format;
use super::work_product;

pub fn dep_graph_tcx_init<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    if !tcx.dep_graph.is_fully_enabled() {
        return
    }

    tcx.allocate_metadata_dep_nodes();
    tcx.precompute_in_scope_traits_hashes();

    if tcx.sess.incr_comp_session_dir_opt().is_none() {
        // If we are only building with -Zquery-dep-graph but without an actual
        // incr. comp. session directory, we exit here. Otherwise we'd fail
        // when trying to load work products.
        return
    }

    let work_products_path = work_products_path(tcx.sess);
    if let Some((work_products_data, start_pos)) = load_data(tcx.sess, &work_products_path) {
        // Decode the list of work_products
        let mut work_product_decoder = Decoder::new(&work_products_data[..], start_pos);
        let work_products: Vec<SerializedWorkProduct> =
            RustcDecodable::decode(&mut work_product_decoder).unwrap_or_else(|e| {
                let msg = format!("Error decoding `work-products` from incremental \
                                   compilation session directory: {}", e);
                tcx.sess.fatal(&msg[..])
            });

        for swp in work_products {
            let mut all_files_exist = true;
            for &(_, ref file_name) in swp.work_product.saved_files.iter() {
                let path = in_incr_comp_dir_sess(tcx.sess, file_name);
                if !path.exists() {
                    all_files_exist = false;

                    if tcx.sess.opts.debugging_opts.incremental_info {
                        eprintln!("incremental: could not find file for work \
                                   product: {}", path.display());
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

fn load_data(sess: &Session, path: &Path) -> Option<(Vec<u8>, usize)> {
    match file_format::read_file(sess, path) {
        Ok(Some(data_and_pos)) => return Some(data_and_pos),
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

fn delete_dirty_work_product(tcx: TyCtxt,
                             swp: SerializedWorkProduct) {
    debug!("delete_dirty_work_product({:?})", swp);
    work_product::delete_workproduct_files(tcx.sess, &swp.work_product);
}

pub fn load_prev_metadata_hashes(tcx: TyCtxt) -> DefIdMap<Fingerprint> {
    let mut output = DefIdMap();

    if !tcx.sess.opts.debugging_opts.query_dep_graph {
        // Previous metadata hashes are only needed for testing.
        return output
    }

    debug!("load_prev_metadata_hashes() - Loading previous metadata hashes");

    let file_path = metadata_hash_export_path(tcx.sess);

    if !file_path.exists() {
        debug!("load_prev_metadata_hashes() - Couldn't find file containing \
                hashes at `{}`", file_path.display());
        return output
    }

    debug!("load_prev_metadata_hashes() - File: {}", file_path.display());

    let (data, start_pos) = match file_format::read_file(tcx.sess, &file_path) {
        Ok(Some(data_and_pos)) => data_and_pos,
        Ok(None) => {
            debug!("load_prev_metadata_hashes() - File produced by incompatible \
                    compiler version: {}", file_path.display());
            return output
        }
        Err(err) => {
            debug!("load_prev_metadata_hashes() - Error reading file `{}`: {}",
                   file_path.display(), err);
            return output
        }
    };

    debug!("load_prev_metadata_hashes() - Decoding hashes");
    let mut decoder = Decoder::new(&data, start_pos);
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

    output
}

pub fn load_dep_graph(sess: &Session) -> PreviousDepGraph {
    let empty = PreviousDepGraph::new(SerializedDepGraph::new());

    if sess.opts.incremental.is_none() {
        return empty
    }

    if let Some((bytes, start_pos)) = load_data(sess, &dep_graph_path(sess)) {
        let mut decoder = Decoder::new(&bytes, start_pos);
        let prev_commandline_args_hash = u64::decode(&mut decoder)
            .expect("Error reading commandline arg hash from cached dep-graph");

        if prev_commandline_args_hash != sess.opts.dep_tracking_hash() {
            if sess.opts.debugging_opts.incremental_info {
                println!("[incremental] completely ignoring cache because of \
                          differing commandline arguments");
            }
            // We can't reuse the cache, purge it.
            debug!("load_dep_graph_new: differing commandline arg hashes");

            delete_all_session_dir_contents(sess)
                .expect("Failed to delete invalidated incr. comp. session \
                         directory contents.");

            // No need to do any further work
            return empty
        }

        let dep_graph = SerializedDepGraph::decode(&mut decoder)
            .expect("Error reading cached dep-graph");

        PreviousDepGraph::new(dep_graph)
    } else {
        empty
    }
}

pub fn load_query_result_cache<'sess>(sess: &'sess Session) -> OnDiskCache<'sess> {
    if sess.opts.incremental.is_none() ||
       !sess.opts.debugging_opts.incremental_queries {
        return OnDiskCache::new_empty(sess.codemap());
    }

    if let Some((bytes, start_pos)) = load_data(sess, &query_cache_path(sess)) {
        OnDiskCache::new(sess, bytes, start_pos)
    } else {
        OnDiskCache::new_empty(sess.codemap())
    }
}
