//! Code to load the dep-graph from files.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::unord::UnordMap;
use rustc_hashes::Hash64;
use rustc_middle::dep_graph::{DepGraph, DepsType, SerializedDepGraph, WorkProductMap};
use rustc_middle::query::on_disk_cache::OnDiskCache;
use rustc_serialize::Decodable;
use rustc_serialize::opaque::MemDecoder;
use rustc_session::Session;
use rustc_session::config::IncrementalStateAssertion;
use rustc_span::Symbol;
use tracing::{debug, warn};

use super::data::*;
use super::fs::*;
use super::save::build_dep_graph;
use super::{file_format, work_product};
use crate::errors;

#[derive(Debug)]
/// Represents the result of an attempt to load incremental compilation data.
pub enum LoadResult<T> {
    /// Loading was successful.
    Ok {
        #[allow(missing_docs)]
        data: T,
    },
    /// The file either didn't exist or was produced by an incompatible compiler version.
    DataOutOfDate,
    /// Loading the dep graph failed.
    LoadDepGraph(PathBuf, std::io::Error),
}

impl<T: Default> LoadResult<T> {
    /// Accesses the data returned in [`LoadResult::Ok`].
    pub fn open(self, sess: &Session) -> T {
        // Check for errors when using `-Zassert-incremental-state`
        match (sess.opts.assert_incr_state, &self) {
            (Some(IncrementalStateAssertion::NotLoaded), LoadResult::Ok { .. }) => {
                sess.dcx().emit_fatal(errors::AssertNotLoaded);
            }
            (
                Some(IncrementalStateAssertion::Loaded),
                LoadResult::LoadDepGraph(..) | LoadResult::DataOutOfDate,
            ) => {
                sess.dcx().emit_fatal(errors::AssertLoaded);
            }
            _ => {}
        };

        match self {
            LoadResult::LoadDepGraph(path, err) => {
                sess.dcx().emit_warn(errors::LoadDepGraph { path, err });
                Default::default()
            }
            LoadResult::DataOutOfDate => {
                if let Err(err) = delete_all_session_dir_contents(sess) {
                    sess.dcx()
                        .emit_err(errors::DeleteIncompatible { path: dep_graph_path(sess), err });
                }
                Default::default()
            }
            LoadResult::Ok { data } => data,
        }
    }
}

fn load_data(path: &Path, sess: &Session) -> LoadResult<(Mmap, usize)> {
    match file_format::read_file(
        path,
        sess.opts.unstable_opts.incremental_info,
        sess.is_nightly_build(),
        sess.cfg_version,
    ) {
        Ok(Some(data_and_pos)) => LoadResult::Ok { data: data_and_pos },
        Ok(None) => {
            // The file either didn't exist or was produced by an incompatible
            // compiler version. Neither is an error.
            LoadResult::DataOutOfDate
        }
        Err(err) => LoadResult::LoadDepGraph(path.to_path_buf(), err),
    }
}

fn delete_dirty_work_product(sess: &Session, swp: SerializedWorkProduct) {
    debug!("delete_dirty_work_product({:?})", swp);
    work_product::delete_workproduct_files(sess, &swp.work_product);
}

fn load_dep_graph(
    sess: &Session,
    deps: &DepsType,
) -> LoadResult<(Arc<SerializedDepGraph>, WorkProductMap)> {
    let prof = sess.prof.clone();

    if sess.opts.incremental.is_none() {
        // No incremental compilation.
        return LoadResult::Ok { data: Default::default() };
    }

    let _timer = sess.prof.generic_activity("incr_comp_prepare_load_dep_graph");

    // Calling `sess.incr_comp_session_dir()` will panic if `sess.opts.incremental.is_none()`.
    // Fortunately, we just checked that this isn't the case.
    let path = dep_graph_path(sess);
    let expected_hash = sess.opts.dep_tracking_hash(false);

    let mut prev_work_products = UnordMap::default();

    // If we are only building with -Zquery-dep-graph but without an actual
    // incr. comp. session directory, we skip this. Otherwise we'd fail
    // when trying to load work products.
    if sess.incr_comp_session_dir_opt().is_some() {
        let work_products_path = work_products_path(sess);
        let load_result = load_data(&work_products_path, sess);

        if let LoadResult::Ok { data: (work_products_data, start_pos) } = load_result {
            // Decode the list of work_products
            let Ok(mut work_product_decoder) = MemDecoder::new(&work_products_data[..], start_pos)
            else {
                sess.dcx().emit_warn(errors::CorruptFile { path: &work_products_path });
                return LoadResult::DataOutOfDate;
            };
            let work_products: Vec<SerializedWorkProduct> =
                Decodable::decode(&mut work_product_decoder);

            for swp in work_products {
                let all_files_exist = swp.work_product.saved_files.items().all(|(_, path)| {
                    let exists = in_incr_comp_dir_sess(sess, path).exists();
                    if !exists && sess.opts.unstable_opts.incremental_info {
                        eprintln!("incremental: could not find file for work product: {path}",);
                    }
                    exists
                });

                if all_files_exist {
                    debug!("reconcile_work_products: all files for {:?} exist", swp);
                    prev_work_products.insert(swp.id, swp.work_product);
                } else {
                    debug!("reconcile_work_products: some file for {:?} does not exist", swp);
                    delete_dirty_work_product(sess, swp);
                }
            }
        }
    }

    let _prof_timer = prof.generic_activity("incr_comp_load_dep_graph");

    match load_data(&path, sess) {
        LoadResult::DataOutOfDate => LoadResult::DataOutOfDate,
        LoadResult::LoadDepGraph(path, err) => LoadResult::LoadDepGraph(path, err),
        LoadResult::Ok { data: (bytes, start_pos) } => {
            let Ok(mut decoder) = MemDecoder::new(&bytes, start_pos) else {
                sess.dcx().emit_warn(errors::CorruptFile { path: &path });
                return LoadResult::DataOutOfDate;
            };
            let prev_commandline_args_hash = Hash64::decode(&mut decoder);

            if prev_commandline_args_hash != expected_hash {
                if sess.opts.unstable_opts.incremental_info {
                    eprintln!(
                        "[incremental] completely ignoring cache because of \
                                    differing commandline arguments"
                    );
                }
                // We can't reuse the cache, purge it.
                debug!("load_dep_graph_new: differing commandline arg hashes");

                // No need to do any further work
                return LoadResult::DataOutOfDate;
            }

            let dep_graph = SerializedDepGraph::decode::<DepsType>(&mut decoder, deps);

            LoadResult::Ok { data: (dep_graph, prev_work_products) }
        }
    }
}

/// Attempts to load the query result cache from disk
///
/// If we are not in incremental compilation mode, returns `None`.
/// Otherwise, tries to load the query result cache from disk,
/// creating an empty cache if it could not be loaded.
pub fn load_query_result_cache(sess: &Session) -> Option<OnDiskCache> {
    if sess.opts.incremental.is_none() {
        return None;
    }

    let _prof_timer = sess.prof.generic_activity("incr_comp_load_query_result_cache");

    let path = query_cache_path(sess);
    match load_data(&path, sess) {
        LoadResult::Ok { data: (bytes, start_pos) } => {
            let cache = OnDiskCache::new(sess, bytes, start_pos).unwrap_or_else(|()| {
                sess.dcx().emit_warn(errors::CorruptFile { path: &path });
                OnDiskCache::new_empty()
            });
            Some(cache)
        }
        _ => Some(OnDiskCache::new_empty()),
    }
}

/// Setups the dependency graph by loading an existing graph from disk and set up streaming of a
/// new graph to an incremental session directory.
pub fn setup_dep_graph(sess: &Session, crate_name: Symbol, deps: &DepsType) -> DepGraph {
    // `load_dep_graph` can only be called after `prepare_session_directory`.
    prepare_session_directory(sess, crate_name);

    let res = sess.opts.build_dep_graph().then(|| load_dep_graph(sess, deps));

    if sess.opts.incremental.is_some() {
        sess.time("incr_comp_garbage_collect_session_directories", || {
            if let Err(e) = garbage_collect_session_directories(sess) {
                warn!(
                    "Error while trying to garbage collect incremental \
                     compilation cache directory: {}",
                    e
                );
            }
        });
    }

    res.and_then(|result| {
        let (prev_graph, prev_work_products) = result.open(sess);
        build_dep_graph(sess, prev_graph, prev_work_products)
    })
    .unwrap_or_else(DepGraph::new_disabled)
}
