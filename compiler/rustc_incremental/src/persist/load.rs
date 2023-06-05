//! Code to save/load the dep-graph from files.

use crate::errors;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::unord::UnordMap;
use rustc_middle::dep_graph::{SerializedDepGraph, WorkProduct, WorkProductId};
use rustc_middle::query::on_disk_cache::OnDiskCache;
use rustc_serialize::opaque::MemDecoder;
use rustc_serialize::Decodable;
use rustc_session::config::IncrementalStateAssertion;
use rustc_session::Session;
use std::path::{Path, PathBuf};

use super::data::*;
use super::file_format;
use super::fs::*;
use super::work_product;

type WorkProductMap = UnordMap<WorkProductId, WorkProduct>;

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
    /// Decoding loaded incremental cache failed.
    DecodeIncrCache(Box<dyn std::any::Any + Send>),
}

impl<T: Default> LoadResult<T> {
    /// Accesses the data returned in [`LoadResult::Ok`].
    pub fn open(self, sess: &Session) -> T {
        // Check for errors when using `-Zassert-incremental-state`
        match (sess.opts.assert_incr_state, &self) {
            (Some(IncrementalStateAssertion::NotLoaded), LoadResult::Ok { .. }) => {
                sess.emit_fatal(errors::AssertNotLoaded);
            }
            (
                Some(IncrementalStateAssertion::Loaded),
                LoadResult::LoadDepGraph(..)
                | LoadResult::DecodeIncrCache(..)
                | LoadResult::DataOutOfDate,
            ) => {
                sess.emit_fatal(errors::AssertLoaded);
            }
            _ => {}
        };

        match self {
            LoadResult::LoadDepGraph(path, err) => {
                sess.emit_warning(errors::LoadDepGraph { path, err });
                Default::default()
            }
            LoadResult::DecodeIncrCache(err) => {
                sess.emit_warning(errors::DecodeIncrCache { err: format!("{err:?}") });
                Default::default()
            }
            LoadResult::DataOutOfDate => {
                if let Err(err) = delete_all_session_dir_contents(sess) {
                    sess.emit_err(errors::DeleteIncompatible { path: dep_graph_path(sess), err });
                }
                Default::default()
            }
            LoadResult::Ok { data } => data,
        }
    }
}

fn load_data(path: &Path, sess: &Session) -> LoadResult<(Mmap, usize)> {
    load_data_no_sess(
        path,
        sess.opts.unstable_opts.incremental_info,
        sess.is_nightly_build(),
        sess.cfg_version,
    )
}

fn load_data_no_sess(
    path: &Path,
    report_incremental_info: bool,
    is_nightly_build: bool,
    cfg_version: &'static str,
) -> LoadResult<(Mmap, usize)> {
    match file_format::read_file(path, report_incremental_info, is_nightly_build, cfg_version) {
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

/// Either a result that has already be computed or a
/// handle that will let us wait until it is computed
/// by a background thread.
pub enum MaybeAsync<T> {
    Sync(T),
    Async(std::thread::JoinHandle<T>),
}

impl<T> MaybeAsync<LoadResult<T>> {
    /// Accesses the data returned in [`LoadResult::Ok`] in an asynchronous way if possible.
    pub fn open(self) -> LoadResult<T> {
        match self {
            MaybeAsync::Sync(result) => result,
            MaybeAsync::Async(handle) => {
                handle.join().unwrap_or_else(|e| LoadResult::DecodeIncrCache(e))
            }
        }
    }
}

/// An asynchronous type for computing the dependency graph.
pub type DepGraphFuture = MaybeAsync<LoadResult<(SerializedDepGraph, WorkProductMap)>>;

/// Launch a thread and load the dependency graph in the background.
pub fn load_dep_graph(sess: &Session) -> DepGraphFuture {
    // Since `sess` isn't `Sync`, we perform all accesses to `sess`
    // before we fire the background thread.

    let prof = sess.prof.clone();

    if sess.opts.incremental.is_none() {
        // No incremental compilation.
        return MaybeAsync::Sync(LoadResult::Ok { data: Default::default() });
    }

    let _timer = sess.prof.generic_activity("incr_comp_prepare_load_dep_graph");

    // Calling `sess.incr_comp_session_dir()` will panic if `sess.opts.incremental.is_none()`.
    // Fortunately, we just checked that this isn't the case.
    let path = dep_graph_path(&sess);
    let report_incremental_info = sess.opts.unstable_opts.incremental_info;
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
            let mut work_product_decoder = MemDecoder::new(&work_products_data[..], start_pos);
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

    let is_nightly_build = sess.is_nightly_build();
    let cfg_version = sess.cfg_version;

    MaybeAsync::Async(std::thread::spawn(move || {
        let _prof_timer = prof.generic_activity("incr_comp_load_dep_graph");

        match load_data_no_sess(&path, report_incremental_info, is_nightly_build, cfg_version) {
            LoadResult::DataOutOfDate => LoadResult::DataOutOfDate,
            LoadResult::LoadDepGraph(path, err) => LoadResult::LoadDepGraph(path, err),
            LoadResult::DecodeIncrCache(err) => LoadResult::DecodeIncrCache(err),
            LoadResult::Ok { data: (bytes, start_pos) } => {
                let mut decoder = MemDecoder::new(&bytes, start_pos);
                let prev_commandline_args_hash = u64::decode(&mut decoder);

                if prev_commandline_args_hash != expected_hash {
                    if report_incremental_info {
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

                let dep_graph = SerializedDepGraph::decode(&mut decoder);

                LoadResult::Ok { data: (dep_graph, prev_work_products) }
            }
        }
    }))
}

/// Attempts to load the query result cache from disk
///
/// If we are not in incremental compilation mode, returns `None`.
/// Otherwise, tries to load the query result cache from disk,
/// creating an empty cache if it could not be loaded.
pub fn load_query_result_cache(sess: &Session) -> Option<OnDiskCache<'_>> {
    if sess.opts.incremental.is_none() {
        return None;
    }

    let _prof_timer = sess.prof.generic_activity("incr_comp_load_query_result_cache");

    match load_data(&query_cache_path(sess), sess) {
        LoadResult::Ok { data: (bytes, start_pos) } => {
            Some(OnDiskCache::new(sess, bytes, start_pos))
        }
        _ => Some(OnDiskCache::new_empty(sess.source_map())),
    }
}
