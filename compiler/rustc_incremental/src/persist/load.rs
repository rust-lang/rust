//! Code to save/load the dep-graph from files.

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::dep_graph::{PreviousDepGraph, SerializedDepGraph, WorkProduct, WorkProductId};
use rustc_middle::ty::query::OnDiskCache;
use rustc_middle::ty::TyCtxt;
use rustc_serialize::opaque::Decoder;
use rustc_serialize::Decodable as RustcDecodable;
use rustc_session::Session;
use std::path::Path;

use super::data::*;
use super::file_format;
use super::fs::*;
use super::work_product;

pub fn dep_graph_tcx_init(tcx: TyCtxt<'_>) {
    if !tcx.dep_graph.is_fully_enabled() {
        return;
    }

    tcx.allocate_metadata_dep_nodes();
}

type WorkProductMap = FxHashMap<WorkProductId, WorkProduct>;

pub enum LoadResult<T> {
    Ok { data: T },
    DataOutOfDate,
    Error { message: String },
}

impl LoadResult<(PreviousDepGraph, WorkProductMap)> {
    pub fn open(self, sess: &Session) -> (PreviousDepGraph, WorkProductMap) {
        match self {
            LoadResult::Error { message } => {
                sess.warn(&message);
                Default::default()
            }
            LoadResult::DataOutOfDate => {
                if let Err(err) = delete_all_session_dir_contents(sess) {
                    sess.err(&format!(
                        "Failed to delete invalidated or incompatible \
                                      incremental compilation session directory contents `{}`: {}.",
                        dep_graph_path(sess).display(),
                        err
                    ));
                }
                Default::default()
            }
            LoadResult::Ok { data } => data,
        }
    }
}

fn load_data(
    report_incremental_info: bool,
    path: &Path,
    nightly_build: bool,
) -> LoadResult<(Vec<u8>, usize)> {
    match file_format::read_file(report_incremental_info, path, nightly_build) {
        Ok(Some(data_and_pos)) => LoadResult::Ok { data: data_and_pos },
        Ok(None) => {
            // The file either didn't exist or was produced by an incompatible
            // compiler version. Neither is an error.
            LoadResult::DataOutOfDate
        }
        Err(err) => LoadResult::Error {
            message: format!("could not load dep-graph from `{}`: {}", path.display(), err),
        },
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
impl<T> MaybeAsync<T> {
    pub fn open(self) -> std::thread::Result<T> {
        match self {
            MaybeAsync::Sync(result) => Ok(result),
            MaybeAsync::Async(handle) => handle.join(),
        }
    }
}

pub type DepGraphFuture = MaybeAsync<LoadResult<(PreviousDepGraph, WorkProductMap)>>;

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
    let path = dep_graph_path_from(&sess.incr_comp_session_dir());
    let report_incremental_info = sess.opts.debugging_opts.incremental_info;
    let expected_hash = sess.opts.dep_tracking_hash();

    let mut prev_work_products = FxHashMap::default();
    let nightly_build = sess.is_nightly_build();

    // If we are only building with -Zquery-dep-graph but without an actual
    // incr. comp. session directory, we skip this. Otherwise we'd fail
    // when trying to load work products.
    if sess.incr_comp_session_dir_opt().is_some() {
        let work_products_path = work_products_path(sess);
        let load_result = load_data(report_incremental_info, &work_products_path, nightly_build);

        if let LoadResult::Ok { data: (work_products_data, start_pos) } = load_result {
            // Decode the list of work_products
            let mut work_product_decoder = Decoder::new(&work_products_data[..], start_pos);
            let work_products: Vec<SerializedWorkProduct> =
                RustcDecodable::decode(&mut work_product_decoder).unwrap_or_else(|e| {
                    let msg = format!(
                        "Error decoding `work-products` from incremental \
                                    compilation session directory: {}",
                        e
                    );
                    sess.fatal(&msg[..])
                });

            for swp in work_products {
                let mut all_files_exist = true;
                if let Some(ref file_name) = swp.work_product.saved_file {
                    let path = in_incr_comp_dir_sess(sess, file_name);
                    if !path.exists() {
                        all_files_exist = false;

                        if sess.opts.debugging_opts.incremental_info {
                            eprintln!(
                                "incremental: could not find file for work \
                                    product: {}",
                                path.display()
                            );
                        }
                    }
                }

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

    MaybeAsync::Async(std::thread::spawn(move || {
        let _prof_timer = prof.generic_activity("incr_comp_load_dep_graph");

        match load_data(report_incremental_info, &path, nightly_build) {
            LoadResult::DataOutOfDate => LoadResult::DataOutOfDate,
            LoadResult::Error { message } => LoadResult::Error { message },
            LoadResult::Ok { data: (bytes, start_pos) } => {
                let mut decoder = Decoder::new(&bytes, start_pos);
                let prev_commandline_args_hash = u64::decode(&mut decoder)
                    .expect("Error reading commandline arg hash from cached dep-graph");

                if prev_commandline_args_hash != expected_hash {
                    if report_incremental_info {
                        println!(
                            "[incremental] completely ignoring cache because of \
                                    differing commandline arguments"
                        );
                    }
                    // We can't reuse the cache, purge it.
                    debug!("load_dep_graph_new: differing commandline arg hashes");

                    // No need to do any further work
                    return LoadResult::DataOutOfDate;
                }

                let dep_graph = SerializedDepGraph::decode(&mut decoder)
                    .expect("Error reading cached dep-graph");

                LoadResult::Ok { data: (PreviousDepGraph::new(dep_graph), prev_work_products) }
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

    match load_data(
        sess.opts.debugging_opts.incremental_info,
        &query_cache_path(sess),
        sess.is_nightly_build(),
    ) {
        LoadResult::Ok { data: (bytes, start_pos) } => {
            Some(OnDiskCache::new(sess, bytes, start_pos))
        }
        _ => Some(OnDiskCache::new_empty(sess.source_map())),
    }
}
