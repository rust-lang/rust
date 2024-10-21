//! rust-analyzer is lazy and doesn't compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This module implements prepopulation of
//! various caches, it's not really advanced at the moment.
mod topologic_sort;

use std::time::Duration;

use hir::db::DefDatabase;

use crate::{
    base_db::{
        ra_salsa::{Database, ParallelDatabase, Snapshot},
        Cancelled, CrateId, SourceDatabase, SourceRootDatabase,
    },
    symbol_index::SymbolsDatabase,
    FxIndexMap, RootDatabase,
};

/// We're indexing many crates.
#[derive(Debug)]
pub struct ParallelPrimeCachesProgress {
    /// the crates that we are currently priming.
    pub crates_currently_indexing: Vec<String>,
    /// the total number of crates we want to prime.
    pub crates_total: usize,
    /// the total number of crates that have finished priming
    pub crates_done: usize,
}

pub fn parallel_prime_caches(
    db: &RootDatabase,
    num_worker_threads: usize,
    cb: &(dyn Fn(ParallelPrimeCachesProgress) + Sync),
) {
    let _p = tracing::info_span!("parallel_prime_caches").entered();

    let graph = db.crate_graph();
    let mut crates_to_prime = {
        let mut builder = topologic_sort::TopologicalSortIter::builder();

        for crate_id in graph.iter() {
            builder.add(crate_id, graph[crate_id].dependencies.iter().map(|d| d.crate_id));
        }

        builder.build()
    };

    enum ParallelPrimeCacheWorkerProgress {
        BeginCrate { crate_id: CrateId, crate_name: String },
        EndCrate { crate_id: CrateId },
    }

    let (work_sender, progress_receiver) = {
        let (progress_sender, progress_receiver) = crossbeam_channel::unbounded();
        let (work_sender, work_receiver) = crossbeam_channel::unbounded();
        let graph = graph.clone();
        let local_roots = db.local_roots();
        let prime_caches_worker = move |db: Snapshot<RootDatabase>| {
            while let Ok((crate_id, crate_name)) = work_receiver.recv() {
                progress_sender
                    .send(ParallelPrimeCacheWorkerProgress::BeginCrate { crate_id, crate_name })?;

                // Compute the DefMap and possibly ImportMap
                let file_id = graph[crate_id].root_file_id;
                let root_id = db.file_source_root(file_id);
                if db.source_root(root_id).is_library {
                    db.crate_def_map(crate_id);
                } else {
                    // This also computes the DefMap
                    db.import_map(crate_id);
                }

                // Compute the symbol search index.
                // This primes the cache for `ide_db::symbol_index::world_symbols()`.
                //
                // We do this for workspace crates only (members of local_roots), because doing it
                // for all dependencies could be *very* unnecessarily slow in a large project.
                //
                // FIXME: We should do it unconditionally if the configuration is set to default to
                // searching dependencies (rust-analyzer.workspace.symbol.search.scope), but we
                // would need to pipe that configuration information down here.
                if local_roots.contains(&root_id) {
                    db.crate_symbols(crate_id.into());
                }

                progress_sender.send(ParallelPrimeCacheWorkerProgress::EndCrate { crate_id })?;
            }

            Ok::<_, crossbeam_channel::SendError<_>>(())
        };

        for id in 0..num_worker_threads {
            let worker = prime_caches_worker.clone();
            let db = db.snapshot();

            stdx::thread::Builder::new(stdx::thread::ThreadIntent::Worker)
                .allow_leak(true)
                .name(format!("PrimeCaches#{id}"))
                .spawn(move || Cancelled::catch(|| worker(db)))
                .expect("failed to spawn thread");
        }

        (work_sender, progress_receiver)
    };

    let crates_total = crates_to_prime.pending();
    let mut crates_done = 0;

    // an index map is used to preserve ordering so we can sort the progress report in order of
    // "longest crate to index" first
    let mut crates_currently_indexing =
        FxIndexMap::with_capacity_and_hasher(num_worker_threads, Default::default());

    while crates_done < crates_total {
        db.unwind_if_cancelled();

        for crate_id in &mut crates_to_prime {
            work_sender
                .send((
                    crate_id,
                    graph[crate_id].display_name.as_deref().unwrap_or_default().to_owned(),
                ))
                .ok();
        }

        // recv_timeout is somewhat a hack, we need a way to from this thread check to see if the current salsa revision
        // is cancelled on a regular basis. workers will only exit if they are processing a task that is cancelled, or
        // if this thread exits, and closes the work channel.
        let worker_progress = match progress_receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(p) => p,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // our workers may have died from a cancelled task, so we'll check and re-raise here.
                db.unwind_if_cancelled();
                break;
            }
        };
        match worker_progress {
            ParallelPrimeCacheWorkerProgress::BeginCrate { crate_id, crate_name } => {
                crates_currently_indexing.insert(crate_id, crate_name);
            }
            ParallelPrimeCacheWorkerProgress::EndCrate { crate_id } => {
                crates_currently_indexing.swap_remove(&crate_id);
                crates_to_prime.mark_done(crate_id);
                crates_done += 1;
            }
        };

        let progress = ParallelPrimeCachesProgress {
            crates_currently_indexing: crates_currently_indexing.values().cloned().collect(),
            crates_done,
            crates_total,
        };

        cb(progress);
    }
}
