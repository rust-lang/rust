//! rust-analyzer is lazy and doesn't compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This modules implemented prepopulation of
//! various caches, it's not really advanced at the moment.
mod topologic_sort;

use std::time::Duration;

use hir::db::DefDatabase;
use ide_db::{
    base_db::{
        salsa::{Database, ParallelDatabase, Snapshot},
        Cancelled, CrateGraph, CrateId, SourceDatabase, SourceDatabaseExt,
    },
    FxHashSet, FxIndexMap,
};

use crate::RootDatabase;

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

pub(crate) fn parallel_prime_caches(
    db: &RootDatabase,
    num_worker_threads: u8,
    cb: &(dyn Fn(ParallelPrimeCachesProgress) + Sync),
) {
    let _p = profile::span("prime_caches");

    let graph = db.crate_graph();
    let mut crates_to_prime = {
        let crate_ids = compute_crates_to_prime(db, &graph);

        let mut builder = topologic_sort::TopologicalSortIter::builder();

        for &crate_id in &crate_ids {
            let crate_data = &graph[crate_id];
            let dependencies = crate_data
                .dependencies
                .iter()
                .map(|d| d.crate_id)
                .filter(|i| crate_ids.contains(i));

            builder.add(crate_id, dependencies);
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
        let prime_caches_worker = move |db: Snapshot<RootDatabase>| {
            while let Ok((crate_id, crate_name)) = work_receiver.recv() {
                progress_sender
                    .send(ParallelPrimeCacheWorkerProgress::BeginCrate { crate_id, crate_name })?;

                // This also computes the DefMap
                db.import_map(crate_id);

                progress_sender.send(ParallelPrimeCacheWorkerProgress::EndCrate { crate_id })?;
            }

            Ok::<_, crossbeam_channel::SendError<_>>(())
        };

        for _ in 0..num_worker_threads {
            let worker = prime_caches_worker.clone();
            let db = db.snapshot();

            stdx::thread::Builder::new(stdx::thread::ThreadIntent::Worker)
                .allow_leak(true)
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
        FxIndexMap::with_capacity_and_hasher(num_worker_threads as _, Default::default());

    while crates_done < crates_total {
        db.unwind_if_cancelled();

        for crate_id in &mut crates_to_prime {
            work_sender
                .send((
                    crate_id,
                    graph[crate_id].display_name.as_deref().unwrap_or_default().to_string(),
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
                crates_currently_indexing.remove(&crate_id);
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

fn compute_crates_to_prime(db: &RootDatabase, graph: &CrateGraph) -> FxHashSet<CrateId> {
    // We're only interested in the workspace crates and the `ImportMap`s of their direct
    // dependencies, though in practice the latter also compute the `DefMap`s.
    // We don't prime transitive dependencies because they're generally not visible in
    // the current workspace.
    graph
        .iter()
        .filter(|&id| {
            let file_id = graph[id].root_file_id;
            let root_id = db.file_source_root(file_id);
            !db.source_root(root_id).is_library
        })
        .flat_map(|id| graph[id].dependencies.iter().map(|krate| krate.crate_id))
        .collect()
}
