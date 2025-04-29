//! rust-analyzer is lazy and doesn't compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This module implements prepopulation of
//! various caches, it's not really advanced at the moment.
mod topologic_sort;

use std::time::Duration;

use hir::{Symbol, db::DefDatabase};
use itertools::Itertools;
use salsa::{Cancelled, Database};

use crate::{
    FxIndexMap, RootDatabase,
    base_db::{Crate, RootQueryDb},
    symbol_index::SymbolsDatabase,
};

/// We're indexing many crates.
#[derive(Debug)]
pub struct ParallelPrimeCachesProgress {
    /// the crates that we are currently priming.
    pub crates_currently_indexing: Vec<Symbol>,
    /// the total number of crates we want to prime.
    pub crates_total: usize,
    /// the total number of crates that have finished priming
    pub crates_done: usize,
    pub work_type: &'static str,
}

pub fn parallel_prime_caches(
    db: &RootDatabase,
    num_worker_threads: usize,
    cb: &(dyn Fn(ParallelPrimeCachesProgress) + Sync),
) {
    let _p = tracing::info_span!("parallel_prime_caches").entered();

    let mut crates_to_prime = {
        // FIXME: We already have the crate list topologically sorted (but without the things
        // `TopologicalSortIter` gives us). Maybe there is a way to avoid using it and rip it out
        // of the codebase?
        let mut builder = topologic_sort::TopologicalSortIter::builder();

        for &crate_id in db.all_crates().iter() {
            builder.add(crate_id, crate_id.data(db).dependencies.iter().map(|d| d.crate_id));
        }

        builder.build()
    };

    enum ParallelPrimeCacheWorkerProgress {
        BeginCrate { crate_id: Crate, crate_name: Symbol },
        EndCrate { crate_id: Crate },
        Cancelled(Cancelled),
    }

    // We split off def map computation from other work,
    // as the def map is the relevant one. Once the defmaps are computed
    // the project is ready to go, the other indices are just nice to have for some IDE features.
    #[derive(PartialOrd, Ord, PartialEq, Eq, Copy, Clone)]
    enum PrimingPhase {
        DefMap,
        ImportMap,
        CrateSymbols,
    }

    let (work_sender, progress_receiver) = {
        let (progress_sender, progress_receiver) = crossbeam_channel::unbounded();
        let (work_sender, work_receiver) = crossbeam_channel::unbounded();
        let prime_caches_worker = move |db: RootDatabase| {
            while let Ok((crate_id, crate_name, kind)) = work_receiver.recv() {
                progress_sender
                    .send(ParallelPrimeCacheWorkerProgress::BeginCrate { crate_id, crate_name })?;

                let cancelled = Cancelled::catch(|| match kind {
                    PrimingPhase::DefMap => _ = db.crate_def_map(crate_id),
                    PrimingPhase::ImportMap => _ = db.import_map(crate_id),
                    PrimingPhase::CrateSymbols => _ = db.crate_symbols(crate_id.into()),
                });

                match cancelled {
                    Ok(()) => progress_sender
                        .send(ParallelPrimeCacheWorkerProgress::EndCrate { crate_id })?,
                    Err(cancelled) => progress_sender
                        .send(ParallelPrimeCacheWorkerProgress::Cancelled(cancelled))?,
                }
            }

            Ok::<_, crossbeam_channel::SendError<_>>(())
        };

        for id in 0..num_worker_threads {
            stdx::thread::Builder::new(
                stdx::thread::ThreadIntent::Worker,
                format!("PrimeCaches#{id}"),
            )
            .allow_leak(true)
            .spawn({
                let worker = prime_caches_worker.clone();
                let db = db.clone();
                move || worker(db)
            })
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

    let mut additional_phases = vec![];

    while crates_done < crates_total {
        db.unwind_if_revision_cancelled();

        for krate in &mut crates_to_prime {
            let name = krate.extra_data(db).display_name.as_deref().cloned().unwrap_or_else(|| {
                Symbol::integer(salsa::plumbing::AsId::as_id(&krate).as_u32() as usize)
            });
            let origin = &krate.data(db).origin;
            if origin.is_lang() {
                additional_phases.push((krate, name.clone(), PrimingPhase::ImportMap));
            } else if origin.is_local() {
                // Compute the symbol search index.
                // This primes the cache for `ide_db::symbol_index::world_symbols()`.
                //
                // We do this for workspace crates only (members of local_roots), because doing it
                // for all dependencies could be *very* unnecessarily slow in a large project.
                //
                // FIXME: We should do it unconditionally if the configuration is set to default to
                // searching dependencies (rust-analyzer.workspace.symbol.search.scope), but we
                // would need to pipe that configuration information down here.
                additional_phases.push((krate, name.clone(), PrimingPhase::CrateSymbols));
            }

            work_sender.send((krate, name, PrimingPhase::DefMap)).ok();
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
                // all our workers have exited, mark us as finished and exit
                cb(ParallelPrimeCachesProgress {
                    crates_currently_indexing: vec![],
                    crates_done,
                    crates_total: crates_done,
                    work_type: "Indexing",
                });
                return;
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
            ParallelPrimeCacheWorkerProgress::Cancelled(cancelled) => {
                // Cancelled::throw should probably be public
                std::panic::resume_unwind(Box::new(cancelled));
            }
        };

        let progress = ParallelPrimeCachesProgress {
            crates_currently_indexing: crates_currently_indexing.values().cloned().collect(),
            crates_done,
            crates_total,
            work_type: "Indexing",
        };

        cb(progress);
    }

    let mut crates_done = 0;
    let crates_total = additional_phases.len();
    for w in additional_phases.into_iter().sorted_by_key(|&(_, _, phase)| phase) {
        work_sender.send(w).ok();
    }

    while crates_done < crates_total {
        db.unwind_if_revision_cancelled();

        // recv_timeout is somewhat a hack, we need a way to from this thread check to see if the current salsa revision
        // is cancelled on a regular basis. workers will only exit if they are processing a task that is cancelled, or
        // if this thread exits, and closes the work channel.
        let worker_progress = match progress_receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(p) => p,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                continue;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // all our workers have exited, mark us as finished and exit
                cb(ParallelPrimeCachesProgress {
                    crates_currently_indexing: vec![],
                    crates_done,
                    crates_total: crates_done,
                    work_type: "Populating symbols",
                });
                return;
            }
        };
        match worker_progress {
            ParallelPrimeCacheWorkerProgress::BeginCrate { crate_id, crate_name } => {
                crates_currently_indexing.insert(crate_id, crate_name);
            }
            ParallelPrimeCacheWorkerProgress::EndCrate { crate_id } => {
                crates_currently_indexing.swap_remove(&crate_id);
                crates_done += 1;
            }
            ParallelPrimeCacheWorkerProgress::Cancelled(cancelled) => {
                // Cancelled::throw should probably be public
                std::panic::resume_unwind(Box::new(cancelled));
            }
        };

        let progress = ParallelPrimeCachesProgress {
            crates_currently_indexing: crates_currently_indexing.values().cloned().collect(),
            crates_done,
            crates_total,
            work_type: "Populating symbols",
        };

        cb(progress);
    }
}
