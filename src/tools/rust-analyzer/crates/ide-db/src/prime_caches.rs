//! rust-analyzer is lazy and doesn't compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This module implements prepopulation of
//! various caches, it's not really advanced at the moment.
use std::panic::AssertUnwindSafe;

use hir::{Symbol, db::DefDatabase};
use rustc_hash::FxHashMap;
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

    enum ParallelPrimeCacheWorkerProgress {
        BeginCrateDefMap { crate_id: Crate, crate_name: Symbol },
        EndCrateDefMap { crate_id: Crate },
        EndCrateImportMap,
        EndModuleSymbols,
        Cancelled(Cancelled),
    }

    // The setup here is a bit complicated. We try to make best use of compute resources.
    // The idea is that if we have a def map available to compute, we should do that first.
    // This is because def map is a dependency of both import map and symbols. So if we have
    // e.g. a def map and a symbols, if we compute the def map we can, after it completes,
    // compute the def maps of dependencies, the existing symbols and the symbols of the
    // new crate, all in parallel. But if we compute the symbols, after that we will only
    // have the def map to compute, and the rest of the CPU cores will rest, which is not
    // good.
    // However, it's better to compute symbols/import map than to compute a def map that
    // isn't ready yet, because one of its dependencies hasn't yet completed its def map.
    // Such def map will just block on the dependency, which is just wasted time. So better
    // to compute the symbols/import map of an already computed def map in that time.

    let (reverse_deps, mut to_be_done_deps) = {
        let all_crates = db.all_crates();
        let to_be_done_deps = all_crates
            .iter()
            .map(|&krate| (krate, krate.data(db).dependencies.len() as u32))
            .collect::<FxHashMap<_, _>>();
        let mut reverse_deps =
            all_crates.iter().map(|&krate| (krate, Vec::new())).collect::<FxHashMap<_, _>>();
        for &krate in &*all_crates {
            for dep in &krate.data(db).dependencies {
                reverse_deps.get_mut(&dep.crate_id).unwrap().push(krate);
            }
        }
        (reverse_deps, to_be_done_deps)
    };

    let (def_map_work_sender, import_map_work_sender, symbols_work_sender, progress_receiver) = {
        let (progress_sender, progress_receiver) = crossbeam_channel::unbounded();
        let (def_map_work_sender, def_map_work_receiver) = crossbeam_channel::unbounded();
        let (import_map_work_sender, import_map_work_receiver) = crossbeam_channel::unbounded();
        let (symbols_work_sender, symbols_work_receiver) = crossbeam_channel::unbounded();
        let prime_caches_worker =
            move |db: RootDatabase| {
                let handle_def_map = |crate_id, crate_name| {
                    progress_sender.send(ParallelPrimeCacheWorkerProgress::BeginCrateDefMap {
                        crate_id,
                        crate_name,
                    })?;

                    let cancelled = Cancelled::catch(|| _ = hir::crate_def_map(&db, crate_id));

                    match cancelled {
                        Ok(()) => progress_sender
                            .send(ParallelPrimeCacheWorkerProgress::EndCrateDefMap { crate_id })?,
                        Err(cancelled) => progress_sender
                            .send(ParallelPrimeCacheWorkerProgress::Cancelled(cancelled))?,
                    }

                    Ok::<_, crossbeam_channel::SendError<_>>(())
                };
                let handle_import_map = |crate_id| {
                    let cancelled = Cancelled::catch(|| _ = db.import_map(crate_id));

                    match cancelled {
                        Ok(()) => progress_sender
                            .send(ParallelPrimeCacheWorkerProgress::EndCrateImportMap)?,
                        Err(cancelled) => progress_sender
                            .send(ParallelPrimeCacheWorkerProgress::Cancelled(cancelled))?,
                    }

                    Ok::<_, crossbeam_channel::SendError<_>>(())
                };
                let handle_symbols = |module| {
                    let cancelled =
                        Cancelled::catch(AssertUnwindSafe(|| _ = db.module_symbols(module)));

                    match cancelled {
                        Ok(()) => progress_sender
                            .send(ParallelPrimeCacheWorkerProgress::EndModuleSymbols)?,
                        Err(cancelled) => progress_sender
                            .send(ParallelPrimeCacheWorkerProgress::Cancelled(cancelled))?,
                    }

                    Ok::<_, crossbeam_channel::SendError<_>>(())
                };

                loop {
                    db.unwind_if_revision_cancelled();

                    // Biased because we want to prefer def maps.
                    crossbeam_channel::select_biased! {
                        recv(def_map_work_receiver) -> work => {
                            let Ok((crate_id, crate_name)) = work else { break };
                            handle_def_map(crate_id, crate_name)?;
                        }
                        recv(import_map_work_receiver) -> work => {
                            let Ok(crate_id) = work else { break };
                            handle_import_map(crate_id)?;
                        }
                        recv(symbols_work_receiver) -> work => {
                            let Ok(module) = work else { break };
                            handle_symbols(module)?;
                        }
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

        (def_map_work_sender, import_map_work_sender, symbols_work_sender, progress_receiver)
    };

    let crate_def_maps_total = db.all_crates().len();
    let mut crate_def_maps_done = 0;
    let (mut crate_import_maps_total, mut crate_import_maps_done) = (0usize, 0usize);
    let (mut module_symbols_total, mut module_symbols_done) = (0usize, 0usize);

    // an index map is used to preserve ordering so we can sort the progress report in order of
    // "longest crate to index" first
    let mut crates_currently_indexing =
        FxIndexMap::with_capacity_and_hasher(num_worker_threads, Default::default());

    for (&krate, &to_be_done_deps) in &to_be_done_deps {
        if to_be_done_deps != 0 {
            continue;
        }

        let name = crate_name(db, krate);
        def_map_work_sender.send((krate, name)).ok();
    }

    while crate_def_maps_done < crate_def_maps_total
        || crate_import_maps_done < crate_import_maps_total
        || module_symbols_done < module_symbols_total
    {
        db.unwind_if_revision_cancelled();

        let progress = ParallelPrimeCachesProgress {
            crates_currently_indexing: crates_currently_indexing.values().cloned().collect(),
            crates_done: crate_def_maps_done,
            crates_total: crate_def_maps_total,
            work_type: "Indexing",
        };

        cb(progress);

        // Biased to prefer progress updates (and because it's faster).
        let progress = match progress_receiver.recv() {
            Ok(p) => p,
            Err(crossbeam_channel::RecvError) => {
                // all our workers have exited, mark us as finished and exit
                cb(ParallelPrimeCachesProgress {
                    crates_currently_indexing: vec![],
                    crates_done: crate_def_maps_done,
                    crates_total: crate_def_maps_done,
                    work_type: "Done",
                });
                return;
            }
        };

        match progress {
            ParallelPrimeCacheWorkerProgress::BeginCrateDefMap { crate_id, crate_name } => {
                crates_currently_indexing.insert(crate_id, crate_name);
            }
            ParallelPrimeCacheWorkerProgress::EndCrateDefMap { crate_id } => {
                crates_currently_indexing.swap_remove(&crate_id);
                crate_def_maps_done += 1;

                // Fire ready dependencies.
                for &dep in &reverse_deps[&crate_id] {
                    let to_be_done = to_be_done_deps.get_mut(&dep).unwrap();
                    *to_be_done -= 1;
                    if *to_be_done == 0 {
                        let dep_name = crate_name(db, dep);
                        def_map_work_sender.send((dep, dep_name)).ok();
                    }
                }

                if crate_def_maps_done == crate_def_maps_total {
                    cb(ParallelPrimeCachesProgress {
                        crates_currently_indexing: vec![],
                        crates_done: crate_def_maps_done,
                        crates_total: crate_def_maps_done,
                        work_type: "Collecting Symbols",
                    });
                }

                let origin = &crate_id.data(db).origin;
                if origin.is_lang() {
                    crate_import_maps_total += 1;
                    import_map_work_sender.send(crate_id).ok();
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
                    let modules = hir::Crate::from(crate_id).modules(db);
                    module_symbols_total += modules.len();
                    for module in modules {
                        symbols_work_sender.send(module).ok();
                    }
                }
            }
            ParallelPrimeCacheWorkerProgress::EndCrateImportMap => crate_import_maps_done += 1,
            ParallelPrimeCacheWorkerProgress::EndModuleSymbols => module_symbols_done += 1,
            ParallelPrimeCacheWorkerProgress::Cancelled(cancelled) => {
                // Cancelled::throw should probably be public
                std::panic::resume_unwind(Box::new(cancelled));
            }
        }
    }
}

fn crate_name(db: &RootDatabase, krate: Crate) -> Symbol {
    krate
        .extra_data(db)
        .display_name
        .as_deref()
        .cloned()
        .unwrap_or_else(|| Symbol::integer(salsa::plumbing::AsId::as_id(&krate).as_u32() as usize))
}
