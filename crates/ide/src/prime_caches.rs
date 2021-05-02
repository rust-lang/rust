//! rust-analyzer is lazy and doesn't not compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This modules implemented prepopulating of
//! various caches, it's not really advanced at the moment.

use hir::db::DefDatabase;
use ide_db::base_db::SourceDatabase;

use crate::RootDatabase;

#[derive(Debug)]
pub enum PrimeCachesProgress {
    Started,
    /// We started indexing a crate.
    StartedOnCrate {
        on_crate: String,
        n_done: usize,
        n_total: usize,
    },
    /// We finished indexing all crates.
    Finished,
}

pub(crate) fn prime_caches(db: &RootDatabase, cb: &(dyn Fn(PrimeCachesProgress) + Sync)) {
    let _p = profile::span("prime_caches");
    let graph = db.crate_graph();
    let topo = &graph.crates_in_topological_order();

    cb(PrimeCachesProgress::Started);
    // Take care to emit the finish signal  even when the computation is canceled.
    let _d = stdx::defer(|| cb(PrimeCachesProgress::Finished));

    // FIXME: This would be easy to parallelize, since it's in the ideal ordering for that.
    // Unfortunately rayon prevents panics from propagation out of a `scope`, which breaks
    // cancellation, so we cannot use rayon.
    for (i, krate) in topo.iter().enumerate() {
        let crate_name = graph[*krate].display_name.as_deref().unwrap_or_default().to_string();

        cb(PrimeCachesProgress::StartedOnCrate {
            on_crate: crate_name,
            n_done: i,
            n_total: topo.len(),
        });
        db.crate_def_map(*krate);
    }
}
