//! rust-analyzer is lazy and doesn't compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This modules implemented prepopulation of
//! various caches, it's not really advanced at the moment.

use hir::db::DefDatabase;
use ide_db::base_db::{SourceDatabase, SourceDatabaseExt};
use rustc_hash::FxHashSet;

use crate::RootDatabase;

/// We started indexing a crate.
#[derive(Debug)]
pub struct PrimeCachesProgress {
    pub on_crate: String,
    pub n_done: usize,
    pub n_total: usize,
}

pub(crate) fn prime_caches(db: &RootDatabase, cb: &(dyn Fn(PrimeCachesProgress) + Sync)) {
    let _p = profile::span("prime_caches");
    let graph = db.crate_graph();
    // We're only interested in the workspace crates and the `ImportMap`s of their direct
    // dependencies, though in practice the latter also compute the `DefMap`s.
    // We don't prime transitive dependencies because they're generally not visible in
    // the current workspace.
    let to_prime: FxHashSet<_> = graph
        .iter()
        .filter(|&id| {
            let file_id = graph[id].root_file_id;
            let root_id = db.file_source_root(file_id);
            !db.source_root(root_id).is_library
        })
        .flat_map(|id| graph[id].dependencies.iter().map(|krate| krate.crate_id))
        .collect();

    // FIXME: This would be easy to parallelize, since it's in the ideal ordering for that.
    // Unfortunately rayon prevents panics from propagation out of a `scope`, which breaks
    // cancellation, so we cannot use rayon.
    let n_total = to_prime.len();
    for (n_done, &crate_id) in to_prime.iter().enumerate() {
        let crate_name = graph[crate_id].display_name.as_deref().unwrap_or_default().to_string();

        cb(PrimeCachesProgress { on_crate: crate_name, n_done, n_total });
        // This also computes the DefMap
        db.import_map(crate_id);
    }
}
