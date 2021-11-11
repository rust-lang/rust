//! rust-analyzer is lazy and doesn't compute anything unless asked. This
//! sometimes is counter productive when, for example, the first goto definition
//! request takes longer to compute. This modules implemented prepopulation of
//! various caches, it's not really advanced at the moment.

use hir::db::DefDatabase;
use ide_db::base_db::{CrateGraph, CrateId, SourceDatabase, SourceDatabaseExt};
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
    // We're only interested in the transitive dependencies of all workspace crates.
    let to_prime: FxHashSet<_> = graph
        .iter()
        .filter(|&id| {
            let file_id = graph[id].root_file_id;
            let root_id = db.file_source_root(file_id);
            !db.source_root(root_id).is_library
        })
        .flat_map(|id| graph.transitive_deps(id))
        .collect();

    let topo = toposort(&graph, &to_prime);

    // FIXME: This would be easy to parallelize, since it's in the ideal ordering for that.
    // Unfortunately rayon prevents panics from propagation out of a `scope`, which breaks
    // cancellation, so we cannot use rayon.
    for (i, &crate_id) in topo.iter().enumerate() {
        let crate_name = graph[crate_id].display_name.as_deref().unwrap_or_default().to_string();

        cb(PrimeCachesProgress { on_crate: crate_name, n_done: i, n_total: topo.len() });
        db.crate_def_map(crate_id);
        db.import_map(crate_id);
    }
}

fn toposort(graph: &CrateGraph, crates: &FxHashSet<CrateId>) -> Vec<CrateId> {
    // Just subset the full topologically sorted set for simplicity.

    let all = graph.crates_in_topological_order();
    let mut result = Vec::with_capacity(crates.len());
    for krate in all {
        if crates.contains(&krate) {
            result.push(krate);
        }
    }
    result
}
