use std::sync::Arc;

use ide_db::{
    base_db::{salsa::Durability, CrateGraph, SourceDatabase},
    FxHashMap, RootDatabase,
};

// Feature: Shuffle Crate Graph
//
// Randomizes all crate IDs in the crate graph, for debugging.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Shuffle Crate Graph**
// |===
pub(crate) fn shuffle_crate_graph(db: &mut RootDatabase) {
    let crate_graph = db.crate_graph();

    let mut shuffled_ids = crate_graph.iter().collect::<Vec<_>>();
    shuffle(&mut shuffled_ids);

    let mut new_graph = CrateGraph::default();

    let mut map = FxHashMap::default();
    for old_id in shuffled_ids.iter().copied() {
        let data = &crate_graph[old_id];
        let new_id = new_graph.add_crate_root(
            data.root_file_id,
            data.edition,
            data.display_name.clone(),
            data.version.clone(),
            data.cfg_options.clone(),
            data.potential_cfg_options.clone(),
            data.env.clone(),
            data.proc_macro.clone(),
            data.is_proc_macro,
            data.origin.clone(),
        );
        map.insert(old_id, new_id);
    }

    for old_id in shuffled_ids.iter().copied() {
        let data = &crate_graph[old_id];
        for dep in &data.dependencies {
            let mut new_dep = dep.clone();
            new_dep.crate_id = map[&dep.crate_id];
            new_graph.add_dep(map[&old_id], new_dep).unwrap();
        }
    }

    db.set_crate_graph_with_durability(Arc::new(new_graph), Durability::HIGH);
}

fn shuffle<T>(slice: &mut [T]) {
    let mut rng = oorandom::Rand32::new(seed());

    let mut remaining = slice.len() - 1;
    while remaining > 0 {
        let index = rng.rand_range(0..remaining as u32);
        slice.swap(remaining, index as usize);
        remaining -= 1;
    }
}

fn seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    RandomState::new().build_hasher().finish()
}
