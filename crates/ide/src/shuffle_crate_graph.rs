use ide_db::{
    base_db::{salsa::Durability, CrateGraph, ProcMacros, SourceDatabase},
    FxHashMap, RootDatabase,
};
use triomphe::Arc;

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
    let proc_macros = db.proc_macros();

    let mut shuffled_ids = crate_graph.iter().collect::<Vec<_>>();

    let mut rng = oorandom::Rand32::new(stdx::rand::seed());
    stdx::rand::shuffle(&mut shuffled_ids, |i| rng.rand_range(0..i as u32) as usize);

    let mut new_graph = CrateGraph::default();
    let mut new_proc_macros = ProcMacros::default();

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
            data.is_proc_macro,
            data.origin.clone(),
            data.target_layout.clone(),
            data.channel,
        );
        new_proc_macros.insert(new_id, proc_macros[&old_id].clone());
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
    db.set_proc_macros_with_durability(Arc::new(new_proc_macros), Durability::HIGH);
}
