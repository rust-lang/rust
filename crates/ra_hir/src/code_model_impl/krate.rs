use ra_db::{CrateId, Cancelable};

use crate::{
    HirFileId, Crate, CrateDependency, AsName, DefLoc, DefKind, Module, SourceItemId,
    db::HirDatabase,
};

impl Crate {
    pub(crate) fn new(crate_id: CrateId) -> Crate {
        Crate { crate_id }
    }
    pub(crate) fn dependencies_impl(&self, db: &impl HirDatabase) -> Vec<CrateDependency> {
        let crate_graph = db.crate_graph();
        crate_graph
            .dependencies(self.crate_id)
            .map(|dep| {
                let krate = Crate::new(dep.crate_id());
                let name = dep.as_name();
                CrateDependency { krate, name }
            })
            .collect()
    }
    pub(crate) fn root_module_impl(&self, db: &impl HirDatabase) -> Cancelable<Option<Module>> {
        let crate_graph = db.crate_graph();
        let file_id = crate_graph.crate_root(self.crate_id);
        let source_root_id = db.file_source_root(file_id);
        let file_id = HirFileId::from(file_id);
        let module_tree = db.module_tree(source_root_id);
        // FIXME: teach module tree about crate roots instead of guessing
        let source = SourceItemId {
            file_id,
            item_id: None,
        };
        let module_id = ctry!(module_tree.find_module_by_source(source));

        let def_loc = DefLoc {
            kind: DefKind::Module,
            source_root_id,
            module_id,
            source_item_id: module_id.source(&module_tree),
        };
        let def_id = def_loc.id(db);

        let module = Module::new(def_id);
        Ok(Some(module))
    }
}
