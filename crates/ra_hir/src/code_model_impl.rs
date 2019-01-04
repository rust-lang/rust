use ra_db::{CrateId, Cancelable};

use crate::{Module, HirFileId, db::HirDatabase, Crate, CrateDependency, AsName};

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
        let module_tree = db.module_tree(source_root_id)?;
        // FIXME: teach module tree about crate roots instead of guessing
        let (module_id, _) = ctry!(module_tree
            .modules_with_sources()
            .find(|(_, src)| src.file_id() == file_id));

        let module = Module::new(db, source_root_id, module_id)?;
        Ok(Some(module))
    }
}
