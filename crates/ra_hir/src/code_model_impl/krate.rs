use ra_db::CrateId;

use crate::{
    Crate, CrateDependency, AsName, Module,
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
    pub(crate) fn root_module_impl(&self, db: &impl HirDatabase) -> Option<Module> {
        let module_tree = db.module_tree(self.crate_id);
        let module_id = module_tree.modules().next()?;

        let module = Module {
            krate: self.crate_id,
            module_id,
        };
        Some(module)
    }
}
