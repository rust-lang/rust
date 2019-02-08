use crate::{
    Crate, CrateDependency, AsName, Module, PersistentHirDatabase,
};

impl Crate {
    pub(crate) fn dependencies_impl(
        &self,
        db: &impl PersistentHirDatabase,
    ) -> Vec<CrateDependency> {
        let crate_graph = db.crate_graph();
        crate_graph
            .dependencies(self.crate_id)
            .map(|dep| {
                let krate = Crate { crate_id: dep.crate_id() };
                let name = dep.as_name();
                CrateDependency { krate, name }
            })
            .collect()
    }
    pub(crate) fn root_module_impl(&self, db: &impl PersistentHirDatabase) -> Option<Module> {
        let module_tree = db.module_tree(*self);
        let module_id = module_tree.modules().next()?;

        let module = Module { krate: *self, module_id };
        Some(module)
    }
}
