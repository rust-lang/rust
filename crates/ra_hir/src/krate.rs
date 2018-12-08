use crate::{HirDatabase, Module, Cancelable};

pub use ra_db::CrateId;

/// hir::Crate describes a single crate. It's the main inteface with which
/// crate's dependencies interact. Mostly, it should be just a proxy for the
/// root module.
#[derive(Debug)]
pub struct Crate {
    crate_id: CrateId,
}

impl Crate {
    pub(crate) fn new(crate_id: CrateId) -> Crate {
        Crate { crate_id }
    }
    pub fn dependencies(&self, db: &impl HirDatabase) -> Vec<Crate> {
        let crate_graph = db.crate_graph();
        crate_graph
            .dependencies(self.crate_id)
            .map(|dep| Crate::new(dep.crate_id()))
            .collect()
    }
    pub fn root_module(&self, db: &impl HirDatabase) -> Cancelable<Option<Module>> {
        let crate_graph = db.crate_graph();
        let file_id = crate_graph.crate_root(self.crate_id);
        let source_root_id = db.file_source_root(file_id);
        let module_tree = db.module_tree(source_root_id)?;
        // FIXME: teach module tree about crate roots instead of guessing
        let (module_id, _) = ctry!(module_tree
            .modules_with_sources()
            .find(|(_, src)| src.file_id() == file_id));

        let module = Module::new(db, source_root_id, module_id)?;
        Ok(Some(module))
    }
}
