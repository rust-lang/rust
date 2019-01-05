use ra_db::{CrateId, Cancelable};

use crate::{HirFileId, db::HirDatabase, Crate, CrateDependency, AsName, DefId, DefLoc, DefKind, Name};

use crate::code_model_api::Module;

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

        let def_loc = DefLoc {
            kind: DefKind::Module,
            source_root_id,
            module_id,
            source_item_id: module_id.source(&module_tree).0,
        };
        let def_id = def_loc.id(db);

        let module = Module::new(def_id);
        Ok(Some(module))
    }
}

impl Module {
    pub(crate) fn new(def_id: DefId) -> Self {
        crate::code_model_api::Module { def_id }
    }

    pub(crate) fn krate_impl(&self, db: &impl HirDatabase) -> Cancelable<Option<Crate>> {
        let root = self.crate_root(db)?;
        let loc = root.def_id.loc(db);
        let file_id = loc.source_item_id.file_id.as_original_file();

        let crate_graph = db.crate_graph();
        let crate_id = ctry!(crate_graph.crate_id_for_crate_root(file_id));
        Ok(Some(Crate::new(crate_id)))
    }

    pub(crate) fn crate_root_impl(&self, db: &impl HirDatabase) -> Cancelable<Module> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id)?;
        let module_id = loc.module_id.crate_root(&module_tree);
        let def_loc = DefLoc {
            module_id,
            source_item_id: module_id.source(&module_tree).0,
            ..loc
        };
        let def_id = def_loc.id(db);
        let module = Module::new(def_id);
        Ok(module)
    }
    /// Finds a child module with the specified name.
    pub fn child_impl(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id)?;
        let child_id = ctry!(loc.module_id.child(&module_tree, name));
        let def_loc = DefLoc {
            module_id: child_id,
            source_item_id: child_id.source(&module_tree).0,
            ..loc
        };
        let def_id = def_loc.id(db);
        let module = Module::new(def_id);
        Ok(Some(module))
    }
}
