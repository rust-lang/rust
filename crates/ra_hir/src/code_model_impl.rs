use ra_db::{CrateId, Cancelable, FileId};
use ra_syntax::{AstNode, ast};

use crate::{HirFileId, db::HirDatabase, Crate, CrateDependency, AsName, DefId, DefLoc, DefKind, Name, Path, PathKind, PerNs, Def};

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

    pub(crate) fn source_impl(&self, db: &impl HirDatabase) -> (FileId, Option<ast::ModuleNode>) {
        let loc = self.def_id.loc(db);
        let source_item_id = loc.source_item_id;
        let module = match source_item_id.item_id {
            None => None,
            Some(_) => {
                let syntax_node = db.file_item(source_item_id);
                let module = ast::Module::cast(syntax_node.borrowed()).unwrap().owned();
                Some(module)
            }
        };
        // FIXME: remove `as_original_file` here
        let file_id = source_item_id.file_id.as_original_file();
        (file_id, module)
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
    pub fn parent_impl(&self, db: &impl HirDatabase) -> Cancelable<Option<Module>> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id)?;
        let parent_id = ctry!(loc.module_id.parent(&module_tree));
        let def_loc = DefLoc {
            module_id: parent_id,
            source_item_id: parent_id.source(&module_tree).0,
            ..loc
        };
        let def_id = def_loc.id(db);
        let module = Module::new(def_id);
        Ok(Some(module))
    }
    pub fn resolve_path_impl(
        &self,
        db: &impl HirDatabase,
        path: &Path,
    ) -> Cancelable<PerNs<DefId>> {
        let mut curr_per_ns = PerNs::types(
            match path.kind {
                PathKind::Crate => self.crate_root(db)?,
                PathKind::Self_ | PathKind::Plain => self.clone(),
                PathKind::Super => {
                    if let Some(p) = self.parent(db)? {
                        p
                    } else {
                        return Ok(PerNs::none());
                    }
                }
            }
            .def_id,
        );

        let segments = &path.segments;
        for name in segments.iter() {
            let curr = if let Some(r) = curr_per_ns.as_ref().take_types() {
                r
            } else {
                return Ok(PerNs::none());
            };
            let module = match curr.resolve(db)? {
                Def::Module(it) => it,
                // TODO here would be the place to handle enum variants...
                _ => return Ok(PerNs::none()),
            };
            let scope = module.scope(db)?;
            curr_per_ns = if let Some(r) = scope.get(&name) {
                r.def_id
            } else {
                return Ok(PerNs::none());
            };
        }
        Ok(curr_per_ns)
    }
}
