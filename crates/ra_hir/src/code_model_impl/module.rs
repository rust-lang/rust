use ra_db::FileId;
use ra_syntax::{ast, SyntaxNode, TreeArc};

use crate::{
    Module, ModuleSource, Problem, ModuleDef,
    Crate, Name, Path, PathKind, PerNs, Def,
    module_tree::ModuleId,
    nameres::{ModuleScope, lower::ImportId},
    db::HirDatabase,
};

impl Module {
    fn with_module_id(&self, module_id: ModuleId) -> Module {
        Module {
            module_id,
            krate: self.krate,
        }
    }

    pub(crate) fn name_impl(&self, db: &impl HirDatabase) -> Option<Name> {
        let module_tree = db.module_tree(self.krate);
        let link = self.module_id.parent_link(&module_tree)?;
        Some(link.name(&module_tree).clone())
    }

    pub(crate) fn definition_source_impl(&self, db: &impl HirDatabase) -> (FileId, ModuleSource) {
        let module_tree = db.module_tree(self.krate);
        let source = self.module_id.source(&module_tree);
        let module_source = ModuleSource::from_source_item_id(db, source);
        let file_id = source.file_id.as_original_file();
        (file_id, module_source)
    }

    pub(crate) fn declaration_source_impl(
        &self,
        db: &impl HirDatabase,
    ) -> Option<(FileId, TreeArc<ast::Module>)> {
        let module_tree = db.module_tree(self.krate);
        let link = self.module_id.parent_link(&module_tree)?;
        let file_id = link
            .owner(&module_tree)
            .source(&module_tree)
            .file_id
            .as_original_file();
        let src = link.source(&module_tree, db);
        Some((file_id, src))
    }

    pub(crate) fn import_source_impl(
        &self,
        db: &impl HirDatabase,
        import: ImportId,
    ) -> TreeArc<ast::PathSegment> {
        let source_map = db.lower_module_source_map(self.clone());
        let (_, source) = self.definition_source(db);
        source_map.get(&source, import)
    }

    pub(crate) fn krate_impl(&self, _db: &impl HirDatabase) -> Option<Crate> {
        Some(Crate::new(self.krate))
    }

    pub(crate) fn crate_root_impl(&self, db: &impl HirDatabase) -> Module {
        let module_tree = db.module_tree(self.krate);
        let module_id = self.module_id.crate_root(&module_tree);
        self.with_module_id(module_id)
    }

    /// Finds a child module with the specified name.
    pub(crate) fn child_impl(&self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
        let module_tree = db.module_tree(self.krate);
        let child_id = self.module_id.child(&module_tree, name)?;
        Some(self.with_module_id(child_id))
    }

    /// Iterates over all child modules.
    pub(crate) fn children_impl(&self, db: &impl HirDatabase) -> impl Iterator<Item = Module> {
        let module_tree = db.module_tree(self.krate);
        let children = self
            .module_id
            .children(&module_tree)
            .map(|(_, module_id)| self.with_module_id(module_id))
            .collect::<Vec<_>>();
        children.into_iter()
    }

    pub(crate) fn parent_impl(&self, db: &impl HirDatabase) -> Option<Module> {
        let module_tree = db.module_tree(self.krate);
        let parent_id = self.module_id.parent(&module_tree)?;
        Some(self.with_module_id(parent_id))
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub(crate) fn scope_impl(&self, db: &impl HirDatabase) -> ModuleScope {
        let item_map = db.item_map(self.krate);
        item_map.per_module[&self.module_id].clone()
    }

    pub(crate) fn resolve_path_impl(&self, db: &impl HirDatabase, path: &Path) -> PerNs<ModuleDef> {
        let mut curr_per_ns: PerNs<ModuleDef> = PerNs::types(match path.kind {
            PathKind::Crate => self.crate_root(db).into(),
            PathKind::Self_ | PathKind::Plain => self.clone().into(),
            PathKind::Super => {
                if let Some(p) = self.parent(db) {
                    p.into()
                } else {
                    return PerNs::none();
                }
            }
            PathKind::Abs => {
                // TODO: absolute use is not supported
                return PerNs::none();
            }
        });

        for segment in path.segments.iter() {
            let curr = match curr_per_ns.as_ref().take_types() {
                Some(r) => r,
                None => {
                    // we still have path segments left, but the path so far
                    // didn't resolve in the types namespace => no resolution
                    // (don't break here because curr_per_ns might contain
                    // something in the value namespace, and it would be wrong
                    // to return that)
                    return PerNs::none();
                }
            };
            // resolve segment in curr

            curr_per_ns = match curr {
                ModuleDef::Module(m) => {
                    let scope = m.scope(db);
                    match scope.get(&segment.name) {
                        Some(r) => r.def_id.clone(),
                        None => PerNs::none(),
                    }
                }
                ModuleDef::Function(_) => PerNs::none(),
                ModuleDef::Def(def) => {
                    match def.resolve(db) {
                        Def::Enum(e) => {
                            // enum variant
                            let matching_variant = e
                                .variants(db)
                                .into_iter()
                                .find(|(n, _variant)| n == &segment.name);

                            match matching_variant {
                                Some((_n, variant)) => {
                                    PerNs::both(variant.def_id().into(), e.def_id().into())
                                }
                                None => PerNs::none(),
                            }
                        }
                        _ => {
                            // could be an inherent method call in UFCS form
                            // (`Struct::method`), or some other kind of associated
                            // item... Which we currently don't handle (TODO)
                            PerNs::none()
                        }
                    }
                }
            };
        }
        curr_per_ns
    }

    pub(crate) fn problems_impl(
        &self,
        db: &impl HirDatabase,
    ) -> Vec<(TreeArc<SyntaxNode>, Problem)> {
        let module_tree = db.module_tree(self.krate);
        self.module_id.problems(&module_tree, db)
    }
}
