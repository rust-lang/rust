use ra_db::{SourceRootId, FileId};
use ra_syntax::{ast, SyntaxNode, AstNode, TreeArc};

use crate::{
    Module, ModuleSource, Problem,
    Crate, DefId, DefLoc, DefKind, Name, Path, PathKind, PerNs, Def,
    module_tree::ModuleId,
    nameres::{ModuleScope, lower::ImportId},
    db::HirDatabase,
};

impl Module {
    pub(crate) fn new(def_id: DefId) -> Self {
        crate::code_model_api::Module { def_id }
    }

    pub(crate) fn from_module_id(
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Self {
        let module_tree = db.module_tree(source_root_id);
        let def_loc = DefLoc {
            kind: DefKind::Module,
            source_root_id,
            module_id,
            source_item_id: module_id.source(&module_tree),
        };
        let def_id = def_loc.id(db);
        Module::new(def_id)
    }

    pub(crate) fn name_impl(&self, db: &impl HirDatabase) -> Option<Name> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        let link = loc.module_id.parent_link(&module_tree)?;
        Some(link.name(&module_tree).clone())
    }

    pub(crate) fn definition_source_impl(&self, db: &impl HirDatabase) -> (FileId, ModuleSource) {
        let loc = self.def_id.loc(db);
        let file_id = loc.source_item_id.file_id.as_original_file();
        let syntax_node = db.file_item(loc.source_item_id);
        let module_source = if let Some(source_file) = ast::SourceFile::cast(&syntax_node) {
            ModuleSource::SourceFile(source_file.to_owned())
        } else {
            let module = ast::Module::cast(&syntax_node).unwrap();
            ModuleSource::Module(module.to_owned())
        };
        (file_id, module_source)
    }

    pub(crate) fn declaration_source_impl(
        &self,
        db: &impl HirDatabase,
    ) -> Option<(FileId, TreeArc<ast::Module>)> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        let link = loc.module_id.parent_link(&module_tree)?;
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
        let loc = self.def_id.loc(db);
        let source_map = db.lower_module_source_map(loc.source_root_id, loc.module_id);
        let (_, source) = self.definition_source(db);
        source_map.get(&source, import)
    }

    pub(crate) fn krate_impl(&self, db: &impl HirDatabase) -> Option<Crate> {
        let root = self.crate_root(db);
        let loc = root.def_id.loc(db);
        let file_id = loc.source_item_id.file_id.as_original_file();

        let crate_graph = db.crate_graph();
        let crate_id = crate_graph.crate_id_for_crate_root(file_id)?;
        Some(Crate::new(crate_id))
    }

    pub(crate) fn crate_root_impl(&self, db: &impl HirDatabase) -> Module {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        let module_id = loc.module_id.crate_root(&module_tree);
        Module::from_module_id(db, loc.source_root_id, module_id)
    }

    /// Finds a child module with the specified name.
    pub fn child_impl(&self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        let child_id = loc.module_id.child(&module_tree, name)?;
        Some(Module::from_module_id(db, loc.source_root_id, child_id))
    }

    /// Iterates over all child modules.
    pub fn children_impl(&self, db: &impl HirDatabase) -> impl Iterator<Item = Module> {
        // FIXME this should be implementable without collecting into a vec, but
        // it's kind of hard since the iterator needs to keep a reference to the
        // module tree.
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        let children = loc
            .module_id
            .children(&module_tree)
            .map(|(_, module_id)| Module::from_module_id(db, loc.source_root_id, module_id))
            .collect::<Vec<_>>();
        children.into_iter()
    }

    pub fn parent_impl(&self, db: &impl HirDatabase) -> Option<Module> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        let parent_id = loc.module_id.parent(&module_tree)?;
        Some(Module::from_module_id(db, loc.source_root_id, parent_id))
    }

    /// Returns a `ModuleScope`: a set of items, visible in this module.
    pub fn scope_impl(&self, db: &impl HirDatabase) -> ModuleScope {
        let loc = self.def_id.loc(db);
        let item_map = db.item_map(loc.source_root_id);
        item_map.per_module[&loc.module_id].clone()
    }

    pub fn resolve_path_impl(&self, db: &impl HirDatabase, path: &Path) -> PerNs<DefId> {
        let mut curr_per_ns = PerNs::types(
            match path.kind {
                PathKind::Crate => self.crate_root(db),
                PathKind::Self_ | PathKind::Plain => self.clone(),
                PathKind::Super => {
                    if let Some(p) = self.parent(db) {
                        p
                    } else {
                        return PerNs::none();
                    }
                }
            }
            .def_id,
        );

        for name in path.segments.iter() {
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
            curr_per_ns = match curr.resolve(db) {
                Def::Module(m) => {
                    let scope = m.scope(db);
                    match scope.get(&name) {
                        Some(r) => r.def_id,
                        None => PerNs::none(),
                    }
                }
                Def::Enum(e) => {
                    // enum variant
                    let matching_variant =
                        e.variants(db).into_iter().find(|(n, _variant)| n == name);

                    match matching_variant {
                        Some((_n, variant)) => PerNs::both(variant.def_id(), e.def_id()),
                        None => PerNs::none(),
                    }
                }
                _ => {
                    // could be an inherent method call in UFCS form
                    // (`Struct::method`), or some other kind of associated
                    // item... Which we currently don't handle (TODO)
                    PerNs::none()
                }
            };
        }
        curr_per_ns
    }

    pub fn problems_impl(&self, db: &impl HirDatabase) -> Vec<(TreeArc<SyntaxNode>, Problem)> {
        let loc = self.def_id.loc(db);
        let module_tree = db.module_tree(loc.source_root_id);
        loc.module_id.problems(&module_tree, db)
    }
}
