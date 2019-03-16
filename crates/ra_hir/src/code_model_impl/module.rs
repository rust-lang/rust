use ra_db::FileId;
use ra_syntax::{ast, SyntaxNode, TreeArc, AstNode};

use crate::{
    Module, ModuleSource, Problem, Name,
    nameres::{ModuleId, ImportId},
    HirDatabase, PersistentHirDatabase,
    HirFileId, SourceItemId,
};

impl ModuleSource {
    pub(crate) fn new(
        db: &impl PersistentHirDatabase,
        file_id: Option<FileId>,
        decl_id: Option<SourceItemId>,
    ) -> ModuleSource {
        match (file_id, decl_id) {
            (Some(file_id), _) => {
                let source_file = db.parse(file_id);
                ModuleSource::SourceFile(source_file)
            }
            (None, Some(item_id)) => {
                let module = db.file_item(item_id);
                let module = ast::Module::cast(&*module).unwrap();
                assert!(module.item_list().is_some(), "expected inline module");
                ModuleSource::Module(module.to_owned())
            }
            (None, None) => panic!(),
        }
    }
}

impl Module {
    fn with_module_id(&self, module_id: ModuleId) -> Module {
        Module { module_id, krate: self.krate }
    }

    pub(crate) fn name_impl(&self, db: &impl HirDatabase) -> Option<Name> {
        let def_map = db.crate_def_map(self.krate);
        let parent = def_map[self.module_id].parent?;
        def_map[parent].children.iter().find_map(|(name, module_id)| {
            if *module_id == self.module_id {
                Some(name.clone())
            } else {
                None
            }
        })
    }

    pub(crate) fn definition_source_impl(
        &self,
        db: &impl PersistentHirDatabase,
    ) -> (HirFileId, ModuleSource) {
        let def_map = db.crate_def_map(self.krate);
        let decl_id = def_map[self.module_id].declaration;
        let file_id = def_map[self.module_id].definition;
        let module_source = ModuleSource::new(db, file_id, decl_id);
        let file_id = file_id.map(HirFileId::from).unwrap_or_else(|| decl_id.unwrap().file_id);
        (file_id, module_source)
    }

    pub(crate) fn declaration_source_impl(
        &self,
        db: &impl HirDatabase,
    ) -> Option<(HirFileId, TreeArc<ast::Module>)> {
        let def_map = db.crate_def_map(self.krate);
        let decl = def_map[self.module_id].declaration?;
        let syntax_node = db.file_item(decl);
        let ast = ast::Module::cast(&syntax_node).unwrap().to_owned();
        Some((decl.file_id, ast))
    }

    pub(crate) fn import_source_impl(
        &self,
        db: &impl HirDatabase,
        import: ImportId,
    ) -> TreeArc<ast::PathSegment> {
        let (file_id, source) = self.definition_source(db);
        let (_, source_map) = db.raw_items_with_source_map(file_id.original_file(db));
        source_map.get(&source, import)
    }

    pub(crate) fn crate_root_impl(&self, db: &impl PersistentHirDatabase) -> Module {
        let def_map = db.crate_def_map(self.krate);
        self.with_module_id(def_map.root())
    }

    /// Finds a child module with the specified name.
    pub(crate) fn child_impl(&self, db: &impl HirDatabase, name: &Name) -> Option<Module> {
        let def_map = db.crate_def_map(self.krate);
        let child_id = def_map[self.module_id].children.get(name)?;
        Some(self.with_module_id(*child_id))
    }

    /// Iterates over all child modules.
    pub(crate) fn children_impl(
        &self,
        db: &impl PersistentHirDatabase,
    ) -> impl Iterator<Item = Module> {
        let def_map = db.crate_def_map(self.krate);
        let children = def_map[self.module_id]
            .children
            .iter()
            .map(|(_, module_id)| self.with_module_id(*module_id))
            .collect::<Vec<_>>();
        children.into_iter()
    }

    pub(crate) fn parent_impl(&self, db: &impl PersistentHirDatabase) -> Option<Module> {
        let def_map = db.crate_def_map(self.krate);
        let parent_id = def_map[self.module_id].parent?;
        Some(self.with_module_id(parent_id))
    }

    pub(crate) fn problems_impl(
        &self,
        db: &impl HirDatabase,
    ) -> Vec<(TreeArc<SyntaxNode>, Problem)> {
        let def_map = db.crate_def_map(self.krate);
        let (my_file_id, _) = self.definition_source(db);
        // FIXME: not entirely corret filterint by module
        def_map
            .problems()
            .iter()
            .filter(|(source_item_id, _problem)| my_file_id == source_item_id.file_id)
            .map(|(source_item_id, problem)| (db.file_item(*source_item_id), problem.clone()))
            .collect()
    }
}
