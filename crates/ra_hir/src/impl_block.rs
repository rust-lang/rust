use std::sync::Arc;
use rustc_hash::FxHashMap;

use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};
use ra_syntax::{
    AstPtr, SourceFile, TreeArc,
    ast::{self, AstNode}
};

use crate::{
    Const, TypeAlias, Function, HirFileId,
    HirDatabase, PersistentHirDatabase,
    ModuleDef, Trait, Resolution,
    type_ref::TypeRef,
    ids::LocationCtx,
    resolve::Resolver,
    ty::Ty, generics::GenericParams,
};

use crate::code_model_api::{Module, ModuleSource};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImplSourceMap {
    map: ArenaMap<ImplId, AstPtr<ast::ImplBlock>>,
}

impl ImplSourceMap {
    fn insert(&mut self, impl_id: ImplId, impl_block: &ast::ImplBlock) {
        self.map.insert(impl_id, AstPtr::new(impl_block))
    }

    pub fn get(&self, source: &ModuleSource, impl_id: ImplId) -> TreeArc<ast::ImplBlock> {
        let file = match source {
            ModuleSource::SourceFile(file) => &*file,
            ModuleSource::Module(m) => m.syntax().ancestors().find_map(SourceFile::cast).unwrap(),
        };

        self.map[impl_id].to_node(file).to_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplBlock {
    module: Module,
    impl_id: ImplId,
}

impl ImplBlock {
    pub(crate) fn containing(
        module_impl_blocks: Arc<ModuleImplBlocks>,
        item: ImplItem,
    ) -> Option<ImplBlock> {
        let impl_id = *module_impl_blocks.impls_by_def.get(&item)?;
        Some(ImplBlock { module: module_impl_blocks.module, impl_id })
    }

    pub(crate) fn from_id(module: Module, impl_id: ImplId) -> ImplBlock {
        ImplBlock { module, impl_id }
    }

    /// Returns the syntax of the impl block
    pub fn source(&self, db: &impl PersistentHirDatabase) -> (HirFileId, TreeArc<ast::ImplBlock>) {
        let source_map = db.impls_in_module_source_map(self.module);
        let (file_id, source) = self.module.definition_source(db);
        (file_id, source_map.get(&source, self.impl_id))
    }

    pub fn id(&self) -> ImplId {
        self.impl_id
    }

    pub fn module(&self) -> Module {
        self.module
    }

    pub fn target_trait_ref(&self, db: &impl PersistentHirDatabase) -> Option<TypeRef> {
        db.impls_in_module(self.module).impls[self.impl_id].target_trait().cloned()
    }

    pub fn target_type(&self, db: &impl PersistentHirDatabase) -> TypeRef {
        db.impls_in_module(self.module).impls[self.impl_id].target_type().clone()
    }

    pub fn target_ty(&self, db: &impl HirDatabase) -> Ty {
        Ty::from_hir(db, &self.resolver(db), &self.target_type(db))
    }

    pub fn target_trait(&self, db: &impl HirDatabase) -> Option<Trait> {
        if let Some(TypeRef::Path(path)) = self.target_trait_ref(db) {
            let resolver = self.resolver(db);
            if let Some(Resolution::Def(ModuleDef::Trait(tr))) =
                resolver.resolve_path(db, &path).take_types()
            {
                return Some(tr);
            }
        }
        None
    }

    pub fn items(&self, db: &impl PersistentHirDatabase) -> Vec<ImplItem> {
        db.impls_in_module(self.module).impls[self.impl_id].items().to_vec()
    }

    pub fn generic_params(&self, db: &impl PersistentHirDatabase) -> Arc<GenericParams> {
        db.generic_params((*self).into())
    }

    pub fn resolver(&self, db: &impl HirDatabase) -> Resolver {
        let r = self.module().resolver(db);
        // add generic params, if present
        let p = self.generic_params(db);
        let r = if !p.params.is_empty() { r.push_generic_params_scope(p) } else { r };
        let r = r.push_impl_block_scope(self.clone());
        r
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplData {
    target_trait: Option<TypeRef>,
    target_type: TypeRef,
    items: Vec<ImplItem>,
}

impl ImplData {
    pub(crate) fn from_ast(
        db: &impl PersistentHirDatabase,
        file_id: HirFileId,
        module: Module,
        node: &ast::ImplBlock,
    ) -> Self {
        let target_trait = node.target_trait().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(node.target_type());
        let ctx = LocationCtx::new(db, module, file_id);
        let items = if let Some(item_list) = node.item_list() {
            item_list
                .impl_items()
                .map(|item_node| match item_node.kind() {
                    ast::ImplItemKind::FnDef(it) => Function { id: ctx.to_def(it) }.into(),
                    ast::ImplItemKind::ConstDef(it) => Const { id: ctx.to_def(it) }.into(),
                    ast::ImplItemKind::TypeAliasDef(it) => TypeAlias { id: ctx.to_def(it) }.into(),
                })
                .collect()
        } else {
            Vec::new()
        };
        ImplData { target_trait, target_type, items }
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        self.target_trait.as_ref()
    }

    pub fn target_type(&self) -> &TypeRef {
        &self.target_type
    }

    pub fn items(&self) -> &[ImplItem] {
        &self.items
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//TODO: rename to ImplDef?
pub enum ImplItem {
    Method(Function),
    Const(Const),
    TypeAlias(TypeAlias),
    // Existential
}
impl_froms!(ImplItem: Const, TypeAlias);

impl From<Function> for ImplItem {
    fn from(func: Function) -> ImplItem {
        ImplItem::Method(func)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImplId(pub RawId);
impl_arena_id!(ImplId);

/// The collection of impl blocks is a two-step process: first we collect the
/// blocks per-module; then we build an index of all impl blocks in the crate.
/// This way, we avoid having to do this process for the whole crate whenever
/// a file is changed; as long as the impl blocks in the file don't change,
/// we don't need to do the second step again.
#[derive(Debug, PartialEq, Eq)]
pub struct ModuleImplBlocks {
    pub(crate) module: Module,
    pub(crate) impls: Arena<ImplId, ImplData>,
    impls_by_def: FxHashMap<ImplItem, ImplId>,
}

impl ModuleImplBlocks {
    fn collect(
        db: &impl PersistentHirDatabase,
        module: Module,
        source_map: &mut ImplSourceMap,
    ) -> Self {
        let mut m = ModuleImplBlocks {
            module,
            impls: Arena::default(),
            impls_by_def: FxHashMap::default(),
        };

        let (file_id, module_source) = m.module.definition_source(db);
        let file_id: HirFileId = file_id.into();
        let node = match &module_source {
            ModuleSource::SourceFile(node) => node.syntax(),
            ModuleSource::Module(node) => {
                node.item_list().expect("inline module should have item list").syntax()
            }
        };

        for impl_block_ast in node.children().filter_map(ast::ImplBlock::cast) {
            let impl_block = ImplData::from_ast(db, file_id, m.module, impl_block_ast);
            let id = m.impls.alloc(impl_block);
            for &impl_item in &m.impls[id].items {
                m.impls_by_def.insert(impl_item, id);
            }

            source_map.insert(id, impl_block_ast);
        }

        m
    }
}

pub(crate) fn impls_in_module_with_source_map_query(
    db: &impl PersistentHirDatabase,
    module: Module,
) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>) {
    let mut source_map = ImplSourceMap::default();

    let result = ModuleImplBlocks::collect(db, module, &mut source_map);

    (Arc::new(result), Arc::new(source_map))
}

pub(crate) fn impls_in_module(
    db: &impl PersistentHirDatabase,
    module: Module,
) -> Arc<ModuleImplBlocks> {
    db.impls_in_module_with_source_map(module).0
}

pub(crate) fn impls_in_module_source_map_query(
    db: &impl PersistentHirDatabase,
    module: Module,
) -> Arc<ImplSourceMap> {
    db.impls_in_module_with_source_map(module).1
}
