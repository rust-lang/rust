use rustc_hash::FxHashMap;
use std::sync::Arc;

use ra_arena::{impl_arena_id, map::ArenaMap, Arena, RawId};
use ra_syntax::{
    ast::{self, AstNode},
    AstPtr,
};

use crate::{
    code_model::{Module, ModuleSource},
    db::{AstDatabase, DefDatabase, HirDatabase},
    generics::HasGenericParams,
    ids::LocationCtx,
    resolve::Resolver,
    ty::Ty,
    type_ref::TypeRef,
    AssocItem, Const, Function, HasSource, HirFileId, Source, TraitRef, TypeAlias,
};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImplSourceMap {
    map: ArenaMap<ImplId, AstPtr<ast::ImplBlock>>,
}

impl ImplSourceMap {
    fn insert(&mut self, impl_id: ImplId, impl_block: &ast::ImplBlock) {
        self.map.insert(impl_id, AstPtr::new(impl_block))
    }

    pub fn get(&self, source: &ModuleSource, impl_id: ImplId) -> ast::ImplBlock {
        let root = match source {
            ModuleSource::SourceFile(file) => file.syntax().clone(),
            ModuleSource::Module(m) => m.syntax().ancestors().last().unwrap(),
        };

        self.map[impl_id].to_node(&root)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplBlock {
    module: Module,
    impl_id: ImplId,
}

impl HasSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::ImplBlock> {
        let source_map = db.impls_in_module_with_source_map(self.module).1;
        let src = self.module.definition_source(db);
        Source { file_id: src.file_id, ast: source_map.get(&src.ast, self.impl_id) }
    }
}

impl ImplBlock {
    pub(crate) fn containing(
        module_impl_blocks: Arc<ModuleImplBlocks>,
        item: AssocItem,
    ) -> Option<ImplBlock> {
        let impl_id = *module_impl_blocks.impls_by_def.get(&item)?;
        Some(ImplBlock { module: module_impl_blocks.module, impl_id })
    }

    pub(crate) fn from_id(module: Module, impl_id: ImplId) -> ImplBlock {
        ImplBlock { module, impl_id }
    }

    pub fn id(&self) -> ImplId {
        self.impl_id
    }

    pub fn module(&self) -> Module {
        self.module
    }

    pub fn target_trait(&self, db: &impl DefDatabase) -> Option<TypeRef> {
        db.impls_in_module(self.module).impls[self.impl_id].target_trait().cloned()
    }

    pub fn target_type(&self, db: &impl DefDatabase) -> TypeRef {
        db.impls_in_module(self.module).impls[self.impl_id].target_type().clone()
    }

    pub fn target_ty(&self, db: &impl HirDatabase) -> Ty {
        Ty::from_hir(db, &self.resolver(db), &self.target_type(db))
    }

    pub fn target_trait_ref(&self, db: &impl HirDatabase) -> Option<TraitRef> {
        let target_ty = self.target_ty(db);
        TraitRef::from_hir(db, &self.resolver(db), &self.target_trait(db)?, Some(target_ty))
    }

    pub fn items(&self, db: &impl DefDatabase) -> Vec<AssocItem> {
        db.impls_in_module(self.module).impls[self.impl_id].items().to_vec()
    }

    pub fn is_negative(&self, db: &impl DefDatabase) -> bool {
        db.impls_in_module(self.module).impls[self.impl_id].negative
    }

    pub(crate) fn resolver(&self, db: &impl DefDatabase) -> Resolver {
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
    items: Vec<AssocItem>,
    negative: bool,
}

impl ImplData {
    pub(crate) fn from_ast(
        db: &(impl DefDatabase + AstDatabase),
        file_id: HirFileId,
        module: Module,
        node: &ast::ImplBlock,
    ) -> Self {
        let target_trait = node.target_trait().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(node.target_type());
        let ctx = LocationCtx::new(db, module, file_id);
        let negative = node.is_negative();
        let items = if let Some(item_list) = node.item_list() {
            item_list
                .impl_items()
                .map(|item_node| match item_node {
                    ast::ImplItem::FnDef(it) => Function { id: ctx.to_def(&it) }.into(),
                    ast::ImplItem::ConstDef(it) => Const { id: ctx.to_def(&it) }.into(),
                    ast::ImplItem::TypeAliasDef(it) => TypeAlias { id: ctx.to_def(&it) }.into(),
                })
                .collect()
        } else {
            Vec::new()
        };
        ImplData { target_trait, target_type, items, negative }
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        self.target_trait.as_ref()
    }

    pub fn target_type(&self) -> &TypeRef {
        &self.target_type
    }

    pub fn items(&self) -> &[AssocItem] {
        &self.items
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
    impls_by_def: FxHashMap<AssocItem, ImplId>,
}

impl ModuleImplBlocks {
    fn collect(
        db: &(impl DefDatabase + AstDatabase),
        module: Module,
        source_map: &mut ImplSourceMap,
    ) -> Self {
        let mut m = ModuleImplBlocks {
            module,
            impls: Arena::default(),
            impls_by_def: FxHashMap::default(),
        };

        let src = m.module.definition_source(db);
        let node = match &src.ast {
            ModuleSource::SourceFile(node) => node.syntax().clone(),
            ModuleSource::Module(node) => {
                node.item_list().expect("inline module should have item list").syntax().clone()
            }
        };

        for impl_block_ast in node.children().filter_map(ast::ImplBlock::cast) {
            let impl_block = ImplData::from_ast(db, src.file_id, m.module, &impl_block_ast);
            let id = m.impls.alloc(impl_block);
            for &impl_item in &m.impls[id].items {
                m.impls_by_def.insert(impl_item, id);
            }

            source_map.insert(id, &impl_block_ast);
        }

        m
    }
}

pub(crate) fn impls_in_module_with_source_map_query(
    db: &(impl DefDatabase + AstDatabase),
    module: Module,
) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>) {
    let mut source_map = ImplSourceMap::default();

    let result = ModuleImplBlocks::collect(db, module, &mut source_map);

    (Arc::new(result), Arc::new(source_map))
}

pub(crate) fn impls_in_module(db: &impl DefDatabase, module: Module) -> Arc<ModuleImplBlocks> {
    db.impls_in_module_with_source_map(module).0
}
