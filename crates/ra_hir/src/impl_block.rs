use std::sync::Arc;
use rustc_hash::FxHashMap;

use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};
use ra_syntax::{
    AstPtr, SourceFile, TreeArc,
ast::{self, AstNode}};

use crate::{
    Const, Type,
    Function, HirFileId,
    PersistentHirDatabase,
    type_ref::TypeRef,
    ids::LocationCtx,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplBlock {
    module_impl_blocks: Arc<ModuleImplBlocks>,
    impl_id: ImplId,
}

impl ImplBlock {
    pub(crate) fn containing(
        module_impl_blocks: Arc<ModuleImplBlocks>,
        item: ImplItem,
    ) -> Option<ImplBlock> {
        let impl_id = *module_impl_blocks.impls_by_def.get(&item)?;
        Some(ImplBlock {
            module_impl_blocks,
            impl_id,
        })
    }

    pub(crate) fn from_id(module_impl_blocks: Arc<ModuleImplBlocks>, impl_id: ImplId) -> ImplBlock {
        ImplBlock {
            module_impl_blocks,
            impl_id,
        }
    }

    pub fn id(&self) -> ImplId {
        self.impl_id
    }

    fn impl_data(&self) -> &ImplData {
        &self.module_impl_blocks.impls[self.impl_id]
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        self.impl_data().target_trait()
    }

    pub fn target_type(&self) -> &TypeRef {
        self.impl_data().target_type()
    }

    pub fn items(&self) -> &[ImplItem] {
        self.impl_data().items()
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
                    ast::ImplItemKind::FnDef(it) => {
                        ImplItem::Method(Function { id: ctx.to_def(it) })
                    }
                    ast::ImplItemKind::ConstDef(it) => {
                        ImplItem::Const(Const { id: ctx.to_def(it) })
                    }
                    ast::ImplItemKind::TypeDef(it) => ImplItem::Type(Type { id: ctx.to_def(it) }),
                })
                .collect()
        } else {
            Vec::new()
        };
        ImplData {
            target_trait,
            target_type,
            items,
        }
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
    Type(Type),
    // Existential
}
impl_froms!(ImplItem: Const, Type);

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
    pub(crate) impls: Arena<ImplId, ImplData>,
    impls_by_def: FxHashMap<ImplItem, ImplId>,
}

impl ModuleImplBlocks {
    fn new() -> Self {
        ModuleImplBlocks {
            impls: Arena::default(),
            impls_by_def: FxHashMap::default(),
        }
    }

    fn collect(
        &mut self,
        db: &impl PersistentHirDatabase,
        module: Module,
        source_map: &mut ImplSourceMap,
    ) {
        let (file_id, module_source) = module.definition_source(db);
        let file_id: HirFileId = file_id.into();
        let node = match &module_source {
            ModuleSource::SourceFile(node) => node.syntax(),
            ModuleSource::Module(node) => node
                .item_list()
                .expect("inline module should have item list")
                .syntax(),
        };

        for impl_block_ast in node.children().filter_map(ast::ImplBlock::cast) {
            let impl_block = ImplData::from_ast(db, file_id, module, impl_block_ast);
            let id = self.impls.alloc(impl_block);
            for &impl_item in &self.impls[id].items {
                self.impls_by_def.insert(impl_item, id);
            }

            source_map.insert(id, impl_block_ast);
        }
    }
}

pub(crate) fn impls_in_module_with_source_map_query(
    db: &impl PersistentHirDatabase,
    module: Module,
) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>) {
    let mut source_map = ImplSourceMap::default();

    let mut result = ModuleImplBlocks::new();
    result.collect(db, module, &mut source_map);

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
