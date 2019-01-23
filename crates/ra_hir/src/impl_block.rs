use std::sync::Arc;
use rustc_hash::FxHashMap;

use ra_arena::{Arena, RawId, impl_arena_id};
use ra_syntax::ast::{self, AstNode};

use crate::{
    DefId, DefLoc, DefKind, SourceItemId, SourceFileItems,
    Function, HirFileId, HirInterner,
    db::HirDatabase,
    type_ref::TypeRef,
};

use crate::code_model_api::{Module, ModuleSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplBlock {
    module_impl_blocks: Arc<ModuleImplBlocks>,
    impl_id: ImplId,
}

impl ImplBlock {
    pub(crate) fn containing(
        module_impl_blocks: Arc<ModuleImplBlocks>,
        def_id: DefId,
    ) -> Option<ImplBlock> {
        let impl_id = *module_impl_blocks.impls_by_def.get(&def_id)?;
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
        db: &impl AsRef<HirInterner>,
        file_id: HirFileId,
        file_items: &SourceFileItems,
        module: Module,
        node: &ast::ImplBlock,
    ) -> Self {
        let target_trait = node.target_trait().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(node.target_type());
        let items = if let Some(item_list) = node.item_list() {
            item_list
                .impl_items()
                .map(|item_node| {
                    let kind = match item_node.kind() {
                        ast::ImplItemKind::FnDef(..) => DefKind::Function,
                        ast::ImplItemKind::ConstDef(..) => DefKind::Item,
                        ast::ImplItemKind::TypeDef(..) => DefKind::Item,
                    };
                    let item_id = file_items.id_of_unchecked(item_node.syntax());
                    let source_item_id = SourceItemId {
                        file_id,
                        item_id: Some(item_id),
                    };
                    let def_loc = DefLoc {
                        module,
                        kind,
                        source_item_id,
                    };
                    let def_id = def_loc.id(db);
                    match item_node.kind() {
                        ast::ImplItemKind::FnDef(..) => ImplItem::Method(Function::new(def_id)),
                        ast::ImplItemKind::ConstDef(..) => ImplItem::Const(def_id),
                        ast::ImplItemKind::TypeDef(..) => ImplItem::Type(def_id),
                    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplItem {
    Method(Function),
    // these don't have their own types yet
    Const(DefId),
    Type(DefId),
    // Existential
}

impl ImplItem {
    pub fn def_id(&self) -> DefId {
        match self {
            ImplItem::Method(f) => f.def_id(),
            ImplItem::Const(def_id) => *def_id,
            ImplItem::Type(def_id) => *def_id,
        }
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
    impls_by_def: FxHashMap<DefId, ImplId>,
}

impl ModuleImplBlocks {
    fn new() -> Self {
        ModuleImplBlocks {
            impls: Arena::default(),
            impls_by_def: FxHashMap::default(),
        }
    }

    fn collect(&mut self, db: &impl HirDatabase, module: Module) {
        let (file_id, module_source) = module.definition_source(db);
        let file_id: HirFileId = file_id.into();
        let node = match &module_source {
            ModuleSource::SourceFile(node) => node.syntax(),
            ModuleSource::Module(node) => node
                .item_list()
                .expect("inline module should have item list")
                .syntax(),
        };

        let source_file_items = db.file_items(file_id);

        for impl_block_ast in node.children().filter_map(ast::ImplBlock::cast) {
            let impl_block =
                ImplData::from_ast(db, file_id, &source_file_items, module, impl_block_ast);
            let id = self.impls.alloc(impl_block);
            for impl_item in &self.impls[id].items {
                self.impls_by_def.insert(impl_item.def_id(), id);
            }
        }
    }
}

pub(crate) fn impls_in_module(db: &impl HirDatabase, module: Module) -> Arc<ModuleImplBlocks> {
    let mut result = ModuleImplBlocks::new();
    result.collect(db, module);
    Arc::new(result)
}
