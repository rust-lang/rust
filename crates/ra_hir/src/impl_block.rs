use std::sync::Arc;
use rustc_hash::FxHashMap;

use ra_arena::{Arena, RawId, impl_arena_id};
use ra_syntax::ast::{self, AstNode};
use ra_db::{LocationIntener, Cancelable};

use crate::{
    Crate, DefId, DefLoc, DefKind, SourceItemId, SourceFileItems,
    Module, Function,
    db::HirDatabase,
    type_ref::TypeRef,
    module::{ModuleSourceNode},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplBlock {
    crate_impl_blocks: Arc<CrateImplBlocks>,
    impl_id: ImplId,
}

impl ImplBlock {
    pub(crate) fn containing(
        crate_impl_blocks: Arc<CrateImplBlocks>,
        def_id: DefId,
    ) -> Option<ImplBlock> {
        let impl_id = *crate_impl_blocks.impls_by_def.get(&def_id)?;
        Some(ImplBlock {
            crate_impl_blocks,
            impl_id,
        })
    }

    fn impl_data(&self) -> &ImplData {
        &self.crate_impl_blocks.impls[self.impl_id]
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        self.impl_data().target_trait.as_ref()
    }

    pub fn target_type(&self) -> &TypeRef {
        &self.impl_data().target_type
    }

    pub fn items(&self) -> &[ImplItem] {
        &self.impl_data().items
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
        db: &impl AsRef<LocationIntener<DefLoc, DefId>>,
        file_items: &SourceFileItems,
        module: &Module,
        node: ast::ImplBlock,
    ) -> Self {
        let target_trait = node.target_type().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(node.target_type());
        let file_id = module.source().file_id();
        let items = if let Some(item_list) = node.item_list() {
            item_list
                .impl_items()
                .map(|item_node| {
                    let kind = match item_node {
                        ast::ImplItem::FnDef(..) => DefKind::Function,
                        ast::ImplItem::ConstDef(..) => DefKind::Item,
                        ast::ImplItem::TypeDef(..) => DefKind::Item,
                    };
                    let item_id = file_items.id_of_unchecked(item_node.syntax());
                    let def_loc = DefLoc {
                        kind,
                        source_root_id: module.source_root_id,
                        module_id: module.module_id,
                        source_item_id: SourceItemId {
                            file_id,
                            item_id: Some(item_id),
                        },
                    };
                    let def_id = def_loc.id(db);
                    match item_node {
                        ast::ImplItem::FnDef(..) => ImplItem::Method(Function::new(def_id)),
                        ast::ImplItem::ConstDef(..) => ImplItem::Const(def_id),
                        ast::ImplItem::TypeDef(..) => ImplItem::Type(def_id),
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

/// We have to collect all impl blocks in a crate, to later be able to find
/// impls for specific types.
#[derive(Debug, PartialEq, Eq)]
pub struct CrateImplBlocks {
    impls: Arena<ImplId, ImplData>,
    impls_by_def: FxHashMap<DefId, ImplId>,
}

impl CrateImplBlocks {
    fn new() -> Self {
        CrateImplBlocks {
            impls: Arena::default(),
            impls_by_def: FxHashMap::default(),
        }
    }

    fn collect(&mut self, db: &impl HirDatabase, module: Module) -> Cancelable<()> {
        let module_source_node = module.source().resolve(db);
        let node = match &module_source_node {
            ModuleSourceNode::SourceFile(node) => node.borrowed().syntax(),
            ModuleSourceNode::Module(node) => node.borrowed().syntax(),
        };

        let source_file_items = db.file_items(module.source().file_id());

        for impl_block_ast in node.children().filter_map(ast::ImplBlock::cast) {
            let impl_block = ImplData::from_ast(db, &source_file_items, &module, impl_block_ast);
            let id = self.impls.alloc(impl_block);
            for impl_item in &self.impls[id].items {
                self.impls_by_def.insert(impl_item.def_id(), id);
            }
        }

        for (_, child) in module.children() {
            self.collect(db, child)?;
        }

        Ok(())
    }
}

pub(crate) fn impls_in_crate(
    db: &impl HirDatabase,
    krate: Crate,
) -> Cancelable<Arc<CrateImplBlocks>> {
    let mut result = CrateImplBlocks::new();
    let root_module = if let Some(root) = krate.root_module(db)? {
        root
    } else {
        return Ok(Arc::new(result));
    };
    result.collect(db, root_module)?;
    Ok(Arc::new(result))
}
