use std::sync::Arc;

use ra_syntax::{
    SyntaxKind, AstNode, SourceFile, TreeArc, AstPtr,
    ast::{self, ModuleItemOwner},
};
use ra_db::SourceRootId;
use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};

use crate::{
    SourceItemId, Path, ModuleSource, HirDatabase, Name, SourceFileItems,
    HirFileId, MacroCallLoc, AsName,
    module_tree::ModuleId
};

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Vis {
    // Priv,
    Other,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ModuleItem {
    pub(crate) id: SourceItemId,
    pub(crate) name: Name,
    pub(super) kind: SyntaxKind,
    pub(super) vis: Vis,
}

impl ModuleItem {
    fn new(
        file_id: HirFileId,
        file_items: &SourceFileItems,
        item: &impl ast::NameOwner,
    ) -> Option<ModuleItem> {
        let name = item.name()?.as_name();
        let kind = item.syntax().kind();
        let vis = Vis::Other;
        let item_id = Some(file_items.id_of_unchecked(item.syntax()));
        let id = SourceItemId { file_id, item_id };
        let res = ModuleItem {
            id,
            name,
            kind,
            vis,
        };
        Some(res)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportId(RawId);
impl_arena_id!(ImportId);

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ImportData {
    pub(super) path: Path,
    pub(super) is_glob: bool,
}

/// A set of items and imports declared inside a module, without relation to
/// other modules.
///
/// This sits in-between raw syntax and name resolution and allows us to avoid
/// recomputing name res: if two instance of `InputModuleItems` are the same, we
/// can avoid redoing name resolution.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LoweredModule {
    pub(crate) items: Vec<ModuleItem>,
    pub(super) imports: Arena<ImportId, ImportData>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImportSourceMap {
    map: ArenaMap<ImportId, AstPtr<ast::PathSegment>>,
}

impl ImportSourceMap {
    fn insert(&mut self, import: ImportId, segment: &ast::PathSegment) {
        self.map.insert(import, AstPtr::new(segment))
    }

    pub fn get(&self, source: &ModuleSource, import: ImportId) -> TreeArc<ast::PathSegment> {
        let file = match source {
            ModuleSource::SourceFile(file) => &*file,
            ModuleSource::Module(m) => m.syntax().ancestors().find_map(SourceFile::cast).unwrap(),
        };

        self.map[import].to_node(file).to_owned()
    }
}

impl LoweredModule {
    pub(crate) fn lower_module_module_query(
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Arc<LoweredModule> {
        db.lower_module(source_root_id, module_id).0
    }

    pub(crate) fn lower_module_source_map_query(
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Arc<ImportSourceMap> {
        db.lower_module(source_root_id, module_id).1
    }

    pub(crate) fn lower_module_query(
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> (Arc<LoweredModule>, Arc<ImportSourceMap>) {
        let module_tree = db.module_tree(source_root_id);
        let source = module_id.source(&module_tree);
        let file_id = source.file_id;
        let source = ModuleSource::from_source_item_id(db, source);
        let mut source_map = ImportSourceMap::default();
        let mut res = LoweredModule::default();
        match source {
            ModuleSource::SourceFile(it) => res.fill(
                &mut source_map,
                db,
                source_root_id,
                module_id,
                file_id,
                &mut it.items_with_macros(),
            ),
            ModuleSource::Module(it) => {
                if let Some(item_list) = it.item_list() {
                    res.fill(
                        &mut source_map,
                        db,
                        source_root_id,
                        module_id,
                        file_id,
                        &mut item_list.items_with_macros(),
                    )
                }
            }
        };
        (Arc::new(res), Arc::new(source_map))
    }

    fn fill(
        &mut self,
        source_map: &mut ImportSourceMap,
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
        file_id: HirFileId,
        items: &mut Iterator<Item = ast::ItemOrMacro>,
    ) {
        let file_items = db.file_items(file_id);

        for item in items {
            match item {
                ast::ItemOrMacro::Item(it) => {
                    self.add_item(source_map, file_id, &file_items, it);
                }
                ast::ItemOrMacro::Macro(macro_call) => {
                    let item_id = file_items.id_of_unchecked(macro_call.syntax());
                    let loc = MacroCallLoc {
                        source_root_id,
                        module_id,
                        source_item_id: SourceItemId {
                            file_id,
                            item_id: Some(item_id),
                        },
                    };
                    let id = loc.id(db);
                    let file_id = HirFileId::from(id);
                    let file_items = db.file_items(file_id);
                    //FIXME: expand recursively
                    for item in db.hir_source_file(file_id).items() {
                        self.add_item(source_map, file_id, &file_items, item);
                    }
                }
            }
        }
    }

    fn add_item(
        &mut self,
        source_map: &mut ImportSourceMap,
        file_id: HirFileId,
        file_items: &SourceFileItems,
        item: &ast::ModuleItem,
    ) -> Option<()> {
        match item.kind() {
            ast::ModuleItemKind::StructDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::EnumDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::FnDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::TraitDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::TypeDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::ImplBlock(_) => {
                // impls don't define items
            }
            ast::ModuleItemKind::UseItem(it) => self.add_use_item(source_map, it),
            ast::ModuleItemKind::ExternCrateItem(_) => {
                // TODO
            }
            ast::ModuleItemKind::ConstDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::StaticDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::Module(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
        }
        Some(())
    }

    fn add_use_item(&mut self, source_map: &mut ImportSourceMap, item: &ast::UseItem) {
        Path::expand_use_item(item, |path, segment| {
            let import = self.imports.alloc(ImportData {
                path,
                is_glob: segment.is_none(),
            });
            if let Some(segment) = segment {
                source_map.insert(import, segment)
            }
        })
    }
}
