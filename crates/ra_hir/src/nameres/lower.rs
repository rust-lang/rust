use std::sync::Arc;

use ra_syntax::{
    TextRange, SyntaxKind, AstNode,
    ast::{self, ModuleItemOwner},
};
use ra_db::{FileId, SourceRootId};

use crate::{
    SourceItemId, SourceFileItemId, Path, ModuleSource, HirDatabase, Name, SourceFileItems,
    HirFileId, MacroCallLoc, AsName,
    module_tree::ModuleId
};
/// A set of items and imports declared inside a module, without relation to
/// other modules.
///
/// This sits in-between raw syntax and name resolution and allows us to avoid
/// recomputing name res: if two instance of `InputModuleItems` are the same, we
/// can avoid redoing name resolution.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct InputModuleItems {
    pub(crate) items: Vec<ModuleItem>,
    pub(super) imports: Vec<Import>,
}

impl InputModuleItems {
    pub(crate) fn input_module_items_query(
        db: &impl HirDatabase,
        source_root_id: SourceRootId,
        module_id: ModuleId,
    ) -> Arc<InputModuleItems> {
        let module_tree = db.module_tree(source_root_id);
        let source = module_id.source(&module_tree);
        let file_id = source.file_id;
        let source = ModuleSource::from_source_item_id(db, source);
        let file_items = db.file_items(file_id);
        let fill = |acc: &mut InputModuleItems, items: &mut Iterator<Item = ast::ItemOrMacro>| {
            for item in items {
                match item {
                    ast::ItemOrMacro::Item(it) => {
                        acc.add_item(file_id, &file_items, it);
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
                            acc.add_item(file_id, &file_items, item);
                        }
                    }
                }
            }
        };

        let mut res = InputModuleItems::default();
        match source {
            ModuleSource::SourceFile(it) => fill(&mut res, &mut it.items_with_macros()),
            ModuleSource::Module(it) => {
                if let Some(item_list) = it.item_list() {
                    fill(&mut res, &mut item_list.items_with_macros())
                }
            }
        };
        Arc::new(res)
    }

    pub(crate) fn add_item(
        &mut self,
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
            ast::ModuleItemKind::UseItem(it) => self.add_use_item(file_items, it),
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

    fn add_use_item(&mut self, file_items: &SourceFileItems, item: &ast::UseItem) {
        let file_item_id = file_items.id_of_unchecked(item.syntax());
        let start_offset = item.syntax().range().start();
        Path::expand_use_item(item, |path, range| {
            let kind = match range {
                None => ImportKind::Glob,
                Some(range) => ImportKind::Named(NamedImport {
                    file_item_id,
                    relative_range: range - start_offset,
                }),
            };
            self.imports.push(Import { kind, path })
        })
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Import {
    pub(super) path: Path,
    pub(super) kind: ImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamedImport {
    pub file_item_id: SourceFileItemId,
    pub relative_range: TextRange,
}

impl NamedImport {
    // FIXME: this is only here for one use-case in completion. Seems like a
    // pretty gross special case.
    pub fn range(&self, db: &impl HirDatabase, file_id: FileId) -> TextRange {
        let source_item_id = SourceItemId {
            file_id: file_id.into(),
            item_id: Some(self.file_item_id),
        };
        let syntax = db.file_item(source_item_id);
        let offset = syntax.range().start();
        self.relative_range + offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ImportKind {
    Glob,
    Named(NamedImport),
}
