use std::sync::Arc;

use ra_syntax::{
    SyntaxKind, AstNode, SourceFile, TreeArc, AstPtr,
    ast::{self, ModuleItemOwner, NameOwner},
};
use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};
use rustc_hash::FxHashMap;

use crate::{
    SourceItemId, Path, ModuleSource, HirDatabase, Name, SourceFileItems,
    HirFileId, MacroCallLoc, AsName, PerNs, DefKind, DefLoc,
    ModuleDef, Module,
};

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
    pub(crate) declarations: FxHashMap<Name, PerNs<ModuleDef>>,
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
        module: Module,
    ) -> Arc<LoweredModule> {
        db.lower_module(module).0
    }

    pub(crate) fn lower_module_source_map_query(
        db: &impl HirDatabase,
        module: Module,
    ) -> Arc<ImportSourceMap> {
        db.lower_module(module).1
    }

    pub(crate) fn lower_module_query(
        db: &impl HirDatabase,
        module: Module,
    ) -> (Arc<LoweredModule>, Arc<ImportSourceMap>) {
        let (file_id, source) = module.definition_source(db);
        let file_id: HirFileId = file_id.into();
        let mut source_map = ImportSourceMap::default();
        let mut res = LoweredModule::default();
        match source {
            ModuleSource::SourceFile(it) => res.fill(
                &mut source_map,
                db,
                module,
                file_id,
                &mut it.items_with_macros(),
            ),
            ModuleSource::Module(it) => {
                if let Some(item_list) = it.item_list() {
                    res.fill(
                        &mut source_map,
                        db,
                        module,
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
        module: Module,
        file_id: HirFileId,
        items: &mut Iterator<Item = ast::ItemOrMacro>,
    ) {
        let file_items = db.file_items(file_id);

        for item in items {
            match item {
                ast::ItemOrMacro::Item(it) => {
                    self.add_def_id(source_map, db, module, file_id, &file_items, it);
                }
                ast::ItemOrMacro::Macro(macro_call) => {
                    let item_id = file_items.id_of_unchecked(macro_call.syntax());
                    let loc = MacroCallLoc {
                        module,
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
                        self.add_def_id(source_map, db, module, file_id, &file_items, item);
                    }
                }
            }
        }
    }

    fn add_def_id(
        &mut self,
        source_map: &mut ImportSourceMap,
        db: &impl HirDatabase,
        module: Module,
        file_id: HirFileId,
        file_items: &SourceFileItems,
        item: &ast::ModuleItem,
    ) {
        let name = match item.kind() {
            ast::ModuleItemKind::StructDef(it) => it.name(),
            ast::ModuleItemKind::EnumDef(it) => it.name(),
            ast::ModuleItemKind::FnDef(it) => it.name(),
            ast::ModuleItemKind::TraitDef(it) => it.name(),
            ast::ModuleItemKind::TypeDef(it) => it.name(),
            ast::ModuleItemKind::ImplBlock(_) => {
                // impls don't define items
                return;
            }
            ast::ModuleItemKind::UseItem(it) => {
                self.add_use_item(source_map, it);
                return;
            }
            ast::ModuleItemKind::ExternCrateItem(_) => {
                // TODO
                return;
            }
            ast::ModuleItemKind::ConstDef(it) => it.name(),
            ast::ModuleItemKind::StaticDef(it) => it.name(),
            ast::ModuleItemKind::Module(_) => {
                // modules are handled separately direclty by nameres
                return;
            }
        };
        if let Some(name) = name {
            let def_id = assign_def_id(db, module, file_id, file_items, item);
            self.declarations.insert(name.as_name(), def_id);
        }
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

fn assign_def_id(
    db: &impl HirDatabase,
    module: Module,
    file_id: HirFileId,
    file_items: &SourceFileItems,
    item: &ast::ModuleItem,
) -> PerNs<ModuleDef> {
    // depending on the item kind, the location can define something in
    // the values namespace, the types namespace, or both
    let kind = DefKind::for_syntax_kind(item.syntax().kind());
    let def_id = kind.map(|k| {
        let item_id = file_items.id_of_unchecked(item.syntax());
        let def_loc = DefLoc {
            kind: k,
            module,
            source_item_id: SourceItemId {
                file_id,
                item_id: Some(item_id),
            },
        };
        def_loc.id(db).into()
    });
    def_id
}

impl DefKind {
    fn for_syntax_kind(kind: SyntaxKind) -> PerNs<DefKind> {
        match kind {
            SyntaxKind::FN_DEF => PerNs::values(DefKind::Function),
            SyntaxKind::STRUCT_DEF => PerNs::both(DefKind::Struct, DefKind::StructCtor),
            SyntaxKind::ENUM_DEF => PerNs::types(DefKind::Enum),
            SyntaxKind::TRAIT_DEF => PerNs::types(DefKind::Trait),
            SyntaxKind::TYPE_DEF => PerNs::types(DefKind::Type),
            SyntaxKind::CONST_DEF => PerNs::values(DefKind::Const),
            SyntaxKind::STATIC_DEF => PerNs::values(DefKind::Static),
            _ => PerNs::none(),
        }
    }
}
