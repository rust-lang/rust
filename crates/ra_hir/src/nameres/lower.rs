use std::sync::Arc;

use ra_syntax::{
    AstNode, SourceFile, TreeArc, AstPtr,
    ast::{self, ModuleItemOwner, NameOwner},
};
use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};
use rustc_hash::FxHashMap;

use crate::{
    SourceItemId, Path, ModuleSource, Name,
    HirFileId, MacroCallLoc, AsName, PerNs, Function,
    ModuleDef, Module, Struct, Enum, Const, Static, Trait, Type,
    ids::LocationCtx, PersistentHirDatabase,
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
        db: &impl PersistentHirDatabase,
        module: Module,
    ) -> Arc<LoweredModule> {
        db.lower_module(module).0
    }

    pub(crate) fn lower_module_source_map_query(
        db: &impl PersistentHirDatabase,
        module: Module,
    ) -> Arc<ImportSourceMap> {
        db.lower_module(module).1
    }

    pub(crate) fn lower_module_query(
        db: &impl PersistentHirDatabase,
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
        db: &impl PersistentHirDatabase,
        module: Module,
        file_id: HirFileId,
        items: &mut Iterator<Item = ast::ItemOrMacro>,
    ) {
        let file_items = db.file_items(file_id);

        for item in items {
            match item {
                ast::ItemOrMacro::Item(it) => {
                    self.add_def_id(source_map, db, module, file_id, it);
                }
                ast::ItemOrMacro::Macro(macro_call) => {
                    let item_id = file_items.id_of_unchecked(macro_call.syntax());
                    let loc = MacroCallLoc {
                        module,
                        source_item_id: SourceItemId { file_id, item_id },
                    };
                    let id = loc.id(db);
                    let file_id = HirFileId::from(id);
                    //FIXME: expand recursively
                    for item in db.hir_parse(file_id).items() {
                        self.add_def_id(source_map, db, module, file_id, item);
                    }
                }
            }
        }
    }

    fn add_def_id(
        &mut self,
        source_map: &mut ImportSourceMap,
        db: &impl PersistentHirDatabase,
        module: Module,
        file_id: HirFileId,
        item: &ast::ModuleItem,
    ) {
        let ctx = LocationCtx::new(db, module, file_id);
        match item.kind() {
            ast::ModuleItemKind::StructDef(it) => {
                if let Some(name) = it.name() {
                    let s = Struct { id: ctx.to_def(it) };
                    let s: ModuleDef = s.into();
                    self.declarations.insert(name.as_name(), PerNs::both(s, s));
                }
            }
            ast::ModuleItemKind::EnumDef(it) => {
                if let Some(name) = it.name() {
                    let e = Enum { id: ctx.to_def(it) };
                    let e: ModuleDef = e.into();
                    self.declarations.insert(name.as_name(), PerNs::types(e));
                }
            }
            ast::ModuleItemKind::FnDef(it) => {
                if let Some(name) = it.name() {
                    let func = Function { id: ctx.to_def(it) };
                    self.declarations
                        .insert(name.as_name(), PerNs::values(func.into()));
                }
            }
            ast::ModuleItemKind::TraitDef(it) => {
                if let Some(name) = it.name() {
                    let t = Trait { id: ctx.to_def(it) };
                    self.declarations
                        .insert(name.as_name(), PerNs::types(t.into()));
                }
            }
            ast::ModuleItemKind::TypeDef(it) => {
                if let Some(name) = it.name() {
                    let t = Type { id: ctx.to_def(it) };
                    self.declarations
                        .insert(name.as_name(), PerNs::types(t.into()));
                }
            }
            ast::ModuleItemKind::ImplBlock(_) => {
                // impls don't define items
            }
            ast::ModuleItemKind::UseItem(it) => {
                self.add_use_item(source_map, it);
            }
            ast::ModuleItemKind::ExternCrateItem(_) => {
                // TODO
            }
            ast::ModuleItemKind::ConstDef(it) => {
                if let Some(name) = it.name() {
                    let c = Const { id: ctx.to_def(it) };
                    self.declarations
                        .insert(name.as_name(), PerNs::values(c.into()));
                }
            }
            ast::ModuleItemKind::StaticDef(it) => {
                if let Some(name) = it.name() {
                    let s = Static { id: ctx.to_def(it) };
                    self.declarations
                        .insert(name.as_name(), PerNs::values(s.into()));
                }
            }
            ast::ModuleItemKind::Module(_) => {
                // modules are handled separately direclty by nameres
            }
        };
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
