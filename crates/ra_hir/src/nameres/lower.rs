use std::sync::Arc;

use ra_syntax::{
    AstNode, SourceFile, TreeArc, AstPtr,
    ast::{self, ModuleItemOwner, NameOwner, AttrsOwner},
};
use ra_arena::{Arena, RawId, impl_arena_id, map::ArenaMap};
use rustc_hash::FxHashMap;

use crate::{
    SourceItemId, Path, ModuleSource, Name,
    HirFileId, MacroCallLoc, AsName, PerNs, Function,
    ModuleDef, Module, Struct, Enum, Const, Static, Trait, TypeAlias,
    ids::LocationCtx, PersistentHirDatabase,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportId(RawId);
impl_arena_id!(ImportId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportData {
    pub(super) path: Path,
    pub(super) alias: Option<Name>,
    pub(super) is_glob: bool,
    pub(super) is_prelude: bool,
    pub(super) is_extern_crate: bool,
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
    pub(crate) fn lower_module_query(
        db: &impl PersistentHirDatabase,
        module: Module,
    ) -> Arc<LoweredModule> {
        db.lower_module_with_source_map(module).0
    }

    pub(crate) fn lower_module_with_source_map_query(
        db: &impl PersistentHirDatabase,
        module: Module,
    ) -> (Arc<LoweredModule>, Arc<ImportSourceMap>) {
        let (file_id, source) = module.definition_source(db);
        let file_id: HirFileId = file_id.into();
        let mut source_map = ImportSourceMap::default();
        let mut res = LoweredModule::default();
        match source {
            ModuleSource::SourceFile(it) => {
                res.fill(&mut source_map, db, module, file_id, &mut it.items_with_macros())
            }
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
                    let loc =
                        MacroCallLoc { module, source_item_id: SourceItemId { file_id, item_id } };
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
                    self.declarations.insert(name.as_name(), PerNs::values(func.into()));
                }
            }
            ast::ModuleItemKind::TraitDef(it) => {
                if let Some(name) = it.name() {
                    let t = Trait { id: ctx.to_def(it) };
                    self.declarations.insert(name.as_name(), PerNs::types(t.into()));
                }
            }
            ast::ModuleItemKind::TypeAliasDef(it) => {
                if let Some(name) = it.name() {
                    let t = TypeAlias { id: ctx.to_def(it) };
                    self.declarations.insert(name.as_name(), PerNs::types(t.into()));
                }
            }
            ast::ModuleItemKind::ImplBlock(_) => {
                // impls don't define items
            }
            ast::ModuleItemKind::UseItem(it) => {
                self.add_use_item(source_map, it);
            }
            ast::ModuleItemKind::ExternCrateItem(it) => {
                if let Some(name_ref) = it.name_ref() {
                    let path = Path::from_name_ref(name_ref);
                    let alias = it.alias().and_then(|a| a.name()).map(AsName::as_name);
                    self.imports.alloc(ImportData {
                        path,
                        alias,
                        is_glob: false,
                        is_prelude: false,
                        is_extern_crate: true,
                    });
                }
            }
            ast::ModuleItemKind::ConstDef(it) => {
                if let Some(name) = it.name() {
                    let c = Const { id: ctx.to_def(it) };
                    self.declarations.insert(name.as_name(), PerNs::values(c.into()));
                }
            }
            ast::ModuleItemKind::StaticDef(it) => {
                if let Some(name) = it.name() {
                    let s = Static { id: ctx.to_def(it) };
                    self.declarations.insert(name.as_name(), PerNs::values(s.into()));
                }
            }
            ast::ModuleItemKind::Module(_) => {
                // modules are handled separately directly by name res
            }
        };
    }

    fn add_use_item(&mut self, source_map: &mut ImportSourceMap, item: &ast::UseItem) {
        let is_prelude =
            item.attrs().any(|attr| attr.as_atom().map(|s| s == "prelude_import").unwrap_or(false));
        Path::expand_use_item(item, |path, segment, alias| {
            let import = self.imports.alloc(ImportData {
                path,
                alias,
                is_glob: segment.is_none(),
                is_prelude,
                is_extern_crate: false,
            });
            if let Some(segment) = segment {
                source_map.insert(import, segment)
            }
        })
    }
}
