//! FIXME: write short doc here

use std::{ops::Index, sync::Arc};

use hir_expand::{
    ast_id_map::AstIdMap,
    db::AstDatabase,
    either::Either,
    hygiene::Hygiene,
    name::{AsName, Name},
};
use ra_arena::{impl_arena_id, map::ArenaMap, Arena, RawId};
use ra_syntax::{
    ast::{self, AttrsOwner, NameOwner},
    AstNode, AstPtr, SourceFile,
};
use test_utils::tested_by;

use crate::{attr::Attr, db::DefDatabase2, path::Path, FileAstId, HirFileId, ModuleSource, Source};

/// `RawItems` is a set of top-level items in a file (except for impls).
///
/// It is the input to name resolution algorithm. `RawItems` are not invalidated
/// on most edits.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RawItems {
    modules: Arena<Module, ModuleData>,
    imports: Arena<ImportId, ImportData>,
    defs: Arena<Def, DefData>,
    macros: Arena<Macro, MacroData>,
    /// items for top-level module
    items: Vec<RawItem>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ImportSourceMap {
    map: ArenaMap<ImportId, ImportSourcePtr>,
}

type ImportSourcePtr = Either<AstPtr<ast::UseTree>, AstPtr<ast::ExternCrateItem>>;
type ImportSource = Either<ast::UseTree, ast::ExternCrateItem>;

fn to_node(ptr: ImportSourcePtr, file: &SourceFile) -> ImportSource {
    ptr.map(|ptr| ptr.to_node(file.syntax()), |ptr| ptr.to_node(file.syntax()))
}

impl ImportSourceMap {
    fn insert(&mut self, import: ImportId, ptr: ImportSourcePtr) {
        self.map.insert(import, ptr)
    }

    pub fn get(&self, source: &ModuleSource, import: ImportId) -> ImportSource {
        let file = match source {
            ModuleSource::SourceFile(file) => file.clone(),
            ModuleSource::Module(m) => m.syntax().ancestors().find_map(SourceFile::cast).unwrap(),
        };

        to_node(self.map[import], &file)
    }
}

impl RawItems {
    pub(crate) fn raw_items_query(
        db: &(impl DefDatabase2 + AstDatabase),
        file_id: HirFileId,
    ) -> Arc<RawItems> {
        db.raw_items_with_source_map(file_id).0
    }

    pub(crate) fn raw_items_with_source_map_query(
        db: &(impl DefDatabase2 + AstDatabase),
        file_id: HirFileId,
    ) -> (Arc<RawItems>, Arc<ImportSourceMap>) {
        let mut collector = RawItemsCollector {
            raw_items: RawItems::default(),
            source_ast_id_map: db.ast_id_map(file_id),
            source_map: ImportSourceMap::default(),
            file_id,
            hygiene: Hygiene::new(db, file_id),
        };
        if let Some(node) = db.parse_or_expand(file_id) {
            if let Some(source_file) = ast::SourceFile::cast(node.clone()) {
                collector.process_module(None, source_file);
            } else if let Some(item_list) = ast::MacroItems::cast(node) {
                collector.process_module(None, item_list);
            }
        }
        (Arc::new(collector.raw_items), Arc::new(collector.source_map))
    }

    pub fn items(&self) -> &[RawItem] {
        &self.items
    }
}

impl Index<Module> for RawItems {
    type Output = ModuleData;
    fn index(&self, idx: Module) -> &ModuleData {
        &self.modules[idx]
    }
}

impl Index<ImportId> for RawItems {
    type Output = ImportData;
    fn index(&self, idx: ImportId) -> &ImportData {
        &self.imports[idx]
    }
}

impl Index<Def> for RawItems {
    type Output = DefData;
    fn index(&self, idx: Def) -> &DefData {
        &self.defs[idx]
    }
}

impl Index<Macro> for RawItems {
    type Output = MacroData;
    fn index(&self, idx: Macro) -> &MacroData {
        &self.macros[idx]
    }
}

// Avoid heap allocation on items without attributes.
type Attrs = Option<Arc<[Attr]>>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RawItem {
    attrs: Attrs,
    pub kind: RawItemKind,
}

impl RawItem {
    pub fn attrs(&self) -> &[Attr] {
        self.attrs.as_ref().map_or(&[], |it| &*it)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RawItemKind {
    Module(Module),
    Import(ImportId),
    Def(Def),
    Macro(Macro),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module(RawId);
impl_arena_id!(Module);

#[derive(Debug, PartialEq, Eq)]
pub enum ModuleData {
    Declaration { name: Name, ast_id: FileAstId<ast::Module> },
    Definition { name: Name, ast_id: FileAstId<ast::Module>, items: Vec<RawItem> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportId(RawId);
impl_arena_id!(ImportId);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportData {
    pub path: Path,
    pub alias: Option<Name>,
    pub is_glob: bool,
    pub is_prelude: bool,
    pub is_extern_crate: bool,
    pub is_macro_use: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Def(RawId);
impl_arena_id!(Def);

#[derive(Debug, PartialEq, Eq)]
pub struct DefData {
    pub name: Name,
    pub kind: DefKind,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DefKind {
    Function(FileAstId<ast::FnDef>),
    Struct(FileAstId<ast::StructDef>),
    Union(FileAstId<ast::StructDef>),
    Enum(FileAstId<ast::EnumDef>),
    Const(FileAstId<ast::ConstDef>),
    Static(FileAstId<ast::StaticDef>),
    Trait(FileAstId<ast::TraitDef>),
    TypeAlias(FileAstId<ast::TypeAliasDef>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Macro(RawId);
impl_arena_id!(Macro);

#[derive(Debug, PartialEq, Eq)]
pub struct MacroData {
    pub ast_id: FileAstId<ast::MacroCall>,
    pub path: Path,
    pub name: Option<Name>,
    pub export: bool,
}

struct RawItemsCollector {
    raw_items: RawItems,
    source_ast_id_map: Arc<AstIdMap>,
    source_map: ImportSourceMap,
    file_id: HirFileId,
    hygiene: Hygiene,
}

impl RawItemsCollector {
    fn process_module(&mut self, current_module: Option<Module>, body: impl ast::ModuleItemOwner) {
        for item_or_macro in body.items_with_macros() {
            match item_or_macro {
                ast::ItemOrMacro::Macro(m) => self.add_macro(current_module, m),
                ast::ItemOrMacro::Item(item) => self.add_item(current_module, item),
            }
        }
    }

    fn add_item(&mut self, current_module: Option<Module>, item: ast::ModuleItem) {
        let attrs = self.parse_attrs(&item);
        let (kind, name) = match item {
            ast::ModuleItem::Module(module) => {
                self.add_module(current_module, module);
                return;
            }
            ast::ModuleItem::UseItem(use_item) => {
                self.add_use_item(current_module, use_item);
                return;
            }
            ast::ModuleItem::ExternCrateItem(extern_crate) => {
                self.add_extern_crate_item(current_module, extern_crate);
                return;
            }
            ast::ModuleItem::ImplBlock(_) => {
                // impls don't participate in name resolution
                return;
            }
            ast::ModuleItem::StructDef(it) => {
                let id = self.source_ast_id_map.ast_id(&it);
                let name = it.name();
                if it.is_union() {
                    (DefKind::Union(id), name)
                } else {
                    (DefKind::Struct(id), name)
                }
            }
            ast::ModuleItem::EnumDef(it) => {
                (DefKind::Enum(self.source_ast_id_map.ast_id(&it)), it.name())
            }
            ast::ModuleItem::FnDef(it) => {
                (DefKind::Function(self.source_ast_id_map.ast_id(&it)), it.name())
            }
            ast::ModuleItem::TraitDef(it) => {
                (DefKind::Trait(self.source_ast_id_map.ast_id(&it)), it.name())
            }
            ast::ModuleItem::TypeAliasDef(it) => {
                (DefKind::TypeAlias(self.source_ast_id_map.ast_id(&it)), it.name())
            }
            ast::ModuleItem::ConstDef(it) => {
                (DefKind::Const(self.source_ast_id_map.ast_id(&it)), it.name())
            }
            ast::ModuleItem::StaticDef(it) => {
                (DefKind::Static(self.source_ast_id_map.ast_id(&it)), it.name())
            }
        };
        if let Some(name) = name {
            let name = name.as_name();
            let def = self.raw_items.defs.alloc(DefData { name, kind });
            self.push_item(current_module, attrs, RawItemKind::Def(def));
        }
    }

    fn add_module(&mut self, current_module: Option<Module>, module: ast::Module) {
        let name = match module.name() {
            Some(it) => it.as_name(),
            None => return,
        };
        let attrs = self.parse_attrs(&module);

        let ast_id = self.source_ast_id_map.ast_id(&module);
        if module.has_semi() {
            let item = self.raw_items.modules.alloc(ModuleData::Declaration { name, ast_id });
            self.push_item(current_module, attrs, RawItemKind::Module(item));
            return;
        }

        if let Some(item_list) = module.item_list() {
            let item = self.raw_items.modules.alloc(ModuleData::Definition {
                name,
                ast_id,
                items: Vec::new(),
            });
            self.process_module(Some(item), item_list);
            self.push_item(current_module, attrs, RawItemKind::Module(item));
            return;
        }
        tested_by!(name_res_works_for_broken_modules);
    }

    fn add_use_item(&mut self, current_module: Option<Module>, use_item: ast::UseItem) {
        // FIXME: cfg_attr
        let is_prelude = use_item.has_atom_attr("prelude_import");
        let attrs = self.parse_attrs(&use_item);

        let mut buf = Vec::new();
        Path::expand_use_item(
            Source { ast: use_item, file_id: self.file_id },
            &self.hygiene,
            |path, use_tree, is_glob, alias| {
                let import_data = ImportData {
                    path,
                    alias,
                    is_glob,
                    is_prelude,
                    is_extern_crate: false,
                    is_macro_use: false,
                };
                buf.push((import_data, Either::A(AstPtr::new(use_tree))));
            },
        );
        for (import_data, ptr) in buf {
            self.push_import(current_module, attrs.clone(), import_data, ptr);
        }
    }

    fn add_extern_crate_item(
        &mut self,
        current_module: Option<Module>,
        extern_crate: ast::ExternCrateItem,
    ) {
        if let Some(name_ref) = extern_crate.name_ref() {
            let path = Path::from_name_ref(&name_ref);
            let alias = extern_crate.alias().and_then(|a| a.name()).map(|it| it.as_name());
            let attrs = self.parse_attrs(&extern_crate);
            // FIXME: cfg_attr
            let is_macro_use = extern_crate.has_atom_attr("macro_use");
            let import_data = ImportData {
                path,
                alias,
                is_glob: false,
                is_prelude: false,
                is_extern_crate: true,
                is_macro_use,
            };
            self.push_import(
                current_module,
                attrs,
                import_data,
                Either::B(AstPtr::new(&extern_crate)),
            );
        }
    }

    fn add_macro(&mut self, current_module: Option<Module>, m: ast::MacroCall) {
        let attrs = self.parse_attrs(&m);
        let path = match m.path().and_then(|path| Path::from_src(path, &self.hygiene)) {
            Some(it) => it,
            _ => return,
        };

        let name = m.name().map(|it| it.as_name());
        let ast_id = self.source_ast_id_map.ast_id(&m);
        // FIXME: cfg_attr
        let export = m.attrs().filter_map(|x| x.simple_name()).any(|name| name == "macro_export");

        let m = self.raw_items.macros.alloc(MacroData { ast_id, path, name, export });
        self.push_item(current_module, attrs, RawItemKind::Macro(m));
    }

    fn push_import(
        &mut self,
        current_module: Option<Module>,
        attrs: Attrs,
        data: ImportData,
        source: ImportSourcePtr,
    ) {
        let import = self.raw_items.imports.alloc(data);
        self.source_map.insert(import, source);
        self.push_item(current_module, attrs, RawItemKind::Import(import))
    }

    fn push_item(&mut self, current_module: Option<Module>, attrs: Attrs, kind: RawItemKind) {
        match current_module {
            Some(module) => match &mut self.raw_items.modules[module] {
                ModuleData::Definition { items, .. } => items,
                ModuleData::Declaration { .. } => unreachable!(),
            },
            None => &mut self.raw_items.items,
        }
        .push(RawItem { attrs, kind })
    }

    fn parse_attrs(&self, item: &impl ast::AttrsOwner) -> Attrs {
        Attr::from_attrs_owner(item, &self.hygiene)
    }
}
