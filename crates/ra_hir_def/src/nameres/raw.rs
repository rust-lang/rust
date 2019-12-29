//! Lowers syntax tree of a rust file into a raw representation of containing
//! items, *without* attaching them to a module structure.
//!
//! That is, raw items don't have semantics, just as syntax, but, unlike syntax,
//! they don't change with trivial source code edits, making them a great tool
//! for building salsa recomputation firewalls.

use std::{ops::Index, sync::Arc};

use hir_expand::{
    ast_id_map::AstIdMap,
    db::AstDatabase,
    hygiene::Hygiene,
    name::{AsName, Name},
};
use ra_arena::{impl_arena_id, Arena, RawId};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AttrsOwner, NameOwner, VisibilityOwner},
    AstNode,
};
use test_utils::tested_by;

use crate::{
    attr::Attrs, db::DefDatabase, path::ModPath, visibility::RawVisibility, FileAstId, HirFileId,
    InFile,
};

/// `RawItems` is a set of top-level items in a file (except for impls).
///
/// It is the input to name resolution algorithm. `RawItems` are not invalidated
/// on most edits.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RawItems {
    modules: Arena<Module, ModuleData>,
    imports: Arena<Import, ImportData>,
    defs: Arena<Def, DefData>,
    macros: Arena<Macro, MacroData>,
    impls: Arena<Impl, ImplData>,
    /// items for top-level module
    items: Vec<RawItem>,
}

impl RawItems {
    pub(crate) fn raw_items_query(
        db: &(impl DefDatabase + AstDatabase),
        file_id: HirFileId,
    ) -> Arc<RawItems> {
        let _p = profile("raw_items_query");
        let mut collector = RawItemsCollector {
            raw_items: RawItems::default(),
            source_ast_id_map: db.ast_id_map(file_id),
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
        let raw_items = collector.raw_items;
        Arc::new(raw_items)
    }

    pub(super) fn items(&self) -> &[RawItem] {
        &self.items
    }
}

impl Index<Module> for RawItems {
    type Output = ModuleData;
    fn index(&self, idx: Module) -> &ModuleData {
        &self.modules[idx]
    }
}

impl Index<Import> for RawItems {
    type Output = ImportData;
    fn index(&self, idx: Import) -> &ImportData {
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

impl Index<Impl> for RawItems {
    type Output = ImplData;
    fn index(&self, idx: Impl) -> &ImplData {
        &self.impls[idx]
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(super) struct RawItem {
    pub(super) attrs: Attrs,
    pub(super) kind: RawItemKind,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum RawItemKind {
    Module(Module),
    Import(Import),
    Def(Def),
    Macro(Macro),
    Impl(Impl),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Module(RawId);
impl_arena_id!(Module);

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ModuleData {
    Declaration {
        name: Name,
        visibility: RawVisibility,
        ast_id: FileAstId<ast::Module>,
    },
    Definition {
        name: Name,
        visibility: RawVisibility,
        ast_id: FileAstId<ast::Module>,
        items: Vec<RawItem>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Import(RawId);
impl_arena_id!(Import);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportData {
    pub(super) path: ModPath,
    pub(super) alias: Option<Name>,
    pub(super) is_glob: bool,
    pub(super) is_prelude: bool,
    pub(super) is_extern_crate: bool,
    pub(super) is_macro_use: bool,
    pub(super) visibility: RawVisibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Def(RawId);
impl_arena_id!(Def);

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DefData {
    pub(super) name: Name,
    pub(super) kind: DefKind,
    pub(super) visibility: RawVisibility,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum DefKind {
    Function(FileAstId<ast::FnDef>),
    Struct(FileAstId<ast::StructDef>),
    Union(FileAstId<ast::UnionDef>),
    Enum(FileAstId<ast::EnumDef>),
    Const(FileAstId<ast::ConstDef>),
    Static(FileAstId<ast::StaticDef>),
    Trait(FileAstId<ast::TraitDef>),
    TypeAlias(FileAstId<ast::TypeAliasDef>),
}

impl DefKind {
    pub fn ast_id(&self) -> FileAstId<ast::ModuleItem> {
        match self {
            DefKind::Function(it) => it.upcast(),
            DefKind::Struct(it) => it.upcast(),
            DefKind::Union(it) => it.upcast(),
            DefKind::Enum(it) => it.upcast(),
            DefKind::Const(it) => it.upcast(),
            DefKind::Static(it) => it.upcast(),
            DefKind::Trait(it) => it.upcast(),
            DefKind::TypeAlias(it) => it.upcast(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Macro(RawId);
impl_arena_id!(Macro);

#[derive(Debug, PartialEq, Eq)]
pub(super) struct MacroData {
    pub(super) ast_id: FileAstId<ast::MacroCall>,
    pub(super) path: ModPath,
    pub(super) name: Option<Name>,
    pub(super) export: bool,
    pub(super) builtin: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Impl(RawId);
impl_arena_id!(Impl);

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ImplData {
    pub(super) ast_id: FileAstId<ast::ImplBlock>,
}

struct RawItemsCollector {
    raw_items: RawItems,
    source_ast_id_map: Arc<AstIdMap>,
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
        let visibility = RawVisibility::from_ast_with_hygiene(item.visibility(), &self.hygiene);
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
            ast::ModuleItem::ImplBlock(it) => {
                self.add_impl(current_module, it);
                return;
            }
            ast::ModuleItem::StructDef(it) => {
                let id = self.source_ast_id_map.ast_id(&it);
                let name = it.name();
                (DefKind::Struct(id), name)
            }
            ast::ModuleItem::UnionDef(it) => {
                let id = self.source_ast_id_map.ast_id(&it);
                let name = it.name();
                (DefKind::Union(id), name)
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
            let def = self.raw_items.defs.alloc(DefData { name, kind, visibility });
            self.push_item(current_module, attrs, RawItemKind::Def(def));
        }
    }

    fn add_module(&mut self, current_module: Option<Module>, module: ast::Module) {
        let name = match module.name() {
            Some(it) => it.as_name(),
            None => return,
        };
        let attrs = self.parse_attrs(&module);
        let visibility = RawVisibility::from_ast_with_hygiene(module.visibility(), &self.hygiene);

        let ast_id = self.source_ast_id_map.ast_id(&module);
        if module.has_semi() {
            let item =
                self.raw_items.modules.alloc(ModuleData::Declaration { name, visibility, ast_id });
            self.push_item(current_module, attrs, RawItemKind::Module(item));
            return;
        }

        if let Some(item_list) = module.item_list() {
            let item = self.raw_items.modules.alloc(ModuleData::Definition {
                name,
                visibility,
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
        let visibility = RawVisibility::from_ast_with_hygiene(use_item.visibility(), &self.hygiene);

        let mut buf = Vec::new();
        ModPath::expand_use_item(
            InFile { value: use_item, file_id: self.file_id },
            &self.hygiene,
            |path, _use_tree, is_glob, alias| {
                let import_data = ImportData {
                    path,
                    alias,
                    is_glob,
                    is_prelude,
                    is_extern_crate: false,
                    is_macro_use: false,
                    visibility: visibility.clone(),
                };
                buf.push(import_data);
            },
        );
        for import_data in buf {
            self.push_import(current_module, attrs.clone(), import_data);
        }
    }

    fn add_extern_crate_item(
        &mut self,
        current_module: Option<Module>,
        extern_crate: ast::ExternCrateItem,
    ) {
        if let Some(name_ref) = extern_crate.name_ref() {
            let path = ModPath::from_name_ref(&name_ref);
            let visibility =
                RawVisibility::from_ast_with_hygiene(extern_crate.visibility(), &self.hygiene);
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
                visibility,
            };
            self.push_import(current_module, attrs, import_data);
        }
    }

    fn add_macro(&mut self, current_module: Option<Module>, m: ast::MacroCall) {
        let attrs = self.parse_attrs(&m);
        let path = match m.path().and_then(|path| ModPath::from_src(path, &self.hygiene)) {
            Some(it) => it,
            _ => return,
        };

        let name = m.name().map(|it| it.as_name());
        let ast_id = self.source_ast_id_map.ast_id(&m);
        // FIXME: cfg_attr
        let export = m.attrs().filter_map(|x| x.simple_name()).any(|name| name == "macro_export");

        // FIXME: cfg_attr
        let builtin =
            m.attrs().filter_map(|x| x.simple_name()).any(|name| name == "rustc_builtin_macro");

        let m = self.raw_items.macros.alloc(MacroData { ast_id, path, name, export, builtin });
        self.push_item(current_module, attrs, RawItemKind::Macro(m));
    }

    fn add_impl(&mut self, current_module: Option<Module>, imp: ast::ImplBlock) {
        let attrs = self.parse_attrs(&imp);
        let ast_id = self.source_ast_id_map.ast_id(&imp);
        let imp = self.raw_items.impls.alloc(ImplData { ast_id });
        self.push_item(current_module, attrs, RawItemKind::Impl(imp))
    }

    fn push_import(&mut self, current_module: Option<Module>, attrs: Attrs, data: ImportData) {
        let import = self.raw_items.imports.alloc(data);
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
        Attrs::new(item, &self.hygiene)
    }
}
