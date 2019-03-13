use std::{
    sync::Arc,
    ops::Index,
};

use ra_db::FileId;
use ra_arena::{Arena, impl_arena_id, RawId};
use ra_syntax::{
    AstNode, SourceFile,
    ast::{self, NameOwner, AttrsOwner},
};

use crate::{
    PersistentHirDatabase, Name, AsName, Path, HirFileId,
    ids::{SourceFileItemId, SourceFileItems},
};

#[derive(Default, PartialEq, Eq)]
pub(crate) struct RawItems {
    modules: Arena<Module, ModuleData>,
    imports: Arena<ImportId, ImportData>,
    defs: Arena<Def, DefData>,
    macros: Arena<Macro, MacroData>,
    /// items for top-level module
    items: Vec<RawItem>,
}

impl RawItems {
    pub(crate) fn items(&self) -> &[RawItem] {
        &self.items
    }

    pub(crate) fn raw_items_query(db: &impl PersistentHirDatabase, file_id: FileId) -> RawItems {
        let mut collector = RawItemsCollector {
            raw_items: RawItems::default(),
            source_file_items: db.file_items(file_id.into()),
        };
        let source_file = db.parse(file_id);
        collector.process_module(None, &*source_file);
        collector.raw_items
    }

    // We can't use queries during name resolution for fear of cycles, so this
    // is a query-less variant of the above function.
    pub(crate) fn from_source_file(source_file: &SourceFile, file_id: HirFileId) -> RawItems {
        let source_file_items = SourceFileItems::from_source_file(source_file, file_id);
        let mut collector = RawItemsCollector {
            raw_items: RawItems::default(),
            source_file_items: Arc::new(source_file_items),
        };
        collector.process_module(None, &*source_file);
        collector.raw_items
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

#[derive(PartialEq, Eq, Clone, Copy)]
pub(crate) enum RawItem {
    Module(Module),
    Import(ImportId),
    Def(Def),
    Macro(Macro),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Module(RawId);
impl_arena_id!(Module);

#[derive(PartialEq, Eq)]
pub(crate) enum ModuleData {
    Declaration { name: Name },
    Definition { name: Name, items: Vec<RawItem> },
}

pub(crate) use crate::nameres::lower::ImportId;
pub(super) use crate::nameres::lower::ImportData;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Def(RawId);
impl_arena_id!(Def);

#[derive(PartialEq, Eq)]
pub(crate) struct DefData {
    pub(crate) source_item_id: SourceFileItemId,
    pub(crate) name: Name,
    pub(crate) kind: DefKind,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum DefKind {
    Function,
    Struct,
    Enum,
    Const,
    Static,
    Trait,
    TypeAlias,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Macro(RawId);
impl_arena_id!(Macro);

#[derive(PartialEq, Eq)]
pub(crate) struct MacroData {
    pub(crate) source_item_id: SourceFileItemId,
    pub(crate) path: Path,
    pub(crate) name: Option<Name>,
    pub(crate) arg: tt::Subtree,
    pub(crate) export: bool,
}

struct RawItemsCollector {
    raw_items: RawItems,
    source_file_items: Arc<SourceFileItems>,
}

impl RawItemsCollector {
    fn process_module(&mut self, current_module: Option<Module>, body: &impl ast::ModuleItemOwner) {
        for item_or_macro in body.items_with_macros() {
            match item_or_macro {
                ast::ItemOrMacro::Macro(m) => self.add_macro(current_module, m),
                ast::ItemOrMacro::Item(item) => self.add_item(current_module, item),
            }
        }
    }

    fn add_item(&mut self, current_module: Option<Module>, item: &ast::ModuleItem) {
        let (kind, name) = match item.kind() {
            ast::ModuleItemKind::Module(module) => {
                self.add_module(current_module, module);
                return;
            }
            ast::ModuleItemKind::UseItem(use_item) => {
                self.add_use_item(current_module, use_item);
                return;
            }
            ast::ModuleItemKind::ExternCrateItem(extern_crate) => {
                self.add_extern_crate_item(current_module, extern_crate);
                return;
            }
            ast::ModuleItemKind::ImplBlock(_) => {
                // impls don't participate in name resolution
                return;
            }
            ast::ModuleItemKind::StructDef(it) => (DefKind::Struct, it.name()),
            ast::ModuleItemKind::EnumDef(it) => (DefKind::Enum, it.name()),
            ast::ModuleItemKind::FnDef(it) => (DefKind::Function, it.name()),
            ast::ModuleItemKind::TraitDef(it) => (DefKind::Trait, it.name()),
            ast::ModuleItemKind::TypeAliasDef(it) => (DefKind::TypeAlias, it.name()),
            ast::ModuleItemKind::ConstDef(it) => (DefKind::Const, it.name()),
            ast::ModuleItemKind::StaticDef(it) => (DefKind::Static, it.name()),
        };
        if let Some(name) = name {
            let name = name.as_name();
            let source_item_id = self.source_file_items.id_of_unchecked(item.syntax());
            let def = self.raw_items.defs.alloc(DefData { name, kind, source_item_id });
            self.push_item(current_module, RawItem::Def(def))
        }
    }

    fn add_module(&mut self, current_module: Option<Module>, module: &ast::Module) {
        let name = match module.name() {
            Some(it) => it.as_name(),
            None => return,
        };
        if module.has_semi() {
            let item = self.raw_items.modules.alloc(ModuleData::Declaration { name });
            self.push_item(current_module, RawItem::Module(item));
            return;
        }

        if let Some(item_list) = module.item_list() {
            let item =
                self.raw_items.modules.alloc(ModuleData::Definition { name, items: Vec::new() });
            self.process_module(Some(item), item_list);
            self.push_item(current_module, RawItem::Module(item));
        }
    }

    fn add_use_item(&mut self, current_module: Option<Module>, use_item: &ast::UseItem) {
        let is_prelude = use_item.has_atom_attr("prelude_import");

        Path::expand_use_item(use_item, |path, segment, alias| {
            let import = self.raw_items.imports.alloc(ImportData {
                path,
                alias,
                is_glob: segment.is_none(),
                is_prelude,
                is_extern_crate: false,
            });
            self.push_item(current_module, RawItem::Import(import))
        })
    }

    fn add_extern_crate_item(
        &mut self,
        current_module: Option<Module>,
        extern_crate: &ast::ExternCrateItem,
    ) {
        if let Some(name_ref) = extern_crate.name_ref() {
            let path = Path::from_name_ref(name_ref);
            let alias = extern_crate.alias().and_then(|a| a.name()).map(AsName::as_name);
            let import = self.raw_items.imports.alloc(ImportData {
                path,
                alias,
                is_glob: false,
                is_prelude: false,
                is_extern_crate: true,
            });
            self.push_item(current_module, RawItem::Import(import))
        }
    }

    fn add_macro(&mut self, current_module: Option<Module>, m: &ast::MacroCall) {
        let (path, arg) = match (
            m.path().and_then(Path::from_ast),
            m.token_tree().and_then(mbe::ast_to_token_tree),
        ) {
            (Some(path), Some((token_tree, _token_map))) => (path, token_tree),
            _ => return,
        };

        let name = m.name().map(|it| it.as_name());
        let source_item_id = self.source_file_items.id_of_unchecked(m.syntax());
        let export = m.has_atom_attr("macro_export");
        let m = self.raw_items.macros.alloc(MacroData { source_item_id, path, arg, name, export });
        self.push_item(current_module, RawItem::Macro(m));
    }

    fn push_item(&mut self, current_module: Option<Module>, item: RawItem) {
        match current_module {
            Some(module) => match &mut self.raw_items.modules[module] {
                ModuleData::Definition { items, .. } => items,
                ModuleData::Declaration { .. } => unreachable!(),
            },
            None => &mut self.raw_items.items,
        }
        .push(item)
    }
}
