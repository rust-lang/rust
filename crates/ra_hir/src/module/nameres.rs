//! Name resolution algorithm. The end result of the algorithm is `ItemMap`: a
//! map with maps each module to it's scope: the set of items, visible in the
//! module. That is, we only resolve imports here, name resolution of item
//! bodies will be done in a separate step.
//!
//! Like Rustc, we use an interative per-crate algorithm: we start with scopes
//! containing only directly defined items, and then iteratively resolve
//! imports.
//!
//! To make this work nicely in the IDE scenarios, we place `InputModuleItems`
//! in between raw syntax and name resolution. `InputModuleItems` are computed
//! using only the module's syntax, and it is all directly defined items plus
//! imports. The plain is to make `InputModuleItems` independent of local
//! modifications (that is, typing inside a function shold not change IMIs),
//! such that the results of name resolution can be preserved unless the module
//! structure itself is modified.
use std::{
    sync::Arc,
};

use rustc_hash::FxHashMap;
use ra_syntax::{
    TextRange,
    SmolStr, SyntaxKind::{self, *},
    ast::{self, AstNode}
};
use ra_db::SourceRootId;

use crate::{
    Cancelable, FileId,
    DefId, DefLoc, DefKind,
    SourceItemId, SourceFileItemId, SourceFileItems,
    Path, PathKind,
    HirDatabase, Crate,
    module::{ModuleId, ModuleTree},
};

/// Item map is the result of the name resolution. Item map contains, for each
/// module, the set of visible items.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ItemMap {
    pub per_module: FxHashMap<ModuleId, ModuleScope>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ModuleScope {
    items: FxHashMap<SmolStr, Resolution>,
}

impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a SmolStr, &Resolution)> + 'a {
        self.items.iter()
    }
    pub fn get(&self, name: &SmolStr) -> Option<&Resolution> {
        self.items.get(name)
    }
}

/// A set of items and imports declared inside a module, without relation to
/// other modules.
///
/// This stands in-between raw syntax and name resolution and alow us to avoid
/// recomputing name res: if `InputModuleItems` are the same, we can avoid
/// running name resolution.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct InputModuleItems {
    items: Vec<ModuleItem>,
    imports: Vec<Import>,
}

#[derive(Debug, PartialEq, Eq)]
struct ModuleItem {
    id: SourceFileItemId,
    name: SmolStr,
    kind: SyntaxKind,
    vis: Vis,
}

#[derive(Debug, PartialEq, Eq)]
enum Vis {
    // Priv,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Import {
    path: Path,
    kind: ImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamedImport {
    pub file_item_id: SourceFileItemId,
    pub relative_range: TextRange,
}

impl NamedImport {
    pub fn range(&self, db: &impl HirDatabase, file_id: FileId) -> TextRange {
        let source_item_id = SourceItemId {
            file_id,
            item_id: Some(self.file_item_id),
        };
        let syntax = db.file_item(source_item_id);
        let offset = syntax.borrowed().range().start();
        self.relative_range + offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ImportKind {
    Glob,
    Named(NamedImport),
}

/// Resolution is basically `DefId` atm, but it should account for stuff like
/// multiple namespaces, ambiguity and errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// None for unresolved
    pub def_id: Option<DefId>,
    /// ident by whitch this is imported into local scope.
    pub import: Option<NamedImport>,
}

// #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
// enum Namespace {
//     Types,
//     Values,
// }

// #[derive(Debug)]
// struct PerNs<T> {
//     types: Option<T>,
//     values: Option<T>,
// }

impl InputModuleItems {
    pub(crate) fn new<'a>(
        file_items: &SourceFileItems,
        items: impl Iterator<Item = ast::ModuleItem<'a>>,
    ) -> InputModuleItems {
        let mut res = InputModuleItems::default();
        for item in items {
            res.add_item(file_items, item);
        }
        res
    }

    fn add_item(&mut self, file_items: &SourceFileItems, item: ast::ModuleItem) -> Option<()> {
        match item {
            ast::ModuleItem::StructDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::EnumDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::FnDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::TraitDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::TypeDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::ImplItem(_) => {
                // impls don't define items
            }
            ast::ModuleItem::UseItem(it) => self.add_use_item(file_items, it),
            ast::ModuleItem::ExternCrateItem(_) => {
                // TODO
            }
            ast::ModuleItem::ConstDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::StaticDef(it) => self.items.push(ModuleItem::new(file_items, it)?),
            ast::ModuleItem::Module(it) => self.items.push(ModuleItem::new(file_items, it)?),
        }
        Some(())
    }

    fn add_use_item(&mut self, file_items: &SourceFileItems, item: ast::UseItem) {
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

impl ModuleItem {
    fn new<'a>(file_items: &SourceFileItems, item: impl ast::NameOwner<'a>) -> Option<ModuleItem> {
        let name = item.name()?.text();
        let kind = item.syntax().kind();
        let vis = Vis::Other;
        let id = file_items.id_of_unchecked(item.syntax());
        let res = ModuleItem {
            id,
            name,
            kind,
            vis,
        };
        Some(res)
    }
}

pub(crate) struct Resolver<'a, DB> {
    db: &'a DB,
    input: &'a FxHashMap<ModuleId, Arc<InputModuleItems>>,
    source_root: SourceRootId,
    module_tree: Arc<ModuleTree>,
    result: ItemMap,
}

impl<'a, DB> Resolver<'a, DB>
where
    DB: HirDatabase,
{
    pub(crate) fn new(
        db: &'a DB,
        input: &'a FxHashMap<ModuleId, Arc<InputModuleItems>>,
        source_root: SourceRootId,
        module_tree: Arc<ModuleTree>,
    ) -> Resolver<'a, DB> {
        Resolver {
            db,
            input,
            source_root,
            module_tree,
            result: ItemMap::default(),
        }
    }

    pub(crate) fn resolve(mut self) -> Cancelable<ItemMap> {
        for (&module_id, items) in self.input.iter() {
            self.populate_module(module_id, items)?;
        }

        for &module_id in self.input.keys() {
            self.db.check_canceled()?;
            self.resolve_imports(module_id)?;
        }
        Ok(self.result)
    }

    fn populate_module(&mut self, module_id: ModuleId, input: &InputModuleItems) -> Cancelable<()> {
        let file_id = module_id.source(&self.module_tree).file_id();

        let mut module_items = ModuleScope::default();

        // Populate extern crates prelude
        {
            let root_id = module_id.crate_root(&self.module_tree);
            let file_id = root_id.source(&self.module_tree).file_id();
            let crate_graph = self.db.crate_graph();
            if let Some(crate_id) = crate_graph.crate_id_for_crate_root(file_id) {
                let krate = Crate::new(crate_id);
                for dep in krate.dependencies(self.db) {
                    if let Some(module) = dep.krate.root_module(self.db)? {
                        let def_id = module.def_id(self.db);
                        self.add_module_item(&mut module_items, dep.name, def_id);
                    }
                }
            };
        }
        for import in input.imports.iter() {
            if let Some(name) = import.path.segments.iter().last() {
                if let ImportKind::Named(import) = import.kind {
                    module_items.items.insert(
                        name.clone(),
                        Resolution {
                            def_id: None,
                            import: Some(import),
                        },
                    );
                }
            }
        }
        // Populate explicitelly declared items, except modules
        for item in input.items.iter() {
            if item.kind == MODULE {
                continue;
            }
            let def_loc = DefLoc {
                kind: DefKind::Item,
                source_root_id: self.source_root,
                module_id,
                source_item_id: SourceItemId {
                    file_id,
                    item_id: Some(item.id),
                },
            };
            let def_id = def_loc.id(self.db);
            let resolution = Resolution {
                def_id: Some(def_id),
                import: None,
            };
            module_items.items.insert(item.name.clone(), resolution);
        }

        // Populate modules
        for (name, module_id) in module_id.children(&self.module_tree) {
            let def_loc = DefLoc {
                kind: DefKind::Module,
                source_root_id: self.source_root,
                module_id,
                source_item_id: module_id.source(&self.module_tree).0,
            };
            let def_id = def_loc.id(self.db);
            self.add_module_item(&mut module_items, name, def_id);
        }

        self.result.per_module.insert(module_id, module_items);
        Ok(())
    }

    fn add_module_item(&self, module_items: &mut ModuleScope, name: SmolStr, def_id: DefId) {
        let resolution = Resolution {
            def_id: Some(def_id),
            import: None,
        };
        module_items.items.insert(name, resolution);
    }

    fn resolve_imports(&mut self, module_id: ModuleId) -> Cancelable<()> {
        for import in self.input[&module_id].imports.iter() {
            self.resolve_import(module_id, import)?;
        }
        Ok(())
    }

    fn resolve_import(&mut self, module_id: ModuleId, import: &Import) -> Cancelable<()> {
        let ptr = match import.kind {
            ImportKind::Glob => return Ok(()),
            ImportKind::Named(ptr) => ptr,
        };

        let mut curr: ModuleId = match import.path.kind {
            PathKind::Plain | PathKind::Self_ => module_id,
            PathKind::Super => {
                match module_id.parent(&self.module_tree) {
                    Some(it) => it,
                    // TODO: error
                    None => return Ok(()),
                }
            }
            PathKind::Crate => module_id.crate_root(&self.module_tree),
        };

        for (i, name) in import.path.segments.iter().enumerate() {
            let is_last = i == import.path.segments.len() - 1;

            let def_id = match self.result.per_module[&curr].items.get(name) {
                None => return Ok(()),
                Some(res) => match res.def_id {
                    Some(it) => it,
                    None => return Ok(()),
                },
            };

            if !is_last {
                curr = match def_id.loc(self.db) {
                    DefLoc {
                        kind: DefKind::Module,
                        module_id,
                        source_root_id,
                        ..
                    } => {
                        if source_root_id == self.source_root {
                            module_id
                        } else {
                            // FIXME: across crates resolve
                            return Ok(());
                        }
                    }
                    _ => return Ok(()),
                }
            } else {
                self.update(module_id, |items| {
                    let res = Resolution {
                        def_id: Some(def_id),
                        import: Some(ptr),
                    };
                    items.items.insert(name.clone(), res);
                })
            }
        }
        Ok(())
    }

    fn update(&mut self, module_id: ModuleId, f: impl FnOnce(&mut ModuleScope)) {
        let module_items = self.result.per_module.get_mut(&module_id).unwrap();
        f(module_items)
    }
}

#[cfg(test)]
mod tests;
