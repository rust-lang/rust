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
    DefId, DefLoc,
    SourceItemId, SourceFileItemId, SourceFileItems,
    Path, PathKind,
    HirDatabase,
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
            item_id: self.file_item_id,
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
        let file_item_id = file_items.id_of(item.syntax());
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
        let id = file_items.id_of(item.syntax());
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
    pub(crate) db: &'a DB,
    pub(crate) input: &'a FxHashMap<ModuleId, Arc<InputModuleItems>>,
    pub(crate) source_root: SourceRootId,
    pub(crate) module_tree: Arc<ModuleTree>,
    pub(crate) result: ItemMap,
}

impl<'a, DB> Resolver<'a, DB>
where
    DB: HirDatabase,
{
    pub(crate) fn resolve(mut self) -> Cancelable<ItemMap> {
        for (&module_id, items) in self.input.iter() {
            self.populate_module(module_id, items)
        }

        for &module_id in self.input.keys() {
            self.db.check_canceled()?;
            self.resolve_imports(module_id);
        }
        Ok(self.result)
    }

    fn populate_module(&mut self, module_id: ModuleId, input: &InputModuleItems) {
        let file_id = module_id.source(&self.module_tree).file_id();

        let mut module_items = ModuleScope::default();

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

        for item in input.items.iter() {
            if item.kind == MODULE {
                // handle submodules separatelly
                continue;
            }
            let def_loc = DefLoc::Item {
                source_item_id: SourceItemId {
                    file_id,
                    item_id: item.id,
                },
            };
            let def_id = def_loc.id(self.db);
            let resolution = Resolution {
                def_id: Some(def_id),
                import: None,
            };
            module_items.items.insert(item.name.clone(), resolution);
        }

        for (name, mod_id) in module_id.children(&self.module_tree) {
            let def_loc = DefLoc::Module {
                id: mod_id,
                source_root: self.source_root,
            };
            let def_id = def_loc.id(self.db);
            let resolution = Resolution {
                def_id: Some(def_id),
                import: None,
            };
            module_items.items.insert(name, resolution);
        }

        self.result.per_module.insert(module_id, module_items);
    }

    fn resolve_imports(&mut self, module_id: ModuleId) {
        for import in self.input[&module_id].imports.iter() {
            self.resolve_import(module_id, import);
        }
    }

    fn resolve_import(&mut self, module_id: ModuleId, import: &Import) {
        let ptr = match import.kind {
            ImportKind::Glob => return,
            ImportKind::Named(ptr) => ptr,
        };

        let mut curr = match import.path.kind {
            // TODO: handle extern crates
            PathKind::Plain => return,
            PathKind::Self_ => module_id,
            PathKind::Super => {
                match module_id.parent(&self.module_tree) {
                    Some(it) => it,
                    // TODO: error
                    None => return,
                }
            }
            PathKind::Crate => module_id.crate_root(&self.module_tree),
        };

        for (i, name) in import.path.segments.iter().enumerate() {
            let is_last = i == import.path.segments.len() - 1;

            let def_id = match self.result.per_module[&curr].items.get(name) {
                None => return,
                Some(res) => match res.def_id {
                    Some(it) => it,
                    None => return,
                },
            };

            if !is_last {
                curr = match def_id.loc(self.db) {
                    DefLoc::Module { id, .. } => id,
                    _ => return,
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
    }

    fn update(&mut self, module_id: ModuleId, f: impl FnOnce(&mut ModuleScope)) {
        let module_items = self.result.per_module.get_mut(&module_id).unwrap();
        f(module_items)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use salsa::Database;
    use ra_db::FilesDatabase;
    use ra_syntax::SmolStr;

    use crate::{
        self as hir,
        db::HirDatabase,
        mock::MockDatabase,
};

    fn item_map(fixture: &str) -> (Arc<hir::ItemMap>, hir::ModuleId) {
        let (db, pos) = MockDatabase::with_position(fixture);
        let source_root = db.file_source_root(pos.file_id);
        let module = hir::Module::guess_from_position(&db, pos).unwrap().unwrap();
        let module_id = module.module_id;
        (db.item_map(source_root).unwrap(), module_id)
    }

    #[test]
    fn test_item_map() {
        let (item_map, module_id) = item_map(
            "
            //- /lib.rs
            mod foo;

            use crate::foo::bar::Baz;
            <|>

            //- /foo/mod.rs
            pub mod bar;

            //- /foo/bar.rs
            pub struct Baz;
        ",
        );
        let name = SmolStr::from("Baz");
        let resolution = &item_map.per_module[&module_id].items[&name];
        assert!(resolution.def_id.is_some());
    }

    #[test]
    fn typing_inside_a_function_should_not_invalidate_item_map() {
        let (mut db, pos) = MockDatabase::with_position(
            "
            //- /lib.rs
            mod foo;<|>

            use crate::foo::bar::Baz;

            fn foo() -> i32 {
                1 + 1
            }
            //- /foo/mod.rs
            pub mod bar;

            //- /foo/bar.rs
            pub struct Baz;
        ",
        );
        let source_root = db.file_source_root(pos.file_id);
        {
            let events = db.log_executed(|| {
                db.item_map(source_root).unwrap();
            });
            assert!(format!("{:?}", events).contains("item_map"))
        }

        let new_text = "
            mod foo;

            use crate::foo::bar::Baz;

            fn foo() -> i32 { 92 }
        "
        .to_string();

        db.query_mut(ra_db::FileTextQuery)
            .set(pos.file_id, Arc::new(new_text));

        {
            let events = db.log_executed(|| {
                db.item_map(source_root).unwrap();
            });
            assert!(
                !format!("{:?}", events).contains("_item_map"),
                "{:#?}",
                events
            )
        }
    }
}
