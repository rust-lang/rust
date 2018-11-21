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
    time::Instant,
};

use rustc_hash::FxHashMap;

use ra_syntax::{
    SmolStr, SyntaxKind::{self, *},
    ast::{self, AstNode, ModuleItemOwner}
};

use crate::{
    Cancelable,
    loc2id::{DefId, DefLoc},
    descriptors::{
        Path, PathKind,
        DescriptorDatabase,
        module::{ModuleId, ModuleTree, ModuleSourceNode},
    },
    syntax_ptr::{LocalSyntaxPtr},
    input::SourceRootId,
};

/// Item map is the result of the name resolution. Item map contains, for each
/// module, the set of visible items.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct ItemMap {
    pub(crate) per_module: FxHashMap<ModuleId, ModuleScope>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub(crate) struct ModuleScope {
    pub(crate) items: FxHashMap<SmolStr, Resolution>,
}

/// A set of items and imports declared inside a module, without relation to
/// other modules.
///
/// This stands in-between raw syntax and name resolution and alow us to avoid
/// recomputing name res: if `InputModuleItems` are the same, we can avoid
/// running name resolution.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct InputModuleItems {
    items: Vec<ModuleItem>,
    imports: Vec<Import>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Import {
    path: Path,
    kind: ImportKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ImportKind {
    Glob,
    // TODO: make offset independent
    Named(LocalSyntaxPtr),
}

pub(crate) fn input_module_items(
    db: &impl DescriptorDatabase,
    source_root: SourceRootId,
    module_id: ModuleId,
) -> Cancelable<Arc<InputModuleItems>> {
    let module_tree = db._module_tree(source_root)?;
    let source = module_id.source(&module_tree);
    let res = match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => {
            let items = it.borrowed().items();
            InputModuleItems::new(items)
        }
        ModuleSourceNode::Module(it) => {
            let items = it
                .borrowed()
                .item_list()
                .into_iter()
                .flat_map(|it| it.items());
            InputModuleItems::new(items)
        }
    };
    Ok(Arc::new(res))
}

pub(crate) fn item_map(
    db: &impl DescriptorDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ItemMap>> {
    let start = Instant::now();
    let module_tree = db._module_tree(source_root)?;
    let input = module_tree
        .modules()
        .map(|id| {
            let items = db._input_module_items(source_root, id)?;
            Ok((id, items))
        })
        .collect::<Cancelable<FxHashMap<_, _>>>()?;

    let mut resolver = Resolver {
        db: db,
        input: &input,
        source_root,
        module_tree,
        result: ItemMap::default(),
    };
    resolver.resolve()?;
    let res = resolver.result;
    let elapsed = start.elapsed();
    log::info!("item_map: {:?}", elapsed);
    Ok(Arc::new(res))
}

/// Resolution is basically `DefId` atm, but it should account for stuff like
/// multiple namespaces, ambiguity and errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Resolution {
    /// None for unresolved
    pub(crate) def_id: Option<DefId>,
    /// ident by whitch this is imported into local scope.
    /// TODO: make this offset-independent.
    pub(crate) import_name: Option<LocalSyntaxPtr>,
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

#[derive(Debug, PartialEq, Eq)]
struct ModuleItem {
    ptr: LocalSyntaxPtr,
    name: SmolStr,
    kind: SyntaxKind,
    vis: Vis,
}

#[derive(Debug, PartialEq, Eq)]
enum Vis {
    // Priv,
    Other,
}

impl InputModuleItems {
    fn new<'a>(items: impl Iterator<Item = ast::ModuleItem<'a>>) -> InputModuleItems {
        let mut res = InputModuleItems::default();
        for item in items {
            res.add_item(item);
        }
        res
    }

    fn add_item(&mut self, item: ast::ModuleItem) -> Option<()> {
        match item {
            ast::ModuleItem::StructDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::EnumDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::FnDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::TraitDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::TypeDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::ImplItem(_) => {
                // impls don't define items
            }
            ast::ModuleItem::UseItem(it) => self.add_use_item(it),
            ast::ModuleItem::ExternCrateItem(_) => {
                // TODO
            }
            ast::ModuleItem::ConstDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::StaticDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::Module(it) => self.items.push(ModuleItem::new(it)?),
        }
        Some(())
    }

    fn add_use_item(&mut self, item: ast::UseItem) {
        Path::expand_use_item(item, |path, ptr| {
            let kind = match ptr {
                None => ImportKind::Glob,
                Some(ptr) => ImportKind::Named(ptr),
            };
            self.imports.push(Import { kind, path })
        })
    }
}

impl ModuleItem {
    fn new<'a>(item: impl ast::NameOwner<'a>) -> Option<ModuleItem> {
        let name = item.name()?.text();
        let ptr = LocalSyntaxPtr::new(item.syntax());
        let kind = item.syntax().kind();
        let vis = Vis::Other;
        let res = ModuleItem {
            ptr,
            name,
            kind,
            vis,
        };
        Some(res)
    }
}

struct Resolver<'a, DB> {
    db: &'a DB,
    input: &'a FxHashMap<ModuleId, Arc<InputModuleItems>>,
    source_root: SourceRootId,
    module_tree: Arc<ModuleTree>,
    result: ItemMap,
}

impl<'a, DB> Resolver<'a, DB>
where
    DB: DescriptorDatabase,
{
    fn resolve(&mut self) -> Cancelable<()> {
        for (&module_id, items) in self.input.iter() {
            self.populate_module(module_id, items)
        }

        for &module_id in self.input.keys() {
            crate::db::check_canceled(self.db)?;
            self.resolve_imports(module_id);
        }
        Ok(())
    }

    fn populate_module(&mut self, module_id: ModuleId, input: &InputModuleItems) {
        let file_id = module_id.source(&self.module_tree).file_id();

        let mut module_items = ModuleScope::default();

        for import in input.imports.iter() {
            if let Some(name) = import.path.segments.iter().last() {
                if let ImportKind::Named(ptr) = import.kind {
                    module_items.items.insert(
                        name.clone(),
                        Resolution {
                            def_id: None,
                            import_name: Some(ptr),
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
            let ptr = item.ptr.into_global(file_id);
            let def_loc = DefLoc::Item { ptr };
            let def_id = self.db.id_maps().def_id(def_loc);
            let resolution = Resolution {
                def_id: Some(def_id),
                import_name: None,
            };
            module_items.items.insert(item.name.clone(), resolution);
        }

        for (name, mod_id) in module_id.children(&self.module_tree) {
            let def_loc = DefLoc::Module {
                id: mod_id,
                source_root: self.source_root,
            };
            let def_id = self.db.id_maps().def_id(def_loc);
            let resolution = Resolution {
                def_id: Some(def_id),
                import_name: None,
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
            PathKind::Abs => return,
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
                curr = match self.db.id_maps().def_loc(def_id) {
                    DefLoc::Module { id, .. } => id,
                    _ => return,
                }
            } else {
                self.update(module_id, |items| {
                    let res = Resolution {
                        def_id: Some(def_id),
                        import_name: Some(ptr),
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
    use crate::{
        mock_analysis::analysis_and_position,
        descriptors::{DescriptorDatabase, module::ModuleDescriptor},
        input::FilesDatabase,
};
    use super::*;

    fn item_map(fixture: &str) -> (Arc<ItemMap>, ModuleId) {
        let (analysis, pos) = analysis_and_position(fixture);
        let db = analysis.imp.db;
        let source_root = db.file_source_root(pos.file_id);
        let descr = ModuleDescriptor::guess_from_position(&*db, pos)
            .unwrap()
            .unwrap();
        let module_id = descr.module_id;
        (db._item_map(source_root).unwrap(), module_id)
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
}
