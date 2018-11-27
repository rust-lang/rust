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
    ops::Index,
};

use rustc_hash::FxHashMap;

use ra_syntax::{
    SyntaxNode, SyntaxNodeRef, TextRange,
    SmolStr, SyntaxKind::{self, *},
    ast::{self, ModuleItemOwner, AstNode}
};

use crate::{
    Cancelable, FileId,
    loc2id::{DefId, DefLoc},
    descriptors::{
        Path, PathKind,
        DescriptorDatabase,
        module::{ModuleId, ModuleTree, ModuleSourceNode},
    },
    input::SourceRootId,
    arena::{Arena, Id}
};

/// Identifier of item within a specific file. This is stable over reparses, so
/// it's OK to use it as a salsa key/value.
pub(crate) type FileItemId = Id<SyntaxNode>;

/// Maps item's `SyntaxNode`s to `FileItemId` and back.
#[derive(Debug, PartialEq, Eq, Default)]
pub(crate) struct FileItems {
    arena: Arena<SyntaxNode>,
}

impl FileItems {
    fn alloc(&mut self, item: SyntaxNode) -> FileItemId {
        self.arena.alloc(item)
    }
    fn id_of(&self, item: SyntaxNodeRef) -> FileItemId {
        let (id, _item) = self
            .arena
            .iter()
            .find(|(_id, i)| i.borrowed() == item)
            .unwrap();
        id
    }
}

impl Index<FileItemId> for FileItems {
    type Output = SyntaxNode;
    fn index(&self, idx: FileItemId) -> &SyntaxNode {
        &self.arena[idx]
    }
}

pub(crate) fn file_items(db: &impl DescriptorDatabase, file_id: FileId) -> Arc<FileItems> {
    let source_file = db.file_syntax(file_id);
    let source_file = source_file.borrowed();
    let mut res = FileItems::default();
    source_file
        .syntax()
        .descendants()
        .filter_map(ast::ModuleItem::cast)
        .map(|it| it.syntax().owned())
        .for_each(|it| {
            res.alloc(it);
        });
    Arc::new(res)
}

pub(crate) fn file_item(
    db: &impl DescriptorDatabase,
    file_id: FileId,
    file_item_id: FileItemId,
) -> SyntaxNode {
    db._file_items(file_id)[file_item_id].clone()
}

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

#[derive(Debug, PartialEq, Eq)]
struct ModuleItem {
    id: FileItemId,
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
pub(crate) struct NamedImport {
    file_item_id: FileItemId,
    relative_range: TextRange,
}

impl NamedImport {
    pub(crate) fn range(&self, db: &impl DescriptorDatabase, file_id: FileId) -> TextRange {
        let syntax = db._file_item(file_id, self.file_item_id);
        let offset = syntax.borrowed().range().start();
        self.relative_range + offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ImportKind {
    Glob,
    Named(NamedImport),
}

pub(crate) fn input_module_items(
    db: &impl DescriptorDatabase,
    source_root: SourceRootId,
    module_id: ModuleId,
) -> Cancelable<Arc<InputModuleItems>> {
    let module_tree = db._module_tree(source_root)?;
    let source = module_id.source(&module_tree);
    let file_items = db._file_items(source.file_id());
    let res = match source.resolve(db) {
        ModuleSourceNode::SourceFile(it) => {
            let items = it.borrowed().items();
            InputModuleItems::new(&file_items, items)
        }
        ModuleSourceNode::Module(it) => {
            let items = it
                .borrowed()
                .item_list()
                .into_iter()
                .flat_map(|it| it.items());
            InputModuleItems::new(&file_items, items)
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
    pub(crate) import: Option<NamedImport>,
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
    fn new<'a>(
        file_items: &FileItems,
        items: impl Iterator<Item = ast::ModuleItem<'a>>,
    ) -> InputModuleItems {
        let mut res = InputModuleItems::default();
        for item in items {
            res.add_item(file_items, item);
        }
        res
    }

    fn add_item(&mut self, file_items: &FileItems, item: ast::ModuleItem) -> Option<()> {
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

    fn add_use_item(&mut self, file_items: &FileItems, item: ast::UseItem) {
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
    fn new<'a>(file_items: &FileItems, item: impl ast::NameOwner<'a>) -> Option<ModuleItem> {
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
                file_id,
                id: item.id,
            };
            let def_id = self.db.id_maps().def_id(def_loc);
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
            let def_id = self.db.id_maps().def_id(def_loc);
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
                curr = match self.db.id_maps().def_loc(def_id) {
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
    use crate::{
        AnalysisChange,
        mock_analysis::{MockAnalysis, analysis_and_position},
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

    #[test]
    fn typing_inside_a_function_should_not_invalidate_item_map() {
        let mock_analysis = MockAnalysis::with_files(
            "
            //- /lib.rs
            mod foo;

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

        let file_id = mock_analysis.id_of("/lib.rs");
        let mut host = mock_analysis.analysis_host();

        let source_root = host.analysis().imp.db.file_source_root(file_id);

        {
            let db = host.analysis().imp.db;
            let events = db.log_executed(|| {
                db._item_map(source_root).unwrap();
            });
            assert!(format!("{:?}", events).contains("_item_map"))
        }

        let mut change = AnalysisChange::new();

        change.change_file(
            file_id,
            "
            mod foo;

            use crate::foo::bar::Baz;

            fn foo() -> i32 { 92 }
        "
            .to_string(),
        );

        host.apply_change(change);

        {
            let db = host.analysis().imp.db;
            let events = db.log_executed(|| {
                db._item_map(source_root).unwrap();
            });
            assert!(
                !format!("{:?}", events).contains("_item_map"),
                "{:#?}",
                events
            )
        }
    }
}
