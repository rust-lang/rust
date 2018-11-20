//! Name resolution algorithm
use std::sync::Arc;

use rustc_hash::FxHashMap;

use ra_syntax::{
    SmolStr, SyntaxKind::{self, *},
    ast::{self, NameOwner, AstNode}
};

use crate::{
    Cancelable,
    loc2id::{DefId, DefLoc},
    descriptors::{
        DescriptorDatabase,
        module::{ModuleId, ModuleTree},
    },
    syntax_ptr::{LocalSyntaxPtr, SyntaxPtr},
    input::SourceRootId,
};

/// A set of items and imports declared inside a module, without relation to
/// other modules.
///
/// This stands in-between raw syntax and name resolution and alow us to avoid
/// recomputing name res: if `InputModuleItems` are the same, we can avoid
/// running name resolution.
#[derive(Debug, Default)]
struct InputModuleItems {
    items: Vec<ModuleItem>,
    glob_imports: Vec<Path>,
    imports: Vec<Path>,
}

#[derive(Debug, Clone)]
struct Path {
    kind: PathKind,
    segments: Vec<(LocalSyntaxPtr, SmolStr)>,
}

#[derive(Debug, Clone, Copy)]
enum PathKind {
    Abs,
    Self_,
    Super,
    Crate,
}

pub(crate) fn item_map(
    db: &impl DescriptorDatabase,
    source_root: SourceRootId,
) -> Cancelable<Arc<ItemMap>> {
    unimplemented!()
}

/// Item map is the result of the name resolution. Item map contains, for each
/// module, the set of visible items.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ItemMap {
    per_module: FxHashMap<ModuleId, ModuleItems>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ModuleItems {
    items: FxHashMap<SmolStr, Resolution>,
    import_resolutions: FxHashMap<LocalSyntaxPtr, DefId>,
}

/// Resolution is basically `DefId` atm, but it should account for stuff like
/// multiple namespaces, ambiguity and errors.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Resolution {
    /// None for unresolved
    def_id: Option<DefId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Namespace {
    Types,
    Values,
}

#[derive(Debug)]
struct PerNs<T> {
    types: Option<T>,
    values: Option<T>,
}

#[derive(Debug)]
struct ModuleItem {
    ptr: LocalSyntaxPtr,
    name: SmolStr,
    kind: SyntaxKind,
    vis: Vis,
}

#[derive(Debug)]
enum Vis {
    Priv,
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
            ast::ModuleItem::ImplItem(it) => {
                // impls don't define items
            }
            ast::ModuleItem::UseItem(it) => self.add_use_item(it),
            ast::ModuleItem::ExternCrateItem(it) => (),
            ast::ModuleItem::ConstDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::StaticDef(it) => self.items.push(ModuleItem::new(it)?),
            ast::ModuleItem::Module(it) => self.items.push(ModuleItem::new(it)?),
        }
        Some(())
    }

    fn add_use_item(&mut self, item: ast::UseItem) {
        if let Some(tree) = item.use_tree() {
            self.add_use_tree(None, tree);
        }
    }

    fn add_use_tree(&mut self, prefix: Option<Path>, tree: ast::UseTree) {
        if let Some(use_tree_list) = tree.use_tree_list() {
            let prefix = match tree.path() {
                None => prefix,
                Some(path) => match convert_path(prefix, path) {
                    Some(it) => Some(it),
                    None => return, // TODO: report errors somewhere
                },
            };
            for tree in use_tree_list.use_trees() {
                self.add_use_tree(prefix.clone(), tree);
            }
        } else {
            if let Some(path) = tree.path() {
                if let Some(path) = convert_path(prefix, path) {
                    if tree.has_star() {
                        &mut self.glob_imports
                    } else {
                        &mut self.imports
                    }
                    .push(path);
                }
            }
        }
    }
}

fn convert_path(prefix: Option<Path>, path: ast::Path) -> Option<Path> {
    let prefix = if let Some(qual) = path.qualifier() {
        Some(convert_path(prefix, qual)?)
    } else {
        None
    };
    let segment = path.segment()?;
    let res = match segment.kind()? {
        ast::PathSegmentKind::Name(name) => {
            let mut res = prefix.unwrap_or_else(|| Path {
                kind: PathKind::Abs,
                segments: Vec::with_capacity(1),
            });
            let ptr = LocalSyntaxPtr::new(name.syntax());
            res.segments.push((ptr, name.text()));
            res
        }
        ast::PathSegmentKind::CrateKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Crate,
                segments: Vec::new(),
            }
        }
        ast::PathSegmentKind::SelfKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Self_,
                segments: Vec::new(),
            }
        }
        ast::PathSegmentKind::SuperKw => {
            if prefix.is_some() {
                return None;
            }
            Path {
                kind: PathKind::Super,
                segments: Vec::new(),
            }
        }
    };
    Some(res)
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
    input: &'a FxHashMap<ModuleId, InputModuleItems>,
    source_root: SourceRootId,
    module_tree: Arc<ModuleTree>,
    result: ItemMap,
}

impl<'a, DB> Resolver<'a, DB>
where
    DB: DescriptorDatabase,
{
    fn resolve(&mut self) {
        for (&module_id, items) in self.input.iter() {
            self.populate_module(module_id, items)
        }

        for &module_id in self.input.keys() {
            self.resolve_imports(module_id);
        }
    }

    fn populate_module(&mut self, module_id: ModuleId, input: &InputModuleItems) {
        let file_id = module_id.source(&self.module_tree).file_id();

        let mut module_items = ModuleItems::default();

        for import in input.imports.iter() {
            if let Some((_, name)) = import.segments.last() {
                module_items
                    .items
                    .insert(name.clone(), Resolution { def_id: None });
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

    fn resolve_import(&mut self, module_id: ModuleId, import: &Path) {
        let mut curr = match import.kind {
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

        for (i, (ptr, name)) in import.segments.iter().enumerate() {
            let is_last = i == import.segments.len() - 1;

            let def_id = match self.result.per_module[&curr].items.get(name) {
                None => return,
                Some(res) => match res.def_id {
                    Some(it) => it,
                    None => return,
                },
            };

            self.update(module_id, |items| {
                items.import_resolutions.insert(*ptr, def_id);
            });

            if !is_last {
                curr = match self.db.id_maps().def_loc(def_id) {
                    DefLoc::Module { id, .. } => id,
                    _ => return,
                }
            } else {
                self.update(module_id, |items| {
                    let res = Resolution {
                        def_id: Some(def_id),
                    };
                    items.items.insert(name.clone(), res);
                })
            }
        }
    }

    fn update(&mut self, module_id: ModuleId, f: impl FnOnce(&mut ModuleItems)) {
        let module_items = self.result.per_module.get_mut(&module_id).unwrap();
        f(module_items)
    }
}
