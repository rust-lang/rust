//! Name resolution algorithm
use std::sync::Arc;

use rustc_hash::FxHashMap;

use ra_syntax::{
    SmolStr, SyntaxKind::{self, *},
    ast::{self, NameOwner, AstNode}
};

use crate::{
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

#[derive(Debug)]
struct ItemMap {
    per_module: FxHashMap<ModuleId, ModuleItems>,
}

#[derive(Debug, Default)]
struct ModuleItems {
    items: FxHashMap<SmolStr, DefId>,
    import_resolutions: FxHashMap<LocalSyntaxPtr, DefId>,
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
            self.populate_module(
                module_id,
                items,
            )
        }
    }

    fn populate_module(
        &mut self,
        module_id: ModuleId,
        input: &InputModuleItems,
    ) {
        let file_id = module_id.source(&self.module_tree).file_id();

        let mut module_items = ModuleItems::default();

        for item in input.items.iter() {
            if item.kind == MODULE {
                // handle submodules separatelly
                continue;
            }
            let ptr = item.ptr.into_global(file_id);
            let def_loc = DefLoc::Item { ptr };
            let def_id = self.db.id_maps().def_id(def_loc);
            module_items.items.insert(item.name.clone(), def_id);
        }

        for (name, mod_id) in module_id.children(&self.module_tree) {
            let def_loc = DefLoc::Module {
                id: mod_id,
                source_root: self.source_root,
            };
            let def_id = self.db.id_maps().def_id(def_loc);
            module_items.items.insert(name, def_id);
        }

        self.result.per_module.insert(module_id, module_items);
    }
}


