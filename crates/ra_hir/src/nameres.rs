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
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use ra_syntax::{
    TextRange,
    SyntaxKind::{self, *},
    ast::{self, AstNode}
};
use ra_db::{SourceRootId, Cancelable, FileId};

use crate::{
    HirFileId,
    DefId, DefLoc, DefKind,
    SourceItemId, SourceFileItemId, SourceFileItems,
    Path, PathKind,
    HirDatabase, Crate,
    Name, AsName,
    module_tree::{ModuleId, ModuleTree},
};

/// Item map is the result of the name resolution. Item map contains, for each
/// module, the set of visible items.
// FIXME: currenty we compute item map per source-root. We should do it per crate instead.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ItemMap {
    pub per_module: FxHashMap<ModuleId, ModuleScope>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ModuleScope {
    items: FxHashMap<Name, Resolution>,
}

impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        self.items.iter()
    }
    pub fn get(&self, name: &Name) -> Option<&Resolution> {
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
    pub(crate) items: Vec<ModuleItem>,
    imports: Vec<Import>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ModuleItem {
    pub(crate) id: SourceItemId,
    pub(crate) name: Name,
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
    // FIXME: this is only here for one use-case in completion. Seems like a
    // pretty gross special case.
    pub fn range(&self, db: &impl HirDatabase, file_id: FileId) -> TextRange {
        let source_item_id = SourceItemId {
            file_id: file_id.into(),
            item_id: Some(self.file_item_id),
        };
        let syntax = db.file_item(source_item_id);
        let offset = syntax.range().start();
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
    pub def_id: PerNs<DefId>,
    /// ident by whitch this is imported into local scope.
    pub import: Option<NamedImport>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Namespace {
    Types,
    Values,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PerNs<T> {
    pub types: Option<T>,
    pub values: Option<T>,
}

impl<T> PerNs<T> {
    pub fn none() -> PerNs<T> {
        PerNs {
            types: None,
            values: None,
        }
    }

    pub fn values(t: T) -> PerNs<T> {
        PerNs {
            types: None,
            values: Some(t),
        }
    }

    pub fn types(t: T) -> PerNs<T> {
        PerNs {
            types: Some(t),
            values: None,
        }
    }

    pub fn both(types: T, values: T) -> PerNs<T> {
        PerNs {
            types: Some(types),
            values: Some(values),
        }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none()
    }

    pub fn take(self, namespace: Namespace) -> Option<T> {
        match namespace {
            Namespace::Types => self.types,
            Namespace::Values => self.values,
        }
    }

    pub fn take_types(self) -> Option<T> {
        self.take(Namespace::Types)
    }

    pub fn take_values(self) -> Option<T> {
        self.take(Namespace::Values)
    }

    pub fn get(&self, namespace: Namespace) -> Option<&T> {
        self.as_ref().take(namespace)
    }

    pub fn as_ref(&self) -> PerNs<&T> {
        PerNs {
            types: self.types.as_ref(),
            values: self.values.as_ref(),
        }
    }

    pub fn and_then<U>(self, f: impl Fn(T) -> Option<U>) -> PerNs<U> {
        PerNs {
            types: self.types.and_then(&f),
            values: self.values.and_then(&f),
        }
    }

    pub fn map<U>(self, f: impl Fn(T) -> U) -> PerNs<U> {
        PerNs {
            types: self.types.map(&f),
            values: self.values.map(&f),
        }
    }
}

impl InputModuleItems {
    pub(crate) fn add_item(
        &mut self,
        file_id: HirFileId,
        file_items: &SourceFileItems,
        item: &ast::ModuleItem,
    ) -> Option<()> {
        match item.kind() {
            ast::ModuleItemKind::StructDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::EnumDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::FnDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::TraitDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::TypeDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::ImplBlock(_) => {
                // impls don't define items
            }
            ast::ModuleItemKind::UseItem(it) => self.add_use_item(file_items, it),
            ast::ModuleItemKind::ExternCrateItem(_) => {
                // TODO
            }
            ast::ModuleItemKind::ConstDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::StaticDef(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
            ast::ModuleItemKind::Module(it) => {
                self.items.push(ModuleItem::new(file_id, file_items, it)?)
            }
        }
        Some(())
    }

    fn add_use_item(&mut self, file_items: &SourceFileItems, item: &ast::UseItem) {
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
    fn new(
        file_id: HirFileId,
        file_items: &SourceFileItems,
        item: &impl ast::NameOwner,
    ) -> Option<ModuleItem> {
        let name = item.name()?.as_name();
        let kind = item.syntax().kind();
        let vis = Vis::Other;
        let item_id = Some(file_items.id_of_unchecked(item.syntax()));
        let id = SourceItemId { file_id, item_id };
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
    processed_imports: FxHashSet<(ModuleId, usize)>,
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
            processed_imports: FxHashSet::default(),
            result: ItemMap::default(),
        }
    }

    pub(crate) fn resolve(mut self) -> Cancelable<ItemMap> {
        for (&module_id, items) in self.input.iter() {
            self.populate_module(module_id, Arc::clone(items))?;
        }

        loop {
            let processed_imports_count = self.processed_imports.len();
            for &module_id in self.input.keys() {
                self.db.check_canceled()?;
                self.resolve_imports(module_id)?;
            }
            if processed_imports_count == self.processed_imports.len() {
                // no new imports resolved
                break;
            }
        }
        Ok(self.result)
    }

    fn populate_module(
        &mut self,
        module_id: ModuleId,
        input: Arc<InputModuleItems>,
    ) -> Cancelable<()> {
        let mut module_items = ModuleScope::default();

        // Populate extern crates prelude
        {
            let root_id = module_id.crate_root(&self.module_tree);
            let file_id = root_id.source(&self.module_tree).file_id;
            let crate_graph = self.db.crate_graph();
            if let Some(crate_id) = crate_graph.crate_id_for_crate_root(file_id.as_original_file())
            {
                let krate = Crate::new(crate_id);
                for dep in krate.dependencies(self.db)? {
                    if let Some(module) = dep.krate.root_module(self.db)? {
                        let def_id = module.def_id;
                        self.add_module_item(
                            &mut module_items,
                            dep.name.clone(),
                            PerNs::types(def_id),
                        );
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
                            def_id: PerNs::none(),
                            import: Some(import),
                        },
                    );
                }
            }
        }
        // Populate explicitly declared items, except modules
        for item in input.items.iter() {
            if item.kind == MODULE {
                continue;
            }
            // depending on the item kind, the location can define something in
            // the values namespace, the types namespace, or both
            let kind = DefKind::for_syntax_kind(item.kind);
            let def_id = kind.map(|k| {
                let def_loc = DefLoc {
                    kind: k,
                    source_root_id: self.source_root,
                    module_id,
                    source_item_id: item.id,
                };
                def_loc.id(self.db)
            });
            let resolution = Resolution {
                def_id,
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
                source_item_id: module_id.source(&self.module_tree),
            };
            let def_id = def_loc.id(self.db);
            self.add_module_item(&mut module_items, name, PerNs::types(def_id));
        }

        self.result.per_module.insert(module_id, module_items);
        Ok(())
    }

    fn add_module_item(&self, module_items: &mut ModuleScope, name: Name, def_id: PerNs<DefId>) {
        let resolution = Resolution {
            def_id,
            import: None,
        };
        module_items.items.insert(name, resolution);
    }

    fn resolve_imports(&mut self, module_id: ModuleId) -> Cancelable<()> {
        for (i, import) in self.input[&module_id].imports.iter().enumerate() {
            if self.processed_imports.contains(&(module_id, i)) {
                // already done
                continue;
            }
            if self.resolve_import(module_id, import)? {
                log::debug!("import {:?} resolved (or definite error)", import);
                self.processed_imports.insert((module_id, i));
            }
        }
        Ok(())
    }

    fn resolve_import(&mut self, module_id: ModuleId, import: &Import) -> Cancelable<bool> {
        log::debug!("resolving import: {:?}", import);
        let ptr = match import.kind {
            ImportKind::Glob => return Ok(false),
            ImportKind::Named(ptr) => ptr,
        };

        let mut curr: ModuleId = match import.path.kind {
            PathKind::Plain | PathKind::Self_ => module_id,
            PathKind::Super => {
                match module_id.parent(&self.module_tree) {
                    Some(it) => it,
                    None => {
                        // TODO: error
                        log::debug!("super path in root module");
                        return Ok(true); // this can't suddenly resolve if we just resolve some other imports
                    }
                }
            }
            PathKind::Crate => module_id.crate_root(&self.module_tree),
        };

        for (i, name) in import.path.segments.iter().enumerate() {
            let is_last = i == import.path.segments.len() - 1;

            let def_id = match self.result.per_module[&curr].items.get(name) {
                Some(res) if !res.def_id.is_none() => res.def_id,
                _ => {
                    log::debug!("path segment {:?} not found", name);
                    return Ok(false);
                }
            };

            if !is_last {
                let type_def_id = if let Some(d) = def_id.take(Namespace::Types) {
                    d
                } else {
                    log::debug!(
                        "path segment {:?} resolved to value only, but is not last",
                        name
                    );
                    return Ok(false);
                };
                curr = match type_def_id.loc(self.db) {
                    DefLoc {
                        kind: DefKind::Module,
                        module_id: target_module_id,
                        source_root_id,
                        ..
                    } => {
                        if source_root_id == self.source_root {
                            target_module_id
                        } else {
                            let module = crate::code_model_api::Module::new(type_def_id);
                            let path = Path {
                                segments: import.path.segments[i + 1..].iter().cloned().collect(),
                                kind: PathKind::Crate,
                            };
                            log::debug!("resolving {:?} in other source root", path);
                            let def_id = module.resolve_path(self.db, &path)?;
                            if !def_id.is_none() {
                                let name = path.segments.last().unwrap();
                                self.update(module_id, |items| {
                                    let res = Resolution {
                                        def_id,
                                        import: Some(ptr),
                                    };
                                    items.items.insert(name.clone(), res);
                                });
                                log::debug!(
                                    "resolved import {:?} ({:?}) cross-source root to {:?}",
                                    name,
                                    import,
                                    def_id.map(|did| did.loc(self.db))
                                );
                                return Ok(true);
                            } else {
                                log::debug!("rest of path did not resolve in other source root");
                                return Ok(true);
                            }
                        }
                    }
                    _ => {
                        log::debug!(
                            "path segment {:?} resolved to non-module {:?}, but is not last",
                            name,
                            type_def_id.loc(self.db)
                        );
                        return Ok(true); // this resolved to a non-module, so the path won't ever resolve
                    }
                }
            } else {
                log::debug!(
                    "resolved import {:?} ({:?}) within source root to {:?}",
                    name,
                    import,
                    def_id.map(|did| did.loc(self.db))
                );
                self.update(module_id, |items| {
                    let res = Resolution {
                        def_id,
                        import: Some(ptr),
                    };
                    items.items.insert(name.clone(), res);
                })
            }
        }
        Ok(true)
    }

    fn update(&mut self, module_id: ModuleId, f: impl FnOnce(&mut ModuleScope)) {
        let module_items = self.result.per_module.get_mut(&module_id).unwrap();
        f(module_items)
    }
}

#[cfg(test)]
mod tests;
