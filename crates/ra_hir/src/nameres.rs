//! Name resolution algorithm. The end result of the algorithm is an `ItemMap`:
//! a map which maps each module to its scope: the set of items visible in the
//! module. That is, we only resolve imports here, name resolution of item
//! bodies will be done in a separate step.
//!
//! Like Rustc, we use an interactive per-crate algorithm: we start with scopes
//! containing only directly defined items, and then iteratively resolve
//! imports.
//!
//! To make this work nicely in the IDE scenario, we place `InputModuleItems`
//! in between raw syntax and name resolution. `InputModuleItems` are computed
//! using only the module's syntax, and it is all directly defined items plus
//! imports. The plan is to make `InputModuleItems` independent of local
//! modifications (that is, typing inside a function should not change IMIs),
//! so that the results of name resolution can be preserved unless the module
//! structure itself is modified.
pub(crate) mod lower;
mod crate_def_map;

use std::{time, sync::Arc};

use rustc_hash::{FxHashMap, FxHashSet};

use ra_arena::map::ArenaMap;
use ra_db::Edition;
use test_utils::tested_by;

use crate::{
    Module, ModuleDef,
    Path, PathKind, PersistentHirDatabase,
    Crate, Name,
    module_tree::{ModuleId, ModuleTree},
    nameres::lower::{ImportId, LoweredModule, ImportData},
};

/// `ItemMap` is the result of module name resolution. It contains, for each
/// module, the set of visible items.
#[derive(Debug, PartialEq, Eq)]
pub struct ItemMap {
    edition: Edition,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    pub(crate) prelude: Option<Module>,
    pub(crate) extern_prelude: FxHashMap<Name, ModuleDef>,
    per_module: ArenaMap<ModuleId, ModuleScope>,
}

impl std::ops::Index<ModuleId> for ItemMap {
    type Output = ModuleScope;
    fn index(&self, id: ModuleId) -> &ModuleScope {
        &self.per_module[id]
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ModuleScope {
    pub(crate) items: FxHashMap<Name, Resolution>,
}

impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        self.items.iter()
    }
    pub fn get(&self, name: &Name) -> Option<&Resolution> {
        self.items.get(name)
    }
}

/// `Resolution` is basically `DefId` atm, but it should account for stuff like
/// multiple namespaces, ambiguity and errors.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Resolution {
    /// None for unresolved
    pub def: PerNs<ModuleDef>,
    /// ident by which this is imported into local scope.
    pub import: Option<ImportId>,
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

impl<T> Default for PerNs<T> {
    fn default() -> Self {
        PerNs { types: None, values: None }
    }
}

impl<T> PerNs<T> {
    pub fn none() -> PerNs<T> {
        PerNs { types: None, values: None }
    }

    pub fn values(t: T) -> PerNs<T> {
        PerNs { types: None, values: Some(t) }
    }

    pub fn types(t: T) -> PerNs<T> {
        PerNs { types: Some(t), values: None }
    }

    pub fn both(types: T, values: T) -> PerNs<T> {
        PerNs { types: Some(types), values: Some(values) }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none()
    }

    pub fn is_both(&self) -> bool {
        self.types.is_some() && self.values.is_some()
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
        PerNs { types: self.types.as_ref(), values: self.values.as_ref() }
    }

    pub fn or(self, other: PerNs<T>) -> PerNs<T> {
        PerNs { types: self.types.or(other.types), values: self.values.or(other.values) }
    }

    pub fn and_then<U>(self, f: impl Fn(T) -> Option<U>) -> PerNs<U> {
        PerNs { types: self.types.and_then(&f), values: self.values.and_then(&f) }
    }

    pub fn map<U>(self, f: impl Fn(T) -> U) -> PerNs<U> {
        PerNs { types: self.types.map(&f), values: self.values.map(&f) }
    }
}

struct Resolver<'a, DB> {
    db: &'a DB,
    input: &'a FxHashMap<ModuleId, Arc<LoweredModule>>,
    krate: Crate,
    module_tree: Arc<ModuleTree>,
    processed_imports: FxHashSet<(ModuleId, ImportId)>,
    /// If module `a` has `use b::*`, then this contains the mapping b -> a (and the import)
    glob_imports: FxHashMap<ModuleId, Vec<(ModuleId, ImportId)>>,
    result: ItemMap,
}

impl<'a, DB> Resolver<'a, DB>
where
    DB: PersistentHirDatabase,
{
    fn new(
        db: &'a DB,
        input: &'a FxHashMap<ModuleId, Arc<LoweredModule>>,
        krate: Crate,
    ) -> Resolver<'a, DB> {
        let module_tree = db.module_tree(krate);
        Resolver {
            db,
            input,
            krate,
            module_tree,
            processed_imports: FxHashSet::default(),
            glob_imports: FxHashMap::default(),
            result: ItemMap {
                edition: krate.edition(db),
                prelude: None,
                extern_prelude: FxHashMap::default(),
                per_module: ArenaMap::default(),
            },
        }
    }

    pub(crate) fn resolve(mut self) -> ItemMap {
        self.populate_extern_prelude();
        for (&module_id, items) in self.input.iter() {
            self.populate_module(module_id, Arc::clone(items));
        }

        let mut iter = 0;
        loop {
            iter += 1;
            if iter > 1000 {
                panic!("failed to reach fixedpoint after 1000 iters")
            }
            let processed_imports_count = self.processed_imports.len();
            for &module_id in self.input.keys() {
                self.db.check_canceled();
                self.resolve_imports(module_id);
            }
            if processed_imports_count == self.processed_imports.len() {
                // no new imports resolved
                break;
            }
        }
        self.result
    }

    fn populate_extern_prelude(&mut self) {
        for dep in self.krate.dependencies(self.db) {
            log::debug!("crate dep {:?} -> {:?}", dep.name, dep.krate);
            if let Some(module) = dep.krate.root_module(self.db) {
                self.result.extern_prelude.insert(dep.name.clone(), module.into());
            }
            // look for the prelude
            if self.result.prelude.is_none() {
                let item_map = self.db.item_map(dep.krate);
                if item_map.prelude.is_some() {
                    self.result.prelude = item_map.prelude;
                }
            }
        }
    }

    fn populate_module(&mut self, module_id: ModuleId, input: Arc<LoweredModule>) {
        let mut module_items = ModuleScope::default();
        for (import_id, import_data) in input.imports.iter() {
            if let Some(last_segment) = import_data.path.segments.iter().last() {
                if !import_data.is_glob {
                    let name =
                        import_data.alias.clone().unwrap_or_else(|| last_segment.name.clone());
                    module_items
                        .items
                        .insert(name, Resolution { def: PerNs::none(), import: Some(import_id) });
                }
            }
        }
        // Populate explicitly declared items, except modules
        for (name, &def) in input.declarations.iter() {
            let resolution = Resolution { def, import: None };
            module_items.items.insert(name.clone(), resolution);
        }

        // Populate modules
        for (name, module_id) in module_id.children(&self.module_tree) {
            let module = Module { module_id, krate: self.krate };
            self.add_module_item(&mut module_items, name, PerNs::types(module.into()));
        }

        self.result.per_module.insert(module_id, module_items);
    }

    fn add_module_item(&self, module_items: &mut ModuleScope, name: Name, def: PerNs<ModuleDef>) {
        let resolution = Resolution { def, import: None };
        module_items.items.insert(name, resolution);
    }

    fn resolve_imports(&mut self, module_id: ModuleId) {
        for (import_id, import_data) in self.input[&module_id].imports.iter() {
            if self.processed_imports.contains(&(module_id, import_id)) {
                // already done
                continue;
            }
            if self.resolve_import(module_id, import_id, import_data) == ReachedFixedPoint::Yes {
                log::debug!("import {:?} resolved (or definite error)", import_id);
                self.processed_imports.insert((module_id, import_id));
            }
        }
    }

    fn resolve_import(
        &mut self,
        module_id: ModuleId,
        import_id: ImportId,
        import: &ImportData,
    ) -> ReachedFixedPoint {
        log::debug!("resolving import: {:?} ({:?})", import, self.result.edition);
        let original_module = Module { krate: self.krate, module_id };

        let (def, reached_fixedpoint) = if import.is_extern_crate {
            let res = self.result.resolve_name_in_extern_prelude(
                &import
                    .path
                    .as_ident()
                    .expect("extern crate should have been desugared to one-element path"),
            );
            (res, if res.is_none() { ReachedFixedPoint::No } else { ReachedFixedPoint::Yes })
        } else {
            let res = self.result.resolve_path_fp(
                self.db,
                ResolveMode::Import,
                original_module,
                &import.path,
            );

            (res.resolved_def, res.reached_fixedpoint)
        };

        if reached_fixedpoint != ReachedFixedPoint::Yes {
            return reached_fixedpoint;
        }

        if import.is_glob {
            log::debug!("glob import: {:?}", import);
            match def.take_types() {
                Some(ModuleDef::Module(m)) => {
                    if import.is_prelude {
                        tested_by!(std_prelude);
                        self.result.prelude = Some(m);
                    } else if m.krate != self.krate {
                        tested_by!(glob_across_crates);
                        // glob import from other crate => we can just import everything once
                        let item_map = self.db.item_map(m.krate);
                        let scope = &item_map[m.module_id];
                        let items = scope
                            .items
                            .iter()
                            .map(|(name, res)| (name.clone(), res.clone()))
                            .collect::<Vec<_>>();
                        self.update(module_id, Some(import_id), &items);
                    } else {
                        // glob import from same crate => we do an initial
                        // import, and then need to propagate any further
                        // additions
                        let scope = &self.result[m.module_id];
                        let items = scope
                            .items
                            .iter()
                            .map(|(name, res)| (name.clone(), res.clone()))
                            .collect::<Vec<_>>();
                        self.update(module_id, Some(import_id), &items);
                        // record the glob import in case we add further items
                        self.glob_imports
                            .entry(m.module_id)
                            .or_default()
                            .push((module_id, import_id));
                    }
                }
                Some(ModuleDef::Enum(e)) => {
                    tested_by!(glob_enum);
                    // glob import from enum => just import all the variants
                    let variants = e.variants(self.db);
                    let resolutions = variants
                        .into_iter()
                        .filter_map(|variant| {
                            let res = Resolution {
                                def: PerNs::both(variant.into(), variant.into()),
                                import: Some(import_id),
                            };
                            let name = variant.name(self.db)?;
                            Some((name, res))
                        })
                        .collect::<Vec<_>>();
                    self.update(module_id, Some(import_id), &resolutions);
                }
                Some(d) => {
                    log::debug!("glob import {:?} from non-module/enum {:?}", import, d);
                }
                None => {
                    log::debug!("glob import {:?} didn't resolve as type", import);
                }
            }
        } else {
            let last_segment = import.path.segments.last().unwrap();
            let name = import.alias.clone().unwrap_or_else(|| last_segment.name.clone());
            log::debug!("resolved import {:?} ({:?}) to {:?}", name, import, def);

            // extern crates in the crate root are special-cased to insert entries into the extern prelude: rust-lang/rust#54658
            if let Some(root_module) = self.krate.root_module(self.db) {
                if import.is_extern_crate && module_id == root_module.module_id {
                    if let Some(def) = def.take_types() {
                        self.result.extern_prelude.insert(name.clone(), def);
                    }
                }
            }
            let resolution = Resolution { def, import: Some(import_id) };
            self.update(module_id, None, &[(name, resolution)]);
        }
        reached_fixedpoint
    }

    fn update(
        &mut self,
        module_id: ModuleId,
        import: Option<ImportId>,
        resolutions: &[(Name, Resolution)],
    ) {
        self.update_recursive(module_id, import, resolutions, 0)
    }

    fn update_recursive(
        &mut self,
        module_id: ModuleId,
        import: Option<ImportId>,
        resolutions: &[(Name, Resolution)],
        depth: usize,
    ) {
        if depth > 100 {
            // prevent stack overflows (but this shouldn't be possible)
            panic!("infinite recursion in glob imports!");
        }
        let module_items = self.result.per_module.get_mut(module_id).unwrap();
        let mut changed = false;
        for (name, res) in resolutions {
            let existing = module_items.items.entry(name.clone()).or_default();
            if existing.def.types.is_none() && res.def.types.is_some() {
                existing.def.types = res.def.types;
                existing.import = import.or(res.import);
                changed = true;
            }
            if existing.def.values.is_none() && res.def.values.is_some() {
                existing.def.values = res.def.values;
                existing.import = import.or(res.import);
                changed = true;
            }
        }
        if !changed {
            return;
        }
        let glob_imports = self
            .glob_imports
            .get(&module_id)
            .into_iter()
            .flat_map(|v| v.iter())
            .cloned()
            .collect::<Vec<_>>();
        for (glob_importing_module, glob_import) in glob_imports {
            // We pass the glob import so that the tracked import in those modules is that glob import
            self.update_recursive(glob_importing_module, Some(glob_import), resolutions, depth + 1);
        }
    }
}

#[derive(Debug, Clone)]
struct ResolvePathResult {
    resolved_def: PerNs<ModuleDef>,
    segment_index: Option<usize>,
    reached_fixedpoint: ReachedFixedPoint,
}

impl ResolvePathResult {
    fn empty(reached_fixedpoint: ReachedFixedPoint) -> ResolvePathResult {
        ResolvePathResult::with(PerNs::none(), reached_fixedpoint, None)
    }

    fn with(
        resolved_def: PerNs<ModuleDef>,
        reached_fixedpoint: ReachedFixedPoint,
        segment_index: Option<usize>,
    ) -> ResolvePathResult {
        ResolvePathResult { resolved_def, reached_fixedpoint, segment_index }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolveMode {
    Import,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReachedFixedPoint {
    Yes,
    No,
}

impl ItemMap {
    pub(crate) fn item_map_query(db: &impl PersistentHirDatabase, krate: Crate) -> Arc<ItemMap> {
        let start = time::Instant::now();
        let module_tree = db.module_tree(krate);
        let input = module_tree
            .modules()
            .map(|module_id| (module_id, db.lower_module(Module { krate, module_id })))
            .collect::<FxHashMap<_, _>>();

        let resolver = Resolver::new(db, &input, krate);
        let res = resolver.resolve();
        let elapsed = start.elapsed();
        log::info!("item_map: {:?}", elapsed);
        Arc::new(res)
    }

    pub(crate) fn resolve_path(
        &self,
        db: &impl PersistentHirDatabase,
        original_module: Module,
        path: &Path,
    ) -> (PerNs<ModuleDef>, Option<usize>) {
        let res = self.resolve_path_fp(db, ResolveMode::Other, original_module, path);
        (res.resolved_def, res.segment_index)
    }

    fn resolve_in_prelude(
        &self,
        db: &impl PersistentHirDatabase,
        original_module: Module,
        name: &Name,
    ) -> PerNs<ModuleDef> {
        if let Some(prelude) = self.prelude {
            let resolution = if prelude.krate == original_module.krate {
                self[prelude.module_id].items.get(name).cloned()
            } else {
                db.item_map(prelude.krate)[prelude.module_id].items.get(name).cloned()
            };
            resolution.map(|r| r.def).unwrap_or_else(PerNs::none)
        } else {
            PerNs::none()
        }
    }

    pub(crate) fn resolve_name_in_module(
        &self,
        db: &impl PersistentHirDatabase,
        module: Module,
        name: &Name,
    ) -> PerNs<ModuleDef> {
        // Resolve in:
        //  - current module / scope
        //  - extern prelude
        //  - std prelude
        let from_scope = self[module.module_id].items.get(name).map_or(PerNs::none(), |it| it.def);
        let from_extern_prelude =
            self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it));
        let from_prelude = self.resolve_in_prelude(db, module, name);

        from_scope.or(from_extern_prelude).or(from_prelude)
    }

    fn resolve_name_in_extern_prelude(&self, name: &Name) -> PerNs<ModuleDef> {
        self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it))
    }

    fn resolve_name_in_crate_root_or_extern_prelude(
        &self,
        db: &impl PersistentHirDatabase,
        module: Module,
        name: &Name,
    ) -> PerNs<ModuleDef> {
        let crate_root = module.crate_root(db);
        let from_crate_root =
            self[crate_root.module_id].items.get(name).map_or(PerNs::none(), |it| it.def);
        let from_extern_prelude = self.resolve_name_in_extern_prelude(name);

        from_crate_root.or(from_extern_prelude)
    }

    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    fn resolve_path_fp(
        &self,
        db: &impl PersistentHirDatabase,
        mode: ResolveMode,
        original_module: Module,
        path: &Path,
    ) -> ResolvePathResult {
        let mut segments = path.segments.iter().enumerate();
        let mut curr_per_ns: PerNs<ModuleDef> = match path.kind {
            PathKind::Crate => PerNs::types(original_module.crate_root(db).into()),
            PathKind::Self_ => PerNs::types(original_module.into()),
            // plain import or absolute path in 2015: crate-relative with
            // fallback to extern prelude (with the simplification in
            // rust-lang/rust#57745)
            // TODO there must be a nicer way to write this condition
            PathKind::Plain | PathKind::Abs
                if self.edition == Edition::Edition2015
                    && (path.kind == PathKind::Abs || mode == ResolveMode::Import) =>
            {
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                log::debug!("resolving {:?} in crate root (+ extern prelude)", segment);
                self.resolve_name_in_crate_root_or_extern_prelude(
                    db,
                    original_module,
                    &segment.name,
                )
            }
            PathKind::Plain => {
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                log::debug!("resolving {:?} in module", segment);
                self.resolve_name_in_module(db, original_module, &segment.name)
            }
            PathKind::Super => {
                if let Some(p) = original_module.parent(db) {
                    PerNs::types(p.into())
                } else {
                    log::debug!("super path in root module");
                    return ResolvePathResult::empty(ReachedFixedPoint::Yes);
                }
            }
            PathKind::Abs => {
                // 2018-style absolute path -- only extern prelude
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                if let Some(def) = self.extern_prelude.get(&segment.name) {
                    log::debug!("absolute path {:?} resolved to crate {:?}", path, def);
                    PerNs::types(*def)
                } else {
                    return ResolvePathResult::empty(ReachedFixedPoint::No); // extern crate declarations can add to the extern prelude
                }
            }
        };

        for (i, segment) in segments {
            let curr = match curr_per_ns.as_ref().take_types() {
                Some(r) => r,
                None => {
                    // we still have path segments left, but the path so far
                    // didn't resolve in the types namespace => no resolution
                    // (don't break here because `curr_per_ns` might contain
                    // something in the value namespace, and it would be wrong
                    // to return that)
                    return ResolvePathResult::empty(ReachedFixedPoint::No);
                }
            };
            // resolve segment in curr

            curr_per_ns = match curr {
                ModuleDef::Module(module) => {
                    if module.krate != original_module.krate {
                        let path = Path {
                            segments: path.segments[i..].iter().cloned().collect(),
                            kind: PathKind::Self_,
                        };
                        log::debug!("resolving {:?} in other crate", path);
                        let item_map = db.item_map(module.krate);
                        let (def, s) = item_map.resolve_path(db, *module, &path);
                        return ResolvePathResult::with(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                        );
                    }

                    match self[module.module_id].items.get(&segment.name) {
                        Some(res) if !res.def.is_none() => res.def,
                        _ => {
                            log::debug!("path segment {:?} not found", segment.name);
                            return ResolvePathResult::empty(ReachedFixedPoint::No);
                        }
                    }
                }
                ModuleDef::Enum(e) => {
                    // enum variant
                    tested_by!(item_map_enum_importing);
                    match e.variant(db, &segment.name) {
                        Some(variant) => PerNs::both(variant.into(), variant.into()),
                        None => {
                            return ResolvePathResult::with(
                                PerNs::types((*e).into()),
                                ReachedFixedPoint::Yes,
                                Some(i),
                            );
                        }
                    }
                }
                s => {
                    // could be an inherent method call in UFCS form
                    // (`Struct::method`), or some other kind of associated item
                    log::debug!(
                        "path segment {:?} resolved to non-module {:?}, but is not last",
                        segment.name,
                        curr,
                    );

                    return ResolvePathResult::with(
                        PerNs::types((*s).into()),
                        ReachedFixedPoint::Yes,
                        Some(i),
                    );
                }
            };
        }
        ResolvePathResult::with(curr_per_ns, ReachedFixedPoint::Yes, None)
    }
}

#[cfg(test)]
mod tests;
