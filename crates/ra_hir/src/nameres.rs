/// This module implements import-resolution/macro expansion algorithm.
///
/// The result of this module is `CrateDefMap`: a datastructure which contains:
///
///   * a tree of modules for the crate
///   * for each module, a set of items visible in the module (directly declared
///     or imported)
///
/// Note that `CrateDefMap` contains fully macro expanded code.
///
/// Computing `CrateDefMap` can be partitioned into several logically
/// independent "phases". The phases are mutually recursive though, there's no
/// stric ordering.
///
/// ## Collecting RawItems
///
///  This happens in the `raw` module, which parses a single source file into a
///  set of top-level items. Nested importa are desugared to flat imports in
///  this phase. Macro calls are represented as a triple of (Path, Option<Name>,
///  TokenTree).
///
/// ## Collecting Modules
///
/// This happens in the `collector` module. In this phase, we recursively walk
/// tree of modules, collect raw items from submodules, populate module scopes
/// with defined items (so, we assign item ids in this phase) and record the set
/// of unresovled imports and macros.
///
/// While we walk tree of modules, we also record macro_rules defenitions and
/// expand calls to macro_rules defined macros.
///
/// ## Resolving Imports
///
/// TBD
///
/// ## Resolving Macros
///
/// While macro_rules from the same crate use a global mutable namespace, macros
/// from other crates (including proc-macros) can be used with `foo::bar!`
/// syntax.
///
/// TBD;

mod per_ns;
mod raw;
mod collector;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use rustc_hash::FxHashMap;
use ra_arena::{Arena, RawId, impl_arena_id};
use ra_db::{FileId, Edition};
use test_utils::tested_by;

use crate::{
    ModuleDef, Name, Crate, Module, Problem,
    PersistentHirDatabase, Path, PathKind, HirFileId,
    ids::{SourceItemId, SourceFileItemId, MacroCallId},
};

pub(crate) use self::raw::{RawItems, ImportId, ImportSourceMap};

pub use self::per_ns::{PerNs, Namespace};

/// Contans all top-level defs from a macro-expanded crate
#[derive(Debug, PartialEq, Eq)]
pub struct CrateDefMap {
    krate: Crate,
    edition: Edition,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    prelude: Option<Module>,
    extern_prelude: FxHashMap<Name, ModuleDef>,
    root: CrateModuleId,
    modules: Arena<CrateModuleId, ModuleData>,
    macros: Arena<CrateMacroId, mbe::MacroRules>,
    public_macros: FxHashMap<Name, CrateMacroId>,
    macro_resolutions: FxHashMap<MacroCallId, (Crate, CrateMacroId)>,
    problems: CrateDefMapProblems,
}

impl std::ops::Index<CrateModuleId> for CrateDefMap {
    type Output = ModuleData;
    fn index(&self, id: CrateModuleId) -> &ModuleData {
        &self.modules[id]
    }
}

impl std::ops::Index<CrateMacroId> for CrateDefMap {
    type Output = mbe::MacroRules;
    fn index(&self, id: CrateMacroId) -> &mbe::MacroRules {
        &self.macros[id]
    }
}

/// An ID of a macro, **local** to a specific crate
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CrateMacroId(RawId);
impl_arena_id!(CrateMacroId);

/// An ID of a module, **local** to a specific crate
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CrateModuleId(RawId);
impl_arena_id!(CrateModuleId);

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct ModuleData {
    pub(crate) parent: Option<CrateModuleId>,
    pub(crate) children: FxHashMap<Name, CrateModuleId>,
    pub(crate) scope: ModuleScope,
    /// None for root
    pub(crate) declaration: Option<SourceItemId>,
    /// None for inline modules.
    ///
    /// Note that non-inline modules, by definition, live inside non-macro file.
    pub(crate) definition: Option<FileId>,
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct CrateDefMapProblems {
    problems: Vec<(SourceItemId, Problem)>,
}

impl CrateDefMapProblems {
    fn add(&mut self, source_item_id: SourceItemId, problem: Problem) {
        self.problems.push((source_item_id, problem))
    }

    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a SourceItemId, &'a Problem)> + 'a {
        self.problems.iter().map(|(s, p)| (s, p))
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Resolution {
    /// None for unresolved
    pub def: PerNs<ModuleDef>,
    /// ident by which this is imported into local scope.
    pub import: Option<ImportId>,
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

impl CrateDefMap {
    pub(crate) fn crate_def_map_query(
        db: &impl PersistentHirDatabase,
        krate: Crate,
    ) -> Arc<CrateDefMap> {
        let start = std::time::Instant::now();
        let def_map = {
            let edition = krate.edition(db);
            let mut modules: Arena<CrateModuleId, ModuleData> = Arena::default();
            let root = modules.alloc(ModuleData::default());
            CrateDefMap {
                krate,
                edition,
                extern_prelude: FxHashMap::default(),
                prelude: None,
                root,
                modules,
                macros: Arena::default(),
                public_macros: FxHashMap::default(),
                macro_resolutions: FxHashMap::default(),
                problems: CrateDefMapProblems::default(),
            }
        };
        let def_map = collector::collect_defs(db, def_map);
        log::info!("crate_def_map_query: {:?}", start.elapsed());
        Arc::new(def_map)
    }

    pub(crate) fn root(&self) -> CrateModuleId {
        self.root
    }

    pub(crate) fn problems(&self) -> &CrateDefMapProblems {
        &self.problems
    }

    pub(crate) fn mk_module(&self, module_id: CrateModuleId) -> Module {
        Module { krate: self.krate, module_id }
    }

    pub(crate) fn prelude(&self) -> Option<Module> {
        self.prelude
    }

    pub(crate) fn extern_prelude(&self) -> &FxHashMap<Name, ModuleDef> {
        &self.extern_prelude
    }

    pub(crate) fn resolve_macro(
        &self,
        macro_call_id: MacroCallId,
    ) -> Option<(Crate, CrateMacroId)> {
        self.macro_resolutions.get(&macro_call_id).map(|&it| it)
    }

    pub(crate) fn find_module_by_source(
        &self,
        file_id: HirFileId,
        decl_id: Option<SourceFileItemId>,
    ) -> Option<CrateModuleId> {
        let decl_id = decl_id.map(|it| it.with_file_id(file_id));
        let (module_id, _module_data) = self.modules.iter().find(|(_module_id, module_data)| {
            if decl_id.is_some() {
                module_data.declaration == decl_id
            } else {
                module_data.definition.map(|it| it.into()) == Some(file_id)
            }
        })?;
        Some(module_id)
    }

    pub(crate) fn resolve_path(
        &self,
        db: &impl PersistentHirDatabase,
        original_module: CrateModuleId,
        path: &Path,
    ) -> (PerNs<ModuleDef>, Option<usize>) {
        let res = self.resolve_path_fp(db, ResolveMode::Other, original_module, path);
        (res.resolved_def, res.segment_index)
    }

    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    fn resolve_path_fp(
        &self,
        db: &impl PersistentHirDatabase,
        mode: ResolveMode,
        original_module: CrateModuleId,
        path: &Path,
    ) -> ResolvePathResult {
        let mut segments = path.segments.iter().enumerate();
        let mut curr_per_ns: PerNs<ModuleDef> = match path.kind {
            PathKind::Crate => {
                PerNs::types(Module { krate: self.krate, module_id: self.root }.into())
            }
            PathKind::Self_ => {
                PerNs::types(Module { krate: self.krate, module_id: original_module }.into())
            }
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
                self.resolve_name_in_crate_root_or_extern_prelude(&segment.name)
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
                if let Some(p) = self.modules[original_module].parent {
                    PerNs::types(Module { krate: self.krate, module_id: p }.into())
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
                    if module.krate != self.krate {
                        let path = Path {
                            segments: path.segments[i..].iter().cloned().collect(),
                            kind: PathKind::Self_,
                        };
                        log::debug!("resolving {:?} in other crate", path);
                        let defp_map = db.crate_def_map(module.krate);
                        let (def, s) = defp_map.resolve_path(db, module.module_id, &path);
                        return ResolvePathResult::with(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                        );
                    }

                    match self[module.module_id].scope.items.get(&segment.name) {
                        Some(res) if !res.def.is_none() => res.def,
                        _ => {
                            log::debug!("path segment {:?} not found", segment.name);
                            return ResolvePathResult::empty(ReachedFixedPoint::No);
                        }
                    }
                }
                ModuleDef::Enum(e) => {
                    // enum variant
                    tested_by!(can_import_enum_variant);
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

    fn resolve_name_in_crate_root_or_extern_prelude(&self, name: &Name) -> PerNs<ModuleDef> {
        let from_crate_root =
            self[self.root].scope.items.get(name).map_or(PerNs::none(), |it| it.def);
        let from_extern_prelude = self.resolve_name_in_extern_prelude(name);

        from_crate_root.or(from_extern_prelude)
    }

    pub(crate) fn resolve_name_in_module(
        &self,
        db: &impl PersistentHirDatabase,
        module: CrateModuleId,
        name: &Name,
    ) -> PerNs<ModuleDef> {
        // Resolve in:
        //  - current module / scope
        //  - extern prelude
        //  - std prelude
        let from_scope = self[module].scope.items.get(name).map_or(PerNs::none(), |it| it.def);
        let from_extern_prelude =
            self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it));
        let from_prelude = self.resolve_in_prelude(db, name);

        from_scope.or(from_extern_prelude).or(from_prelude)
    }

    fn resolve_name_in_extern_prelude(&self, name: &Name) -> PerNs<ModuleDef> {
        self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it))
    }

    fn resolve_in_prelude(&self, db: &impl PersistentHirDatabase, name: &Name) -> PerNs<ModuleDef> {
        if let Some(prelude) = self.prelude {
            let resolution = if prelude.krate == self.krate {
                self[prelude.module_id].scope.items.get(name).cloned()
            } else {
                db.crate_def_map(prelude.krate)[prelude.module_id].scope.items.get(name).cloned()
            };
            resolution.map(|r| r.def).unwrap_or_else(PerNs::none)
        } else {
            PerNs::none()
        }
    }
}
