/// This module implements new import-resolution/macro expansion algorithm.
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

mod raw;
mod collector;
#[cfg(test)]
mod tests;

use rustc_hash::FxHashMap;
use test_utils::tested_by;
use ra_arena::Arena;

use crate::{
    Name, Module, Path, PathKind, ModuleDef, Crate,
    PersistentHirDatabase,
    module_tree::ModuleId,
    nameres::{ModuleScope, ResolveMode, ResolvePathResult, PerNs, Edition, ReachedFixedPoint},
};

#[derive(Default, Debug)]
struct ModuleData {
    parent: Option<ModuleId>,
    children: FxHashMap<Name, ModuleId>,
    scope: ModuleScope,
}

/// Contans all top-level defs from a macro-expanded crate
#[derive(Debug)]
pub(crate) struct CrateDefMap {
    krate: Crate,
    edition: Edition,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    prelude: Option<Module>,
    extern_prelude: FxHashMap<Name, ModuleDef>,
    root: ModuleId,
    modules: Arena<ModuleId, ModuleData>,
    public_macros: FxHashMap<Name, mbe::MacroRules>,
}

impl std::ops::Index<ModuleId> for CrateDefMap {
    type Output = ModuleScope;
    fn index(&self, id: ModuleId) -> &ModuleScope {
        &self.modules[id].scope
    }
}

impl CrateDefMap {
    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    #[allow(unused)]
    fn resolve_path_fp(
        &self,
        db: &impl PersistentHirDatabase,
        mode: ResolveMode,
        original_module: ModuleId,
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

    fn resolve_name_in_crate_root_or_extern_prelude(&self, name: &Name) -> PerNs<ModuleDef> {
        let from_crate_root = self[self.root].items.get(name).map_or(PerNs::none(), |it| it.def);
        let from_extern_prelude = self.resolve_name_in_extern_prelude(name);

        from_crate_root.or(from_extern_prelude)
    }

    fn resolve_name_in_module(
        &self,
        db: &impl PersistentHirDatabase,
        module: ModuleId,
        name: &Name,
    ) -> PerNs<ModuleDef> {
        // Resolve in:
        //  - current module / scope
        //  - extern prelude
        //  - std prelude
        let from_scope = self[module].items.get(name).map_or(PerNs::none(), |it| it.def);
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
                self[prelude.module_id].items.get(name).cloned()
            } else {
                db.item_map(prelude.krate)[prelude.module_id].items.get(name).cloned()
            };
            resolution.map(|r| r.def).unwrap_or_else(PerNs::none)
        } else {
            PerNs::none()
        }
    }
}
