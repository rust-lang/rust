//! This modules implements a function to resolve a path `foo::bar::baz` to a
//! def, which is used within the name resolution.
//!
//! When name resolution is finished, the result of resolving a path is either
//! `Some(def)` or `None`. However, when we are in process of resolving imports
//! or macros, there's a third possibility:
//!
//!   I can't resolve this path right now, but I might be resolve this path
//!   later, when more macros are expanded.
//!
//! `ReachedFixedPoint` signals about this.

use std::iter::successors;

use base_db::Edition;
use hir_expand::name::Name;
use test_utils::mark;

use crate::{
    db::DefDatabase,
    item_scope::BUILTIN_SCOPE,
    nameres::{BuiltinShadowMode, CrateDefMap},
    path::{ModPath, PathKind},
    per_ns::PerNs,
    visibility::{RawVisibility, Visibility},
    AdtId, CrateId, EnumVariantId, LocalModuleId, ModuleDefId, ModuleId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResolveMode {
    Import,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReachedFixedPoint {
    Yes,
    No,
}

#[derive(Debug, Clone)]
pub(super) struct ResolvePathResult {
    pub(super) resolved_def: PerNs,
    pub(super) segment_index: Option<usize>,
    pub(super) reached_fixedpoint: ReachedFixedPoint,
    pub(super) krate: Option<CrateId>,
}

impl ResolvePathResult {
    fn empty(reached_fixedpoint: ReachedFixedPoint) -> ResolvePathResult {
        ResolvePathResult::with(PerNs::none(), reached_fixedpoint, None, None)
    }

    fn with(
        resolved_def: PerNs,
        reached_fixedpoint: ReachedFixedPoint,
        segment_index: Option<usize>,
        krate: Option<CrateId>,
    ) -> ResolvePathResult {
        ResolvePathResult { resolved_def, reached_fixedpoint, segment_index, krate }
    }
}

impl CrateDefMap {
    pub(super) fn resolve_name_in_extern_prelude(&self, name: &Name) -> PerNs {
        self.extern_prelude
            .get(name)
            .map_or(PerNs::none(), |&it| PerNs::types(it, Visibility::Public))
    }

    pub(crate) fn resolve_visibility(
        &self,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        visibility: &RawVisibility,
    ) -> Option<Visibility> {
        match visibility {
            RawVisibility::Module(path) => {
                let (result, remaining) =
                    self.resolve_path(db, original_module, &path, BuiltinShadowMode::Module);
                if remaining.is_some() {
                    return None;
                }
                let types = result.take_types()?;
                match types {
                    ModuleDefId::ModuleId(m) => Some(Visibility::Module(m)),
                    _ => {
                        // error: visibility needs to refer to module
                        None
                    }
                }
            }
            RawVisibility::Public => Some(Visibility::Public),
        }
    }

    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    pub(super) fn resolve_path_fp_with_macro(
        &self,
        db: &dyn DefDatabase,
        mode: ResolveMode,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> ResolvePathResult {
        let mut segments = path.segments.iter().enumerate();
        let mut curr_per_ns: PerNs = match path.kind {
            PathKind::DollarCrate(krate) => {
                if krate == self.krate {
                    mark::hit!(macro_dollar_crate_self);
                    PerNs::types(
                        ModuleId { krate: self.krate, local_id: self.root }.into(),
                        Visibility::Public,
                    )
                } else {
                    let def_map = db.crate_def_map(krate);
                    let module = ModuleId { krate, local_id: def_map.root };
                    mark::hit!(macro_dollar_crate_other);
                    PerNs::types(module.into(), Visibility::Public)
                }
            }
            PathKind::Crate => PerNs::types(
                ModuleId { krate: self.krate, local_id: self.root }.into(),
                Visibility::Public,
            ),
            // plain import or absolute path in 2015: crate-relative with
            // fallback to extern prelude (with the simplification in
            // rust-lang/rust#57745)
            // FIXME there must be a nicer way to write this condition
            PathKind::Plain | PathKind::Abs
                if self.edition == Edition::Edition2015
                    && (path.kind == PathKind::Abs || mode == ResolveMode::Import) =>
            {
                let (_, segment) = match segments.next() {
                    Some((idx, segment)) => (idx, segment),
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                log::debug!("resolving {:?} in crate root (+ extern prelude)", segment);
                self.resolve_name_in_crate_root_or_extern_prelude(&segment)
            }
            PathKind::Plain => {
                let (_, segment) = match segments.next() {
                    Some((idx, segment)) => (idx, segment),
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                // The first segment may be a builtin type. If the path has more
                // than one segment, we first try resolving it as a module
                // anyway.
                // FIXME: If the next segment doesn't resolve in the module and
                // BuiltinShadowMode wasn't Module, then we need to try
                // resolving it as a builtin.
                let prefer_module =
                    if path.segments.len() == 1 { shadow } else { BuiltinShadowMode::Module };

                log::debug!("resolving {:?} in module", segment);
                self.resolve_name_in_module(db, original_module, &segment, prefer_module)
            }
            PathKind::Super(lvl) => {
                let m = successors(Some(original_module), |m| self.modules[*m].parent)
                    .nth(lvl as usize);
                if let Some(local_id) = m {
                    PerNs::types(
                        ModuleId { krate: self.krate, local_id }.into(),
                        Visibility::Public,
                    )
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
                if let Some(def) = self.extern_prelude.get(&segment) {
                    log::debug!("absolute path {:?} resolved to crate {:?}", path, def);
                    PerNs::types(*def, Visibility::Public)
                } else {
                    return ResolvePathResult::empty(ReachedFixedPoint::No); // extern crate declarations can add to the extern prelude
                }
            }
        };

        for (i, segment) in segments {
            let (curr, vis) = match curr_per_ns.take_types_vis() {
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
                ModuleDefId::ModuleId(module) => {
                    if module.krate != self.krate {
                        let path = ModPath {
                            segments: path.segments[i..].to_vec(),
                            kind: PathKind::Super(0),
                        };
                        log::debug!("resolving {:?} in other crate", path);
                        let defp_map = db.crate_def_map(module.krate);
                        let (def, s) = defp_map.resolve_path(db, module.local_id, &path, shadow);
                        return ResolvePathResult::with(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                            Some(module.krate),
                        );
                    }

                    // Since it is a qualified path here, it should not contains legacy macros
                    self[module.local_id].scope.get(&segment)
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    // enum variant
                    mark::hit!(can_import_enum_variant);
                    let enum_data = db.enum_data(e);
                    match enum_data.variant(&segment) {
                        Some(local_id) => {
                            let variant = EnumVariantId { parent: e, local_id };
                            match &*enum_data.variants[local_id].variant_data {
                                crate::adt::VariantData::Record(_) => {
                                    PerNs::types(variant.into(), Visibility::Public)
                                }
                                crate::adt::VariantData::Tuple(_)
                                | crate::adt::VariantData::Unit => {
                                    PerNs::both(variant.into(), variant.into(), Visibility::Public)
                                }
                            }
                        }
                        None => {
                            return ResolvePathResult::with(
                                PerNs::types(e.into(), vis),
                                ReachedFixedPoint::Yes,
                                Some(i),
                                Some(self.krate),
                            );
                        }
                    }
                }
                s => {
                    // could be an inherent method call in UFCS form
                    // (`Struct::method`), or some other kind of associated item
                    log::debug!(
                        "path segment {:?} resolved to non-module {:?}, but is not last",
                        segment,
                        curr,
                    );

                    return ResolvePathResult::with(
                        PerNs::types(s, vis),
                        ReachedFixedPoint::Yes,
                        Some(i),
                        Some(self.krate),
                    );
                }
            };
        }

        ResolvePathResult::with(curr_per_ns, ReachedFixedPoint::Yes, None, Some(self.krate))
    }

    fn resolve_name_in_module(
        &self,
        db: &dyn DefDatabase,
        module: LocalModuleId,
        name: &Name,
        shadow: BuiltinShadowMode,
    ) -> PerNs {
        // Resolve in:
        //  - legacy scope of macro
        //  - current module / scope
        //  - extern prelude
        //  - std prelude
        let from_legacy_macro = self[module]
            .scope
            .get_legacy_macro(name)
            .map_or_else(PerNs::none, |m| PerNs::macros(m, Visibility::Public));
        let from_scope = self[module].scope.get(name);
        let from_builtin = BUILTIN_SCOPE.get(name).copied().unwrap_or_else(PerNs::none);
        let from_scope_or_builtin = match shadow {
            BuiltinShadowMode::Module => from_scope.or(from_builtin),
            BuiltinShadowMode::Other => {
                if let Some(ModuleDefId::ModuleId(_)) = from_scope.take_types() {
                    from_builtin.or(from_scope)
                } else {
                    from_scope.or(from_builtin)
                }
            }
        };
        let from_extern_prelude = self
            .extern_prelude
            .get(name)
            .map_or(PerNs::none(), |&it| PerNs::types(it, Visibility::Public));
        let from_prelude = self.resolve_in_prelude(db, name);

        from_legacy_macro.or(from_scope_or_builtin).or(from_extern_prelude).or(from_prelude)
    }

    fn resolve_name_in_crate_root_or_extern_prelude(&self, name: &Name) -> PerNs {
        let from_crate_root = self[self.root].scope.get(name);
        let from_extern_prelude = self.resolve_name_in_extern_prelude(name);

        from_crate_root.or(from_extern_prelude)
    }

    fn resolve_in_prelude(&self, db: &dyn DefDatabase, name: &Name) -> PerNs {
        if let Some(prelude) = self.prelude {
            let keep;
            let def_map = if prelude.krate == self.krate {
                self
            } else {
                // Extend lifetime
                keep = db.crate_def_map(prelude.krate);
                &keep
            };
            def_map[prelude.local_id].scope.get(name)
        } else {
            PerNs::none()
        }
    }
}
