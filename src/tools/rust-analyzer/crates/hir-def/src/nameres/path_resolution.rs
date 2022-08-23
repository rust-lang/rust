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

use base_db::Edition;
use hir_expand::name::Name;

use crate::{
    db::DefDatabase,
    item_scope::BUILTIN_SCOPE,
    nameres::{BuiltinShadowMode, DefMap},
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
        ResolvePathResult { resolved_def, segment_index, reached_fixedpoint, krate }
    }
}

impl DefMap {
    pub(super) fn resolve_name_in_extern_prelude(
        &self,
        db: &dyn DefDatabase,
        name: &Name,
    ) -> Option<ModuleId> {
        match self.block {
            Some(_) => self.crate_root(db).def_map(db).extern_prelude.get(name).copied(),
            None => self.extern_prelude.get(name).copied(),
        }
    }

    pub(crate) fn resolve_visibility(
        &self,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        visibility: &RawVisibility,
    ) -> Option<Visibility> {
        let mut vis = match visibility {
            RawVisibility::Module(path) => {
                let (result, remaining) =
                    self.resolve_path(db, original_module, path, BuiltinShadowMode::Module);
                if remaining.is_some() {
                    return None;
                }
                let types = result.take_types()?;
                match types {
                    ModuleDefId::ModuleId(m) => Visibility::Module(m),
                    _ => {
                        // error: visibility needs to refer to module
                        return None;
                    }
                }
            }
            RawVisibility::Public => Visibility::Public,
        };

        // In block expressions, `self` normally refers to the containing non-block module, and
        // `super` to its parent (etc.). However, visibilities must only refer to a module in the
        // DefMap they're written in, so we restrict them when that happens.
        if let Visibility::Module(m) = vis {
            if self.block_id() != m.block {
                cov_mark::hit!(adjust_vis_in_block_def_map);
                vis = Visibility::Module(self.module_id(self.root()));
                tracing::debug!("visibility {:?} points outside DefMap, adjusting to {:?}", m, vis);
            }
        }

        Some(vis)
    }

    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    pub(super) fn resolve_path_fp_with_macro(
        &self,
        db: &dyn DefDatabase,
        mode: ResolveMode,
        mut original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> ResolvePathResult {
        let mut result = ResolvePathResult::empty(ReachedFixedPoint::No);

        let mut arc;
        let mut current_map = self;
        loop {
            let new = current_map.resolve_path_fp_with_macro_single(
                db,
                mode,
                original_module,
                path,
                shadow,
            );

            // Merge `new` into `result`.
            result.resolved_def = result.resolved_def.or(new.resolved_def);
            if result.reached_fixedpoint == ReachedFixedPoint::No {
                result.reached_fixedpoint = new.reached_fixedpoint;
            }
            // FIXME: this doesn't seem right; what if the different namespace resolutions come from different crates?
            result.krate = result.krate.or(new.krate);
            result.segment_index = match (result.segment_index, new.segment_index) {
                (Some(idx), None) => Some(idx),
                (Some(old), Some(new)) => Some(old.max(new)),
                (None, new) => new,
            };

            match &current_map.block {
                Some(block) => {
                    original_module = block.parent.local_id;
                    arc = block.parent.def_map(db);
                    current_map = &*arc;
                }
                None => return result,
            }
        }
    }

    pub(super) fn resolve_path_fp_with_macro_single(
        &self,
        db: &dyn DefDatabase,
        mode: ResolveMode,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> ResolvePathResult {
        let graph = db.crate_graph();
        let _cx = stdx::panic_context::enter(format!(
            "DefMap {:?} crate_name={:?} block={:?} path={}",
            self.krate, graph[self.krate].display_name, self.block, path
        ));

        let mut segments = path.segments().iter().enumerate();
        let mut curr_per_ns: PerNs = match path.kind {
            PathKind::DollarCrate(krate) => {
                if krate == self.krate {
                    cov_mark::hit!(macro_dollar_crate_self);
                    PerNs::types(self.crate_root(db).into(), Visibility::Public)
                } else {
                    let def_map = db.crate_def_map(krate);
                    let module = def_map.module_id(def_map.root);
                    cov_mark::hit!(macro_dollar_crate_other);
                    PerNs::types(module.into(), Visibility::Public)
                }
            }
            PathKind::Crate => PerNs::types(self.crate_root(db).into(), Visibility::Public),
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
                tracing::debug!("resolving {:?} in crate root (+ extern prelude)", segment);
                self.resolve_name_in_crate_root_or_extern_prelude(db, segment)
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
                    if path.segments().len() == 1 { shadow } else { BuiltinShadowMode::Module };

                tracing::debug!("resolving {:?} in module", segment);
                self.resolve_name_in_module(db, original_module, segment, prefer_module)
            }
            PathKind::Super(lvl) => {
                let mut module = original_module;
                for i in 0..lvl {
                    match self.modules[module].parent {
                        Some(it) => module = it,
                        None => match &self.block {
                            Some(block) => {
                                // Look up remaining path in parent `DefMap`
                                let new_path = ModPath::from_segments(
                                    PathKind::Super(lvl - i),
                                    path.segments().to_vec(),
                                );
                                tracing::debug!(
                                    "`super` path: {} -> {} in parent map",
                                    path,
                                    new_path
                                );
                                return block.parent.def_map(db).resolve_path_fp_with_macro(
                                    db,
                                    mode,
                                    block.parent.local_id,
                                    &new_path,
                                    shadow,
                                );
                            }
                            None => {
                                tracing::debug!("super path in root module");
                                return ResolvePathResult::empty(ReachedFixedPoint::Yes);
                            }
                        },
                    }
                }

                // Resolve `self` to the containing crate-rooted module if we're a block
                self.with_ancestor_maps(db, module, &mut |def_map, module| {
                    if def_map.block.is_some() {
                        None // keep ascending
                    } else {
                        Some(PerNs::types(def_map.module_id(module).into(), Visibility::Public))
                    }
                })
                .expect("block DefMap not rooted in crate DefMap")
            }
            PathKind::Abs => {
                // 2018-style absolute path -- only extern prelude
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                if let Some(&def) = self.extern_prelude.get(segment) {
                    tracing::debug!("absolute path {:?} resolved to crate {:?}", path, def);
                    PerNs::types(def.into(), Visibility::Public)
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
                        let path = ModPath::from_segments(
                            PathKind::Super(0),
                            path.segments()[i..].iter().cloned(),
                        );
                        tracing::debug!("resolving {:?} in other crate", path);
                        let defp_map = module.def_map(db);
                        let (def, s) = defp_map.resolve_path(db, module.local_id, &path, shadow);
                        return ResolvePathResult::with(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                            Some(module.krate),
                        );
                    }

                    let def_map;
                    let module_data = if module.block == self.block_id() {
                        &self[module.local_id]
                    } else {
                        def_map = module.def_map(db);
                        &def_map[module.local_id]
                    };

                    // Since it is a qualified path here, it should not contains legacy macros
                    module_data.scope.get(segment)
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    // enum variant
                    cov_mark::hit!(can_import_enum_variant);
                    let enum_data = db.enum_data(e);
                    match enum_data.variant(segment) {
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
                    tracing::debug!(
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
            // FIXME: shadowing
            .and_then(|it| it.last())
            .map_or_else(PerNs::none, |&m| PerNs::macros(m.into(), Visibility::Public));
        let from_scope = self[module].scope.get(name);
        let from_builtin = match self.block {
            Some(_) => {
                // Only resolve to builtins in the root `DefMap`.
                PerNs::none()
            }
            None => BUILTIN_SCOPE.get(name).copied().unwrap_or_else(PerNs::none),
        };
        let from_scope_or_builtin = match shadow {
            BuiltinShadowMode::Module => from_scope.or(from_builtin),
            BuiltinShadowMode::Other => match from_scope.take_types() {
                Some(ModuleDefId::ModuleId(_)) => from_builtin.or(from_scope),
                Some(_) | None => from_scope.or(from_builtin),
            },
        };

        let extern_prelude = || {
            self.extern_prelude
                .get(name)
                .map_or(PerNs::none(), |&it| PerNs::types(it.into(), Visibility::Public))
        };
        let prelude = || self.resolve_in_prelude(db, name);

        from_legacy_macro.or(from_scope_or_builtin).or_else(extern_prelude).or_else(prelude)
    }

    fn resolve_name_in_crate_root_or_extern_prelude(
        &self,
        db: &dyn DefDatabase,
        name: &Name,
    ) -> PerNs {
        let from_crate_root = match self.block {
            Some(_) => {
                let def_map = self.crate_root(db).def_map(db);
                def_map[def_map.root].scope.get(name)
            }
            None => self[self.root].scope.get(name),
        };
        let from_extern_prelude = || {
            self.resolve_name_in_extern_prelude(db, name)
                .map_or(PerNs::none(), |it| PerNs::types(it.into(), Visibility::Public))
        };

        from_crate_root.or_else(from_extern_prelude)
    }

    fn resolve_in_prelude(&self, db: &dyn DefDatabase, name: &Name) -> PerNs {
        if let Some(prelude) = self.prelude {
            let keep;
            let def_map = if prelude.krate == self.krate {
                self
            } else {
                // Extend lifetime
                keep = prelude.def_map(db);
                &keep
            };
            def_map[prelude.local_id].scope.get(name)
        } else {
            PerNs::none()
        }
    }
}
