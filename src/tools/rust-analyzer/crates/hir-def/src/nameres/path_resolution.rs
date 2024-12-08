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

use either::Either;
use hir_expand::{name::Name, Lookup};
use span::Edition;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    item_scope::{ImportOrExternCrate, BUILTIN_SCOPE},
    item_tree::FieldsShape,
    nameres::{sub_namespace_match, BlockInfo, BuiltinShadowMode, DefMap, MacroSubNs},
    path::{ModPath, PathKind},
    per_ns::PerNs,
    visibility::{RawVisibility, Visibility},
    AdtId, LocalModuleId, ModuleDefId,
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
    pub(super) from_differing_crate: bool,
}

impl ResolvePathResult {
    fn empty(reached_fixedpoint: ReachedFixedPoint) -> ResolvePathResult {
        ResolvePathResult::new(PerNs::none(), reached_fixedpoint, None, false)
    }

    fn new(
        resolved_def: PerNs,
        reached_fixedpoint: ReachedFixedPoint,
        segment_index: Option<usize>,
        from_differing_crate: bool,
    ) -> ResolvePathResult {
        ResolvePathResult { resolved_def, segment_index, reached_fixedpoint, from_differing_crate }
    }
}

impl PerNs {
    pub(super) fn filter_macro(
        mut self,
        db: &dyn DefDatabase,
        expected: Option<MacroSubNs>,
    ) -> Self {
        self.macros = self.macros.filter(|&(id, _, _)| {
            let this = MacroSubNs::from_id(db, id);
            sub_namespace_match(Some(this), expected)
        });

        self
    }
}

impl DefMap {
    pub(crate) fn resolve_visibility(
        &self,
        db: &dyn DefDatabase,
        // module to import to
        original_module: LocalModuleId,
        // pub(path)
        //     ^^^^ this
        visibility: &RawVisibility,
        within_impl: bool,
    ) -> Option<Visibility> {
        let mut vis = match visibility {
            RawVisibility::Module(path, explicitness) => {
                let (result, remaining) =
                    self.resolve_path(db, original_module, path, BuiltinShadowMode::Module, None);
                if remaining.is_some() {
                    return None;
                }
                let types = result.take_types()?;
                match types {
                    ModuleDefId::ModuleId(m) => Visibility::Module(m, *explicitness),
                    // error: visibility needs to refer to module
                    _ => {
                        return None;
                    }
                }
            }
            RawVisibility::Public => Visibility::Public,
        };

        // In block expressions, `self` normally refers to the containing non-block module, and
        // `super` to its parent (etc.). However, visibilities must only refer to a module in the
        // DefMap they're written in, so we restrict them when that happens.
        if let Visibility::Module(m, mv) = vis {
            // ...unless we're resolving visibility for an associated item in an impl.
            if self.block_id() != m.block && !within_impl {
                cov_mark::hit!(adjust_vis_in_block_def_map);
                vis = Visibility::Module(self.module_id(Self::ROOT), mv);
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
        // module to import to
        mut original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
        // Pass `MacroSubNs` if we know we're resolving macro names and which kind of macro we're
        // resolving them to. Pass `None` otherwise, e.g. when we're resolving import paths.
        expected_macro_subns: Option<MacroSubNs>,
    ) -> ResolvePathResult {
        let mut result = self.resolve_path_fp_with_macro_single(
            db,
            mode,
            original_module,
            path,
            shadow,
            expected_macro_subns,
        );

        if self.block.is_none() {
            // If we're in the root `DefMap`, we can resolve the path directly.
            return result;
        }

        let mut arc;
        let mut current_map = self;

        let mut merge = |new: ResolvePathResult| {
            result.resolved_def = result.resolved_def.or(new.resolved_def);
            if result.reached_fixedpoint == ReachedFixedPoint::No {
                result.reached_fixedpoint = new.reached_fixedpoint;
            }
            result.from_differing_crate |= new.from_differing_crate;
            result.segment_index = match (result.segment_index, new.segment_index) {
                (Some(idx), None) => Some(idx),
                (Some(old), Some(new)) => Some(old.max(new)),
                (None, new) => new,
            };
        };

        loop {
            match current_map.block {
                Some(block) if original_module == Self::ROOT => {
                    // Block modules "inherit" names from its parent module.
                    original_module = block.parent.local_id;
                    arc = block.parent.def_map(db, current_map.krate);
                    current_map = &arc;
                }
                // Proper (non-block) modules, including those in block `DefMap`s, don't.
                _ => {
                    if original_module != Self::ROOT && current_map.block.is_some() {
                        // A module inside a block. Do not resolve items declared in upper blocks, but we do need to get
                        // the prelude items (which are not inserted into blocks because they can be overridden there).
                        original_module = Self::ROOT;
                        arc = db.crate_def_map(self.krate);
                        current_map = &arc;

                        let new = current_map.resolve_path_fp_in_all_preludes(
                            db,
                            mode,
                            original_module,
                            path,
                            shadow,
                        );
                        merge(new);
                    }

                    return result;
                }
            }

            let new = current_map.resolve_path_fp_with_macro_single(
                db,
                mode,
                original_module,
                path,
                shadow,
                expected_macro_subns,
            );

            merge(new);
        }
    }

    pub(super) fn resolve_path_fp_with_macro_single(
        &self,
        db: &dyn DefDatabase,
        mode: ResolveMode,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
        expected_macro_subns: Option<MacroSubNs>,
    ) -> ResolvePathResult {
        let mut segments = path.segments().iter().enumerate();
        let curr_per_ns = match path.kind {
            PathKind::DollarCrate(krate) => {
                if krate == self.krate {
                    cov_mark::hit!(macro_dollar_crate_self);
                    PerNs::types(self.crate_root().into(), Visibility::Public, None)
                } else {
                    let def_map = db.crate_def_map(krate);
                    let module = def_map.module_id(Self::ROOT);
                    cov_mark::hit!(macro_dollar_crate_other);
                    PerNs::types(module.into(), Visibility::Public, None)
                }
            }
            PathKind::Crate => PerNs::types(self.crate_root().into(), Visibility::Public, None),
            // plain import or absolute path in 2015: crate-relative with
            // fallback to extern prelude (with the simplification in
            // rust-lang/rust#57745)
            // FIXME there must be a nicer way to write this condition
            PathKind::Plain | PathKind::Abs
                if self.data.edition == Edition::Edition2015
                    && (path.kind == PathKind::Abs || mode == ResolveMode::Import) =>
            {
                let (_, segment) = match segments.next() {
                    Some((idx, segment)) => (idx, segment),
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                tracing::debug!("resolving {:?} in crate root (+ extern prelude)", segment);
                self.resolve_name_in_crate_root_or_extern_prelude(db, original_module, segment)
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
                self.resolve_name_in_module(
                    db,
                    original_module,
                    segment,
                    prefer_module,
                    expected_macro_subns,
                )
            }
            PathKind::Super(lvl) => {
                let mut local_id = original_module;
                let mut ext;
                let mut def_map = self;

                // Adjust `local_id` to `self`, i.e. the nearest non-block module.
                if def_map.module_id(local_id).is_block_module() {
                    (ext, local_id) = adjust_to_nearest_non_block_module(db, def_map, local_id);
                    def_map = &ext;
                }

                // Go up the module tree but skip block modules as `super` always refers to the
                // nearest non-block module.
                for _ in 0..lvl {
                    // Loop invariant: at the beginning of each loop, `local_id` must refer to a
                    // non-block module.
                    if let Some(parent) = def_map.modules[local_id].parent {
                        local_id = parent;
                        if def_map.module_id(local_id).is_block_module() {
                            (ext, local_id) =
                                adjust_to_nearest_non_block_module(db, def_map, local_id);
                            def_map = &ext;
                        }
                    } else {
                        stdx::always!(def_map.block.is_none());
                        tracing::debug!("super path in root module");
                        return ResolvePathResult::empty(ReachedFixedPoint::Yes);
                    }
                }

                let module = def_map.module_id(local_id);
                stdx::never!(module.is_block_module());

                if self.block != def_map.block {
                    // If we have a different `DefMap` from `self` (the original `DefMap` we started
                    // with), resolve the remaining path segments in that `DefMap`.
                    let path =
                        ModPath::from_segments(PathKind::SELF, path.segments().iter().cloned());
                    return def_map.resolve_path_fp_with_macro(
                        db,
                        mode,
                        local_id,
                        &path,
                        shadow,
                        expected_macro_subns,
                    );
                }

                PerNs::types(module.into(), Visibility::Public, None)
            }
            PathKind::Abs => match self.resolve_path_abs(&mut segments, path) {
                Either::Left(it) => it,
                Either::Right(reached_fixed_point) => {
                    return ResolvePathResult::empty(reached_fixed_point)
                }
            },
        };

        self.resolve_remaining_segments(segments, curr_per_ns, path, db, shadow, original_module)
    }

    /// Resolves a path only in the preludes, without accounting for item scopes.
    pub(super) fn resolve_path_fp_in_all_preludes(
        &self,
        db: &dyn DefDatabase,
        mode: ResolveMode,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> ResolvePathResult {
        let mut segments = path.segments().iter().enumerate();
        let curr_per_ns = match path.kind {
            // plain import or absolute path in 2015: crate-relative with
            // fallback to extern prelude (with the simplification in
            // rust-lang/rust#57745)
            // FIXME there must be a nicer way to write this condition
            PathKind::Plain | PathKind::Abs
                if self.data.edition == Edition::Edition2015
                    && (path.kind == PathKind::Abs || mode == ResolveMode::Import) =>
            {
                let (_, segment) = match segments.next() {
                    Some((idx, segment)) => (idx, segment),
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                tracing::debug!("resolving {:?} in crate root (+ extern prelude)", segment);
                self.resolve_name_in_extern_prelude(segment)
            }
            PathKind::Plain => {
                let (_, segment) = match segments.next() {
                    Some((idx, segment)) => (idx, segment),
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                tracing::debug!("resolving {:?} in module", segment);
                self.resolve_name_in_all_preludes(db, segment)
            }
            PathKind::Abs => match self.resolve_path_abs(&mut segments, path) {
                Either::Left(it) => it,
                Either::Right(reached_fixed_point) => {
                    return ResolvePathResult::empty(reached_fixed_point)
                }
            },
            PathKind::DollarCrate(_) | PathKind::Crate | PathKind::Super(_) => {
                return ResolvePathResult::empty(ReachedFixedPoint::Yes)
            }
        };

        self.resolve_remaining_segments(segments, curr_per_ns, path, db, shadow, original_module)
    }

    /// 2018-style absolute path -- only extern prelude
    fn resolve_path_abs<'a>(
        &self,
        segments: &mut impl Iterator<Item = (usize, &'a Name)>,
        path: &ModPath,
    ) -> Either<PerNs, ReachedFixedPoint> {
        let segment = match segments.next() {
            Some((_, segment)) => segment,
            None => return Either::Right(ReachedFixedPoint::Yes),
        };
        if let Some(&(def, extern_crate)) = self.data.extern_prelude.get(segment) {
            tracing::debug!("absolute path {:?} resolved to crate {:?}", path, def);
            Either::Left(PerNs::types(
                def.into(),
                Visibility::Public,
                extern_crate.map(ImportOrExternCrate::ExternCrate),
            ))
        } else {
            Either::Right(ReachedFixedPoint::No) // extern crate declarations can add to the extern prelude
        }
    }

    fn resolve_remaining_segments<'a>(
        &self,
        segments: impl Iterator<Item = (usize, &'a Name)>,
        mut curr_per_ns: PerNs,
        path: &ModPath,
        db: &dyn DefDatabase,
        shadow: BuiltinShadowMode,
        original_module: LocalModuleId,
    ) -> ResolvePathResult {
        for (i, segment) in segments {
            let (curr, vis, imp) = match curr_per_ns.take_types_full() {
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
                            PathKind::SELF,
                            path.segments()[i..].iter().cloned(),
                        );
                        tracing::debug!("resolving {:?} in other crate", path);
                        let defp_map = module.def_map(db);
                        // Macro sub-namespaces only matter when resolving single-segment paths
                        // because `macro_use` and other preludes should be taken into account. At
                        // this point, we know we're resolving a multi-segment path so macro kind
                        // expectation is discarded.
                        let (def, s) =
                            defp_map.resolve_path(db, module.local_id, &path, shadow, None);
                        return ResolvePathResult::new(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                            true,
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
                    let def_map;

                    let loc = e.lookup(db);
                    let tree = loc.id.item_tree(db);
                    let current_def_map =
                        self.krate == loc.container.krate && self.block_id() == loc.container.block;
                    let res = if current_def_map {
                        &self.enum_definitions[&e]
                    } else {
                        def_map = loc.container.def_map(db);
                        &def_map.enum_definitions[&e]
                    }
                    .iter()
                    .find_map(|&variant| {
                        let variant_data = &tree[variant.lookup(db).id.value];
                        (variant_data.name == *segment).then(|| match variant_data.shape {
                            FieldsShape::Record => {
                                PerNs::types(variant.into(), Visibility::Public, None)
                            }
                            FieldsShape::Tuple | FieldsShape::Unit => PerNs::both(
                                variant.into(),
                                variant.into(),
                                Visibility::Public,
                                None,
                            ),
                        })
                    });
                    match res {
                        Some(res) => res,
                        None => {
                            return ResolvePathResult::new(
                                PerNs::types(e.into(), vis, imp),
                                ReachedFixedPoint::Yes,
                                Some(i),
                                false,
                            )
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

                    return ResolvePathResult::new(
                        PerNs::types(s, vis, imp),
                        ReachedFixedPoint::Yes,
                        Some(i),
                        false,
                    );
                }
            };

            curr_per_ns = curr_per_ns
                .filter_visibility(|vis| vis.is_visible_from_def_map(db, self, original_module));
        }

        ResolvePathResult::new(curr_per_ns, ReachedFixedPoint::Yes, None, false)
    }

    fn resolve_name_in_module(
        &self,
        db: &dyn DefDatabase,
        module: LocalModuleId,
        name: &Name,
        shadow: BuiltinShadowMode,
        expected_macro_subns: Option<MacroSubNs>,
    ) -> PerNs {
        // Resolve in:
        //  - legacy scope of macro
        //  - current module / scope
        //  - extern prelude / macro_use prelude
        //  - std prelude
        let from_legacy_macro = self[module]
            .scope
            .get_legacy_macro(name)
            // FIXME: shadowing
            .and_then(|it| it.last())
            .copied()
            .filter(|&id| {
                sub_namespace_match(Some(MacroSubNs::from_id(db, id)), expected_macro_subns)
            })
            .map_or_else(PerNs::none, |m| PerNs::macros(m, Visibility::Public, None));
        let from_scope = self[module].scope.get(name).filter_macro(db, expected_macro_subns);
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
            if self.block.is_some() && module == DefMap::ROOT {
                // Don't resolve extern prelude in pseudo-modules of blocks, because
                // they might been shadowed by local names.
                return PerNs::none();
            }
            self.resolve_name_in_extern_prelude(name)
        };
        let macro_use_prelude = || self.resolve_in_macro_use_prelude(name);
        let prelude = || {
            if self.block.is_some() && module == DefMap::ROOT {
                return PerNs::none();
            }
            self.resolve_in_prelude(db, name)
        };

        from_legacy_macro
            .or(from_scope_or_builtin)
            .or_else(extern_prelude)
            .or_else(macro_use_prelude)
            .or_else(prelude)
    }

    fn resolve_name_in_all_preludes(&self, db: &dyn DefDatabase, name: &Name) -> PerNs {
        // Resolve in:
        //  - extern prelude / macro_use prelude
        //  - std prelude
        let extern_prelude = self.resolve_name_in_extern_prelude(name);
        let macro_use_prelude = || self.resolve_in_macro_use_prelude(name);
        let prelude = || self.resolve_in_prelude(db, name);

        extern_prelude.or_else(macro_use_prelude).or_else(prelude)
    }

    fn resolve_name_in_extern_prelude(&self, name: &Name) -> PerNs {
        self.data.extern_prelude.get(name).map_or(PerNs::none(), |&(it, extern_crate)| {
            PerNs::types(
                it.into(),
                Visibility::Public,
                extern_crate.map(ImportOrExternCrate::ExternCrate),
            )
        })
    }

    fn resolve_in_macro_use_prelude(&self, name: &Name) -> PerNs {
        self.macro_use_prelude.get(name).map_or(PerNs::none(), |&(it, _extern_crate)| {
            PerNs::macros(
                it,
                Visibility::Public,
                // FIXME?
                None, // extern_crate.map(ImportOrExternCrate::ExternCrate),
            )
        })
    }

    fn resolve_name_in_crate_root_or_extern_prelude(
        &self,
        db: &dyn DefDatabase,
        module: LocalModuleId,
        name: &Name,
    ) -> PerNs {
        let from_crate_root = match self.block {
            Some(_) => {
                let def_map = self.crate_root().def_map(db);
                def_map[Self::ROOT].scope.get(name)
            }
            None => self[Self::ROOT].scope.get(name),
        };
        let from_extern_prelude = || {
            if self.block.is_some() && module == DefMap::ROOT {
                // Don't resolve extern prelude in pseudo-module of a block.
                return PerNs::none();
            }
            self.resolve_name_in_extern_prelude(name)
        };

        from_crate_root.or_else(from_extern_prelude)
    }

    fn resolve_in_prelude(&self, db: &dyn DefDatabase, name: &Name) -> PerNs {
        if let Some((prelude, _use)) = self.prelude {
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

/// Given a block module, returns its nearest non-block module and the `DefMap` it belongs to.
fn adjust_to_nearest_non_block_module(
    db: &dyn DefDatabase,
    def_map: &DefMap,
    mut local_id: LocalModuleId,
) -> (Arc<DefMap>, LocalModuleId) {
    // INVARIANT: `local_id` in `def_map` must be a block module.
    stdx::always!(def_map.module_id(local_id).is_block_module());

    let mut ext;
    // This needs to be a local variable due to our mighty lifetime.
    let mut def_map = def_map;
    loop {
        let BlockInfo { parent, .. } = def_map.block.expect("block module without parent module");

        ext = parent.def_map(db, def_map.krate);
        def_map = &ext;
        local_id = parent.local_id;
        if !parent.is_block_module() {
            return (ext, local_id);
        }
    }
}
