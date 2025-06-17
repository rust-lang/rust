//! Defines hir-level representation of visibility (e.g. `pub` and `pub(crate)`).

use std::iter;

use base_db::Crate;
use hir_expand::{InFile, Lookup};
use la_arena::ArenaMap;
use syntax::ast::{self, HasVisibility};
use triomphe::Arc;

use crate::{
    AssocItemId, HasModule, ItemContainerId, LocalFieldId, LocalModuleId, ModuleId, TraitId,
    VariantId, db::DefDatabase, nameres::DefMap, resolver::HasResolver, src::HasSource,
};

pub use crate::item_tree::{RawVisibility, VisibilityExplicitness};

/// Visibility of an item, with the path resolved.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Visibility is restricted to a certain module.
    Module(ModuleId, VisibilityExplicitness),
    /// Visibility is restricted to the crate.
    PubCrate(Crate),
    /// Visibility is unrestricted.
    Public,
}

impl Visibility {
    pub fn resolve(
        db: &dyn DefDatabase,
        resolver: &crate::resolver::Resolver<'_>,
        raw_vis: &RawVisibility,
    ) -> Self {
        // we fall back to public visibility (i.e. fail open) if the path can't be resolved
        resolver.resolve_visibility(db, raw_vis).unwrap_or(Visibility::Public)
    }

    pub(crate) fn is_visible_from_other_crate(self) -> bool {
        matches!(self, Visibility::Public)
    }

    #[tracing::instrument(skip_all)]
    pub fn is_visible_from(self, db: &dyn DefDatabase, from_module: ModuleId) -> bool {
        let to_module = match self {
            Visibility::Module(m, _) => m,
            Visibility::PubCrate(krate) => return from_module.krate == krate,
            Visibility::Public => return true,
        };
        if from_module == to_module {
            // if the modules are the same, visibility is trivially satisfied
            return true;
        }
        // if they're not in the same crate, it can't be visible
        if from_module.krate != to_module.krate {
            return false;
        }
        let def_map = from_module.def_map(db);
        Self::is_visible_from_def_map_(db, def_map, to_module, from_module.local_id)
    }

    pub(crate) fn is_visible_from_def_map(
        self,
        db: &dyn DefDatabase,
        def_map: &DefMap,
        from_module: LocalModuleId,
    ) -> bool {
        let to_module = match self {
            Visibility::Module(m, _) => m,
            Visibility::PubCrate(krate) => return def_map.krate() == krate,
            Visibility::Public => return true,
        };
        // if they're not in the same crate, it can't be visible
        if def_map.krate() != to_module.krate {
            return false;
        }

        if from_module == to_module.local_id && def_map.block_id() == to_module.block {
            // if the modules are the same, visibility is trivially satisfied
            return true;
        }
        Self::is_visible_from_def_map_(db, def_map, to_module, from_module)
    }

    fn is_visible_from_def_map_(
        db: &dyn DefDatabase,
        def_map: &DefMap,
        mut to_module: ModuleId,
        mut from_module: LocalModuleId,
    ) -> bool {
        debug_assert_eq!(to_module.krate, def_map.krate());
        // `to_module` might be the root module of a block expression. Those have the same
        // visibility as the containing module (even though no items are directly nameable from
        // there, getting this right is important for method resolution).
        // In that case, we adjust the visibility of `to_module` to point to the containing module.

        // Additional complication: `to_module` might be in `from_module`'s `DefMap`, which we're
        // currently computing, so we must not call the `def_map` query for it.
        let def_map_block = def_map.block_id();
        loop {
            match (to_module.block, def_map_block) {
                // `to_module` is not a block, so there is no parent def map to use.
                (None, _) => (),
                // `to_module` is at `def_map`'s block, no need to move further.
                (Some(a), Some(b)) if a == b => {}
                _ => {
                    if let Some(parent) = to_module.def_map(db).parent() {
                        to_module = parent;
                        continue;
                    }
                }
            }
            break;
        }

        // from_module needs to be a descendant of to_module
        let mut def_map = def_map;
        let mut parent_arc;
        loop {
            if def_map.module_id(from_module) == to_module {
                return true;
            }
            match def_map[from_module].parent {
                Some(parent) => from_module = parent,
                None => {
                    match def_map.parent() {
                        Some(module) => {
                            parent_arc = module.def_map(db);
                            def_map = parent_arc;
                            from_module = module.local_id;
                        }
                        // Reached the root module, nothing left to check.
                        None => return false,
                    }
                }
            }
        }
    }

    /// Returns the most permissive visibility of `self` and `other`.
    ///
    /// If there is no subset relation between `self` and `other`, returns `None` (ie. they're only
    /// visible in unrelated modules).
    pub(crate) fn max(self, other: Visibility, def_map: &DefMap) -> Option<Visibility> {
        match (self, other) {
            (_, Visibility::Public) | (Visibility::Public, _) => Some(Visibility::Public),
            (Visibility::PubCrate(krate), Visibility::PubCrate(krateb)) => {
                if krate == krateb {
                    Some(Visibility::PubCrate(krate))
                } else {
                    None
                }
            }
            (Visibility::Module(mod_, _), Visibility::PubCrate(krate))
            | (Visibility::PubCrate(krate), Visibility::Module(mod_, _)) => {
                if mod_.krate == krate {
                    Some(Visibility::PubCrate(krate))
                } else {
                    None
                }
            }
            (Visibility::Module(mod_a, expl_a), Visibility::Module(mod_b, expl_b)) => {
                if mod_a == mod_b {
                    // Most module visibilities are `pub(self)`, and assuming no errors
                    // this will be the common and thus fast path.
                    return Some(Visibility::Module(
                        mod_a,
                        match (expl_a, expl_b) {
                            (VisibilityExplicitness::Explicit, _)
                            | (_, VisibilityExplicitness::Explicit) => {
                                VisibilityExplicitness::Explicit
                            }
                            _ => VisibilityExplicitness::Implicit,
                        },
                    ));
                }

                if mod_a.krate() != def_map.krate() || mod_b.krate() != def_map.krate() {
                    return None;
                }

                let def_block = def_map.block_id();
                if mod_a.containing_block() != def_block || mod_b.containing_block() != def_block {
                    return None;
                }

                let mut a_ancestors =
                    iter::successors(Some(mod_a.local_id), |&m| def_map[m].parent);

                if a_ancestors.any(|m| m == mod_b.local_id) {
                    // B is above A
                    return Some(Visibility::Module(mod_b, expl_b));
                }

                let mut b_ancestors =
                    iter::successors(Some(mod_b.local_id), |&m| def_map[m].parent);
                if b_ancestors.any(|m| m == mod_a.local_id) {
                    // A is above B
                    return Some(Visibility::Module(mod_a, expl_a));
                }

                None
            }
        }
    }

    /// Returns the least permissive visibility of `self` and `other`.
    ///
    /// If there is no subset relation between `self` and `other`, returns `None` (ie. they're only
    /// visible in unrelated modules).
    pub(crate) fn min(self, other: Visibility, def_map: &DefMap) -> Option<Visibility> {
        match (self, other) {
            (vis, Visibility::Public) | (Visibility::Public, vis) => Some(vis),
            (Visibility::PubCrate(krate), Visibility::PubCrate(krateb)) => {
                if krate == krateb {
                    Some(Visibility::PubCrate(krate))
                } else {
                    None
                }
            }
            (Visibility::Module(mod_, exp), Visibility::PubCrate(krate))
            | (Visibility::PubCrate(krate), Visibility::Module(mod_, exp)) => {
                if mod_.krate == krate { Some(Visibility::Module(mod_, exp)) } else { None }
            }
            (Visibility::Module(mod_a, expl_a), Visibility::Module(mod_b, expl_b)) => {
                if mod_a == mod_b {
                    // Most module visibilities are `pub(self)`, and assuming no errors
                    // this will be the common and thus fast path.
                    return Some(Visibility::Module(
                        mod_a,
                        match (expl_a, expl_b) {
                            (VisibilityExplicitness::Explicit, _)
                            | (_, VisibilityExplicitness::Explicit) => {
                                VisibilityExplicitness::Explicit
                            }
                            _ => VisibilityExplicitness::Implicit,
                        },
                    ));
                }

                if mod_a.krate() != def_map.krate() || mod_b.krate() != def_map.krate() {
                    return None;
                }

                let def_block = def_map.block_id();
                if mod_a.containing_block() != def_block || mod_b.containing_block() != def_block {
                    return None;
                }

                let mut a_ancestors =
                    iter::successors(Some(mod_a.local_id), |&m| def_map[m].parent);

                if a_ancestors.any(|m| m == mod_b.local_id) {
                    // B is above A
                    return Some(Visibility::Module(mod_a, expl_a));
                }

                let mut b_ancestors =
                    iter::successors(Some(mod_b.local_id), |&m| def_map[m].parent);
                if b_ancestors.any(|m| m == mod_a.local_id) {
                    // A is above B
                    return Some(Visibility::Module(mod_b, expl_b));
                }

                None
            }
        }
    }
}

/// Resolve visibility of all specific fields of a struct or union variant.
pub(crate) fn field_visibilities_query(
    db: &dyn DefDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Visibility>> {
    let variant_fields = db.variant_fields(variant_id);
    let fields = variant_fields.fields();
    if fields.is_empty() {
        return Arc::default();
    }
    let resolver = variant_id.module(db).resolver(db);
    let mut res = ArenaMap::default();
    for (field_id, field_data) in fields.iter() {
        res.insert(field_id, Visibility::resolve(db, &resolver, &field_data.visibility));
    }
    res.shrink_to_fit();
    Arc::new(res)
}

pub fn visibility_from_ast(
    db: &dyn DefDatabase,
    has_resolver: impl HasResolver,
    ast_vis: InFile<Option<ast::Visibility>>,
) -> Visibility {
    let mut span_map = None;
    let raw_vis = crate::item_tree::visibility_from_ast(db, ast_vis.value, &mut |range| {
        span_map.get_or_insert_with(|| db.span_map(ast_vis.file_id)).span_for_range(range).ctx
    });
    if raw_vis == RawVisibility::Public {
        return Visibility::Public;
    }

    Visibility::resolve(db, &has_resolver.resolver(db), &raw_vis)
}

/// Resolve visibility of a type alias.
pub(crate) fn assoc_visibility_query(db: &dyn DefDatabase, def: AssocItemId) -> Visibility {
    match def {
        AssocItemId::FunctionId(function_id) => {
            let loc = function_id.lookup(db);
            trait_item_visibility(db, loc.container).unwrap_or_else(|| {
                let source = loc.source(db);
                visibility_from_ast(db, function_id, source.map(|src| src.visibility()))
            })
        }
        AssocItemId::ConstId(const_id) => {
            let loc = const_id.lookup(db);
            trait_item_visibility(db, loc.container).unwrap_or_else(|| {
                let source = loc.source(db);
                visibility_from_ast(db, const_id, source.map(|src| src.visibility()))
            })
        }
        AssocItemId::TypeAliasId(type_alias_id) => {
            let loc = type_alias_id.lookup(db);
            trait_item_visibility(db, loc.container).unwrap_or_else(|| {
                let source = loc.source(db);
                visibility_from_ast(db, type_alias_id, source.map(|src| src.visibility()))
            })
        }
    }
}

fn trait_item_visibility(db: &dyn DefDatabase, container: ItemContainerId) -> Option<Visibility> {
    match container {
        ItemContainerId::TraitId(trait_) => Some(trait_visibility(db, trait_)),
        _ => None,
    }
}

fn trait_visibility(db: &dyn DefDatabase, def: TraitId) -> Visibility {
    let loc = def.lookup(db);
    let source = loc.source(db);
    visibility_from_ast(db, def, source.map(|src| src.visibility()))
}
