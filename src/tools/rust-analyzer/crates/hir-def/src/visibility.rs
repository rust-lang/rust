//! Defines hir-level representation of visibility (e.g. `pub` and `pub(crate)`).

use std::iter;

use intern::Interned;
use la_arena::ArenaMap;
use span::SyntaxContextId;
use syntax::ast;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    nameres::DefMap,
    path::{ModPath, PathKind},
    resolver::HasResolver,
    ConstId, FunctionId, HasModule, LocalFieldId, LocalModuleId, ModuleId, VariantId,
};

/// Visibility of an item, not yet resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawVisibility {
    /// `pub(in module)`, `pub(crate)` or `pub(super)`. Also private, which is
    /// equivalent to `pub(self)`.
    Module(Interned<ModPath>, VisibilityExplicitness),
    /// `pub`.
    Public,
}

impl RawVisibility {
    pub(crate) fn private() -> RawVisibility {
        RawVisibility::Module(
            Interned::new(ModPath::from_kind(PathKind::SELF)),
            VisibilityExplicitness::Implicit,
        )
    }

    pub(crate) fn from_ast(
        db: &dyn DefDatabase,
        node: Option<ast::Visibility>,
        span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContextId,
    ) -> RawVisibility {
        let node = match node {
            None => return RawVisibility::private(),
            Some(node) => node,
        };
        Self::from_ast_with_span_map(db, node, span_for_range)
    }

    fn from_ast_with_span_map(
        db: &dyn DefDatabase,
        node: ast::Visibility,
        span_for_range: &mut dyn FnMut(::tt::TextRange) -> SyntaxContextId,
    ) -> RawVisibility {
        let path = match node.kind() {
            ast::VisibilityKind::In(path) => {
                let path = ModPath::from_src(db.upcast(), path, span_for_range);
                match path {
                    None => return RawVisibility::private(),
                    Some(path) => path,
                }
            }
            ast::VisibilityKind::PubCrate => ModPath::from_kind(PathKind::Crate),
            ast::VisibilityKind::PubSuper => ModPath::from_kind(PathKind::Super(1)),
            ast::VisibilityKind::PubSelf => ModPath::from_kind(PathKind::SELF),
            ast::VisibilityKind::Pub => return RawVisibility::Public,
        };
        RawVisibility::Module(Interned::new(path), VisibilityExplicitness::Explicit)
    }

    pub fn resolve(
        &self,
        db: &dyn DefDatabase,
        resolver: &crate::resolver::Resolver,
    ) -> Visibility {
        // we fall back to public visibility (i.e. fail open) if the path can't be resolved
        resolver.resolve_visibility(db, self).unwrap_or(Visibility::Public)
    }
}

/// Visibility of an item, with the path resolved.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Visibility is restricted to a certain module.
    Module(ModuleId, VisibilityExplicitness),
    /// Visibility is unrestricted.
    Public,
}

impl Visibility {
    pub(crate) fn is_visible_from_other_crate(self) -> bool {
        matches!(self, Visibility::Public)
    }

    #[tracing::instrument(skip_all)]
    pub fn is_visible_from(self, db: &dyn DefDatabase, from_module: ModuleId) -> bool {
        let to_module = match self {
            Visibility::Module(m, _) => m,
            Visibility::Public => return true,
        };
        // if they're not in the same crate, it can't be visible
        if from_module.krate != to_module.krate {
            return false;
        }
        let def_map = from_module.def_map(db);
        Self::is_visible_from_def_map_(db, &def_map, to_module, from_module.local_id)
    }

    pub(crate) fn is_visible_from_def_map(
        self,
        db: &dyn DefDatabase,
        def_map: &DefMap,
        from_module: LocalModuleId,
    ) -> bool {
        let to_module = match self {
            Visibility::Module(m, _) => m,
            Visibility::Public => return true,
        };
        // if they're not in the same crate, it can't be visible
        if def_map.krate() != to_module.krate {
            return false;
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
                (Some(a), Some(b)) if a == b => {
                    cov_mark::hit!(is_visible_from_same_block_def_map);
                }
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
                            def_map = &*parent_arc;
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
            (Visibility::Module(mod_a, expl_a), Visibility::Module(mod_b, expl_b)) => {
                if mod_a.krate != mod_b.krate {
                    return None;
                }

                let def_block = def_map.block_id();
                if (mod_a.containing_block(), mod_b.containing_block()) != (def_block, def_block) {
                    return None;
                }

                let mut a_ancestors =
                    iter::successors(Some(mod_a.local_id), |&m| def_map[m].parent);
                let mut b_ancestors =
                    iter::successors(Some(mod_b.local_id), |&m| def_map[m].parent);

                if a_ancestors.any(|m| m == mod_b.local_id) {
                    // B is above A
                    return Some(Visibility::Module(mod_b, expl_b));
                }

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
            (Visibility::Module(mod_a, expl_a), Visibility::Module(mod_b, expl_b)) => {
                if mod_a.krate != mod_b.krate {
                    return None;
                }

                let def_block = def_map.block_id();
                if (mod_a.containing_block(), mod_b.containing_block()) != (def_block, def_block) {
                    return None;
                }

                let mut a_ancestors =
                    iter::successors(Some(mod_a.local_id), |&m| def_map[m].parent);
                let mut b_ancestors =
                    iter::successors(Some(mod_b.local_id), |&m| def_map[m].parent);

                if a_ancestors.any(|m| m == mod_b.local_id) {
                    // B is above A
                    return Some(Visibility::Module(mod_a, expl_b));
                }

                if b_ancestors.any(|m| m == mod_a.local_id) {
                    // A is above B
                    return Some(Visibility::Module(mod_b, expl_a));
                }

                None
            }
        }
    }
}

/// Whether the item was imported through an explicit `pub(crate) use` or just a `use` without
/// visibility.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum VisibilityExplicitness {
    Explicit,
    Implicit,
}

impl VisibilityExplicitness {
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit)
    }
}

/// Resolve visibility of all specific fields of a struct or union variant.
pub(crate) fn field_visibilities_query(
    db: &dyn DefDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Visibility>> {
    let var_data = variant_id.variant_data(db);
    let resolver = variant_id.module(db).resolver(db);
    let mut res = ArenaMap::default();
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, field_data.visibility.resolve(db, &resolver));
    }
    Arc::new(res)
}

/// Resolve visibility of a function.
pub(crate) fn function_visibility_query(db: &dyn DefDatabase, def: FunctionId) -> Visibility {
    let resolver = def.resolver(db);
    db.function_data(def).visibility.resolve(db, &resolver)
}

/// Resolve visibility of a const.
pub(crate) fn const_visibility_query(db: &dyn DefDatabase, def: ConstId) -> Visibility {
    let resolver = def.resolver(db);
    db.const_data(def).visibility.resolve(db, &resolver)
}
