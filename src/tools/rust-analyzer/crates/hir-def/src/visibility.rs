//! Defines hir-level representation of visibility (e.g. `pub` and `pub(crate)`).

use std::{iter, sync::Arc};

use hir_expand::{hygiene::Hygiene, InFile};
use la_arena::ArenaMap;
use syntax::ast;

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
    Module(ModPath),
    /// `pub`.
    Public,
}

impl RawVisibility {
    pub(crate) const fn private() -> RawVisibility {
        RawVisibility::Module(ModPath::from_kind(PathKind::Super(0)))
    }

    pub(crate) fn from_ast(
        db: &dyn DefDatabase,
        node: InFile<Option<ast::Visibility>>,
    ) -> RawVisibility {
        Self::from_ast_with_hygiene(db, node.value, &Hygiene::new(db.upcast(), node.file_id))
    }

    pub(crate) fn from_ast_with_hygiene(
        db: &dyn DefDatabase,
        node: Option<ast::Visibility>,
        hygiene: &Hygiene,
    ) -> RawVisibility {
        Self::from_ast_with_hygiene_and_default(db, node, RawVisibility::private(), hygiene)
    }

    pub(crate) fn from_ast_with_hygiene_and_default(
        db: &dyn DefDatabase,
        node: Option<ast::Visibility>,
        default: RawVisibility,
        hygiene: &Hygiene,
    ) -> RawVisibility {
        let node = match node {
            None => return default,
            Some(node) => node,
        };
        match node.kind() {
            ast::VisibilityKind::In(path) => {
                let path = ModPath::from_src(db.upcast(), path, hygiene);
                let path = match path {
                    None => return RawVisibility::private(),
                    Some(path) => path,
                };
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::PubCrate => {
                let path = ModPath::from_kind(PathKind::Crate);
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::PubSuper => {
                let path = ModPath::from_kind(PathKind::Super(1));
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::PubSelf => {
                let path = ModPath::from_kind(PathKind::Plain);
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::Pub => RawVisibility::Public,
        }
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
    Module(ModuleId),
    /// Visibility is unrestricted.
    Public,
}

impl Visibility {
    pub fn is_visible_from(self, db: &dyn DefDatabase, from_module: ModuleId) -> bool {
        let to_module = match self {
            Visibility::Module(m) => m,
            Visibility::Public => return true,
        };
        // if they're not in the same crate, it can't be visible
        if from_module.krate != to_module.krate {
            return false;
        }
        let def_map = from_module.def_map(db);
        self.is_visible_from_def_map(db, &def_map, from_module.local_id)
    }

    pub(crate) fn is_visible_from_other_crate(self) -> bool {
        matches!(self, Visibility::Public)
    }

    pub(crate) fn is_visible_from_def_map(
        self,
        db: &dyn DefDatabase,
        def_map: &DefMap,
        mut from_module: LocalModuleId,
    ) -> bool {
        let mut to_module = match self {
            Visibility::Module(m) => m,
            Visibility::Public => return true,
        };

        // `to_module` might be the root module of a block expression. Those have the same
        // visibility as the containing module (even though no items are directly nameable from
        // there, getting this right is important for method resolution).
        // In that case, we adjust the visibility of `to_module` to point to the containing module.

        // Additional complication: `to_module` might be in `from_module`'s `DefMap`, which we're
        // currently computing, so we must not call the `def_map` query for it.
        let mut arc;
        loop {
            let to_module_def_map =
                if to_module.krate == def_map.krate() && to_module.block == def_map.block_id() {
                    cov_mark::hit!(is_visible_from_same_block_def_map);
                    def_map
                } else {
                    arc = to_module.def_map(db);
                    &arc
                };
            match to_module_def_map.parent() {
                Some(parent) => to_module = parent,
                None => break,
            }
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
            (Visibility::Module(_) | Visibility::Public, Visibility::Public)
            | (Visibility::Public, Visibility::Module(_)) => Some(Visibility::Public),
            (Visibility::Module(mod_a), Visibility::Module(mod_b)) => {
                if mod_a.krate != mod_b.krate {
                    return None;
                }

                let mut a_ancestors = iter::successors(Some(mod_a.local_id), |&m| {
                    let parent_id = def_map[m].parent?;
                    Some(parent_id)
                });
                let mut b_ancestors = iter::successors(Some(mod_b.local_id), |&m| {
                    let parent_id = def_map[m].parent?;
                    Some(parent_id)
                });

                if a_ancestors.any(|m| m == mod_b.local_id) {
                    // B is above A
                    return Some(Visibility::Module(mod_b));
                }

                if b_ancestors.any(|m| m == mod_a.local_id) {
                    // A is above B
                    return Some(Visibility::Module(mod_a));
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
    let var_data = match variant_id {
        VariantId::StructId(it) => db.struct_data(it).variant_data.clone(),
        VariantId::UnionId(it) => db.union_data(it).variant_data.clone(),
        VariantId::EnumVariantId(it) => {
            db.enum_data(it.parent).variants[it.local_id].variant_data.clone()
        }
    };
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
