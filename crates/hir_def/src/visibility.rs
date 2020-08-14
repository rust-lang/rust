//! Defines hir-level representation of visibility (e.g. `pub` and `pub(crate)`).

use hir_expand::{hygiene::Hygiene, InFile};
use syntax::ast;

use crate::{
    db::DefDatabase,
    nameres::CrateDefMap,
    path::{ModPath, PathKind},
    ModuleId,
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
        let path = ModPath { kind: PathKind::Super(0), segments: Vec::new() };
        RawVisibility::Module(path)
    }

    pub(crate) fn from_ast(
        db: &dyn DefDatabase,
        node: InFile<Option<ast::Visibility>>,
    ) -> RawVisibility {
        Self::from_ast_with_hygiene(node.value, &Hygiene::new(db.upcast(), node.file_id))
    }

    pub(crate) fn from_ast_with_hygiene(
        node: Option<ast::Visibility>,
        hygiene: &Hygiene,
    ) -> RawVisibility {
        Self::from_ast_with_hygiene_and_default(node, RawVisibility::private(), hygiene)
    }

    pub(crate) fn from_ast_with_hygiene_and_default(
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
                let path = ModPath::from_src(path, hygiene);
                let path = match path {
                    None => return RawVisibility::private(),
                    Some(path) => path,
                };
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::PubCrate => {
                let path = ModPath { kind: PathKind::Crate, segments: Vec::new() };
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::PubSuper => {
                let path = ModPath { kind: PathKind::Super(1), segments: Vec::new() };
                RawVisibility::Module(path)
            }
            ast::VisibilityKind::PubSelf => {
                let path = ModPath { kind: PathKind::Plain, segments: Vec::new() };
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
        let def_map = db.crate_def_map(from_module.krate);
        self.is_visible_from_def_map(&def_map, from_module.local_id)
    }

    pub(crate) fn is_visible_from_other_crate(self) -> bool {
        match self {
            Visibility::Module(_) => false,
            Visibility::Public => true,
        }
    }

    pub(crate) fn is_visible_from_def_map(
        self,
        def_map: &CrateDefMap,
        from_module: crate::LocalModuleId,
    ) -> bool {
        let to_module = match self {
            Visibility::Module(m) => m,
            Visibility::Public => return true,
        };
        // from_module needs to be a descendant of to_module
        let mut ancestors = std::iter::successors(Some(from_module), |m| {
            let parent_id = def_map[*m].parent?;
            Some(parent_id)
        });
        ancestors.any(|m| m == to_module.local_id)
    }

    /// Returns the most permissive visibility of `self` and `other`.
    ///
    /// If there is no subset relation between `self` and `other`, returns `None` (ie. they're only
    /// visible in unrelated modules).
    pub(crate) fn max(self, other: Visibility, def_map: &CrateDefMap) -> Option<Visibility> {
        match (self, other) {
            (Visibility::Module(_), Visibility::Public)
            | (Visibility::Public, Visibility::Module(_))
            | (Visibility::Public, Visibility::Public) => Some(Visibility::Public),
            (Visibility::Module(mod_a), Visibility::Module(mod_b)) => {
                if mod_a.krate != mod_b.krate {
                    return None;
                }

                let mut a_ancestors = std::iter::successors(Some(mod_a.local_id), |m| {
                    let parent_id = def_map[*m].parent?;
                    Some(parent_id)
                });
                let mut b_ancestors = std::iter::successors(Some(mod_b.local_id), |m| {
                    let parent_id = def_map[*m].parent?;
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
