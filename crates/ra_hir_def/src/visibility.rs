//! Defines hir-level representation of visibility (e.g. `pub` and `pub(crate)`).

use std::sync::Arc;

use either::Either;

use hir_expand::InFile;
use ra_syntax::ast::{self, VisibilityOwner};

use crate::{
    db::DefDatabase,
    path::{ModPath, PathKind},
    src::{HasChildSource, HasSource},
    AdtId, Lookup, ModuleId, VisibilityDefId,
};

/// Visibility of an item, not yet resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Visibility {
    // FIXME: We could avoid the allocation in many cases by special-casing
    // pub(crate), pub(super) and private. Alternatively, `ModPath` could be
    // made to contain an Arc<[Segment]> instead of a Vec?
    /// `pub(in module)`, `pub(crate)` or `pub(super)`. Also private, which is
    /// equivalent to `pub(self)`.
    Module(Arc<ModPath>),
    /// `pub`.
    Public,
}

impl Visibility {
    pub(crate) fn visibility_query(db: &impl DefDatabase, def: VisibilityDefId) -> Visibility {
        match def {
            VisibilityDefId::ModuleId(module) => {
                let def_map = db.crate_def_map(module.krate);
                let src = match def_map[module.local_id].declaration_source(db) {
                    Some(it) => it,
                    None => return Visibility::private(),
                };
                Visibility::from_ast(db, src.map(|it| it.visibility()))
            }
            VisibilityDefId::StructFieldId(it) => {
                let src = it.parent.child_source(db);
                let is_enum = match it.parent {
                    crate::VariantId::EnumVariantId(_) => true,
                    _ => false,
                };
                let vis_node = src.map(|m| match &m[it.local_id] {
                    Either::Left(tuple) => tuple.visibility(),
                    Either::Right(record) => record.visibility(),
                });
                if vis_node.value.is_none() && is_enum {
                    Visibility::Public
                } else {
                    Visibility::from_ast(db, vis_node)
                }
            }
            VisibilityDefId::AdtId(it) => match it {
                AdtId::StructId(it) => visibility_from_loc(it.lookup(db), db),
                AdtId::EnumId(it) => visibility_from_loc(it.lookup(db), db),
                AdtId::UnionId(it) => visibility_from_loc(it.lookup(db), db),
            },
            VisibilityDefId::TraitId(it) => visibility_from_loc(it.lookup(db), db),
            VisibilityDefId::ConstId(it) => visibility_from_loc(it.lookup(db), db),
            VisibilityDefId::StaticId(it) => visibility_from_loc(it.lookup(db), db),
            VisibilityDefId::FunctionId(it) => visibility_from_loc(it.lookup(db), db),
            VisibilityDefId::TypeAliasId(it) => visibility_from_loc(it.lookup(db), db),
        }
    }

    fn private() -> Visibility {
        let path = ModPath { kind: PathKind::Super(0), segments: Vec::new() };
        Visibility::Module(Arc::new(path))
    }

    fn from_ast(db: &impl DefDatabase, node: InFile<Option<ast::Visibility>>) -> Visibility {
        let file_id = node.file_id;
        let node = match node.value {
            None => return Visibility::private(),
            Some(node) => node,
        };
        match node.kind() {
            ast::VisibilityKind::In(path) => {
                let path = ModPath::from_src(path, &hir_expand::hygiene::Hygiene::new(db, file_id));
                let path = match path {
                    None => return Visibility::private(),
                    Some(path) => path,
                };
                Visibility::Module(Arc::new(path))
            }
            ast::VisibilityKind::PubCrate => {
                let path = ModPath { kind: PathKind::Crate, segments: Vec::new() };
                Visibility::Module(Arc::new(path))
            }
            ast::VisibilityKind::PubSuper => {
                let path = ModPath { kind: PathKind::Super(1), segments: Vec::new() };
                Visibility::Module(Arc::new(path))
            }
            ast::VisibilityKind::Pub => Visibility::Public,
        }
    }

    pub fn resolve(
        &self,
        db: &impl DefDatabase,
        resolver: &crate::resolver::Resolver,
    ) -> ResolvedVisibility {
        // we fall back to public visibility (i.e. fail open) if the path can't be resolved
        resolver.resolve_visibility(db, self).unwrap_or(ResolvedVisibility::Public)
    }
}

/// Visibility of an item, with the path resolved.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResolvedVisibility {
    /// Visibility is restricted to a certain module.
    Module(ModuleId),
    /// Visibility is unrestricted.
    Public,
}

impl ResolvedVisibility {
    pub fn visible_from(self, db: &impl DefDatabase, from_module: ModuleId) -> bool {
        let to_module = match self {
            ResolvedVisibility::Module(m) => m,
            ResolvedVisibility::Public => return true,
        };
        // if they're not in the same crate, it can't be visible
        if from_module.krate != to_module.krate {
            return false;
        }
        // from_module needs to be a descendant of to_module
        let def_map = db.crate_def_map(from_module.krate);
        let mut ancestors = std::iter::successors(Some(from_module), |m| {
            let parent_id = def_map[m.local_id].parent?;
            Some(ModuleId { local_id: parent_id, ..*m })
        });
        ancestors.any(|m| m == to_module)
    }
}

fn visibility_from_loc<T>(node: T, db: &impl DefDatabase) -> Visibility
where
    T: HasSource,
    T::Value: ast::VisibilityOwner,
{
    let src = node.source(db);
    Visibility::from_ast(db, src.map(|n| n.visibility()))
}
