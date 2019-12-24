use std::sync::Arc;

use either::Either;

use hir_expand::InFile;
use ra_syntax::ast::{self, VisibilityOwner};

use crate::{
    db::DefDatabase,
    path::{ModPath, PathKind},
    src::{HasChildSource, HasSource},
    AdtId, Lookup, VisibilityDefId,
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
                // TODO: enum variant fields should be public by default
                let vis_node = src.map(|m| match &m[it.local_id] {
                    Either::Left(tuple) => tuple.visibility(),
                    Either::Right(record) => record.visibility(),
                });
                Visibility::from_ast(db, vis_node)
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
}

fn visibility_from_loc<T>(node: T, db: &impl DefDatabase) -> Visibility
where
    T: HasSource,
    T::Value: ast::VisibilityOwner,
{
    let src = node.source(db);
    Visibility::from_ast(db, src.map(|n| n.visibility()))
}
