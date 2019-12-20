//! Utilities for mapping between hir IDs and the surface syntax.

use hir_expand::InFile;
use ra_arena::map::ArenaMap;
use ra_syntax::AstNode;

use crate::{db::DefDatabase, AssocItemLoc, ItemLoc};

pub trait HasSource {
    type Value;
    fn source(&self, db: &impl DefDatabase) -> InFile<Self::Value>;
}

impl<N: AstNode> HasSource for AssocItemLoc<N> {
    type Value = N;

    fn source(&self, db: &impl DefDatabase) -> InFile<N> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl<N: AstNode> HasSource for ItemLoc<N> {
    type Value = N;

    fn source(&self, db: &impl DefDatabase) -> InFile<N> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

pub trait HasChildSource {
    type ChildId;
    type Value;
    fn child_source(&self, db: &impl DefDatabase) -> InFile<ArenaMap<Self::ChildId, Self::Value>>;
}
