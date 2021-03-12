//! Utilities for mapping between hir IDs and the surface syntax.

use hir_expand::InFile;
use la_arena::ArenaMap;

use crate::{db::DefDatabase, item_tree::ItemTreeNode, AssocItemLoc, ItemLoc};

pub trait HasSource {
    type Value;
    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value>;
}

impl<N: ItemTreeNode> HasSource for AssocItemLoc<N> {
    type Value = N::Source;

    fn source(&self, db: &dyn DefDatabase) -> InFile<N::Source> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id()).unwrap();
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

impl<N: ItemTreeNode> HasSource for ItemLoc<N> {
    type Value = N::Source;

    fn source(&self, db: &dyn DefDatabase) -> InFile<N::Source> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id()).unwrap();
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

pub trait HasChildSource<ChildId> {
    type Value;
    fn child_source(&self, db: &dyn DefDatabase) -> InFile<ArenaMap<ChildId, Self::Value>>;
}
