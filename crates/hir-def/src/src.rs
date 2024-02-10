//! Utilities for mapping between hir IDs and the surface syntax.

use hir_expand::InFile;
use la_arena::ArenaMap;
use syntax::ast;

use crate::{db::DefDatabase, item_tree::ItemTreeNode, ItemTreeLoc, Lookup, UseId};

pub trait HasSource {
    type Value;
    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value>;
}

impl<T> HasSource for T
where
    T: ItemTreeLoc,
    T::Id: ItemTreeNode,
{
    type Value = <T::Id as ItemTreeNode>::Source;

    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value> {
        let id = self.item_tree_id();
        let file_id = id.file_id();
        let tree = id.item_tree(db);
        let ast_id_map = db.ast_id_map(file_id);
        let root = db.parse_or_expand(file_id);
        let node = &tree[id.value];

        InFile::new(file_id, ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

pub trait HasChildSource<ChildId> {
    type Value;
    fn child_source(&self, db: &dyn DefDatabase) -> InFile<ArenaMap<ChildId, Self::Value>>;
}

impl HasChildSource<la_arena::Idx<ast::UseTree>> for UseId {
    type Value = ast::UseTree;
    fn child_source(
        &self,
        db: &dyn DefDatabase,
    ) -> InFile<ArenaMap<la_arena::Idx<ast::UseTree>, Self::Value>> {
        let loc = &self.lookup(db);
        let use_ = &loc.id.item_tree(db)[loc.id.value];
        InFile::new(
            loc.id.file_id(),
            use_.use_tree_source_map(db, loc.id.file_id()).into_iter().collect(),
        )
    }
}
