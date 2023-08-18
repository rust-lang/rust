//! Utilities for mapping between hir IDs and the surface syntax.

use hir_expand::InFile;
use la_arena::ArenaMap;
use syntax::ast;

use crate::{
    db::DefDatabase, item_tree::ItemTreeNode, AssocItemLoc, ItemLoc, Lookup, Macro2Loc,
    MacroRulesLoc, ProcMacroLoc, UseId,
};

pub trait HasSource {
    type Value;
    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value>;
}

impl<N: ItemTreeNode> HasSource for AssocItemLoc<N> {
    type Value = N::Source;

    fn source(&self, db: &dyn DefDatabase) -> InFile<N::Source> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id());
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

impl<N: ItemTreeNode> HasSource for ItemLoc<N> {
    type Value = N::Source;

    fn source(&self, db: &dyn DefDatabase) -> InFile<N::Source> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id());
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

impl HasSource for Macro2Loc {
    type Value = ast::MacroDef;

    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id());
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

impl HasSource for MacroRulesLoc {
    type Value = ast::MacroRules;

    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id());
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
    }
}

impl HasSource for ProcMacroLoc {
    type Value = ast::Fn;

    fn source(&self, db: &dyn DefDatabase) -> InFile<Self::Value> {
        let tree = self.id.item_tree(db);
        let ast_id_map = db.ast_id_map(self.id.file_id());
        let root = db.parse_or_expand(self.id.file_id());
        let node = &tree[self.id.value];

        InFile::new(self.id.file_id(), ast_id_map.get(node.ast_id()).to_node(&root))
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
