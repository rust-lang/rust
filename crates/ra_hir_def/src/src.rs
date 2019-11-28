//! Utilities for mapping between hir IDs and the surface syntax.

use hir_expand::InFile;
use ra_arena::map::ArenaMap;
use ra_syntax::ast;

use crate::{db::DefDatabase, ConstLoc, FunctionLoc, StaticLoc, TypeAliasLoc};

pub trait HasSource {
    type Value;
    fn source(&self, db: &impl DefDatabase) -> InFile<Self::Value>;
}

impl HasSource for FunctionLoc {
    type Value = ast::FnDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::FnDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for TypeAliasLoc {
    type Value = ast::TypeAliasDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::TypeAliasDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for ConstLoc {
    type Value = ast::ConstDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::ConstDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for StaticLoc {
    type Value = ast::StaticDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::StaticDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

pub trait HasChildSource {
    type ChildId;
    type Value;
    fn child_source(&self, db: &impl DefDatabase) -> InFile<ArenaMap<Self::ChildId, Self::Value>>;
}
