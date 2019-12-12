//! Utilities for mapping between hir IDs and the surface syntax.

use hir_expand::InFile;
use ra_arena::map::ArenaMap;
use ra_syntax::ast;

use crate::{
    db::DefDatabase, ConstLoc, EnumLoc, FunctionLoc, ImplLoc, StaticLoc, StructLoc, TraitLoc,
    TypeAliasLoc, UnionLoc,
};

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

impl HasSource for ImplLoc {
    type Value = ast::ImplBlock;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::ImplBlock> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for TraitLoc {
    type Value = ast::TraitDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::TraitDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for StructLoc {
    type Value = ast::StructDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::StructDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for UnionLoc {
    type Value = ast::UnionDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::UnionDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

impl HasSource for EnumLoc {
    type Value = ast::EnumDef;

    fn source(&self, db: &impl DefDatabase) -> InFile<ast::EnumDef> {
        let node = self.ast_id.to_node(db);
        InFile::new(self.ast_id.file_id, node)
    }
}

pub trait HasChildSource {
    type ChildId;
    type Value;
    fn child_source(&self, db: &impl DefDatabase) -> InFile<ArenaMap<Self::ChildId, Self::Value>>;
}
