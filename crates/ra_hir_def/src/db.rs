//! Defines database & queries for name resolution.

use ra_db::{salsa, SourceDatabase};
use ra_syntax::ast;

#[salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    #[salsa::interned]
    fn intern_function(&self, loc: crate::ItemLoc<ast::FnDef>) -> crate::FunctionId;
    #[salsa::interned]
    fn intern_struct(&self, loc: crate::ItemLoc<ast::StructDef>) -> crate::StructId;
    #[salsa::interned]
    fn intern_enum(&self, loc: crate::ItemLoc<ast::EnumDef>) -> crate::EnumId;
    #[salsa::interned]
    fn intern_const(&self, loc: crate::ItemLoc<ast::ConstDef>) -> crate::ConstId;
    #[salsa::interned]
    fn intern_static(&self, loc: crate::ItemLoc<ast::StaticDef>) -> crate::StaticId;
    #[salsa::interned]
    fn intern_trait(&self, loc: crate::ItemLoc<ast::TraitDef>) -> crate::TraitId;
    #[salsa::interned]
    fn intern_type_alias(&self, loc: crate::ItemLoc<ast::TypeAliasDef>) -> crate::TypeAliasId;
}
