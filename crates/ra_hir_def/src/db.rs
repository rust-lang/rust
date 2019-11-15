//! Defines database & queries for name resolution.
use std::sync::Arc;

use hir_expand::{db::AstDatabase, HirFileId};
use ra_db::{salsa, CrateId, SourceDatabase};
use ra_syntax::ast;

use crate::{
    adt::{EnumData, StructData},
    body::{scope::ExprScopes, Body, BodySourceMap},
    imp::ImplData,
    nameres::{
        raw::{ImportSourceMap, RawItems},
        CrateDefMap,
    },
    DefWithBodyId, EnumId, ImplId, ItemLoc, StructOrUnionId,
};

#[salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    #[salsa::interned]
    fn intern_function(&self, loc: ItemLoc<ast::FnDef>) -> crate::FunctionId;
    #[salsa::interned]
    fn intern_struct_or_union(&self, loc: ItemLoc<ast::StructDef>) -> crate::StructOrUnionId;
    #[salsa::interned]
    fn intern_enum(&self, loc: ItemLoc<ast::EnumDef>) -> crate::EnumId;
    #[salsa::interned]
    fn intern_const(&self, loc: ItemLoc<ast::ConstDef>) -> crate::ConstId;
    #[salsa::interned]
    fn intern_static(&self, loc: ItemLoc<ast::StaticDef>) -> crate::StaticId;
    #[salsa::interned]
    fn intern_trait(&self, loc: ItemLoc<ast::TraitDef>) -> crate::TraitId;
    #[salsa::interned]
    fn intern_type_alias(&self, loc: ItemLoc<ast::TypeAliasDef>) -> crate::TypeAliasId;
    #[salsa::interned]
    fn intern_impl(&self, loc: ItemLoc<ast::ImplBlock>) -> crate::ImplId;
}

#[salsa::query_group(DefDatabase2Storage)]
pub trait DefDatabase2: InternDatabase + AstDatabase {
    #[salsa::invoke(RawItems::raw_items_with_source_map_query)]
    fn raw_items_with_source_map(
        &self,
        file_id: HirFileId,
    ) -> (Arc<RawItems>, Arc<ImportSourceMap>);

    #[salsa::invoke(RawItems::raw_items_query)]
    fn raw_items(&self, file_id: HirFileId) -> Arc<RawItems>;

    #[salsa::invoke(CrateDefMap::crate_def_map_query)]
    fn crate_def_map(&self, krate: CrateId) -> Arc<CrateDefMap>;

    #[salsa::invoke(StructData::struct_data_query)]
    fn struct_data(&self, id: StructOrUnionId) -> Arc<StructData>;

    #[salsa::invoke(EnumData::enum_data_query)]
    fn enum_data(&self, e: EnumId) -> Arc<EnumData>;

    #[salsa::invoke(ImplData::impl_data_query)]
    fn impl_data(&self, e: ImplId) -> Arc<ImplData>;

    #[salsa::invoke(Body::body_with_source_map_query)]
    fn body_with_source_map(&self, def: DefWithBodyId) -> (Arc<Body>, Arc<BodySourceMap>);

    #[salsa::invoke(Body::body_query)]
    fn body(&self, def: DefWithBodyId) -> Arc<Body>;

    #[salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, def: DefWithBodyId) -> Arc<ExprScopes>;
}
