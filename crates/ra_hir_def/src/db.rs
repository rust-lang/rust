//! Defines database & queries for name resolution.
use std::sync::Arc;

use hir_expand::{db::AstDatabase, HirFileId};
use ra_db::{salsa, CrateId, SourceDatabase};
use ra_syntax::ast;

use crate::{
    adt::{EnumData, StructData},
    nameres::{
        raw::{ImportSourceMap, RawItems},
        CrateDefMap,
    },
    EnumId, StructId, UnionId,
};

#[salsa::query_group(InternDatabaseStorage)]
pub trait InternDatabase: SourceDatabase {
    #[salsa::interned]
    fn intern_function(&self, loc: crate::ItemLoc<ast::FnDef>) -> crate::FunctionId;
    #[salsa::interned]
    fn intern_struct(&self, loc: crate::ItemLoc<ast::StructDef>) -> crate::StructId;
    #[salsa::interned]
    fn intern_union(&self, loc: crate::ItemLoc<ast::StructDef>) -> crate::UnionId;
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
    fn struct_data(&self, s: StructId) -> Arc<StructData>;

    #[salsa::invoke(StructData::union_data_query)]
    fn union_data(&self, s: UnionId) -> Arc<StructData>;

    #[salsa::invoke(EnumData::enum_data_query)]
    fn enum_data(&self, e: EnumId) -> Arc<EnumData>;
}
