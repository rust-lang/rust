use std::sync::Arc;

use ra_syntax::{SyntaxNode, TreeArc, SourceFile};
use ra_db::{SourceDatabase, salsa};

use crate::{
    HirFileId, MacroDefId, AstIdMap, ErasedFileAstId, Crate, Module, HirInterner,
    Function, FnSignature, ExprScopes, TypeAlias,
    Struct, Enum, StructField,
    Const, ConstSignature, Static,
    nameres::{Namespace, ImportSourceMap, RawItems, CrateDefMap},
    ty::{InferenceResult, Ty, method_resolution::CrateImplBlocks, TypableDef, CallableDef, FnSig},
    adt::{StructData, EnumData},
    impl_block::{ModuleImplBlocks, ImplSourceMap},
    generics::{GenericParams, GenericDef},
    type_ref::TypeRef,
    traits::TraitData, Trait, ty::TraitRef
};

#[salsa::query_group(DefDatabaseStorage)]
pub trait DefDatabase: SourceDatabase + AsRef<HirInterner> {
    #[salsa::invoke(crate::ids::macro_def_query)]
    fn macro_def(&self, macro_id: MacroDefId) -> Option<Arc<mbe::MacroRules>>;

    #[salsa::invoke(HirFileId::hir_parse_query)]
    fn hir_parse(&self, file_id: HirFileId) -> TreeArc<SourceFile>;

    #[salsa::invoke(crate::adt::StructData::struct_data_query)]
    fn struct_data(&self, s: Struct) -> Arc<StructData>;

    #[salsa::invoke(crate::adt::EnumData::enum_data_query)]
    fn enum_data(&self, e: Enum) -> Arc<EnumData>;

    #[salsa::invoke(crate::traits::TraitData::trait_data_query)]
    fn trait_data(&self, t: Trait) -> Arc<TraitData>;

    #[salsa::invoke(crate::source_id::AstIdMap::ast_id_map_query)]
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;

    #[salsa::invoke(crate::source_id::AstIdMap::file_item_query)]
    fn ast_id_to_node(&self, file_id: HirFileId, ast_id: ErasedFileAstId) -> TreeArc<SyntaxNode>;

    #[salsa::invoke(RawItems::raw_items_query)]
    fn raw_items(&self, file_id: HirFileId) -> Arc<RawItems>;

    #[salsa::invoke(RawItems::raw_items_with_source_map_query)]
    fn raw_items_with_source_map(
        &self,
        file_id: HirFileId,
    ) -> (Arc<RawItems>, Arc<ImportSourceMap>);

    #[salsa::invoke(CrateDefMap::crate_def_map_query)]
    fn crate_def_map(&self, krate: Crate) -> Arc<CrateDefMap>;

    #[salsa::invoke(crate::impl_block::impls_in_module)]
    fn impls_in_module(&self, module: Module) -> Arc<ModuleImplBlocks>;

    #[salsa::invoke(crate::impl_block::impls_in_module_source_map_query)]
    fn impls_in_module_source_map(&self, module: Module) -> Arc<ImplSourceMap>;

    #[salsa::invoke(crate::impl_block::impls_in_module_with_source_map_query)]
    fn impls_in_module_with_source_map(
        &self,
        module: Module,
    ) -> (Arc<ModuleImplBlocks>, Arc<ImplSourceMap>);

    #[salsa::invoke(crate::generics::GenericParams::generic_params_query)]
    fn generic_params(&self, def: GenericDef) -> Arc<GenericParams>;

    #[salsa::invoke(crate::FnSignature::fn_signature_query)]
    fn fn_signature(&self, func: Function) -> Arc<FnSignature>;

    #[salsa::invoke(crate::type_alias::type_alias_ref_query)]
    fn type_alias_ref(&self, typ: TypeAlias) -> Arc<TypeRef>;

    #[salsa::invoke(crate::ConstSignature::const_signature_query)]
    fn const_signature(&self, konst: Const) -> Arc<ConstSignature>;

    #[salsa::invoke(crate::ConstSignature::static_signature_query)]
    fn static_signature(&self, konst: Static) -> Arc<ConstSignature>;
}

#[salsa::query_group(HirDatabaseStorage)]
pub trait HirDatabase: DefDatabase {
    #[salsa::invoke(ExprScopes::expr_scopes_query)]
    fn expr_scopes(&self, func: Function) -> Arc<ExprScopes>;

    #[salsa::invoke(crate::ty::infer)]
    fn infer(&self, func: Function) -> Arc<InferenceResult>;

    #[salsa::invoke(crate::ty::type_for_def)]
    fn type_for_def(&self, def: TypableDef, ns: Namespace) -> Ty;

    #[salsa::invoke(crate::ty::type_for_field)]
    fn type_for_field(&self, field: StructField) -> Ty;

    #[salsa::invoke(crate::ty::callable_item_sig)]
    fn callable_item_signature(&self, def: CallableDef) -> FnSig;

    #[salsa::invoke(crate::expr::body_with_source_map_query)]
    fn body_with_source_map(
        &self,
        func: Function,
    ) -> (Arc<crate::expr::Body>, Arc<crate::expr::BodySourceMap>);

    #[salsa::invoke(crate::expr::body_hir_query)]
    fn body_hir(&self, func: Function) -> Arc<crate::expr::Body>;

    #[salsa::invoke(crate::ty::method_resolution::CrateImplBlocks::impls_in_crate_query)]
    fn impls_in_crate(&self, krate: Crate) -> Arc<CrateImplBlocks>;

    #[salsa::invoke(crate::ty::method_resolution::implements)]
    fn implements(&self, trait_ref: TraitRef) -> bool;
}

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
