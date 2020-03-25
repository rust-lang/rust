//! FIXME: write short doc here

pub use hir_def::db::{
    AttrsQuery, BodyQuery, BodyWithSourceMapQuery, ConstDataQuery, CrateDefMapQueryQuery,
    CrateLangItemsQuery, DefDatabase, DefDatabaseStorage, DocumentationQuery, EnumDataQuery,
    ExprScopesQuery, FunctionDataQuery, GenericParamsQuery, ImplDataQuery, InternConstQuery,
    InternDatabase, InternDatabaseStorage, InternEnumQuery, InternFunctionQuery, InternImplQuery,
    InternStaticQuery, InternStructQuery, InternTraitQuery, InternTypeAliasQuery, InternUnionQuery,
    LangItemQuery, ModuleLangItemsQuery, RawItemsQuery, StaticDataQuery, StructDataQuery,
    TraitDataQuery, TypeAliasDataQuery, UnionDataQuery,
};
pub use hir_expand::db::{
    AstDatabase, AstDatabaseStorage, AstIdMapQuery, InternEagerExpansionQuery, InternMacroQuery,
    MacroArgQuery, MacroDefQuery, MacroExpandQuery, ParseMacroQuery,
};
pub use hir_ty::db::{
    AssociatedTyDataQuery, AssociatedTyValueQuery, CallableItemSignatureQuery, FieldTypesQuery,
    GenericDefaultsQuery, GenericPredicatesForParamQuery, GenericPredicatesQuery, HirDatabase,
    HirDatabaseStorage, ImplDatumQuery, ImplSelfTyQuery, ImplTraitQuery, ImplsForTraitQuery,
    ImplsInCrateQuery, InferQueryQuery, InternAssocTyValueQuery, InternChalkImplQuery,
    InternTypeCtorQuery, InternTypeParamIdQuery, StructDatumQuery, TraitDatumQuery,
    TraitSolveQuery, TyQuery, ValueTyQuery,
};

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
