//! FIXME: write short doc here

pub use hir_def::db::{
    AttrsQuery, BodyQuery, BodyWithSourceMapQuery, ConstDataQuery, CrateDefMapQueryQuery,
    CrateLangItemsQuery, DefDatabase, DefDatabaseStorage, DocumentationQuery, EnumDataQuery,
    ExprScopesQuery, FunctionDataQuery, GenericParamsQuery, ImplDataQuery, ImportMapQuery,
    InternConstQuery, InternDatabase, InternDatabaseStorage, InternEnumQuery, InternFunctionQuery,
    InternImplQuery, InternStaticQuery, InternStructQuery, InternTraitQuery, InternTypeAliasQuery,
    InternUnionQuery, ItemTreeQuery, LangItemQuery, ModuleLangItemsQuery, StaticDataQuery,
    StructDataQuery, TraitDataQuery, TypeAliasDataQuery, UnionDataQuery,
};
pub use hir_expand::db::{
    AstDatabase, AstDatabaseStorage, AstIdMapQuery, InternEagerExpansionQuery, InternMacroQuery,
    MacroArgTextQuery, MacroDefQuery, MacroExpandQuery, ParseMacroQuery,
};
pub use hir_ty::db::*;

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
