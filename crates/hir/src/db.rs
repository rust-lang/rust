//! FIXME: write short doc here

pub use hir_def::db::{
    AttrsQuery, BlockDefMapQuery, BodyQuery, BodyWithSourceMapQuery, ConstDataQuery,
    CrateDefMapQueryQuery, CrateLangItemsQuery, DefDatabase, DefDatabaseStorage, EnumDataQuery,
    ExprScopesQuery, FileItemTreeQuery, FunctionDataQuery, GenericParamsQuery, ImplDataQuery,
    ImportMapQuery, InternConstQuery, InternDatabase, InternDatabaseStorage, InternEnumQuery,
    InternFunctionQuery, InternImplQuery, InternStaticQuery, InternStructQuery, InternTraitQuery,
    InternTypeAliasQuery, InternUnionQuery, LangItemQuery, StaticDataQuery, StructDataQuery,
    TraitDataQuery, TypeAliasDataQuery, UnionDataQuery,
};
pub use hir_expand::db::{
    AstDatabase, AstDatabaseStorage, AstIdMapQuery, HygieneFrameQuery, InternEagerExpansionQuery,
    InternMacroQuery, MacroArgTextQuery, MacroDefQuery, MacroExpandQuery, ParseMacroExpansionQuery,
};
pub use hir_ty::db::*;

#[test]
fn hir_database_is_object_safe() {
    fn _assert_object_safe(_: &dyn HirDatabase) {}
}
