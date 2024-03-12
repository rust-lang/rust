//! Re-exports various subcrates databases so that the calling code can depend
//! only on `hir`. This breaks abstraction boundary a bit, it would be cool if
//! we didn't do that.
//!
//! But we need this for at least LRU caching at the query level.
pub use hir_def::db::{
    AttrsQuery, BlockDefMapQuery, BodyQuery, BodyWithSourceMapQuery, ConstDataQuery,
    ConstVisibilityQuery, CrateLangItemsQuery, CrateSupportsNoStdQuery, DefDatabase,
    DefDatabaseStorage, EnumDataQuery, EnumVariantDataWithDiagnosticsQuery, ExprScopesQuery,
    ExternCrateDeclDataQuery, FieldVisibilitiesQuery, FieldsAttrsQuery, FieldsAttrsSourceMapQuery,
    FileItemTreeQuery, FunctionDataQuery, FunctionVisibilityQuery, GenericParamsQuery,
    ImplDataWithDiagnosticsQuery, ImportMapQuery, InternAnonymousConstQuery, InternBlockQuery,
    InternConstQuery, InternDatabase, InternDatabaseStorage, InternEnumQuery,
    InternExternBlockQuery, InternExternCrateQuery, InternFunctionQuery, InternImplQuery,
    InternInTypeConstQuery, InternMacro2Query, InternMacroRulesQuery, InternProcMacroQuery,
    InternStaticQuery, InternStructQuery, InternTraitAliasQuery, InternTraitQuery,
    InternTypeAliasQuery, InternUnionQuery, InternUseQuery, LangItemQuery, Macro2DataQuery,
    MacroRulesDataQuery, ProcMacroDataQuery, StaticDataQuery, StructDataWithDiagnosticsQuery,
    TraitAliasDataQuery, TraitDataWithDiagnosticsQuery, TypeAliasDataQuery,
    UnionDataWithDiagnosticsQuery,
};
pub use hir_expand::db::{
    AstIdMapQuery, DeclMacroExpanderQuery, ExpandDatabase, ExpandDatabaseStorage,
    ExpandProcMacroQuery, InternMacroCallQuery, InternSyntaxContextQuery, MacroArgQuery,
    ParseMacroExpansionErrorQuery, ParseMacroExpansionQuery, ProcMacrosQuery, RealSpanMapQuery,
};
pub use hir_ty::db::*;
