//! Re-exports various subcrates databases so that the calling code can depend
//! only on `hir`. This breaks abstraction boundary a bit, it would be cool if
//! we didn't do that.
//!
//! But we need this for at least LRU caching at the query level.
pub use hir_def::db::DefDatabase;
//     AttrsQuery, BlockDefMapQuery, BlockItemTreeQuery, BlockItemTreeWithSourceMapQuery, BodyQuery,
//     BodyWithSourceMapQuery, ConstDataQuery, ConstVisibilityQuery, CrateDefMapQuery,
//     CrateLangItemsQuery, CrateNotableTraitsQuery, CrateSupportsNoStdQuery, DefDatabase,
//     DefDatabaseStorage, EnumDataQuery, EnumVariantDataWithDiagnosticsQuery,
//     ExpandProcAttrMacrosQuery, ExprScopesQuery, ExternCrateDeclDataQuery, FieldVisibilitiesQuery,
//     FieldsAttrsQuery, FieldsAttrsSourceMapQuery, FileItemTreeQuery, FileItemTreeWithSourceMapQuery,
//     FunctionDataQuery, FunctionVisibilityQuery, GenericParamsQuery,
//     GenericParamsWithSourceMapQuery, ImplItemsWithDiagnosticsQuery, ImportMapQuery,
//     IncludeMacroInvocQuery, InternAnonymousConstQuery, InternBlockQuery, InternConstQuery,
//     InternDatabase, InternDatabaseStorage, InternEnumQuery, InternExternBlockQuery,
//     InternExternCrateQuery, InternFunctionQuery, InternImplQuery, InternInTypeConstQuery,
//     InternMacro2Query, InternMacroRulesQuery, InternProcMacroQuery, InternStaticQuery,
//     InternStructQuery, InternTraitAliasQuery, InternTraitQuery, InternTypeAliasQuery,
//     InternUnionQuery, InternUseQuery, LangItemQuery, Macro2DataQuery, MacroDefQuery,
//     MacroRulesDataQuery, NotableTraitsInDepsQuery, ProcMacroDataQuery, StaticDataQuery,
//     StructDataWithDiagnosticsQuery, TraitAliasDataQuery, TraitItemsWithDiagnosticsQuery,
//     TypeAliasDataQuery, UnionDataWithDiagnosticsQuery,
// };
pub use hir_expand::db::ExpandDatabase;
// AstIdMapQuery, DeclMacroExpanderQuery, ExpandDatabase, ExpandDatabaseStorage,
// ExpandProcMacroQuery, InternMacroCallQuery, InternSyntaxContextQuery, MacroArgQuery,
// ParseMacroExpansionErrorQuery, ParseMacroExpansionQuery, ProcMacroSpanQuery, ProcMacrosQuery,
// RealSpanMapQuery,
pub use hir_ty::db::HirDatabase;
//     AdtDatumQuery, AdtVarianceQuery, AssociatedTyDataQuery, AssociatedTyValueQuery, BorrowckQuery,
//     CallableItemSignatureQuery, ConstEvalDiscriminantQuery, ConstEvalQuery, ConstEvalStaticQuery,
//     ConstParamTyQuery, DynCompatibilityOfTraitQuery, FieldTypesQuery, FnDefDatumQuery,
//     FnDefVarianceQuery, GenericDefaultsQuery, GenericPredicatesForParamQuery,
//     GenericPredicatesQuery, GenericPredicatesWithoutParentQuery, HirDatabase, HirDatabaseStorage,
//     ImplDatumQuery, ImplSelfTyQuery, ImplTraitQuery, IncoherentInherentImplCratesQuery, InferQuery,
//     InherentImplsInBlockQuery, InherentImplsInCrateQuery, InternCallableDefQuery,
//     InternClosureQuery, InternCoroutineQuery, InternImplTraitIdQuery, InternLifetimeParamIdQuery,
//     InternTypeOrConstParamIdQuery, LayoutOfAdtQuery, LayoutOfTyQuery, LookupImplMethodQuery,
//     MirBodyForClosureQuery, MirBodyQuery, MonomorphizedMirBodyForClosureQuery,
//     MonomorphizedMirBodyQuery, ProgramClausesForChalkEnvQuery, ReturnTypeImplTraitsQuery,
//     TargetDataLayoutQuery, TraitDatumQuery, TraitEnvironmentQuery, TraitImplsInBlockQuery,
//     TraitImplsInCrateQuery, TraitImplsInDepsQuery, TraitSolveQuery, TyQuery,
//     TypeAliasImplTraitsQuery, ValueTyQuery,
// };
