//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnostics should
//! be expressed in terms of hir types themselves.
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_def::{
    DefWithBodyId, GenericParamId, HasModule, SyntheticSyntax,
    expr_store::{
        ExprOrPatPtr, ExpressionStoreSourceMap, hir_assoc_type_binding_to_ast,
        hir_generic_arg_to_ast, hir_segment_to_ast_segment,
    },
    hir::ExprOrPatId,
};
use hir_expand::{HirFileId, InFile, mod_path::ModPath, name::Name};
use hir_ty::{
    CastError, InferenceDiagnostic, InferenceTyDiagnosticSource, ParamEnvAndCrate,
    PathGenericsSource, PathLoweringDiagnostic, TyLoweringDiagnostic, TyLoweringDiagnosticKind,
    db::HirDatabase,
    diagnostics::{BodyValidationDiagnostic, UnsafetyReason},
    display::{DisplayTarget, HirDisplay},
    next_solver::DbInterner,
    solver_errors::SolverDiagnosticKind,
};
use stdx::{impl_from, never};
use syntax::{
    AstNode, AstPtr, SyntaxError, SyntaxNodePtr, TextRange,
    ast::{self, HasGenericArgs},
    match_ast,
};
use triomphe::Arc;

use crate::{AssocItem, Field, Function, GenericDef, Local, Trait, Type, Variant};

pub use hir_def::VariantId;
pub use hir_ty::{
    GenericArgsProhibitedReason, IncorrectGenericsLenKind,
    diagnostics::{CaseType, IncorrectCase},
};

#[derive(Debug, Clone)]
pub enum SpanAst {
    Expr(ast::Expr),
    Pat(ast::Pat),
    Type(ast::Type),
}
const _: () = {
    use syntax::ast::*;
    impl_from!(Expr, Pat, Type for SpanAst);
};

impl From<Either<ast::Expr, ast::Pat>> for SpanAst {
    fn from(value: Either<ast::Expr, ast::Pat>) -> Self {
        match value {
            Either::Left(it) => it.into(),
            Either::Right(it) => it.into(),
        }
    }
}

impl ast::AstNode for SpanAst {
    fn can_cast(kind: syntax::SyntaxKind) -> bool {
        ast::Expr::can_cast(kind) || ast::Pat::can_cast(kind) || ast::Type::can_cast(kind)
    }

    fn cast(syntax: syntax::SyntaxNode) -> Option<Self> {
        ast::Expr::cast(syntax.clone())
            .map(SpanAst::Expr)
            .or_else(|| ast::Pat::cast(syntax.clone()).map(SpanAst::Pat))
            .or_else(|| ast::Type::cast(syntax).map(SpanAst::Type))
    }

    fn syntax(&self) -> &syntax::SyntaxNode {
        match self {
            SpanAst::Expr(it) => it.syntax(),
            SpanAst::Pat(it) => it.syntax(),
            SpanAst::Type(it) => it.syntax(),
        }
    }
}

pub type SpanSyntax = InFile<AstPtr<SpanAst>>;

macro_rules! diagnostics {
    ($AnyDiagnostic:ident <$db:lifetime> -> $($diag:ident $(<$lt:lifetime>)?,)*) => {
        #[derive(Debug)]
        pub enum $AnyDiagnostic<$db> {$(
            $diag(Box<$diag $(<$lt>)?>),
        )*}

        $(
            impl<$db> From<$diag $(<$lt>)?> for $AnyDiagnostic<$db> {
                fn from(d: $diag $(<$lt>)?) -> $AnyDiagnostic<$db> {
                    $AnyDiagnostic::$diag(Box::new(d))
                }
            }
        )*
    };
}

diagnostics![AnyDiagnostic<'db> ->
    AwaitOutsideOfAsync,
    BreakOutsideOfLoop,
    CannotBeDereferenced<'db>,
    CastToUnsized<'db>,
    ExpectedArrayOrSlicePat<'db>,
    ExpectedFunction<'db>,
    FruInDestructuringAssignment,
    FunctionalRecordUpdateOnNonStruct,
    GenericDefaultRefersToSelf,
    InactiveCode,
    IncoherentImpl,
    IncorrectCase,
    IncorrectGenericsLen,
    IncorrectGenericsOrder,
    InvalidCast<'db>,
    InvalidDeriveTarget,
    InvalidLhsOfAssignment,
    InvalidRangePatType,
    MacroDefError,
    MacroError,
    MacroExpansionParseError,
    MalformedDerive,
    MethodCallIllegalSizedBound,
    MismatchedArgCount,
    MismatchedTupleStructPatArgCount,
    MissingFields,
    MissingMatchArms,
    MissingUnsafe,
    MovedOutOfRef<'db>,
    NeedMut,
    NonExhaustiveLet,
    NonExhaustiveRecordExpr,
    NonExhaustiveRecordPat,
    NoSuchField,
    MismatchedArrayPatLen,
    DuplicateField,
    PatternArgInExternFn,
    PrivateAssocItem,
    PrivateField,
    RemoveTrailingReturn,
    RemoveUnnecessaryElse,
    UnusedMustUse<'db>,
    ReplaceFilterMapNextWithFindMap,
    TraitImplIncorrectSafety,
    TraitImplMissingAssocItems,
    TraitImplOrphan,
    TraitImplRedundantAssocItems,
    TypedHole<'db>,
    TypeMismatch<'db>,
    UndeclaredLabel,
    UnimplementedBuiltinMacro,
    UnreachableLabel,
    UnresolvedAssocItem,
    UnresolvedExternCrate,
    UnresolvedField<'db>,
    UnresolvedImport,
    UnresolvedMacroCall,
    UnresolvedMethodCall<'db>,
    UnresolvedModule,
    UnresolvedIdent,
    UnusedMut,
    UnusedVariable,
    GenericArgsProhibited,
    ParenthesizedGenericArgsWithoutFnTrait,
    BadRtn,
    MissingLifetime,
    ElidedLifetimesInPath,
    TypeMustBeKnown<'db>,
    UnionExprMustHaveExactlyOneField,
    UnimplementedTrait<'db>,
];

#[derive(Debug)]
pub struct BreakOutsideOfLoop {
    pub expr: InFile<ExprOrPatPtr>,
    pub is_break: bool,
    pub bad_value_break: bool,
}

#[derive(Debug)]
pub struct TypedHole<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub expected: Type<'db>,
}

#[derive(Debug)]
pub struct UnresolvedModule {
    pub decl: InFile<AstPtr<ast::Module>>,
    pub candidates: Box<[String]>,
}

#[derive(Debug)]
pub struct UnresolvedExternCrate {
    pub decl: InFile<AstPtr<ast::ExternCrate>>,
}

#[derive(Debug)]
pub struct UnresolvedImport {
    pub decl: InFile<AstPtr<ast::UseTree>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnresolvedMacroCall {
    pub range: InFile<TextRange>,
    pub path: ModPath,
    pub is_bang: bool,
}
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnreachableLabel {
    pub node: InFile<AstPtr<ast::Lifetime>>,
    pub name: Name,
}

#[derive(Debug)]
pub struct AwaitOutsideOfAsync {
    pub node: InFile<AstPtr<ast::AwaitExpr>>,
    pub location: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UndeclaredLabel {
    pub node: InFile<AstPtr<ast::Lifetime>>,
    pub name: Name,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InactiveCode {
    pub node: InFile<SyntaxNodePtr>,
    pub cfg: CfgExpr,
    pub opts: CfgOptions,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroError {
    pub range: InFile<TextRange>,
    pub message: String,
    pub error: bool,
    pub kind: &'static str,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroExpansionParseError {
    pub range: InFile<TextRange>,
    pub errors: Arc<[SyntaxError]>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroDefError {
    pub node: InFile<AstPtr<ast::Macro>>,
    pub message: String,
    pub name: Option<TextRange>,
}

#[derive(Debug)]
pub struct UnimplementedBuiltinMacro {
    pub node: InFile<SyntaxNodePtr>,
}

#[derive(Debug)]
pub struct InvalidDeriveTarget {
    pub range: InFile<TextRange>,
}

#[derive(Debug)]
pub struct MalformedDerive {
    pub range: InFile<TextRange>,
}

#[derive(Debug)]
pub struct NoSuchField {
    pub field: InFile<AstPtr<Either<ast::RecordExprField, ast::RecordPatField>>>,
    pub private: Option<Field>,
    pub variant: VariantId,
}

#[derive(Debug)]
pub struct DuplicateField {
    pub field: InFile<AstPtr<Either<ast::RecordExprField, ast::RecordPatField>>>,
    pub variant: Variant,
}

#[derive(Debug)]
pub struct PrivateAssocItem {
    pub expr_or_pat: InFile<ExprOrPatPtr>,
    pub item: AssocItem,
}

#[derive(Debug)]
pub struct MismatchedTupleStructPatArgCount {
    pub expr_or_pat: InFile<ExprOrPatPtr>,
    pub expected: usize,
    pub found: usize,
}

#[derive(Debug)]
pub struct MismatchedArrayPatLen {
    pub pat: InFile<ExprOrPatPtr>,
    pub expected: u128,
    pub found: u128,
    pub has_rest: bool,
}

#[derive(Debug)]
pub struct ExpectedArrayOrSlicePat<'db> {
    pub pat: InFile<ExprOrPatPtr>,
    pub found: Type<'db>,
}

#[derive(Debug)]
pub struct InvalidRangePatType {
    pub pat: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct ExpectedFunction<'db> {
    pub call: InFile<ExprOrPatPtr>,
    pub found: Type<'db>,
}

#[derive(Debug)]
pub struct CannotBeDereferenced<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub found: Type<'db>,
}

#[derive(Debug)]
pub struct FruInDestructuringAssignment {
    pub node: InFile<AstPtr<ast::Expr>>,
}

#[derive(Debug)]
pub struct FunctionalRecordUpdateOnNonStruct {
    pub base_expr: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct UnresolvedField<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub receiver: Type<'db>,
    pub name: Name,
    pub method_with_same_name_exists: bool,
}

#[derive(Debug)]
pub struct UnresolvedMethodCall<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub receiver: Type<'db>,
    pub name: Name,
    pub field_with_same_name: Option<Type<'db>>,
    pub assoc_func_with_same_name: Option<Function>,
}

#[derive(Debug)]
pub struct UnresolvedAssocItem {
    pub expr_or_pat: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct UnresolvedIdent {
    pub node: InFile<(ExprOrPatPtr, Option<TextRange>)>,
}

#[derive(Debug)]
pub struct PrivateField {
    pub expr: InFile<ExprOrPatPtr>,
    pub field: Field,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsafeLint {
    HardError,
    UnsafeOpInUnsafeFn,
    DeprecatedSafe2024,
}

#[derive(Debug)]
pub struct MissingUnsafe {
    pub node: InFile<ExprOrPatPtr>,
    pub lint: UnsafeLint,
    pub reason: UnsafetyReason,
}

#[derive(Debug)]
pub struct MissingFields {
    pub file: HirFileId,
    pub field_list_parent: AstPtr<Either<ast::RecordExpr, ast::RecordPat>>,
    pub field_list_parent_path: Option<AstPtr<ast::Path>>,
    pub missed_fields: Vec<(Name, Field)>,
}

#[derive(Debug)]
pub struct ReplaceFilterMapNextWithFindMap {
    pub file: HirFileId,
    /// This expression is the whole method chain up to and including `.filter_map(..).next()`.
    pub next_expr: AstPtr<ast::Expr>,
}

#[derive(Debug)]
pub struct MismatchedArgCount {
    pub call_expr: InFile<ExprOrPatPtr>,
    pub expected: usize,
    pub found: usize,
}

#[derive(Debug)]
pub struct MissingMatchArms {
    pub scrutinee_expr: InFile<AstPtr<ast::Expr>>,
    pub uncovered_patterns: String,
}

#[derive(Debug)]
pub struct NonExhaustiveLet {
    pub pat: InFile<AstPtr<ast::Pat>>,
    pub uncovered_patterns: String,
}

#[derive(Debug)]
pub struct NonExhaustiveRecordExpr {
    pub expr: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct NonExhaustiveRecordPat {
    pub pat: InFile<ExprOrPatPtr>,
    pub variant: Variant,
}

#[derive(Debug)]
pub struct TypeMismatch<'db> {
    pub expr_or_pat: InFile<ExprOrPatPtr>,
    pub expected: Type<'db>,
    pub actual: Type<'db>,
}

#[derive(Debug)]
pub struct NeedMut {
    pub local: Local,
    pub span: InFile<SyntaxNodePtr>,
}

#[derive(Debug)]
pub struct UnusedMut {
    pub local: Local,
}

#[derive(Debug)]
pub struct UnusedVariable {
    pub local: Local,
}

#[derive(Debug)]
pub struct MovedOutOfRef<'db> {
    pub ty: Type<'db>,
    pub span: InFile<SyntaxNodePtr>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct IncoherentImpl {
    pub file_id: HirFileId,
    pub impl_: AstPtr<ast::Impl>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TraitImplOrphan {
    pub file_id: HirFileId,
    pub impl_: AstPtr<ast::Impl>,
}

// FIXME: Split this off into the corresponding 4 rustc errors
#[derive(Debug, PartialEq, Eq)]
pub struct TraitImplIncorrectSafety {
    pub file_id: HirFileId,
    pub impl_: AstPtr<ast::Impl>,
    pub should_be_safe: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TraitImplMissingAssocItems {
    pub file_id: HirFileId,
    pub impl_: AstPtr<ast::Impl>,
    pub missing: Vec<(Name, AssocItem)>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TraitImplRedundantAssocItems {
    pub file_id: HirFileId,
    pub trait_: Trait,
    pub impl_: AstPtr<ast::Impl>,
    pub assoc_item: (Name, AssocItem),
}

#[derive(Debug)]
pub struct RemoveTrailingReturn {
    pub return_expr: InFile<AstPtr<ast::ReturnExpr>>,
}

#[derive(Debug)]
pub struct RemoveUnnecessaryElse {
    pub if_expr: InFile<AstPtr<ast::IfExpr>>,
}

#[derive(Debug)]
pub struct UnusedMustUse<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub message: Option<&'db str>,
}

#[derive(Debug)]
pub struct CastToUnsized<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub cast_ty: Type<'db>,
}

#[derive(Debug)]
pub struct InvalidCast<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub error: CastError,
    pub expr_ty: Type<'db>,
    pub cast_ty: Type<'db>,
}

#[derive(Debug)]
pub struct GenericArgsProhibited {
    pub args: InFile<AstPtr<Either<ast::GenericArgList, ast::ParenthesizedArgList>>>,
    pub reason: GenericArgsProhibitedReason,
}

#[derive(Debug)]
pub struct ParenthesizedGenericArgsWithoutFnTrait {
    pub args: InFile<AstPtr<ast::ParenthesizedArgList>>,
}

#[derive(Debug)]
pub struct BadRtn {
    pub rtn: InFile<AstPtr<ast::ReturnTypeSyntax>>,
}

#[derive(Debug)]
pub struct IncorrectGenericsLen {
    /// Points at the name if there are no generics.
    pub generics_or_segment: InFile<AstPtr<Either<ast::GenericArgList, ast::NameRef>>>,
    pub kind: IncorrectGenericsLenKind,
    pub provided: u32,
    pub expected: u32,
    pub def: GenericDef,
}

#[derive(Debug)]
pub struct MissingLifetime {
    /// Points at the name if there are no generics.
    pub generics_or_segment: InFile<AstPtr<Either<ast::GenericArgList, ast::NameRef>>>,
    pub expected: u32,
    pub def: GenericDef,
}

#[derive(Debug)]
pub struct ElidedLifetimesInPath {
    /// Points at the name if there are no generics.
    pub generics_or_segment: InFile<AstPtr<Either<ast::GenericArgList, ast::NameRef>>>,
    pub expected: u32,
    pub def: GenericDef,
    pub hard_error: bool,
}

#[derive(Debug)]
pub struct TypeMustBeKnown<'db> {
    pub at_point: SpanSyntax,
    pub top_term: Option<Either<Type<'db>, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericArgKind {
    Lifetime,
    Type,
    Const,
}

impl GenericArgKind {
    fn from_id(id: GenericParamId) -> Self {
        match id {
            GenericParamId::TypeParamId(_) => GenericArgKind::Type,
            GenericParamId::ConstParamId(_) => GenericArgKind::Const,
            GenericParamId::LifetimeParamId(_) => GenericArgKind::Lifetime,
        }
    }
}

#[derive(Debug)]
pub struct IncorrectGenericsOrder {
    pub provided_arg: InFile<AstPtr<ast::GenericArg>>,
    pub expected_kind: GenericArgKind,
}

#[derive(Debug)]
pub struct GenericDefaultRefersToSelf {
    /// The `Self` segment.
    pub segment: InFile<AstPtr<ast::PathSegment>>,
}

#[derive(Debug)]
pub struct UnionExprMustHaveExactlyOneField {
    pub expr: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct InvalidLhsOfAssignment {
    pub lhs: InFile<AstPtr<Either<ast::Expr, ast::Pat>>>,
}

#[derive(Debug)]
pub struct MethodCallIllegalSizedBound {
    pub call_expr: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct PatternArgInExternFn {
    pub node: InFile<AstPtr<ast::Pat>>,
}

#[derive(Debug)]
pub struct UnimplementedTrait<'db> {
    pub span: SpanSyntax,
    pub trait_predicate: crate::TraitPredicate<'db>,
    pub root_trait_predicate: Option<crate::TraitPredicate<'db>>,
}

impl<'db> AnyDiagnostic<'db> {
    pub(crate) fn body_validation_diagnostic(
        db: &'db dyn HirDatabase,
        diagnostic: BodyValidationDiagnostic<'db>,
        source_map: &hir_def::expr_store::BodySourceMap,
    ) -> Option<AnyDiagnostic<'db>> {
        match diagnostic {
            BodyValidationDiagnostic::RecordMissingFields { record, variant, missed_fields } => {
                let variant_data = variant.fields(db);
                let missed_fields = missed_fields
                    .into_iter()
                    .map(|idx| {
                        (
                            variant_data.fields()[idx].name.clone(),
                            Field { parent: variant.into(), id: idx },
                        )
                    })
                    .collect();

                let record = match record {
                    Either::Left(record_expr) => source_map.expr_syntax(record_expr).ok()?,
                    Either::Right(record_pat) => source_map.pat_syntax(record_pat).ok()?,
                };
                let file = record.file_id;
                let root = record.file_syntax(db);
                match record.value.to_node(&root) {
                    Either::Left(ast::Expr::RecordExpr(record_expr))
                        if record_expr.record_expr_field_list().is_some() =>
                    {
                        let field_list_parent_path =
                            record_expr.path().map(|path| AstPtr::new(&path));
                        return Some(
                            MissingFields {
                                file,
                                field_list_parent: AstPtr::new(&Either::Left(record_expr)),
                                field_list_parent_path,
                                missed_fields,
                            }
                            .into(),
                        );
                    }
                    Either::Right(ast::Pat::RecordPat(record_pat))
                        if record_pat.record_pat_field_list().is_some() =>
                    {
                        let field_list_parent_path =
                            record_pat.path().map(|path| AstPtr::new(&path));
                        return Some(
                            MissingFields {
                                file,
                                field_list_parent: AstPtr::new(&Either::Right(record_pat)),
                                field_list_parent_path,
                                missed_fields,
                            }
                            .into(),
                        );
                    }
                    _ => {}
                }
            }
            BodyValidationDiagnostic::ReplaceFilterMapNextWithFindMap { method_call_expr } => {
                if let Ok(next_source_ptr) = source_map.expr_syntax(method_call_expr) {
                    return Some(
                        ReplaceFilterMapNextWithFindMap {
                            file: next_source_ptr.file_id,
                            next_expr: next_source_ptr.value.cast()?,
                        }
                        .into(),
                    );
                }
            }
            BodyValidationDiagnostic::MissingMatchArms { match_expr, uncovered_patterns } => {
                if let Ok(source_ptr) = source_map.expr_syntax(match_expr)
                    && let root = source_ptr.file_syntax(db)
                    && let Either::Left(ast::Expr::MatchExpr(match_expr)) =
                        source_ptr.value.to_node(&root)
                    && let Some(scrut_expr) = match_expr.expr()
                    && match_expr.match_arm_list().is_some()
                {
                    return Some(
                        MissingMatchArms {
                            scrutinee_expr: InFile::new(
                                source_ptr.file_id,
                                AstPtr::new(&scrut_expr),
                            ),
                            uncovered_patterns,
                        }
                        .into(),
                    );
                }
            }
            BodyValidationDiagnostic::NonExhaustiveLet { pat, uncovered_patterns } => {
                if let Ok(source_ptr) = source_map.pat_syntax(pat)
                    && let Some(ast_pat) = source_ptr.value.cast::<ast::Pat>()
                {
                    return Some(
                        NonExhaustiveLet {
                            pat: InFile::new(source_ptr.file_id, ast_pat),
                            uncovered_patterns,
                        }
                        .into(),
                    );
                }
            }
            BodyValidationDiagnostic::RemoveTrailingReturn { return_expr } => {
                if let Ok(source_ptr) = source_map.expr_syntax(return_expr)
                    // Filters out desugared return expressions (e.g. desugared try operators).
                    && let Some(ptr) = source_ptr.value.cast::<ast::ReturnExpr>()
                {
                    return Some(
                        RemoveTrailingReturn { return_expr: InFile::new(source_ptr.file_id, ptr) }
                            .into(),
                    );
                }
            }
            BodyValidationDiagnostic::RemoveUnnecessaryElse { if_expr } => {
                if let Ok(source_ptr) = source_map.expr_syntax(if_expr)
                    && let Some(ptr) = source_ptr.value.cast::<ast::IfExpr>()
                {
                    return Some(
                        RemoveUnnecessaryElse { if_expr: InFile::new(source_ptr.file_id, ptr) }
                            .into(),
                    );
                }
            }
            BodyValidationDiagnostic::UnusedMustUse { expr, message } => {
                if let Ok(source_ptr) = source_map.expr_syntax(expr) {
                    return Some(UnusedMustUse { expr: source_ptr, message }.into());
                }
            }
        }
        None
    }

    pub(crate) fn inference_diagnostic(
        db: &'db dyn HirDatabase,
        def: DefWithBodyId,
        d: &'db InferenceDiagnostic,
        source_map: &hir_def::expr_store::BodySourceMap,
        sig_map: &hir_def::expr_store::ExpressionStoreSourceMap,
        env: ParamEnvAndCrate<'db>,
    ) -> Option<AnyDiagnostic<'db>> {
        let expr_syntax = |expr| {
            source_map
                .expr_syntax(expr)
                .inspect_err(|_| stdx::never!("inference diagnostic in desugared expr"))
                .ok()
        };
        let pat_syntax = |pat| {
            source_map
                .pat_syntax(pat)
                .inspect_err(|_| stdx::never!("inference diagnostic in desugared pattern"))
                .ok()
        };
        let type_syntax = |pat| {
            source_map
                .type_syntax(pat)
                .inspect_err(|_| stdx::never!("inference diagnostic in desugared type"))
                .ok()
        };
        let expr_or_pat_syntax = |id| match id {
            ExprOrPatId::ExprId(expr) => expr_syntax(expr),
            ExprOrPatId::PatId(pat) => pat_syntax(pat),
        };
        let span_syntax = |span| match span {
            hir_ty::Span::ExprId(idx) => expr_syntax(idx).map(|it| it.upcast()),
            hir_ty::Span::PatId(idx) => pat_syntax(idx).map(|it| it.upcast()),
            hir_ty::Span::TypeRefId(idx) => type_syntax(idx).map(|it| it.upcast()),
            hir_ty::Span::BindingId(idx) => {
                pat_syntax(source_map.patterns_for_binding(idx)[0]).map(|it| it.upcast())
            }
            hir_ty::Span::Dummy => {
                never!("should never create a diagnostic for dummy spans");
                None
            }
        };
        Some(match d {
            &InferenceDiagnostic::NoSuchField { field: expr, private, variant } => {
                let expr_or_pat = match expr {
                    ExprOrPatId::ExprId(expr) => {
                        source_map.field_syntax(expr).map(AstPtr::wrap_left)
                    }
                    ExprOrPatId::PatId(pat) => source_map.pat_field_syntax(pat),
                };
                let private = private.map(|id| Field { id, parent: variant.into() });
                NoSuchField { field: expr_or_pat, private, variant }.into()
            }
            &InferenceDiagnostic::MismatchedArrayPatLen { pat, expected, found, has_rest } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                MismatchedArrayPatLen { pat, expected, found, has_rest }.into()
            }
            InferenceDiagnostic::ExpectedArrayOrSlicePat { pat, found } => {
                let pat = pat_syntax(*pat)?.map(Into::into);
                ExpectedArrayOrSlicePat { pat, found: Type::new(db, def, found.as_ref()) }.into()
            }
            &InferenceDiagnostic::InvalidRangePatType { pat } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                InvalidRangePatType { pat }.into()
            }
            &InferenceDiagnostic::DuplicateField { field: expr, variant } => {
                let expr_or_pat = match expr {
                    ExprOrPatId::ExprId(expr) => {
                        source_map.field_syntax(expr).map(AstPtr::wrap_left)
                    }
                    ExprOrPatId::PatId(pat) => source_map.pat_field_syntax(pat),
                };
                DuplicateField { field: expr_or_pat, variant: variant.into() }.into()
            }
            &InferenceDiagnostic::MismatchedArgCount { call_expr, expected, found } => {
                MismatchedArgCount { call_expr: expr_syntax(call_expr)?, expected, found }.into()
            }
            &InferenceDiagnostic::PrivateField { expr, field } => {
                let expr = expr_syntax(expr)?;
                let field = field.into();
                PrivateField { expr, field }.into()
            }
            &InferenceDiagnostic::PrivateAssocItem { id, item } => {
                let expr_or_pat = expr_or_pat_syntax(id)?;
                let item = item.into();
                PrivateAssocItem { expr_or_pat, item }.into()
            }
            InferenceDiagnostic::ExpectedFunction { call_expr, found } => {
                let call_expr = expr_syntax(*call_expr)?;
                ExpectedFunction { call: call_expr, found: Type::new(db, def, found.as_ref()) }
                    .into()
            }
            InferenceDiagnostic::UnresolvedField {
                expr,
                receiver,
                name,
                method_with_same_name_exists,
            } => {
                let expr = expr_syntax(*expr)?;
                UnresolvedField {
                    expr,
                    name: name.clone(),
                    receiver: Type::new(db, def, receiver.as_ref()),
                    method_with_same_name_exists: *method_with_same_name_exists,
                }
                .into()
            }
            InferenceDiagnostic::UnresolvedMethodCall {
                expr,
                receiver,
                name,
                field_with_same_name,
                assoc_func_with_same_name,
            } => {
                let expr = expr_syntax(*expr)?;
                UnresolvedMethodCall {
                    expr,
                    name: name.clone(),
                    receiver: Type::new(db, def, receiver.as_ref()),
                    field_with_same_name: field_with_same_name
                        .as_ref()
                        .map(|ty| Type::new(db, def, ty.as_ref())),
                    assoc_func_with_same_name: assoc_func_with_same_name.map(Into::into),
                }
                .into()
            }
            &InferenceDiagnostic::UnresolvedAssocItem { id } => {
                let expr_or_pat = expr_or_pat_syntax(id)?;
                UnresolvedAssocItem { expr_or_pat }.into()
            }
            &InferenceDiagnostic::UnresolvedIdent { id } => {
                let node = match id {
                    ExprOrPatId::ExprId(id) => match source_map.expr_syntax(id) {
                        Ok(syntax) => syntax.map(|it| (it, None)),
                        Err(SyntheticSyntax) => source_map
                            .format_args_implicit_capture(id)?
                            .map(|(node, range)| (node.wrap_left(), Some(range))),
                    },
                    ExprOrPatId::PatId(id) => pat_syntax(id)?.map(|it| (it, None)),
                };
                UnresolvedIdent { node }.into()
            }
            &InferenceDiagnostic::BreakOutsideOfLoop { expr, is_break, bad_value_break } => {
                let expr = expr_syntax(expr)?;
                BreakOutsideOfLoop { expr, is_break, bad_value_break }.into()
            }
            &InferenceDiagnostic::NonExhaustiveRecordExpr { expr } => {
                NonExhaustiveRecordExpr { expr: expr_syntax(expr)? }.into()
            }
            &InferenceDiagnostic::NonExhaustiveRecordPat { pat, variant } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                NonExhaustiveRecordPat { pat, variant: variant.into() }.into()
            }
            &InferenceDiagnostic::FunctionalRecordUpdateOnNonStruct { base_expr } => {
                FunctionalRecordUpdateOnNonStruct { base_expr: expr_syntax(base_expr)? }.into()
            }
            InferenceDiagnostic::TypedHole { expr, expected } => {
                let expr = expr_syntax(*expr)?;
                TypedHole { expr, expected: Type::new(db, def, expected.as_ref()) }.into()
            }
            &InferenceDiagnostic::MismatchedTupleStructPatArgCount { pat, expected, found } => {
                let InFile { file_id, value } = pat_syntax(pat)?;
                // cast from Either<Pat, SelfParam> -> Either<_, Pat>
                let ptr = AstPtr::try_from_raw(value.syntax_node_ptr())?;
                let expr_or_pat = InFile { file_id, value: ptr };
                MismatchedTupleStructPatArgCount { expr_or_pat, expected, found }.into()
            }
            InferenceDiagnostic::CastToUnsized { expr, cast_ty } => {
                let expr = expr_syntax(*expr)?;
                CastToUnsized { expr, cast_ty: Type::new(db, def, cast_ty.as_ref()) }.into()
            }
            InferenceDiagnostic::InvalidCast { expr, error, expr_ty, cast_ty } => {
                let expr = expr_syntax(*expr)?;
                let expr_ty = Type::new(db, def, expr_ty.as_ref());
                let cast_ty = Type::new(db, def, cast_ty.as_ref());
                InvalidCast { expr, error: *error, expr_ty, cast_ty }.into()
            }
            InferenceDiagnostic::CannotBeDereferenced { expr, found } => {
                let expr = expr_syntax(*expr)?;
                CannotBeDereferenced { expr, found: Type::new(db, def, found.as_ref()) }.into()
            }
            InferenceDiagnostic::TyDiagnostic { source, diag } => {
                let source_map = match source {
                    InferenceTyDiagnosticSource::Body => source_map,
                    InferenceTyDiagnosticSource::Signature => sig_map,
                };
                Self::ty_diagnostic(diag, source_map, db)?
            }
            InferenceDiagnostic::PathDiagnostic { node, diag } => {
                let source = expr_or_pat_syntax(*node)?;
                let syntax = source.value.to_node(&db.parse_or_expand(source.file_id));
                let path = match_ast! {
                    match (syntax.syntax()) {
                        ast::RecordExpr(it) => it.path()?,
                        ast::RecordPat(it) => it.path()?,
                        ast::TupleStructPat(it) => it.path()?,
                        ast::PathExpr(it) => it.path()?,
                        ast::PathPat(it) => it.path()?,
                        _ => return None,
                    }
                };
                Self::path_diagnostic(diag, source.with_value(path))?
            }
            &InferenceDiagnostic::MethodCallIncorrectGenericsLen {
                expr,
                provided_count,
                expected_count,
                kind,
                def,
            } => {
                let syntax = expr_syntax(expr)?;
                let file_id = syntax.file_id;
                let syntax =
                    syntax.with_value(syntax.value.cast::<ast::MethodCallExpr>()?).to_node(db);
                let generics_or_name = syntax
                    .generic_arg_list()
                    .map(Either::Left)
                    .or_else(|| syntax.name_ref().map(Either::Right))?;
                let generics_or_name = InFile::new(file_id, AstPtr::new(&generics_or_name));
                IncorrectGenericsLen {
                    generics_or_segment: generics_or_name,
                    kind,
                    provided: provided_count,
                    expected: expected_count,
                    def: def.into(),
                }
                .into()
            }
            &InferenceDiagnostic::MethodCallIncorrectGenericsOrder {
                expr,
                param_id,
                arg_idx,
                has_self_arg,
            } => {
                let syntax = expr_syntax(expr)?;
                let file_id = syntax.file_id;
                let syntax =
                    syntax.with_value(syntax.value.cast::<ast::MethodCallExpr>()?).to_node(db);
                let generic_args = syntax.generic_arg_list()?;
                let provided_arg = hir_generic_arg_to_ast(&generic_args, arg_idx, has_self_arg)?;
                let provided_arg = InFile::new(file_id, AstPtr::new(&provided_arg));
                let expected_kind = GenericArgKind::from_id(param_id);
                IncorrectGenericsOrder { provided_arg, expected_kind }.into()
            }
            &InferenceDiagnostic::InvalidLhsOfAssignment { lhs } => {
                let lhs = expr_syntax(lhs)?;
                InvalidLhsOfAssignment { lhs }.into()
            }
            &InferenceDiagnostic::MethodCallIllegalSizedBound { call_expr } => {
                MethodCallIllegalSizedBound { call_expr: expr_syntax(call_expr)? }.into()
            }
            &InferenceDiagnostic::TypeMustBeKnown { at_point, ref top_term } => {
                let at_point = span_syntax(at_point)?;
                let top_term = top_term.as_ref().map(|top_term| match top_term.as_ref().kind() {
                    rustc_type_ir::GenericArgKind::Type(ty) => Either::Left(Type {
                        ty,
                        env: crate::body_param_env_from_has_crate(db, def),
                    }),
                    // FIXME: Printing the const to string is definitely not the correct thing to do here.
                    rustc_type_ir::GenericArgKind::Const(konst) => Either::Right(
                        konst.display(db, DisplayTarget::from_crate(db, def.krate(db))).to_string(),
                    ),
                    rustc_type_ir::GenericArgKind::Lifetime(_) => {
                        unreachable!("we currently don't emit TypeMustBeKnown for lifetimes")
                    }
                });
                TypeMustBeKnown { at_point, top_term }.into()
            }
            &InferenceDiagnostic::UnionExprMustHaveExactlyOneField { expr } => {
                let expr = expr_syntax(expr)?;
                UnionExprMustHaveExactlyOneField { expr }.into()
            }
            InferenceDiagnostic::TypeMismatch { node, expected, found } => {
                let expr_or_pat = expr_or_pat_syntax(*node)?;
                TypeMismatch {
                    expr_or_pat,
                    expected: Type { env, ty: expected.as_ref() },
                    actual: Type { env, ty: found.as_ref() },
                }
                .into()
            }
            InferenceDiagnostic::SolverDiagnostic(d) => {
                let span = span_syntax(d.span)?;
                Self::solver_diagnostic(db, &d.kind, span, env)?
            }
        })
    }

    fn solver_diagnostic(
        db: &'db dyn HirDatabase,
        d: &'db SolverDiagnosticKind,
        span: SpanSyntax,
        env: ParamEnvAndCrate<'db>,
    ) -> Option<AnyDiagnostic<'db>> {
        let interner = DbInterner::new_no_crate(db);
        Some(match d {
            SolverDiagnosticKind::TraitUnimplemented { trait_predicate, root_trait_predicate } => {
                let trait_predicate =
                    crate::TraitPredicate { inner: trait_predicate.get(interner), env };
                let root_trait_predicate =
                    root_trait_predicate.as_ref().map(|root_trait_predicate| {
                        crate::TraitPredicate { inner: root_trait_predicate.get(interner), env }
                    });
                UnimplementedTrait { span, trait_predicate, root_trait_predicate }.into()
            }
        })
    }

    fn path_diagnostic(
        diag: &PathLoweringDiagnostic,
        path: InFile<ast::Path>,
    ) -> Option<AnyDiagnostic<'db>> {
        Some(match *diag {
            PathLoweringDiagnostic::GenericArgsProhibited { segment, reason } => {
                let segment = hir_segment_to_ast_segment(&path.value, segment)?;

                if let Some(rtn) = segment.return_type_syntax() {
                    // RTN errors are emitted as `GenericArgsProhibited` or `ParenthesizedGenericArgsWithoutFnTrait`.
                    return Some(BadRtn { rtn: path.with_value(AstPtr::new(&rtn)) }.into());
                }

                let args = if let Some(generics) = segment.generic_arg_list() {
                    AstPtr::new(&generics).wrap_left()
                } else {
                    AstPtr::new(&segment.parenthesized_arg_list()?).wrap_right()
                };
                let args = path.with_value(args);
                GenericArgsProhibited { args, reason }.into()
            }
            PathLoweringDiagnostic::ParenthesizedGenericArgsWithoutFnTrait { segment } => {
                let segment = hir_segment_to_ast_segment(&path.value, segment)?;

                if let Some(rtn) = segment.return_type_syntax() {
                    // RTN errors are emitted as `GenericArgsProhibited` or `ParenthesizedGenericArgsWithoutFnTrait`.
                    return Some(BadRtn { rtn: path.with_value(AstPtr::new(&rtn)) }.into());
                }

                let args = AstPtr::new(&segment.parenthesized_arg_list()?);
                let args = path.with_value(args);
                ParenthesizedGenericArgsWithoutFnTrait { args }.into()
            }
            PathLoweringDiagnostic::IncorrectGenericsLen {
                generics_source,
                provided_count,
                expected_count,
                kind,
                def,
            } => {
                let generics_or_segment =
                    path_generics_source_to_ast(&path.value, generics_source)?;
                let generics_or_segment = path.with_value(AstPtr::new(&generics_or_segment));
                IncorrectGenericsLen {
                    generics_or_segment,
                    kind,
                    provided: provided_count,
                    expected: expected_count,
                    def: def.into(),
                }
                .into()
            }
            PathLoweringDiagnostic::IncorrectGenericsOrder {
                generics_source,
                param_id,
                arg_idx,
                has_self_arg,
            } => {
                let generic_args =
                    path_generics_source_to_ast(&path.value, generics_source)?.left()?;
                let provided_arg = hir_generic_arg_to_ast(&generic_args, arg_idx, has_self_arg)?;
                let provided_arg = path.with_value(AstPtr::new(&provided_arg));
                let expected_kind = GenericArgKind::from_id(param_id);
                IncorrectGenericsOrder { provided_arg, expected_kind }.into()
            }
            PathLoweringDiagnostic::MissingLifetime { generics_source, expected_count, def }
            | PathLoweringDiagnostic::ElisionFailure { generics_source, expected_count, def } => {
                let generics_or_segment =
                    path_generics_source_to_ast(&path.value, generics_source)?;
                let generics_or_segment = path.with_value(AstPtr::new(&generics_or_segment));
                MissingLifetime { generics_or_segment, expected: expected_count, def: def.into() }
                    .into()
            }
            PathLoweringDiagnostic::ElidedLifetimesInPath {
                generics_source,
                expected_count,
                def,
                hard_error,
            } => {
                let generics_or_segment =
                    path_generics_source_to_ast(&path.value, generics_source)?;
                let generics_or_segment = path.with_value(AstPtr::new(&generics_or_segment));
                ElidedLifetimesInPath {
                    generics_or_segment,
                    expected: expected_count,
                    def: def.into(),
                    hard_error,
                }
                .into()
            }
            PathLoweringDiagnostic::GenericDefaultRefersToSelf { segment } => {
                let segment = hir_segment_to_ast_segment(&path.value, segment)?;
                let segment = path.with_value(AstPtr::new(&segment));
                GenericDefaultRefersToSelf { segment }.into()
            }
        })
    }

    pub(crate) fn ty_diagnostic(
        diag: &TyLoweringDiagnostic,
        source_map: &ExpressionStoreSourceMap,
        db: &'db dyn HirDatabase,
    ) -> Option<AnyDiagnostic<'db>> {
        let Ok(source) = source_map.type_syntax(diag.source) else {
            stdx::never!("error on synthetic type syntax");
            return None;
        };
        let syntax = || source.value.to_node(&db.parse_or_expand(source.file_id));
        Some(match &diag.kind {
            TyLoweringDiagnosticKind::PathDiagnostic(diag) => {
                let ast::Type::PathType(syntax) = syntax() else { return None };
                Self::path_diagnostic(diag, source.with_value(syntax.path()?))?
            }
        })
    }
}

fn path_generics_source_to_ast(
    path: &ast::Path,
    generics_source: PathGenericsSource,
) -> Option<Either<ast::GenericArgList, ast::NameRef>> {
    Some(match generics_source {
        PathGenericsSource::Segment(segment) => {
            let segment = hir_segment_to_ast_segment(path, segment)?;
            segment
                .generic_arg_list()
                .map(Either::Left)
                .or_else(|| segment.name_ref().map(Either::Right))?
        }
        PathGenericsSource::AssocType { segment, assoc_type } => {
            let segment = hir_segment_to_ast_segment(path, segment)?;
            let segment_args = segment.generic_arg_list()?;
            let assoc = hir_assoc_type_binding_to_ast(&segment_args, assoc_type)?;
            assoc
                .generic_arg_list()
                .map(Either::Left)
                .or_else(|| assoc.name_ref().map(Either::Right))?
        }
    })
}
