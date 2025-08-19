//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnostics should
//! be expressed in terms of hir types themselves.
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_def::{
    DefWithBodyId, GenericParamId, SyntheticSyntax,
    expr_store::{
        ExprOrPatPtr, ExpressionStoreSourceMap, hir_assoc_type_binding_to_ast,
        hir_generic_arg_to_ast, hir_segment_to_ast_segment,
    },
    hir::ExprOrPatId,
};
use hir_expand::{HirFileId, InFile, mod_path::ModPath, name::Name};
use hir_ty::{
    CastError, InferenceDiagnostic, InferenceTyDiagnosticSource, PathGenericsSource,
    PathLoweringDiagnostic, TyLoweringDiagnostic, TyLoweringDiagnosticKind,
    db::HirDatabase,
    diagnostics::{BodyValidationDiagnostic, UnsafetyReason},
};
use syntax::{
    AstNode, AstPtr, SyntaxError, SyntaxNodePtr, TextRange,
    ast::{self, HasGenericArgs},
    match_ast,
};
use triomphe::Arc;

use crate::{AssocItem, Field, Function, GenericDef, Local, Trait, Type};

pub use hir_def::VariantId;
pub use hir_ty::{
    GenericArgsProhibitedReason, IncorrectGenericsLenKind,
    diagnostics::{CaseType, IncorrectCase},
};

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
// FIXME Accept something like the following in the macro call instead
// diagnostics![
// pub struct BreakOutsideOfLoop {
//     pub expr: InFile<AstPtr<ast::Expr>>,
//     pub is_break: bool,
//     pub bad_value_break: bool,
// }, ...
// or more concisely
// BreakOutsideOfLoop {
//     expr: InFile<AstPtr<ast::Expr>>,
//     is_break: bool,
//     bad_value_break: bool,
// }, ...
// ]

diagnostics![AnyDiagnostic<'db> ->
    AwaitOutsideOfAsync,
    BreakOutsideOfLoop,
    CastToUnsized<'db>,
    ExpectedFunction<'db>,
    InactiveCode,
    IncoherentImpl,
    IncorrectCase,
    InvalidCast<'db>,
    InvalidDeriveTarget,
    MacroDefError,
    MacroError,
    MacroExpansionParseError,
    MalformedDerive,
    MismatchedArgCount,
    MismatchedTupleStructPatArgCount,
    MissingFields,
    MissingMatchArms,
    MissingUnsafe,
    MovedOutOfRef<'db>,
    NeedMut,
    NonExhaustiveLet,
    NoSuchField,
    PrivateAssocItem,
    PrivateField,
    RemoveTrailingReturn,
    RemoveUnnecessaryElse,
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
    IncorrectGenericsLen,
    IncorrectGenericsOrder,
    MissingLifetime,
    ElidedLifetimesInPath,
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
    pub macro_call: InFile<SyntaxNodePtr>,
    pub precise_location: Option<TextRange>,
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
    pub node: InFile<SyntaxNodePtr>,
    pub precise_location: Option<TextRange>,
    pub message: String,
    pub error: bool,
    pub kind: &'static str,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroExpansionParseError {
    pub node: InFile<SyntaxNodePtr>,
    pub precise_location: Option<TextRange>,
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
    pub node: InFile<SyntaxNodePtr>,
}

#[derive(Debug)]
pub struct MalformedDerive {
    pub node: InFile<SyntaxNodePtr>,
}

#[derive(Debug)]
pub struct NoSuchField {
    pub field: InFile<AstPtr<Either<ast::RecordExprField, ast::RecordPatField>>>,
    pub private: Option<Field>,
    pub variant: VariantId,
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
pub struct ExpectedFunction<'db> {
    pub call: InFile<ExprOrPatPtr>,
    pub found: Type<'db>,
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
    pub missed_fields: Vec<Name>,
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

impl<'db> AnyDiagnostic<'db> {
    pub(crate) fn body_validation_diagnostic(
        db: &'db dyn HirDatabase,
        diagnostic: BodyValidationDiagnostic,
        source_map: &hir_def::expr_store::BodySourceMap,
    ) -> Option<AnyDiagnostic<'db>> {
        match diagnostic {
            BodyValidationDiagnostic::RecordMissingFields { record, variant, missed_fields } => {
                let variant_data = variant.fields(db);
                let missed_fields = missed_fields
                    .into_iter()
                    .map(|idx| variant_data.fields()[idx].name.clone())
                    .collect();

                let record = match record {
                    Either::Left(record_expr) => source_map.expr_syntax(record_expr).ok()?,
                    Either::Right(record_pat) => source_map.pat_syntax(record_pat).ok()?,
                };
                let file = record.file_id;
                let root = record.file_syntax(db);
                match record.value.to_node(&root) {
                    Either::Left(ast::Expr::RecordExpr(record_expr)) => {
                        if record_expr.record_expr_field_list().is_some() {
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
                    }
                    Either::Right(ast::Pat::RecordPat(record_pat)) => {
                        if record_pat.record_pat_field_list().is_some() {
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
                match source_map.expr_syntax(match_expr) {
                    Ok(source_ptr) => {
                        let root = source_ptr.file_syntax(db);
                        if let Either::Left(ast::Expr::MatchExpr(match_expr)) =
                            &source_ptr.value.to_node(&root)
                        {
                            match match_expr.expr() {
                                Some(scrut_expr) if match_expr.match_arm_list().is_some() => {
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
                                _ => {}
                            }
                        }
                    }
                    Err(SyntheticSyntax) => (),
                }
            }
            BodyValidationDiagnostic::NonExhaustiveLet { pat, uncovered_patterns } => {
                match source_map.pat_syntax(pat) {
                    Ok(source_ptr) => {
                        if let Some(ast_pat) = source_ptr.value.cast::<ast::Pat>() {
                            return Some(
                                NonExhaustiveLet {
                                    pat: InFile::new(source_ptr.file_id, ast_pat),
                                    uncovered_patterns,
                                }
                                .into(),
                            );
                        }
                    }
                    Err(SyntheticSyntax) => {}
                }
            }
            BodyValidationDiagnostic::RemoveTrailingReturn { return_expr } => {
                if let Ok(source_ptr) = source_map.expr_syntax(return_expr) {
                    // Filters out desugared return expressions (e.g. desugared try operators).
                    if let Some(ptr) = source_ptr.value.cast::<ast::ReturnExpr>() {
                        return Some(
                            RemoveTrailingReturn {
                                return_expr: InFile::new(source_ptr.file_id, ptr),
                            }
                            .into(),
                        );
                    }
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
        }
        None
    }

    pub(crate) fn inference_diagnostic(
        db: &'db dyn HirDatabase,
        def: DefWithBodyId,
        d: &InferenceDiagnostic,
        source_map: &hir_def::expr_store::BodySourceMap,
        sig_map: &hir_def::expr_store::ExpressionStoreSourceMap,
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
        let expr_or_pat_syntax = |id| match id {
            ExprOrPatId::ExprId(expr) => expr_syntax(expr),
            ExprOrPatId::PatId(pat) => pat_syntax(pat),
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
                ExpectedFunction { call: call_expr, found: Type::new(db, def, found.clone()) }
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
                    receiver: Type::new(db, def, receiver.clone()),
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
                    receiver: Type::new(db, def, receiver.clone()),
                    field_with_same_name: field_with_same_name
                        .clone()
                        .map(|ty| Type::new(db, def, ty)),
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
            InferenceDiagnostic::TypedHole { expr, expected } => {
                let expr = expr_syntax(*expr)?;
                TypedHole { expr, expected: Type::new(db, def, expected.clone()) }.into()
            }
            &InferenceDiagnostic::MismatchedTupleStructPatArgCount { pat, expected, found } => {
                let expr_or_pat = match pat {
                    ExprOrPatId::ExprId(expr) => expr_syntax(expr)?,
                    ExprOrPatId::PatId(pat) => {
                        let InFile { file_id, value } = pat_syntax(pat)?;

                        // cast from Either<Pat, SelfParam> -> Either<_, Pat>
                        let ptr = AstPtr::try_from_raw(value.syntax_node_ptr())?;
                        InFile { file_id, value: ptr }
                    }
                };
                MismatchedTupleStructPatArgCount { expr_or_pat, expected, found }.into()
            }
            InferenceDiagnostic::CastToUnsized { expr, cast_ty } => {
                let expr = expr_syntax(*expr)?;
                CastToUnsized { expr, cast_ty: Type::new(db, def, cast_ty.clone()) }.into()
            }
            InferenceDiagnostic::InvalidCast { expr, error, expr_ty, cast_ty } => {
                let expr = expr_syntax(*expr)?;
                let expr_ty = Type::new(db, def, expr_ty.clone());
                let cast_ty = Type::new(db, def, cast_ty.clone());
                InvalidCast { expr, error: *error, expr_ty, cast_ty }.into()
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
