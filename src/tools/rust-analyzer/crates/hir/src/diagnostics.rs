//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnostics should
//! be expressed in terms of hir types themselves.
use std::mem::discriminant;

use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_def::{
    AdtId, AssocItemId, DefWithBodyId, EnumId, EnumVariantId, GenericDefId, GenericParamId, ImplId,
    Lookup, MacroId, ModuleDefId, ModuleId, SyntheticSyntax, TraitId,
    attrs::AttrFlags,
    expr_store::{
        Body, ExprOrPatPtr, ExpressionStore, ExpressionStoreDiagnostics, ExpressionStoreSourceMap,
        hir_assoc_type_binding_to_ast, hir_generic_arg_to_ast, hir_segment_to_ast_segment,
    },
    hir::{ExprId, ExprOrPatId, PatId},
    nameres::{
        DefMap,
        assoc::{ImplItems, TraitItems},
        diagnostics::{DefDiagnosticKind, DefDiagnostics},
    },
    signatures::{
        ConstSignature, FunctionSignature, ImplFlags, ImplSignature, TraitFlags, TraitSignature,
        TypeAliasSignature,
    },
    type_ref::TypeRefId,
    unstable_features::UnstableFeatures,
};
use hir_expand::{
    HirFileId, InFile, MacroCallId, MacroCallKind, MacroKind, RenderedExpandError, ValueResult,
    mod_path::ModPath, name::Name,
};
use hir_ty::{
    CastError, ExplicitDropMethodUseKind, InferBodyId, InferenceDiagnostic, InferenceResult,
    ParamEnvAndCrate, PathGenericsSource, PathLoweringDiagnostic, TyLoweringDiagnostic,
    check_orphan_rules,
    db::{AnonConstId, HirDatabase, signature_anon_consts_and_diagnostics},
    diagnostics::{BodyValidationDiagnostic, UnsafetyReason},
    display::{DisplayTarget, HirDisplay},
    method_resolution::TraitImpls,
    next_solver::{
        DbInterner, EarlyBinder, TyKind, TypingMode,
        infer::{DbInternerInferExt, InferCtxt},
    },
    solver_errors::SolverDiagnosticKind,
    traits::{is_inherent_impl_coherent, structurally_normalize_ty},
};
use rustc_type_ir::inherent::IntoKind as _;
use span::Edition;
use stdx::{impl_from, never};
use syntax::{
    AstNode, AstPtr, SyntaxError, SyntaxNodePtr, TextRange,
    ast::{self, HasGenericArgs, HasName},
    match_ast,
};
use triomphe::Arc;

use crate::{
    AnyFunctionId, AssocItem, Field, Function, GenericDef, Trait, Type, TypeOwnerId, Variant,
    struct_tail_raw,
};

pub use hir_def::{VariantId, expr_store::MissingBodyItemKind};
pub use hir_ty::{
    GenericArgsProhibitedReason, IncorrectGenericsLenKind, ReturnKind,
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
    ArrayPatternWithoutFixedLength,
    AwaitOutsideOfAsync,
    BreakOutsideOfLoop,
    CannotBeDereferenced<'db>,
    UnaryOperatorCannotBeApplied<'db>,
    CannotImplicitlyDerefTraitObject<'db>,
    CannotIndexInto<'db>,
    CastToUnsized<'db>,
    ExpectedArrayOrSlicePat<'db>,
    ExpectedFunction<'db>,
    ExplicitDropMethodUse,
    FruInDestructuringAssignment,
    MissingBody,
    FunctionalRecordUpdateOnNonStruct,
    GenericDefaultRefersToSelf,
    InactiveCode,
    IncoherentImpl,
    IncorrectCase,
    IncorrectGenericsLen,
    IncorrectGenericsOrder,
    InferVarsNotAllowed,
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
    MutRefInImmRefPat,
    MutableRefBinding,
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
    GenericArgsProhibited,
    ParenthesizedGenericArgsWithoutFnTrait,
    BadRtn,
    MissingLifetime,
    ElidedLifetimesInPath,
    TypeMustBeKnown<'db>,
    UnionExprMustHaveExactlyOneField,
    UnionPatMustHaveExactlyOneField,
    UnionPatHasRest,
    UnimplementedTrait<'db>,
    YieldOutsideCoroutine,
    ReturnOutsideFunction,
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
    pub expected: u64,
    pub found: u64,
    pub has_rest: bool,
}

#[derive(Debug)]
pub struct ArrayPatternWithoutFixedLength {
    pub pat: InFile<ExprOrPatPtr>,
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
pub struct UnaryOperatorCannotBeApplied<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub op: ast::UnaryOp,
    pub found: Type<'db>,
}

#[derive(Debug)]
pub struct MutRefInImmRefPat {
    pub pat: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct CannotImplicitlyDerefTraitObject<'db> {
    pub pat: InFile<ExprOrPatPtr>,
    pub found: Type<'db>,
}

#[derive(Debug)]
pub struct CannotIndexInto<'db> {
    pub expr: InFile<ExprOrPatPtr>,
    pub found: Type<'db>,
}

#[derive(Debug)]
pub struct ExplicitDropMethodUse {
    pub expr_or_path: Either<InFile<AstPtr<ast::MethodCallExpr>>, InFile<AstPtr<ast::Path>>>,
}

#[derive(Debug)]
pub struct FruInDestructuringAssignment {
    pub node: InFile<AstPtr<ast::Expr>>,
}

#[derive(Debug)]
pub struct MissingBody {
    pub node: InFile<SyntaxNodePtr>,
    pub kind: MissingBodyItemKind,
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
    /// True when the call is through a `Fn`/`FnMut`/`FnOnce` trait (E0057)
    /// rather than a regular function call (E0061).
    pub is_fn_trait_call: bool,
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
pub struct InferVarsNotAllowed {
    pub node: InFile<SyntaxNodePtr>,
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
pub struct UnionPatMustHaveExactlyOneField {
    pub pat: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct UnionPatHasRest {
    pub pat: InFile<ExprOrPatPtr>,
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
    pub parent_trait_predicates: Vec<crate::TraitPredicate<'db>>,
}

#[derive(Debug)]
pub struct MutableRefBinding {
    pub pat: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct YieldOutsideCoroutine {
    pub expr: InFile<ExprOrPatPtr>,
}

#[derive(Debug)]
pub struct ReturnOutsideFunction {
    pub expr: InFile<ExprOrPatPtr>,
    pub kind: ReturnKind,
}

pub(crate) struct DiagnosticsCollector<'a, 'db> {
    db: &'db dyn HirDatabase,
    krate: base_db::Crate,
    edition: Edition,
    style_lints: bool,
    acc: &'a mut Vec<AnyDiagnostic<'db>>,
}

fn precise_macro_call_location(
    ast: &MacroCallKind,
    db: &dyn HirDatabase,
    krate: base_db::Crate,
) -> InFile<TextRange> {
    // FIXME: maybe we actually want slightly different ranges for the different macro diagnostics
    // - e.g. the full attribute for macro errors, but only the name for name resolution
    match ast {
        MacroCallKind::FnLike { ast_id, .. } => {
            let node = ast_id.to_node(db);
            let range = node
                .path()
                .and_then(|it| it.segment())
                .and_then(|it| it.name_ref())
                .map(|it| it.syntax().text_range());
            let range = range.unwrap_or_else(|| node.syntax().text_range());
            ast_id.with_value(range)
        }
        MacroCallKind::Derive { ast_id, derive_attr_index, derive_index, .. } => {
            let range = derive_attr_index.find_derive_range(db, krate, *ast_id, *derive_index);
            ast_id.with_value(range)
        }
        MacroCallKind::Attr { ast_id, censored_attr_ids: attr_ids, .. } => {
            let attr_range =
                attr_ids.invoc_attr().find_attr_range(db, krate, *ast_id).1.syntax().text_range();
            ast_id.with_value(attr_range)
        }
    }
}

impl<'a, 'db> DiagnosticsCollector<'a, 'db> {
    pub(crate) fn collect(
        db: &'db dyn HirDatabase,
        module: ModuleId,
        acc: &'a mut Vec<AnyDiagnostic<'db>>,
        style_lints: bool,
    ) {
        let krate = module.krate(db);
        DiagnosticsCollector { db, krate, edition: krate.data(db).edition, style_lints, acc }
            .collect_module(module);
    }

    fn emit_def_diagnostic(&mut self, diag: &DefDiagnosticKind) {
        match diag {
            DefDiagnosticKind::UnresolvedModule { ast: declaration, candidates } => {
                let decl = declaration.to_ptr(self.db);
                self.acc.push(
                    UnresolvedModule {
                        decl: InFile::new(declaration.file_id, decl),
                        candidates: candidates.clone(),
                    }
                    .into(),
                )
            }
            DefDiagnosticKind::UnresolvedExternCrate { ast } => {
                let item = ast.to_ptr(self.db);
                self.acc
                    .push(UnresolvedExternCrate { decl: InFile::new(ast.file_id, item) }.into());
            }

            DefDiagnosticKind::MacroError { ast, path, err } => {
                let item = ast.to_ptr(self.db);
                let RenderedExpandError { message, error, kind } = err.render_to_string(self.db);
                self.acc.push(
                    MacroError {
                        range: InFile::new(ast.file_id, item.text_range()),
                        message: format!("{}: {message}", path.display(self.db, self.edition)),
                        error,
                        kind,
                    }
                    .into(),
                )
            }
            DefDiagnosticKind::UnresolvedImport { id, index } => {
                let file_id = id.file_id;

                let use_tree = hir_def::src::use_tree_to_ast(self.db, *id, *index);
                self.acc.push(
                    UnresolvedImport { decl: InFile::new(file_id, AstPtr::new(&use_tree)) }.into(),
                );
            }

            DefDiagnosticKind::UnconfiguredCode { ast_id, cfg, opts } => {
                let ast_id_map = ast_id.file_id.ast_id_map(self.db);
                let ptr = ast_id_map.get_erased(ast_id.value);
                self.acc.push(
                    InactiveCode {
                        node: InFile::new(ast_id.file_id, ptr),
                        cfg: cfg.clone(),
                        opts: opts.clone(),
                    }
                    .into(),
                );
            }
            DefDiagnosticKind::UnresolvedMacroCall { ast, path } => {
                let location = precise_macro_call_location(ast, self.db, self.krate);
                self.acc.push(
                    UnresolvedMacroCall {
                        range: location,
                        path: path.clone(),
                        is_bang: matches!(ast, MacroCallKind::FnLike { .. }),
                    }
                    .into(),
                );
            }
            DefDiagnosticKind::UnimplementedBuiltinMacro { ast } => {
                let node = ast.to_node(self.db);
                // Must have a name, otherwise we wouldn't emit it.
                let name = node.name().expect("unimplemented builtin macro with no name");
                self.acc.push(
                    UnimplementedBuiltinMacro {
                        node: ast.with_value(SyntaxNodePtr::from(AstPtr::new(&name))),
                    }
                    .into(),
                );
            }
            DefDiagnosticKind::InvalidDeriveTarget { ast, id } => {
                let (_, attr) = id.find_attr_range(self.db, self.krate, *ast);
                let derive = attr
                    .path()
                    .map(|path| path.syntax().text_range())
                    .unwrap_or_else(|| attr.syntax().text_range());
                self.acc.push(InvalidDeriveTarget { range: ast.with_value(derive) }.into());
            }
            DefDiagnosticKind::MalformedDerive { ast, id } => {
                let derive = id.find_attr_range(self.db, self.krate, *ast).1.syntax().text_range();
                self.acc.push(MalformedDerive { range: ast.with_value(derive) }.into());
            }
            DefDiagnosticKind::MacroDefError { ast, message } => {
                let node = ast.to_node(self.db);
                self.acc.push(
                    MacroDefError {
                        node: InFile::new(ast.file_id, AstPtr::new(&node)),
                        name: node.name().map(|it| it.syntax().text_range()),
                        message: message.clone(),
                    }
                    .into(),
                );
            }
        }
    }

    fn emit_def_diagnostics(&mut self, diagnostics: &DefDiagnostics) {
        diagnostics.iter().for_each(|diag| self.emit_def_diagnostic(&diag.kind));
    }

    fn emit_case_diagnostics(&mut self, def: ModuleDefId) {
        self.acc
            .extend(hir_ty::diagnostics::incorrect_case(self.db, def).into_iter().map(Into::into));
    }

    fn collect_macro_call(&mut self, macro_call_id: MacroCallId) {
        let Some(e) = macro_call_id.parse_macro_expansion_error(self.db) else {
            return;
        };
        let ValueResult { value: parse_errors, err } = e;
        if let Some(err) = err {
            let loc = macro_call_id.loc(self.db);
            let file_id = loc.kind.file_id();
            let mut range = precise_macro_call_location(&loc.kind, self.db, loc.krate);
            let RenderedExpandError { message, error, kind } = err.render_to_string(self.db);
            if Some(err.span().anchor.file_id)
                == file_id.file_id().map(|it| it.span_file_id(self.db))
            {
                range.value = err.span().range
                    + file_id
                        .ast_id_map(self.db)
                        .get_erased(err.span().anchor.ast_id)
                        .text_range()
                        .start();
            }
            self.acc.push(MacroError { range, message, error, kind }.into());
        }

        if !parse_errors.is_empty() {
            let loc = macro_call_id.loc(self.db);
            let range = precise_macro_call_location(&loc.kind, self.db, loc.krate);
            self.acc.push(MacroExpansionParseError { range, errors: parse_errors.clone() }.into())
        }
    }

    fn collect_assoc_items(&mut self, defs: &[(Name, AssocItemId)], def_map: &DefMap) {
        for &(_, def) in defs {
            self.collect_module_def(def.into(), def_map);
        }
    }

    fn collect_trait(&mut self, def: TraitId, def_map: &DefMap) {
        let (signature, source_map) = TraitSignature::with_source_map(self.db, def);
        let items = TraitItems::query_with_diagnostics(self.db, def);

        self.collect_generic_def(&signature.store, source_map, def.into());
        self.emit_def_diagnostics(&items.1);
        items.0.macro_calls().for_each(|(_, call)| self.collect_macro_call(call));
        self.collect_assoc_items(&items.0.items, def_map);
    }

    fn collect_impl(
        &mut self,
        def: ImplId,
        infcx: &InferCtxt<'db>,
        def_map: &'db DefMap,
        impl_assoc_items_scratch: &mut Vec<(Name, AssocItemId)>,
    ) {
        let (impl_signature, source_map) = ImplSignature::with_source_map(self.db, def);
        let impl_items = ImplItems::of(self.db, def);

        self.collect_generic_def(&impl_signature.store, source_map, def.into());
        self.emit_def_diagnostics(&impl_items.1);
        impl_items.0.macro_calls().for_each(|(_, call)| self.collect_macro_call(call));
        self.collect_assoc_items(&impl_items.0.items, def_map);

        let loc = def.lookup(self.db);

        let file_id = loc.id.file_id;
        if file_id.macro_file().is_some_and(|it| it.kind(self.db) == MacroKind::DeriveBuiltIn) {
            // these expansion come from us, diagnosing them is a waste of resources
            // FIXME: Once we diagnose the inputs to builtin derives, we should at least extract those diagnostics somehow
            return;
        }

        let ast_id_map = file_id.ast_id_map(self.db);

        let trait_impl = impl_signature.target_trait.is_some();
        if !trait_impl && !is_inherent_impl_coherent(self.db, def_map, def) {
            self.acc.push(IncoherentImpl { impl_: ast_id_map.get(loc.id.value), file_id }.into())
        }

        if trait_impl && !check_orphan_rules(self.db, def) {
            self.acc.push(TraitImplOrphan { impl_: ast_id_map.get(loc.id.value), file_id }.into())
        }

        let trait_ = trait_impl
            .then(|| self.db.impl_trait(def))
            .flatten()
            .map(|trait_ref| trait_ref.instantiate_identity().skip_norm_wip().def_id.0);
        let mut trait_is_unsafe = trait_.is_some_and(|trait_| {
            TraitSignature::of(self.db, trait_).flags.contains(TraitFlags::UNSAFE)
        });
        let impl_is_negative = impl_signature.is_negative();
        let impl_is_unsafe = impl_signature.flags.contains(ImplFlags::UNSAFE);

        let trait_is_unresolved = trait_.is_none() && trait_impl;
        if trait_is_unresolved {
            // Ignore trait safety errors when the trait is unresolved, as otherwise we'll treat it as safe,
            // which may not be correct.
            trait_is_unsafe = impl_is_unsafe;
        }

        let drop_maybe_dangle = (|| {
            let trait_ = trait_?;
            let drop_trait = infcx.interner.lang_items().Drop?;
            if drop_trait != trait_ {
                return None;
            }
            let parent = def.into();
            let (lifetimes_attrs, type_and_consts_attrs) =
                AttrFlags::query_generic_params(self.db, parent);
            let res = lifetimes_attrs.values().any(|it| it.contains(AttrFlags::MAY_DANGLE))
                || type_and_consts_attrs.values().any(|it| it.contains(AttrFlags::MAY_DANGLE));
            Some(res)
        })()
        .unwrap_or(false);

        match (impl_is_unsafe, trait_is_unsafe, impl_is_negative, drop_maybe_dangle) {
                // unsafe negative impl
                (true, _, true, _) |
                // unsafe impl for safe trait
                (true, false, _, false) => self.acc.push(TraitImplIncorrectSafety { impl_: ast_id_map.get(loc.id.value), file_id, should_be_safe: true }.into()),
                // safe impl for unsafe trait
                (false, true, false, _) |
                // safe impl of dangling drop
                (false, false, _, true) => self.acc.push(TraitImplIncorrectSafety { impl_: ast_id_map.get(loc.id.value), file_id, should_be_safe: false }.into()),
                _ => (),
            };

        // Negative impls can't have items, don't emit missing items diagnostic for them
        if let (false, Some(trait_)) = (impl_is_negative, trait_) {
            let trait_items = &trait_.trait_items(self.db).items;
            let required_items = trait_items.iter().filter(|&(_, assoc)| match *assoc {
                AssocItemId::FunctionId(it) => !FunctionSignature::of(self.db, it).has_body(),
                AssocItemId::ConstId(id) => !ConstSignature::of(self.db, id).has_body(),
                AssocItemId::TypeAliasId(it) => TypeAliasSignature::of(self.db, it).ty.is_none(),
            });
            impl_assoc_items_scratch.extend(impl_items.0.items.iter().cloned());

            let redundant = impl_assoc_items_scratch
                .iter()
                .filter(|(name, id)| {
                    !trait_items.iter().any(|(impl_name, impl_item)| {
                        discriminant(impl_item) == discriminant(id) && impl_name == name
                    })
                })
                .map(|(name, item)| (name.clone(), AssocItem::from(*item)));
            for (name, assoc_item) in redundant {
                self.acc.push(
                    TraitImplRedundantAssocItems {
                        trait_: trait_.into(),
                        file_id,
                        impl_: ast_id_map.get(loc.id.value),
                        assoc_item: (name, assoc_item),
                    }
                    .into(),
                )
            }

            let mut missing: Vec<_> = required_items
                .filter(|(name, id)| {
                    !impl_assoc_items_scratch.iter().any(|(impl_name, impl_item)| {
                        discriminant(impl_item) == discriminant(id) && impl_name == name
                    })
                })
                .map(|(name, item)| (name.clone(), AssocItem::from(*item)))
                .collect();

            if !missing.is_empty() {
                let env = ParamEnvAndCrate {
                    param_env: self.db.trait_environment(def.into()),
                    krate: self.krate,
                };
                let self_ty = self.db.impl_self_ty(def).instantiate_identity().skip_norm_wip();
                let self_ty = structurally_normalize_ty(infcx, self_ty, env.param_env);
                let tail_ty = struct_tail_raw(self.db, infcx.interner, self_ty, |ty| {
                    structurally_normalize_ty(infcx, ty, env.param_env)
                });
                let self_ty_is_guaranteed_unsized =
                    matches!(tail_ty.kind(), TyKind::Dynamic(..) | TyKind::Slice(..) | TyKind::Str);
                if self_ty_is_guaranteed_unsized {
                    missing.retain(|(_, assoc_item)| {
                            let assoc_item = match *assoc_item {
                                AssocItem::Function(it) => match it.id {
                                    AnyFunctionId::FunctionId(id) => id.into(),
                                    AnyFunctionId::BuiltinDeriveImplMethod { .. } => {
                                        never!("should not have an `AnyFunctionId::BuiltinDeriveImplMethod` here");
                                        return false;
                                    },
                                },
                                AssocItem::Const(it) => it.id.into(),
                                AssocItem::TypeAlias(it) => it.id.into(),
                            };
                            !hir_ty::dyn_compatibility::generics_require_sized_self(self.db, assoc_item)
                        });
                }
            }

            // HACK: When specialization is enabled in the current crate, and there exists
            // *any* blanket impl that provides a default implementation for the missing item,
            // suppress the missing associated item diagnostic.
            // This can lead to false negatives when the impl in question does not actually
            // specialize that blanket impl, but determining the exact specialization
            // relationship here would be significantly more expensive.
            if !missing.is_empty() {
                let features = UnstableFeatures::query(self.db, self.krate);
                if features.specialization || features.min_specialization {
                    missing.retain(|(assoc_name, assoc_item)| {
                        let AssocItem::Function(_) = assoc_item else {
                            return true;
                        };

                        for &impl_ in
                            TraitImpls::for_crate(self.db, self.krate).blanket_impls(trait_)
                        {
                            if impl_ == def {
                                continue;
                            }

                            for (name, item) in &impl_.impl_items(self.db).items {
                                let AssocItemId::FunctionId(fn_) = item else {
                                    continue;
                                };
                                if name != assoc_name {
                                    continue;
                                }

                                if FunctionSignature::of(self.db, *fn_).is_default() {
                                    return false;
                                }
                            }
                        }

                        true
                    });
                }
            }

            if !missing.is_empty() {
                self.acc.push(
                    TraitImplMissingAssocItems {
                        impl_: ast_id_map.get(loc.id.value),
                        file_id,
                        missing,
                    }
                    .into(),
                )
            }
            impl_assoc_items_scratch.clear();
        }
    }

    fn collect_module(&mut self, def: ModuleId) {
        let _p = tracing::info_span!("diagnostics", name = ?def.name(self.db)).entered();

        let def_map = def.def_map(self.db);
        let scope = &def_map[def].scope;

        for diag in def_map.diagnostics() {
            if diag.in_module != def {
                // FIXME: This is accidentally quadratic.
                continue;
            }
            self.emit_def_diagnostic(&diag.kind);
        }

        if !def.is_block_module(self.db) {
            // These are reported by the body of block modules
            scope.all_macro_calls().for_each(|call| self.collect_macro_call(call));
        }

        scope
            .declarations()
            .chain(scope.unnamed_consts().map(ModuleDefId::ConstId))
            .for_each(|def| self.collect_module_def(def, def_map));

        scope.legacy_macros().flat_map(|(_, it)| it).for_each(|&def| {
            self.emit_case_diagnostics(def.into());
            self.collect_macro_def(def);
        });

        let interner = DbInterner::new_with(self.db, self.krate);
        let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());
        let mut impl_assoc_items_scratch = Vec::new();
        scope.impls().for_each(|def| {
            impl_assoc_items_scratch.clear();
            self.collect_impl(def, &infcx, def_map, &mut impl_assoc_items_scratch)
        });
    }

    fn collect_macro_def(&mut self, def: MacroId) {
        let id = def.definition(self.db);
        if let hir_expand::MacroDefKind::Declarative(ast, _) = id.kind
            && let expander = ast.decl_macro_expander(self.db, id.krate)
            && let Some(e) = expander.mac.err()
        {
            self.emit_def_diagnostic(&DefDiagnosticKind::MacroDefError {
                ast,
                message: e.to_string(),
            });
        }
    }

    fn collect_anon_const(&mut self, source_map: &ExpressionStoreSourceMap, def: AnonConstId<'db>) {
        self.emit_inference_errors(def.into(), source_map, def.into());
    }

    fn collect_enum(&mut self, def: EnumId) {
        self.collect_only_generic_def(def);

        let variants = def.enum_variants_with_diagnostics(self.db);
        variants.0.variants.values().for_each(|&(def, _)| self.collect_enum_variant(def));

        let file = def.lookup(self.db).id.file_id;
        let ast_id_map = file.ast_id_map(self.db);
        for diag in &variants.1 {
            self.acc.push(
                InactiveCode {
                    node: InFile::new(file, ast_id_map.get(diag.ast_id).syntax_node_ptr()),
                    cfg: diag.cfg.clone(),
                    opts: diag.opts.clone(),
                }
                .into(),
            );
        }
    }

    fn collect_enum_variant(&mut self, def: EnumVariantId) {
        self.collect_def_with_body(
            def.into(),
            TypeOwnerId::GenericDefId(def.loc(self.db).parent.into()),
        );
        self.collect_variant(def.into());
    }

    fn collect_anon_consts_and_ty_diagnostics(
        &mut self,
        source_map: &ExpressionStoreSourceMap,
        anon_consts: &[AnonConstId<'db>],
        diagnostics: &[TyLoweringDiagnostic],
    ) {
        anon_consts.iter().for_each(|&anon_const| self.collect_anon_const(source_map, anon_const));

        diagnostics
            .iter()
            .filter_map(|diag| AnyDiagnostic::ty_diagnostic(diag, source_map, self.db))
            .for_each(|diag| self.acc.push(diag));
    }

    fn collect_generic_def(
        &mut self,
        store: &ExpressionStore,
        source_map: &ExpressionStoreSourceMap,
        def: GenericDefId,
    ) {
        self.collect_expr_store(store, source_map);
        for (anon_consts, diagnostics) in signature_anon_consts_and_diagnostics(self.db, def) {
            self.collect_anon_consts_and_ty_diagnostics(source_map, anon_consts, diagnostics);
        }
    }

    fn collect_def_with_body(&mut self, def: DefWithBodyId, type_owner: TypeOwnerId<'db>) {
        let (body, source_map) = Body::with_source_map(self.db, def);

        self.collect_expr_store(body, source_map);
        self.emit_inference_errors(def.into(), source_map, type_owner);

        // FIXME: Missing unsafe and body validation should be defined for any `InferBodyId`.
        let missing_unsafe = hir_ty::diagnostics::missing_unsafe(self.db, def);
        for (node, reason) in missing_unsafe.unsafe_exprs {
            match source_map.expr_or_pat_syntax(node) {
                Ok(node) => self.acc.push(
                    MissingUnsafe {
                        node,
                        lint: if missing_unsafe.fn_is_unsafe {
                            UnsafeLint::UnsafeOpInUnsafeFn
                        } else {
                            UnsafeLint::HardError
                        },
                        reason,
                    }
                    .into(),
                ),
                Err(SyntheticSyntax) => {
                    // FIXME: Here and elsewhere in this file, the `expr` was
                    // desugared, report or assert that this doesn't happen.
                }
            }
        }
        for node in missing_unsafe.deprecated_safe_calls {
            match source_map.expr_syntax(node) {
                Ok(node) => self.acc.push(
                    MissingUnsafe {
                        node,
                        lint: UnsafeLint::DeprecatedSafe2024,
                        reason: UnsafetyReason::UnsafeFnCall,
                    }
                    .into(),
                ),
                Err(SyntheticSyntax) => never!("synthetic DeprecatedSafe2024"),
            }
        }

        for diagnostic in BodyValidationDiagnostic::collect(self.db, def, self.style_lints) {
            self.acc
                .extend(AnyDiagnostic::body_validation_diagnostic(self.db, diagnostic, source_map));
        }
    }

    fn emit_inference_errors(
        &mut self,
        def: InferBodyId<'db>,
        source_map: &ExpressionStoreSourceMap,
        type_owner: TypeOwnerId<'db>,
    ) {
        let infer = InferenceResult::of(self.db, def);

        self.acc.extend(infer.diagnostics().iter().filter_map(|diag| {
            AnyDiagnostic::inference_diagnostic(
                self.db,
                self.krate,
                self.edition,
                diag,
                source_map,
                type_owner,
            )
        }));
    }

    fn collect_variant(&mut self, def: VariantId) {
        let (fields, source_map) = def.fields_with_source_map(self.db);
        self.collect_expr_store(&fields.store, source_map);

        let lowering = self.db.field_types_with_diagnostics(def);
        self.collect_anon_consts_and_ty_diagnostics(
            source_map,
            lowering.defined_anon_consts(),
            lowering.diagnostics(),
        );
    }

    fn collect_generic_def_with_body(
        &mut self,
        def: impl Into<GenericDefId> + Into<DefWithBodyId> + Copy,
    ) {
        let generic_def: GenericDefId = def.into();
        let (signature_store, signature_source_map) =
            ExpressionStore::with_source_map(self.db, generic_def.into());
        self.collect_generic_def(signature_store, signature_source_map, generic_def);

        let def_with_body: DefWithBodyId = def.into();
        self.collect_def_with_body(def_with_body, generic_def.into());
    }

    fn collect_generic_variant(&mut self, def: impl Into<GenericDefId> + Into<VariantId> + Copy) {
        let generic_def: GenericDefId = def.into();
        let (store, source_map) = ExpressionStore::with_source_map(self.db, generic_def.into());
        self.collect_generic_def(store, source_map, generic_def);
        self.collect_variant(def.into());
    }

    fn collect_only_generic_def(&mut self, def: impl Into<GenericDefId>) {
        let generic_def: GenericDefId = def.into();
        let (store, source_map) = ExpressionStore::with_source_map(self.db, generic_def.into());
        self.collect_generic_def(store, source_map, generic_def);
    }

    fn collect_module_def(&mut self, def: ModuleDefId, def_map: &DefMap) {
        self.emit_case_diagnostics(def);

        match def {
            ModuleDefId::ModuleId(def) => {
                // Only add diagnostics from inline modules
                if def_map[def].origin.is_inline() {
                    self.collect_module(def);
                }
            }
            ModuleDefId::TraitId(def) => self.collect_trait(def, def_map),
            ModuleDefId::MacroId(def) => self.collect_macro_def(def),
            ModuleDefId::FunctionId(def) => self.collect_generic_def_with_body(def),
            ModuleDefId::ConstId(def) => self.collect_generic_def_with_body(def),
            ModuleDefId::StaticId(def) => self.collect_generic_def_with_body(def),
            ModuleDefId::EnumVariantId(def) => self.collect_enum_variant(def),
            ModuleDefId::AdtId(AdtId::StructId(def)) => self.collect_generic_variant(def),
            ModuleDefId::AdtId(AdtId::UnionId(def)) => self.collect_generic_variant(def),
            ModuleDefId::AdtId(AdtId::EnumId(def)) => self.collect_enum(def),
            ModuleDefId::TypeAliasId(def) => self.collect_only_generic_def(def),
            ModuleDefId::BuiltinType(_) => {}
        }
    }

    fn collect_expr_store(
        &mut self,
        store: &ExpressionStore,
        source_map: &ExpressionStoreSourceMap,
    ) {
        for (_, def_map) in store.blocks(self.db) {
            self.collect_module(def_map.root_module_id());
        }

        for diag in source_map.diagnostics() {
            self.acc.push(match diag {
                ExpressionStoreDiagnostics::InactiveCode { node, cfg, opts } => {
                    InactiveCode { node: *node, cfg: cfg.clone(), opts: opts.clone() }.into()
                }
                ExpressionStoreDiagnostics::UnresolvedMacroCall { node, path } => {
                    UnresolvedMacroCall {
                        range: node.map(|ptr| ptr.text_range()),
                        path: path.clone(),
                        is_bang: true,
                    }
                    .into()
                }
                ExpressionStoreDiagnostics::AwaitOutsideOfAsync { node, location } => {
                    AwaitOutsideOfAsync { node: *node, location: location.clone() }.into()
                }
                ExpressionStoreDiagnostics::UnreachableLabel { node, name } => {
                    UnreachableLabel { node: *node, name: name.clone() }.into()
                }
                ExpressionStoreDiagnostics::UndeclaredLabel { node, name } => {
                    UndeclaredLabel { node: *node, name: name.clone() }.into()
                }
                ExpressionStoreDiagnostics::PatternArgInExternFn { node } => {
                    PatternArgInExternFn { node: *node }.into()
                }
                ExpressionStoreDiagnostics::FruInDestructuringAssignment { node } => {
                    FruInDestructuringAssignment { node: *node }.into()
                }
                ExpressionStoreDiagnostics::MissingBody { node, kind } => {
                    MissingBody { node: *node, kind: *kind }.into()
                }
            });
        }

        source_map.macro_calls().for_each(|(_ast_id, call_id)| self.collect_macro_call(call_id));
    }
}

impl<'db> AnyDiagnostic<'db> {
    fn body_validation_diagnostic(
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

    fn inference_diagnostic(
        db: &'db dyn HirDatabase,
        krate: base_db::Crate,
        edition: Edition,
        d: &'db InferenceDiagnostic,
        source_map: &ExpressionStoreSourceMap,
        type_owner: TypeOwnerId<'db>,
    ) -> Option<AnyDiagnostic<'db>> {
        let expr_syntax = |expr| Self::expr_syntax(expr, source_map);
        let pat_syntax = |pat| Self::pat_syntax(pat, source_map);
        let expr_or_pat_syntax = |id| match id {
            ExprOrPatId::ExprId(expr) => expr_syntax(expr),
            ExprOrPatId::PatId(pat) => pat_syntax(pat),
        };
        let new_ty = |ty| Type { owner: type_owner, ty: EarlyBinder::bind(ty) };
        let span_syntax = |span| Self::span_syntax(span, source_map);
        Some(match d {
            &InferenceDiagnostic::NoSuchField { field: expr, private, variant } => {
                let expr_or_pat = match expr.unpack() {
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
            &InferenceDiagnostic::ArrayPatternWithoutFixedLength { pat } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                ArrayPatternWithoutFixedLength { pat }.into()
            }
            InferenceDiagnostic::ExpectedArrayOrSlicePat { pat, found } => {
                let pat = pat_syntax(*pat)?.map(Into::into);
                ExpectedArrayOrSlicePat {
                    pat,
                    found: Type { owner: type_owner, ty: EarlyBinder::bind(found.as_ref()) },
                }
                .into()
            }
            &InferenceDiagnostic::InvalidRangePatType { pat } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                InvalidRangePatType { pat }.into()
            }
            &InferenceDiagnostic::DuplicateField { field: expr, variant } => {
                let expr_or_pat = match expr.unpack() {
                    ExprOrPatId::ExprId(expr) => {
                        source_map.field_syntax(expr).map(AstPtr::wrap_left)
                    }
                    ExprOrPatId::PatId(pat) => source_map.pat_field_syntax(pat),
                };
                DuplicateField { field: expr_or_pat, variant: variant.into() }.into()
            }
            &InferenceDiagnostic::MismatchedArgCount {
                call_expr,
                expected,
                found,
                is_fn_trait_call,
            } => MismatchedArgCount {
                call_expr: expr_syntax(call_expr)?,
                expected,
                found,
                is_fn_trait_call,
            }
            .into(),
            &InferenceDiagnostic::PrivateField { expr, field } => {
                let expr = expr_syntax(expr)?;
                let field = field.into();
                PrivateField { expr, field }.into()
            }
            &InferenceDiagnostic::PrivateAssocItem { id, item } => {
                let expr_or_pat = expr_or_pat_syntax(id.unpack())?;
                let item = item.into();
                PrivateAssocItem { expr_or_pat, item }.into()
            }
            InferenceDiagnostic::ExpectedFunction { call_expr, found } => {
                let call_expr = expr_syntax(*call_expr)?;
                ExpectedFunction { call: call_expr, found: new_ty(found.as_ref()) }.into()
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
                    receiver: new_ty(receiver.as_ref()),
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
                    receiver: new_ty(receiver.as_ref()),
                    field_with_same_name: field_with_same_name
                        .as_ref()
                        .map(|ty| new_ty(ty.as_ref())),
                    assoc_func_with_same_name: assoc_func_with_same_name.map(Into::into),
                }
                .into()
            }
            &InferenceDiagnostic::UnresolvedAssocItem { id } => {
                let expr_or_pat = expr_or_pat_syntax(id.unpack())?;
                UnresolvedAssocItem { expr_or_pat }.into()
            }
            &InferenceDiagnostic::UnresolvedIdent { id } => {
                let node = match id.unpack() {
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
            &InferenceDiagnostic::UnionPatMustHaveExactlyOneField { pat } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                UnionPatMustHaveExactlyOneField { pat }.into()
            }
            &InferenceDiagnostic::UnionPatHasRest { pat } => {
                let pat = pat_syntax(pat)?.map(Into::into);
                UnionPatHasRest { pat }.into()
            }
            &InferenceDiagnostic::FunctionalRecordUpdateOnNonStruct { base_expr } => {
                FunctionalRecordUpdateOnNonStruct { base_expr: expr_syntax(base_expr)? }.into()
            }
            InferenceDiagnostic::TypedHole { expr, expected } => {
                let expr = expr_syntax(*expr)?;
                TypedHole { expr, expected: new_ty(expected.as_ref()) }.into()
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
                CastToUnsized { expr, cast_ty: new_ty(cast_ty.as_ref()) }.into()
            }
            InferenceDiagnostic::InvalidCast { expr, error, expr_ty, cast_ty } => {
                let expr = expr_syntax(*expr)?;
                let expr_ty = new_ty(expr_ty.as_ref());
                let cast_ty = new_ty(cast_ty.as_ref());
                InvalidCast { expr, error: *error, expr_ty, cast_ty }.into()
            }
            InferenceDiagnostic::CannotBeDereferenced { expr, found } => {
                let expr = expr_syntax(*expr)?;
                CannotBeDereferenced { expr, found: new_ty(found.as_ref()) }.into()
            }
            InferenceDiagnostic::UnaryOperatorCannotBeApplied { expr, op, found } => {
                let expr = expr_syntax(*expr)?;
                UnaryOperatorCannotBeApplied { expr, op: *op, found: new_ty(found.as_ref()) }.into()
            }
            InferenceDiagnostic::MutRefInImmRefPat { pat } => {
                let pat = pat_syntax(*pat)?.map(Into::into);
                MutRefInImmRefPat { pat }.into()
            }
            InferenceDiagnostic::CannotImplicitlyDerefTraitObject { pat, found } => {
                let pat = pat_syntax(*pat)?.map(Into::into);
                CannotImplicitlyDerefTraitObject { pat, found: new_ty(found.as_ref()) }.into()
            }
            InferenceDiagnostic::CannotIndexInto { expr, found } => {
                let expr = expr_syntax(*expr)?;
                CannotIndexInto { expr, found: new_ty(found.as_ref()) }.into()
            }
            InferenceDiagnostic::TyDiagnostic { diag } => {
                Self::ty_diagnostic(diag, source_map, db)?
            }
            InferenceDiagnostic::PathDiagnostic { node, diag } => {
                let source = expr_or_pat_syntax(node.unpack())?;
                let syntax = source.value.to_node(&source.file_id.parse_or_expand(db));
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
                    rustc_type_ir::GenericArgKind::Type(ty) => Either::Left(new_ty(ty)),
                    // FIXME: Printing the const to string is definitely not the correct thing to do here.
                    rustc_type_ir::GenericArgKind::Const(konst) => Either::Right(
                        konst
                            .display(db, DisplayTarget::from_crate_and_edition(db, krate, edition))
                            .to_string(),
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
                let expr_or_pat = expr_or_pat_syntax(node.unpack())?;
                TypeMismatch {
                    expr_or_pat,
                    expected: Type { owner: type_owner, ty: EarlyBinder::bind(expected.as_ref()) },
                    actual: Type { owner: type_owner, ty: EarlyBinder::bind(found.as_ref()) },
                }
                .into()
            }
            InferenceDiagnostic::SolverDiagnostic(d) => {
                let span = span_syntax(d.span)?;
                Self::solver_diagnostic(db, &d.kind, span, type_owner)?
            }
            InferenceDiagnostic::ExplicitDropMethodUse { kind } => {
                let expr_or_path = match kind {
                    ExplicitDropMethodUseKind::MethodCall(expr) => {
                        let expr = expr_syntax(*expr)?;
                        let expr = expr.with_value(expr.value.cast::<ast::MethodCallExpr>()?);
                        Either::Left(expr)
                    }
                    ExplicitDropMethodUseKind::Path(path_expr_id) => {
                        let syntax = expr_or_pat_syntax(path_expr_id.unpack())?;
                        let file_id = syntax.file_id;
                        let syntax =
                            syntax.with_value(syntax.value.cast::<ast::PathExpr>()?).to_node(db);
                        let path = syntax.path()?;
                        let path = InFile::new(file_id, AstPtr::new(&path));
                        Either::Right(path)
                    }
                };
                ExplicitDropMethodUse { expr_or_path }.into()
            }
            InferenceDiagnostic::MutableRefBinding { pat } => {
                let pat = pat_syntax(*pat)?.map(Into::into);
                MutableRefBinding { pat }.into()
            }
            &InferenceDiagnostic::YieldOutsideCoroutine { expr } => {
                YieldOutsideCoroutine { expr: expr_syntax(expr)? }.into()
            }
            &InferenceDiagnostic::ReturnOutsideFunction { expr, kind } => {
                ReturnOutsideFunction { expr: expr_syntax(expr)?, kind }.into()
            }
            &InferenceDiagnostic::RecordMissingFields { record, variant, ref missed_fields } => {
                let record = expr_or_pat_syntax(record)?;
                let file = record.file_id;
                let root = record.file_syntax(db);
                let variant_data = variant.fields(db);
                let missed_fields = missed_fields
                    .iter()
                    .map(|&idx| {
                        (
                            variant_data.fields()[idx].name.clone(),
                            Field { parent: variant.into(), id: idx },
                        )
                    })
                    .collect();
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
                        MissingFields {
                            file,
                            field_list_parent: AstPtr::new(&Either::Right(record_pat)),
                            field_list_parent_path,
                            missed_fields,
                        }
                        .into()
                    }
                    _ => return None,
                }
            }
        })
    }

    fn solver_diagnostic(
        db: &'db dyn HirDatabase,
        d: &'db SolverDiagnosticKind,
        span: SpanSyntax,
        type_owner: TypeOwnerId<'db>,
    ) -> Option<AnyDiagnostic<'db>> {
        let interner = DbInterner::new_no_crate(db);
        Some(match d {
            SolverDiagnosticKind::TraitUnimplemented {
                trait_predicate,
                parent_trait_predicates,
            } => {
                let trait_predicate = crate::TraitPredicate {
                    inner: trait_predicate.get(interner),
                    owner: type_owner,
                };
                let parent_trait_predicates = parent_trait_predicates
                    .iter()
                    .map(|trait_predicate| crate::TraitPredicate {
                        inner: trait_predicate.get(interner),
                        owner: type_owner,
                    })
                    .collect();
                UnimplementedTrait { span, trait_predicate, parent_trait_predicates }.into()
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

    fn expr_syntax(
        expr: ExprId,
        source_map: &ExpressionStoreSourceMap,
    ) -> Option<InFile<ExprOrPatPtr>> {
        source_map
            .expr_syntax(expr)
            .inspect_err(|_| stdx::never!("inference diagnostic in desugared expr"))
            .ok()
    }

    fn pat_syntax(
        pat: PatId,
        source_map: &ExpressionStoreSourceMap,
    ) -> Option<InFile<ExprOrPatPtr>> {
        source_map
            .pat_syntax(pat)
            .inspect_err(|_| stdx::never!("inference diagnostic in desugared pattern"))
            .ok()
    }

    fn type_syntax(
        type_ref: TypeRefId,
        source_map: &ExpressionStoreSourceMap,
    ) -> Option<InFile<AstPtr<ast::Type>>> {
        source_map
            .type_syntax(type_ref)
            .inspect_err(|_| stdx::never!("inference diagnostic in desugared type"))
            .ok()
    }

    fn span_syntax(
        span: hir_ty::Span,
        source_map: &ExpressionStoreSourceMap,
    ) -> Option<InFile<AstPtr<SpanAst>>> {
        Some(match span {
            hir_ty::Span::ExprId(idx) => Self::expr_syntax(idx, source_map)?.map(|it| it.upcast()),
            hir_ty::Span::PatId(idx) => Self::pat_syntax(idx, source_map)?.map(|it| it.upcast()),
            hir_ty::Span::TypeRefId(idx) => {
                Self::type_syntax(idx, source_map)?.map(|it| it.upcast())
            }
            hir_ty::Span::BindingId(idx) => {
                let &pat = source_map.patterns_for_binding(idx).first()?;
                Self::pat_syntax(pat, source_map)?.map(|it| it.upcast())
            }
            hir_ty::Span::Dummy => {
                never!("should never create a diagnostic for dummy spans");
                return None;
            }
        })
    }

    fn ty_diagnostic(
        diag: &TyLoweringDiagnostic,
        source_map: &ExpressionStoreSourceMap,
        db: &'db dyn HirDatabase,
    ) -> Option<AnyDiagnostic<'db>> {
        Some(match diag {
            TyLoweringDiagnostic::PathDiagnostic { source, diag } => {
                let source = Self::type_syntax(*source, source_map)?;
                let syntax = source.value.to_node(&source.file_id.parse_or_expand(db));
                let ast::Type::PathType(syntax) = syntax else { return None };
                Self::path_diagnostic(diag, source.with_value(syntax.path()?))?
            }
            TyLoweringDiagnostic::InferVarsNotAllowed { source } => {
                let source = Self::span_syntax(*source, source_map)?;
                InferVarsNotAllowed { node: source.map(Into::into) }.into()
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
