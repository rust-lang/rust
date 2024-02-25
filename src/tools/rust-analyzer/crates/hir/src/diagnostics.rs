//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnostics should
//! be expressed in terms of hir types themselves.
pub use hir_ty::diagnostics::{CaseType, IncorrectCase};
use hir_ty::{db::HirDatabase, diagnostics::BodyValidationDiagnostic, InferenceDiagnostic};

use base_db::CrateId;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_def::{body::SyntheticSyntax, hir::ExprOrPatId, path::ModPath, AssocItemId, DefWithBodyId};
use hir_expand::{name::Name, HirFileId, InFile};
use syntax::{ast, AstPtr, SyntaxError, SyntaxNodePtr, TextRange};

use crate::{AssocItem, Field, Local, MacroKind, Trait, Type};

macro_rules! diagnostics {
    ($($diag:ident,)*) => {
        #[derive(Debug)]
        pub enum AnyDiagnostic {$(
            $diag(Box<$diag>),
        )*}

        $(
            impl From<$diag> for AnyDiagnostic {
                fn from(d: $diag) -> AnyDiagnostic {
                    AnyDiagnostic::$diag(Box::new(d))
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

diagnostics![
    BreakOutsideOfLoop,
    ExpectedFunction,
    InactiveCode,
    IncoherentImpl,
    IncorrectCase,
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
    MovedOutOfRef,
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
    TypedHole,
    TypeMismatch,
    UndeclaredLabel,
    UnimplementedBuiltinMacro,
    UnreachableLabel,
    UnresolvedAssocItem,
    UnresolvedExternCrate,
    UnresolvedField,
    UnresolvedImport,
    UnresolvedMacroCall,
    UnresolvedMethodCall,
    UnresolvedModule,
    UnresolvedIdent,
    UnresolvedProcMacro,
    UnusedMut,
    UnusedVariable,
];

#[derive(Debug)]
pub struct BreakOutsideOfLoop {
    pub expr: InFile<AstPtr<ast::Expr>>,
    pub is_break: bool,
    pub bad_value_break: bool,
}

#[derive(Debug)]
pub struct TypedHole {
    pub expr: InFile<AstPtr<ast::Expr>>,
    pub expected: Type,
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
pub struct UnresolvedProcMacro {
    pub node: InFile<SyntaxNodePtr>,
    /// If the diagnostic can be pinpointed more accurately than via `node`, this is the `TextRange`
    /// to use instead.
    pub precise_location: Option<TextRange>,
    pub macro_name: Option<String>,
    pub kind: MacroKind,
    /// The crate id of the proc-macro this macro belongs to, or `None` if the proc-macro can't be found.
    pub krate: CrateId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroError {
    pub node: InFile<SyntaxNodePtr>,
    pub precise_location: Option<TextRange>,
    pub message: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroExpansionParseError {
    pub node: InFile<SyntaxNodePtr>,
    pub precise_location: Option<TextRange>,
    pub errors: Box<[SyntaxError]>,
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
    pub private: bool,
}

#[derive(Debug)]
pub struct PrivateAssocItem {
    pub expr_or_pat: InFile<AstPtr<Either<ast::Expr, Either<ast::Pat, ast::SelfParam>>>>,
    pub item: AssocItem,
}

#[derive(Debug)]
pub struct MismatchedTupleStructPatArgCount {
    pub expr_or_pat: InFile<AstPtr<Either<ast::Expr, ast::Pat>>>,
    pub expected: usize,
    pub found: usize,
}

#[derive(Debug)]
pub struct ExpectedFunction {
    pub call: InFile<AstPtr<ast::Expr>>,
    pub found: Type,
}

#[derive(Debug)]
pub struct UnresolvedField {
    pub expr: InFile<AstPtr<ast::Expr>>,
    pub receiver: Type,
    pub name: Name,
    pub method_with_same_name_exists: bool,
}

#[derive(Debug)]
pub struct UnresolvedMethodCall {
    pub expr: InFile<AstPtr<ast::Expr>>,
    pub receiver: Type,
    pub name: Name,
    pub field_with_same_name: Option<Type>,
    pub assoc_func_with_same_name: Option<AssocItemId>,
}

#[derive(Debug)]
pub struct UnresolvedAssocItem {
    pub expr_or_pat: InFile<AstPtr<Either<ast::Expr, Either<ast::Pat, ast::SelfParam>>>>,
}

#[derive(Debug)]
pub struct UnresolvedIdent {
    pub expr: InFile<AstPtr<ast::Expr>>,
}

#[derive(Debug)]
pub struct PrivateField {
    pub expr: InFile<AstPtr<ast::Expr>>,
    pub field: Field,
}

#[derive(Debug)]
pub struct MissingUnsafe {
    pub expr: InFile<AstPtr<ast::Expr>>,
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
    pub call_expr: InFile<AstPtr<ast::Expr>>,
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
pub struct TypeMismatch {
    pub expr_or_pat: InFile<AstPtr<Either<ast::Expr, ast::Pat>>>,
    pub expected: Type,
    pub actual: Type,
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
pub struct MovedOutOfRef {
    pub ty: Type,
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

impl AnyDiagnostic {
    pub(crate) fn body_validation_diagnostic(
        db: &dyn HirDatabase,
        diagnostic: BodyValidationDiagnostic,
        source_map: &hir_def::body::BodySourceMap,
    ) -> Option<AnyDiagnostic> {
        match diagnostic {
            BodyValidationDiagnostic::RecordMissingFields { record, variant, missed_fields } => {
                let variant_data = variant.variant_data(db.upcast());
                let missed_fields = missed_fields
                    .into_iter()
                    .map(|idx| variant_data.fields()[idx].name.clone())
                    .collect();

                match record {
                    Either::Left(record_expr) => match source_map.expr_syntax(record_expr) {
                        Ok(source_ptr) => {
                            let root = source_ptr.file_syntax(db.upcast());
                            if let ast::Expr::RecordExpr(record_expr) =
                                source_ptr.value.to_node(&root)
                            {
                                if record_expr.record_expr_field_list().is_some() {
                                    let field_list_parent_path =
                                        record_expr.path().map(|path| AstPtr::new(&path));
                                    return Some(
                                        MissingFields {
                                            file: source_ptr.file_id,
                                            field_list_parent: AstPtr::new(&Either::Left(
                                                record_expr,
                                            )),
                                            field_list_parent_path,
                                            missed_fields,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                        Err(SyntheticSyntax) => (),
                    },
                    Either::Right(record_pat) => match source_map.pat_syntax(record_pat) {
                        Ok(source_ptr) => {
                            if let Some(ptr) = source_ptr.value.cast::<ast::RecordPat>() {
                                let root = source_ptr.file_syntax(db.upcast());
                                let record_pat = ptr.to_node(&root);
                                if record_pat.record_pat_field_list().is_some() {
                                    let field_list_parent_path =
                                        record_pat.path().map(|path| AstPtr::new(&path));
                                    return Some(
                                        MissingFields {
                                            file: source_ptr.file_id,
                                            field_list_parent: AstPtr::new(&Either::Right(
                                                record_pat,
                                            )),
                                            field_list_parent_path,
                                            missed_fields,
                                        }
                                        .into(),
                                    );
                                }
                            }
                        }
                        Err(SyntheticSyntax) => (),
                    },
                }
            }
            BodyValidationDiagnostic::ReplaceFilterMapNextWithFindMap { method_call_expr } => {
                if let Ok(next_source_ptr) = source_map.expr_syntax(method_call_expr) {
                    return Some(
                        ReplaceFilterMapNextWithFindMap {
                            file: next_source_ptr.file_id,
                            next_expr: next_source_ptr.value,
                        }
                        .into(),
                    );
                }
            }
            BodyValidationDiagnostic::MissingMatchArms { match_expr, uncovered_patterns } => {
                match source_map.expr_syntax(match_expr) {
                    Ok(source_ptr) => {
                        let root = source_ptr.file_syntax(db.upcast());
                        if let ast::Expr::MatchExpr(match_expr) = &source_ptr.value.to_node(&root) {
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
                if let Ok(source_ptr) = source_map.expr_syntax(if_expr) {
                    if let Some(ptr) = source_ptr.value.cast::<ast::IfExpr>() {
                        return Some(
                            RemoveUnnecessaryElse { if_expr: InFile::new(source_ptr.file_id, ptr) }
                                .into(),
                        );
                    }
                }
            }
        }
        None
    }

    pub(crate) fn inference_diagnostic(
        db: &dyn HirDatabase,
        def: DefWithBodyId,
        d: &InferenceDiagnostic,
        source_map: &hir_def::body::BodySourceMap,
    ) -> Option<AnyDiagnostic> {
        let expr_syntax = |expr| source_map.expr_syntax(expr).expect("unexpected synthetic");
        let pat_syntax = |pat| source_map.pat_syntax(pat).expect("unexpected synthetic");
        Some(match d {
            &InferenceDiagnostic::NoSuchField { field: expr, private } => {
                let expr_or_pat = match expr {
                    ExprOrPatId::ExprId(expr) => {
                        source_map.field_syntax(expr).map(AstPtr::wrap_left)
                    }
                    ExprOrPatId::PatId(pat) => {
                        source_map.pat_field_syntax(pat).map(AstPtr::wrap_right)
                    }
                };
                NoSuchField { field: expr_or_pat, private }.into()
            }
            &InferenceDiagnostic::MismatchedArgCount { call_expr, expected, found } => {
                MismatchedArgCount { call_expr: expr_syntax(call_expr), expected, found }.into()
            }
            &InferenceDiagnostic::PrivateField { expr, field } => {
                let expr = expr_syntax(expr);
                let field = field.into();
                PrivateField { expr, field }.into()
            }
            &InferenceDiagnostic::PrivateAssocItem { id, item } => {
                let expr_or_pat = match id {
                    ExprOrPatId::ExprId(expr) => expr_syntax(expr).map(AstPtr::wrap_left),
                    ExprOrPatId::PatId(pat) => pat_syntax(pat).map(AstPtr::wrap_right),
                };
                let item = item.into();
                PrivateAssocItem { expr_or_pat, item }.into()
            }
            InferenceDiagnostic::ExpectedFunction { call_expr, found } => {
                let call_expr = expr_syntax(*call_expr);
                ExpectedFunction { call: call_expr, found: Type::new(db, def, found.clone()) }
                    .into()
            }
            InferenceDiagnostic::UnresolvedField {
                expr,
                receiver,
                name,
                method_with_same_name_exists,
            } => {
                let expr = expr_syntax(*expr);
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
                let expr = expr_syntax(*expr);
                UnresolvedMethodCall {
                    expr,
                    name: name.clone(),
                    receiver: Type::new(db, def, receiver.clone()),
                    field_with_same_name: field_with_same_name
                        .clone()
                        .map(|ty| Type::new(db, def, ty)),
                    assoc_func_with_same_name: *assoc_func_with_same_name,
                }
                .into()
            }
            &InferenceDiagnostic::UnresolvedAssocItem { id } => {
                let expr_or_pat = match id {
                    ExprOrPatId::ExprId(expr) => expr_syntax(expr).map(AstPtr::wrap_left),
                    ExprOrPatId::PatId(pat) => pat_syntax(pat).map(AstPtr::wrap_right),
                };
                UnresolvedAssocItem { expr_or_pat }.into()
            }
            &InferenceDiagnostic::UnresolvedIdent { expr } => {
                let expr = expr_syntax(expr);
                UnresolvedIdent { expr }.into()
            }
            &InferenceDiagnostic::BreakOutsideOfLoop { expr, is_break, bad_value_break } => {
                let expr = expr_syntax(expr);
                BreakOutsideOfLoop { expr, is_break, bad_value_break }.into()
            }
            InferenceDiagnostic::TypedHole { expr, expected } => {
                let expr = expr_syntax(*expr);
                TypedHole { expr, expected: Type::new(db, def, expected.clone()) }.into()
            }
            &InferenceDiagnostic::MismatchedTupleStructPatArgCount { pat, expected, found } => {
                let expr_or_pat = match pat {
                    ExprOrPatId::ExprId(expr) => expr_syntax(expr).map(AstPtr::wrap_left),
                    ExprOrPatId::PatId(pat) => {
                        let InFile { file_id, value } =
                            source_map.pat_syntax(pat).expect("unexpected synthetic");

                        // cast from Either<Pat, SelfParam> -> Either<_, Pat>
                        let ptr = AstPtr::try_from_raw(value.syntax_node_ptr())?;
                        InFile { file_id, value: ptr }
                    }
                };
                MismatchedTupleStructPatArgCount { expr_or_pat, expected, found }.into()
            }
        })
    }
}
