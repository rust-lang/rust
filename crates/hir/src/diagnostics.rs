//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnistics should
//! be expressed in terms of hir types themselves.
use std::any::Any;

use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_def::path::ModPath;
use hir_expand::{name::Name, HirFileId, InFile};
use syntax::{ast, AstPtr, SyntaxNodePtr, TextRange};

pub use crate::diagnostics_sink::{
    Diagnostic, DiagnosticCode, DiagnosticSink, DiagnosticSinkBuilder,
};

macro_rules! diagnostics {
    ($($diag:ident,)*) => {
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

diagnostics![
    InactiveCode,
    MacroError,
    MissingFields,
    UnimplementedBuiltinMacro,
    UnresolvedExternCrate,
    UnresolvedImport,
    UnresolvedMacroCall,
    UnresolvedModule,
    UnresolvedProcMacro,
];

#[derive(Debug)]
pub struct UnresolvedModule {
    pub decl: InFile<AstPtr<ast::Module>>,
    pub candidate: String,
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
    pub macro_call: InFile<AstPtr<ast::MacroCall>>,
    pub path: ModPath,
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
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroError {
    pub node: InFile<SyntaxNodePtr>,
    pub message: String,
}

#[derive(Debug)]
pub struct UnimplementedBuiltinMacro {
    pub node: InFile<SyntaxNodePtr>,
}

// Diagnostic: no-such-field
//
// This diagnostic is triggered if created structure does not have field provided in record.
#[derive(Debug)]
pub struct NoSuchField {
    pub file: HirFileId,
    pub field: AstPtr<ast::RecordExprField>,
}

impl Diagnostic for NoSuchField {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("no-such-field")
    }

    fn message(&self) -> String {
        "no such field".to_string()
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.field.clone().into())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: break-outside-of-loop
//
// This diagnostic is triggered if the `break` keyword is used outside of a loop.
#[derive(Debug)]
pub struct BreakOutsideOfLoop {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for BreakOutsideOfLoop {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("break-outside-of-loop")
    }
    fn message(&self) -> String {
        "break outside of loop".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-unsafe
//
// This diagnostic is triggered if an operation marked as `unsafe` is used outside of an `unsafe` function or block.
#[derive(Debug)]
pub struct MissingUnsafe {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for MissingUnsafe {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-unsafe")
    }
    fn message(&self) -> String {
        format!("This operation is unsafe and requires an unsafe function or block")
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct MissingFields {
    pub file: HirFileId,
    pub field_list_parent: Either<AstPtr<ast::RecordExpr>, AstPtr<ast::RecordPat>>,
    pub field_list_parent_path: Option<AstPtr<ast::Path>>,
    pub missed_fields: Vec<Name>,
}

// Diagnostic: replace-filter-map-next-with-find-map
//
// This diagnostic is triggered when `.filter_map(..).next()` is used, rather than the more concise `.find_map(..)`.
#[derive(Debug)]
pub struct ReplaceFilterMapNextWithFindMap {
    pub file: HirFileId,
    /// This expression is the whole method chain up to and including `.filter_map(..).next()`.
    pub next_expr: AstPtr<ast::Expr>,
}

impl Diagnostic for ReplaceFilterMapNextWithFindMap {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("replace-filter-map-next-with-find-map")
    }
    fn message(&self) -> String {
        "replace filter_map(..).next() with find_map(..)".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.next_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: mismatched-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
#[derive(Debug)]
pub struct MismatchedArgCount {
    pub file: HirFileId,
    pub call_expr: AstPtr<ast::Expr>,
    pub expected: usize,
    pub found: usize,
}

impl Diagnostic for MismatchedArgCount {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("mismatched-arg-count")
    }
    fn message(&self) -> String {
        let s = if self.expected == 1 { "" } else { "s" };
        format!("Expected {} argument{}, found {}", self.expected, s, self.found)
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.call_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
    fn is_experimental(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct RemoveThisSemicolon {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for RemoveThisSemicolon {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("remove-this-semicolon")
    }

    fn message(&self) -> String {
        "Remove this semicolon".to_string()
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-ok-or-some-in-tail-expr
//
// This diagnostic is triggered if a block that should return `Result` returns a value not wrapped in `Ok`,
// or if a block that should return `Option` returns a value not wrapped in `Some`.
//
// Example:
//
// ```rust
// fn foo() -> Result<u8, ()> {
//     10
// }
// ```
#[derive(Debug)]
pub struct MissingOkOrSomeInTailExpr {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
    // `Some` or `Ok` depending on whether the return type is Result or Option
    pub required: String,
}

impl Diagnostic for MissingOkOrSomeInTailExpr {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-ok-or-some-in-tail-expr")
    }
    fn message(&self) -> String {
        format!("wrap return expression in {}", self.required)
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-match-arm
//
// This diagnostic is triggered if `match` block is missing one or more match arms.
#[derive(Debug)]
pub struct MissingMatchArms {
    pub file: HirFileId,
    pub match_expr: AstPtr<ast::Expr>,
    pub arms: AstPtr<ast::MatchArmList>,
}

impl Diagnostic for MissingMatchArms {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-match-arm")
    }
    fn message(&self) -> String {
        String::from("Missing match arm")
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.match_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct InternalBailedOut {
    pub file: HirFileId,
    pub pat_syntax_ptr: SyntaxNodePtr,
}

impl Diagnostic for InternalBailedOut {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("internal:match-check-bailed-out")
    }
    fn message(&self) -> String {
        format!("Internal: match check bailed out")
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.pat_syntax_ptr.clone() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

pub use hir_ty::diagnostics::IncorrectCase;

impl Diagnostic for IncorrectCase {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("incorrect-ident-case")
    }

    fn message(&self) -> String {
        format!(
            "{} `{}` should have {} name, e.g. `{}`",
            self.ident_type,
            self.ident_text,
            self.expected_case.to_string(),
            self.suggested_text
        )
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.ident.clone().into())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }

    fn is_experimental(&self) -> bool {
        true
    }
}
