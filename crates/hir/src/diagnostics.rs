//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnistics should
//! be expressed in terms of hir types themselves.
use std::any::Any;

use cfg::{CfgExpr, CfgOptions, DnfExpr};
use hir_def::path::ModPath;
use hir_expand::{name::Name, HirFileId, InFile};
use stdx::format_to;
use syntax::{ast, AstPtr, SyntaxNodePtr, TextRange};

pub use crate::diagnostics_sink::{
    Diagnostic, DiagnosticCode, DiagnosticSink, DiagnosticSinkBuilder,
};

macro_rules! diagnostics {
    ($($diag:ident)*) => {
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

diagnostics![UnresolvedModule];

#[derive(Debug)]
pub struct UnresolvedModule {
    pub decl: InFile<AstPtr<ast::Module>>,
    pub candidate: String,
}

// Diagnostic: unresolved-extern-crate
//
// This diagnostic is triggered if rust-analyzer is unable to discover referred extern crate.
#[derive(Debug)]
pub struct UnresolvedExternCrate {
    pub file: HirFileId,
    pub item: AstPtr<ast::ExternCrate>,
}

impl Diagnostic for UnresolvedExternCrate {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-extern-crate")
    }
    fn message(&self) -> String {
        "unresolved extern crate".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.item.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct UnresolvedImport {
    pub file: HirFileId,
    pub node: AstPtr<ast::UseTree>,
}

impl Diagnostic for UnresolvedImport {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-import")
    }
    fn message(&self) -> String {
        "unresolved import".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
    fn is_experimental(&self) -> bool {
        // This currently results in false positives in the following cases:
        // - `cfg_if!`-generated code in libstd (we don't load the sysroot correctly)
        // - `core::arch` (we don't handle `#[path = "../<path>"]` correctly)
        // - proc macros and/or proc macro generated code
        true
    }
}

// Diagnostic: unresolved-macro-call
//
// This diagnostic is triggered if rust-analyzer is unable to resolve the path to a
// macro in a macro invocation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnresolvedMacroCall {
    pub file: HirFileId,
    pub node: AstPtr<ast::MacroCall>,
    pub path: ModPath,
}

impl Diagnostic for UnresolvedMacroCall {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-macro-call")
    }
    fn message(&self) -> String {
        format!("unresolved macro `{}!`", self.path)
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone().into())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
    fn is_experimental(&self) -> bool {
        true
    }
}

// Diagnostic: inactive-code
//
// This diagnostic is shown for code with inactive `#[cfg]` attributes.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InactiveCode {
    pub file: HirFileId,
    pub node: SyntaxNodePtr,
    pub cfg: CfgExpr,
    pub opts: CfgOptions,
}

impl Diagnostic for InactiveCode {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("inactive-code")
    }
    fn message(&self) -> String {
        let inactive = DnfExpr::new(self.cfg.clone()).why_inactive(&self.opts);
        let mut buf = "code is inactive due to #[cfg] directives".to_string();

        if let Some(inactive) = inactive {
            format_to!(buf, ": {}", inactive);
        }

        buf
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: unresolved-proc-macro
//
// This diagnostic is shown when a procedural macro can not be found. This usually means that
// procedural macro support is simply disabled (and hence is only a weak hint instead of an error),
// but can also indicate project setup problems.
//
// If you are seeing a lot of "proc macro not expanded" warnings, you can add this option to the
// `rust-analyzer.diagnostics.disabled` list to prevent them from showing. Alternatively you can
// enable support for procedural macros (see `rust-analyzer.procMacro.enable`).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UnresolvedProcMacro {
    pub file: HirFileId,
    pub node: SyntaxNodePtr,
    /// If the diagnostic can be pinpointed more accurately than via `node`, this is the `TextRange`
    /// to use instead.
    pub precise_location: Option<TextRange>,
    pub macro_name: Option<String>,
}

impl Diagnostic for UnresolvedProcMacro {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unresolved-proc-macro")
    }

    fn message(&self) -> String {
        match &self.macro_name {
            Some(name) => format!("proc macro `{}` not expanded", name),
            None => "proc macro not expanded".to_string(),
        }
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: macro-error
//
// This diagnostic is shown for macro expansion errors.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MacroError {
    pub file: HirFileId,
    pub node: SyntaxNodePtr,
    pub message: String,
}

impl Diagnostic for MacroError {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("macro-error")
    }
    fn message(&self) -> String {
        self.message.clone()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone())
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
    fn is_experimental(&self) -> bool {
        // Newly added and not very well-tested, might contain false positives.
        true
    }
}

#[derive(Debug)]
pub struct UnimplementedBuiltinMacro {
    pub file: HirFileId,
    pub node: SyntaxNodePtr,
}

impl Diagnostic for UnimplementedBuiltinMacro {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("unimplemented-builtin-macro")
    }

    fn message(&self) -> String {
        "unimplemented built-in macro".to_string()
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.node.clone())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
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

// Diagnostic: missing-structure-fields
//
// This diagnostic is triggered if record lacks some fields that exist in the corresponding structure.
//
// Example:
//
// ```rust
// struct A { a: u8, b: u8 }
//
// let a = A { a: 10 };
// ```
#[derive(Debug)]
pub struct MissingFields {
    pub file: HirFileId,
    pub field_list_parent: AstPtr<ast::RecordExpr>,
    pub field_list_parent_path: Option<AstPtr<ast::Path>>,
    pub missed_fields: Vec<Name>,
}

impl Diagnostic for MissingFields {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-structure-fields")
    }
    fn message(&self) -> String {
        let mut buf = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            format_to!(buf, "- {}\n", field);
        }
        buf
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile {
            file_id: self.file,
            value: self
                .field_list_parent_path
                .clone()
                .map(SyntaxNodePtr::from)
                .unwrap_or_else(|| self.field_list_parent.clone().into()),
        }
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-pat-fields
//
// This diagnostic is triggered if pattern lacks some fields that exist in the corresponding structure.
//
// Example:
//
// ```rust
// struct A { a: u8, b: u8 }
//
// let a = A { a: 10, b: 20 };
//
// if let A { a } = a {
//     // ...
// }
// ```
#[derive(Debug)]
pub struct MissingPatFields {
    pub file: HirFileId,
    pub field_list_parent: AstPtr<ast::RecordPat>,
    pub field_list_parent_path: Option<AstPtr<ast::Path>>,
    pub missed_fields: Vec<Name>,
}

impl Diagnostic for MissingPatFields {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-pat-fields")
    }
    fn message(&self) -> String {
        let mut buf = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            format_to!(buf, "- {}\n", field);
        }
        buf
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile {
            file_id: self.file,
            value: self
                .field_list_parent_path
                .clone()
                .map(SyntaxNodePtr::from)
                .unwrap_or_else(|| self.field_list_parent.clone().into()),
        }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
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
