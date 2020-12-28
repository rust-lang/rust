//! FIXME: write short doc here
pub use hir_def::diagnostics::{InactiveCode, UnresolvedModule, UnresolvedProcMacro};
pub use hir_expand::diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticSink, DiagnosticSinkBuilder,
};
pub use hir_ty::diagnostics::{
    IncorrectCase, MismatchedArgCount, MissingFields, MissingMatchArms, MissingOkOrSomeInTailExpr,
    NoSuchField, RemoveThisSemicolon, ReplaceFilterMapNextWithFindMap,
};

// PHIL:
// hir/src/diagnostics.rs - just pub uses the type from hir_ty::diagnostics (DONE)
// hir_ty/src/diagnostics.rs - defines the type (DONE)
// hir_ty/src/diagnostics.rs - plus a test (DONE) <--- one example found, need to copy the not-applicable tests from the assist version
// ide/src/diagnostics.rs - define handler for when this diagnostic is raised (DONE)

// ide/src/diagnostics/fixes.rs - pulls in type from hir, and impls DiagnosticWithFix (TODO)
// hir_ty/src/diagnostics/expr.rs - do the real work (TODO)
