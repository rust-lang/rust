//! FIXME: write short doc here
pub use hir_def::diagnostics::{InactiveCode, UnresolvedModule, UnresolvedProcMacro};
pub use hir_expand::diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticSink, DiagnosticSinkBuilder,
};
pub use hir_ty::diagnostics::{
    IncorrectCase, MismatchedArgCount, MissingFields, MissingMatchArms, MissingOkOrSomeInTailExpr,
    NoSuchField, RemoveThisSemicolon, ReplaceFilterMapNextWithFindMap,
};
