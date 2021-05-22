//! Re-export diagnostics such that clients of `hir` don't have to depend on
//! low-level crates.
//!
//! This probably isn't the best way to do this -- ideally, diagnistics should
//! be expressed in terms of hir types themselves.
pub use hir_def::diagnostics::{
    InactiveCode, UnresolvedMacroCall, UnresolvedModule, UnresolvedProcMacro,
};
pub use hir_expand::diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticSink, DiagnosticSinkBuilder,
};
pub use hir_ty::diagnostics::{
    IncorrectCase, MismatchedArgCount, MissingFields, MissingMatchArms, MissingOkOrSomeInTailExpr,
    NoSuchField, RemoveThisSemicolon, ReplaceFilterMapNextWithFindMap,
};
