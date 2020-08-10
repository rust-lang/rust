//! FIXME: write short doc here
pub use hir_def::diagnostics::UnresolvedModule;
pub use hir_expand::diagnostics::{
    Diagnostic, DiagnosticSink, DiagnosticSinkBuilder, DiagnosticWithFix,
};
pub use hir_ty::diagnostics::{
    MismatchedArgCount, MissingFields, MissingMatchArms, MissingOkInTailExpr, NoSuchField,
};
