//! FIXME: write short doc here
pub use hir_def::diagnostics::{UnconfiguredCode, UnresolvedModule};
pub use hir_expand::diagnostics::{Diagnostic, DiagnosticSink, DiagnosticSinkBuilder};
pub use hir_ty::diagnostics::{
    IncorrectCase, MismatchedArgCount, MissingFields, MissingMatchArms, MissingOkInTailExpr,
    NoSuchField,
};
