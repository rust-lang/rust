//! Type inference-based diagnostics.
mod expr;
mod match_check;
mod unsafe_check;
mod decl_check;

pub use crate::diagnostics::{
    decl_check::{incorrect_case, CaseType, IncorrectCase},
    expr::{
        record_literal_missing_fields, record_pattern_missing_fields, BodyValidationDiagnostic,
    },
    unsafe_check::{missing_unsafe, unsafe_expressions, UnsafeExpr},
};

#[derive(Debug, PartialEq, Eq)]
pub struct IncoherentImpl {
    pub file_id: hir_expand::HirFileId,
    pub impl_: syntax::AstPtr<syntax::ast::Impl>,
}
