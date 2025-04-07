//! Type inference-based diagnostics.
mod decl_check;
mod expr;
mod match_check;
mod unsafe_check;

pub use crate::diagnostics::{
    decl_check::{incorrect_case, CaseType, IncorrectCase},
    expr::{
        record_literal_missing_fields, record_pattern_missing_fields, BodyValidationDiagnostic,
    },
    unsafe_check::{
        missing_unsafe, unsafe_operations, unsafe_operations_for_body, InsideUnsafeBlock,
        UnsafetyReason,
    },
};
