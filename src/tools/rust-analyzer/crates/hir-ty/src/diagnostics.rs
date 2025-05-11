//! Type inference-based diagnostics.
mod decl_check;
mod expr;
mod match_check;
mod unsafe_check;

pub use crate::diagnostics::{
    decl_check::{CaseType, IncorrectCase, incorrect_case},
    expr::{
        BodyValidationDiagnostic, record_literal_missing_fields, record_pattern_missing_fields,
    },
    unsafe_check::{
        InsideUnsafeBlock, UnsafetyReason, missing_unsafe, unsafe_operations,
        unsafe_operations_for_body,
    },
};
