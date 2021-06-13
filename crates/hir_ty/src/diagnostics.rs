//! Type inference-based diagnostics.
mod expr;
mod match_check;
mod unsafe_check;
mod decl_check;

use std::fmt;

use base_db::CrateId;
use hir_def::ModuleDefId;
use hir_expand::HirFileId;
use syntax::{ast, AstPtr};

use crate::db::HirDatabase;

pub use crate::diagnostics::{
    expr::{
        record_literal_missing_fields, record_pattern_missing_fields, BodyValidationDiagnostic,
    },
    unsafe_check::missing_unsafe,
};

pub fn validate_module_item(
    db: &dyn HirDatabase,
    krate: CrateId,
    owner: ModuleDefId,
) -> Vec<IncorrectCase> {
    let _p = profile::span("validate_module_item");
    let mut validator = decl_check::DeclValidator::new(db, krate);
    validator.validate_item(owner);
    validator.sink
}

#[derive(Debug)]
pub enum CaseType {
    // `some_var`
    LowerSnakeCase,
    // `SOME_CONST`
    UpperSnakeCase,
    // `SomeStruct`
    UpperCamelCase,
}

impl fmt::Display for CaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            CaseType::LowerSnakeCase => "snake_case",
            CaseType::UpperSnakeCase => "UPPER_SNAKE_CASE",
            CaseType::UpperCamelCase => "CamelCase",
        };

        write!(f, "{}", repr)
    }
}

#[derive(Debug)]
pub enum IdentType {
    Constant,
    Enum,
    Field,
    Function,
    Parameter,
    StaticVariable,
    Structure,
    Variable,
    Variant,
}

impl fmt::Display for IdentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            IdentType::Constant => "Constant",
            IdentType::Enum => "Enum",
            IdentType::Field => "Field",
            IdentType::Function => "Function",
            IdentType::Parameter => "Parameter",
            IdentType::StaticVariable => "Static variable",
            IdentType::Structure => "Structure",
            IdentType::Variable => "Variable",
            IdentType::Variant => "Variant",
        };

        write!(f, "{}", repr)
    }
}

#[derive(Debug)]
pub struct IncorrectCase {
    pub file: HirFileId,
    pub ident: AstPtr<ast::Name>,
    pub expected_case: CaseType,
    pub ident_type: IdentType,
    pub ident_text: String,
    pub suggested_text: String,
}
