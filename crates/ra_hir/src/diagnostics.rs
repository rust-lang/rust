use crate::{expr::ExprId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionDiagnostic {
    NoSuchField { expr: ExprId, field: usize },
}
