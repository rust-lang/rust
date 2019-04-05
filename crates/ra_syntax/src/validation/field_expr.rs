use crate::{ast::{self, FieldKind},
            SyntaxError,
            SyntaxErrorKind::*,
};

pub(crate) fn validate_field_expr_node(node: &ast::FieldExpr, errors: &mut Vec<SyntaxError>) {
    if let Some(FieldKind::Index(idx)) = node.field_access() {
        if idx.text().chars().any(|c| c < '0' || c > '9') {
            errors.push(SyntaxError::new(InvalidTupleIndexFormat, idx.range()));
        }
    }
}
