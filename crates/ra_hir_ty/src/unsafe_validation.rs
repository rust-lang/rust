//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use std::sync::Arc;

use hir_def::FunctionId;
use hir_expand::diagnostics::DiagnosticSink;

use crate::{
    db::HirDatabase, diagnostics::MissingUnsafe, expr::unsafe_expressions, InferenceResult,
};

pub use hir_def::{
    body::{
        scope::{ExprScopes, ScopeEntry, ScopeId},
        Body, BodySourceMap, ExprPtr, ExprSource, PatPtr, PatSource,
    },
    expr::{
        ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal, LogicOp,
        MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement, UnaryOp,
    },
    LocalFieldId, VariantId,
};

pub struct UnsafeValidator<'a, 'b: 'a> {
    func: FunctionId,
    infer: Arc<InferenceResult>,
    sink: &'a mut DiagnosticSink<'b>,
}

impl<'a, 'b> UnsafeValidator<'a, 'b> {
    pub fn new(
        func: FunctionId,
        infer: Arc<InferenceResult>,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> UnsafeValidator<'a, 'b> {
        UnsafeValidator { func, infer, sink }
    }

    pub fn validate_body(&mut self, db: &dyn HirDatabase) {
        let def = self.func.into();
        let unsafe_expressions = unsafe_expressions(db, self.infer.as_ref(), def);
        let func_data = db.function_data(self.func);
        if func_data.is_unsafe
            || unsafe_expressions
                .iter()
                .filter(|unsafe_expr| !unsafe_expr.inside_unsafe_block)
                .count()
                == 0
        {
            return;
        }

        let (_, body_source) = db.body_with_source_map(def);
        for unsafe_expr in unsafe_expressions {
            if !unsafe_expr.inside_unsafe_block {
                if let Ok(in_file) = body_source.as_ref().expr_syntax(unsafe_expr.expr) {
                    self.sink.push(MissingUnsafe { file: in_file.file_id, expr: in_file.value })
                }
            }
        }
    }
}
