//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use std::sync::Arc;

use hir_def::{
    body::Body,
    expr::{Expr, ExprId, UnaryOp},
    DefWithBodyId, FunctionId,
};
use hir_expand::diagnostics::DiagnosticSink;

use crate::{
    db::HirDatabase, diagnostics::MissingUnsafe, lower::CallableDef, ApplicationTy,
    InferenceResult, Ty, TypeCtor,
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

pub struct UnsafeExpr {
    pub expr: ExprId,
    pub inside_unsafe_block: bool,
}

pub fn unsafe_expressions(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
) -> Vec<UnsafeExpr> {
    let mut unsafe_exprs = vec![];
    let body = db.body(def);
    walk_unsafe(&mut unsafe_exprs, db, infer, &body, body.body_expr, false);

    unsafe_exprs
}

fn walk_unsafe(
    unsafe_exprs: &mut Vec<UnsafeExpr>,
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    body: &Body,
    current: ExprId,
    inside_unsafe_block: bool,
) {
    let expr = &body.exprs[current];
    match expr {
        Expr::Call { callee, .. } => {
            let ty = &infer[*callee];
            if let &Ty::Apply(ApplicationTy {
                ctor: TypeCtor::FnDef(CallableDef::FunctionId(func)),
                ..
            }) = ty
            {
                if db.function_data(func).is_unsafe {
                    unsafe_exprs.push(UnsafeExpr { expr: current, inside_unsafe_block });
                }
            }
        }
        Expr::MethodCall { .. } => {
            if infer
                .method_resolution(current)
                .map(|func| db.function_data(func).is_unsafe)
                .unwrap_or(false)
            {
                unsafe_exprs.push(UnsafeExpr { expr: current, inside_unsafe_block });
            }
        }
        Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
            if let Ty::Apply(ApplicationTy { ctor: TypeCtor::RawPtr(..), .. }) = &infer[*expr] {
                unsafe_exprs.push(UnsafeExpr { expr: current, inside_unsafe_block });
            }
        }
        Expr::Unsafe { body: child } => {
            return walk_unsafe(unsafe_exprs, db, infer, body, *child, true);
        }
        _ => {}
    }

    expr.walk_child_exprs(|child| {
        walk_unsafe(unsafe_exprs, db, infer, body, child, inside_unsafe_block);
    });
}
