//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use std::sync::Arc;

use hir_def::{DefWithBodyId, FunctionId};
use hir_expand::diagnostics::DiagnosticSink;

use crate::{
    db::HirDatabase, diagnostics::MissingUnsafe, lower::CallableDef, ApplicationTy,
    InferenceResult, Ty, TypeCtor,
};

use rustc_hash::FxHashSet;

use hir_def::{
    body::Body,
    expr::{Expr, ExprId, UnaryOp},
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

impl UnsafeExpr {
    fn new(expr: ExprId) -> Self {
        Self { expr, inside_unsafe_block: false }
    }
}

pub fn unsafe_expressions(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
) -> Vec<UnsafeExpr> {
    let mut unsafe_exprs = vec![];
    let mut unsafe_block_exprs = FxHashSet::default();
    let body = db.body(def);
    for (id, expr) in body.exprs.iter() {
        match expr {
            Expr::Unsafe { .. } => {
                unsafe_block_exprs.insert(id);
            }
            Expr::Call { callee, .. } => {
                let ty = &infer[*callee];
                if let &Ty::Apply(ApplicationTy {
                    ctor: TypeCtor::FnDef(CallableDef::FunctionId(func)),
                    ..
                }) = ty
                {
                    if db.function_data(func).is_unsafe {
                        unsafe_exprs.push(UnsafeExpr::new(id));
                    }
                }
            }
            Expr::MethodCall { .. } => {
                if infer
                    .method_resolution(id)
                    .map(|func| db.function_data(func).is_unsafe)
                    .unwrap_or(false)
                {
                    unsafe_exprs.push(UnsafeExpr::new(id));
                }
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if let Ty::Apply(ApplicationTy { ctor: TypeCtor::RawPtr(..), .. }) = &infer[*expr] {
                    unsafe_exprs.push(UnsafeExpr::new(id));
                }
            }
            _ => {}
        }
    }

    for unsafe_expr in &mut unsafe_exprs {
        unsafe_expr.inside_unsafe_block =
            is_in_unsafe(&body, body.body_expr, unsafe_expr.expr, false);
    }

    unsafe_exprs
}

fn is_in_unsafe(body: &Body, current: ExprId, needle: ExprId, within_unsafe: bool) -> bool {
    if current == needle {
        return within_unsafe;
    }

    let expr = &body.exprs[current];
    if let &Expr::Unsafe { body: child } = expr {
        return is_in_unsafe(body, child, needle, true);
    }

    let mut found = false;
    expr.walk_child_exprs(|child| {
        found = found || is_in_unsafe(body, child, needle, within_unsafe);
    });
    found
}
