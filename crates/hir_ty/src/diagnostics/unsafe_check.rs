//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use std::sync::Arc;

use hir_def::{
    body::Body,
    expr::{Expr, ExprId, UnaryOp},
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId,
};
use hir_expand::diagnostics::DiagnosticSink;

use crate::{
    db::HirDatabase, diagnostics::MissingUnsafe, lower::CallableDefId, ApplicationTy,
    InferenceResult, Ty, TypeCtor,
};

pub(super) struct UnsafeValidator<'a, 'b: 'a> {
    owner: DefWithBodyId,
    infer: Arc<InferenceResult>,
    sink: &'a mut DiagnosticSink<'b>,
}

impl<'a, 'b> UnsafeValidator<'a, 'b> {
    pub(super) fn new(
        owner: DefWithBodyId,
        infer: Arc<InferenceResult>,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> UnsafeValidator<'a, 'b> {
        UnsafeValidator { owner, infer, sink }
    }

    pub(super) fn validate_body(&mut self, db: &dyn HirDatabase) {
        let def = self.owner.into();
        let unsafe_expressions = unsafe_expressions(db, self.infer.as_ref(), def);
        let is_unsafe = match self.owner {
            DefWithBodyId::FunctionId(it) => db.function_data(it).is_unsafe,
            DefWithBodyId::StaticId(_) | DefWithBodyId::ConstId(_) => false,
        };
        if is_unsafe
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
    walk_unsafe(&mut unsafe_exprs, db, infer, def, &body, body.body_expr, false);

    unsafe_exprs
}

fn walk_unsafe(
    unsafe_exprs: &mut Vec<UnsafeExpr>,
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
    body: &Body,
    current: ExprId,
    inside_unsafe_block: bool,
) {
    let expr = &body.exprs[current];
    match expr {
        Expr::Call { callee, .. } => {
            let ty = &infer[*callee];
            if let &Ty::Apply(ApplicationTy {
                ctor: TypeCtor::FnDef(CallableDefId::FunctionId(func)),
                ..
            }) = ty
            {
                if db.function_data(func).is_unsafe {
                    unsafe_exprs.push(UnsafeExpr { expr: current, inside_unsafe_block });
                }
            }
        }
        Expr::Path(path) => {
            let resolver = resolver_for_expr(db.upcast(), def, current);
            let value_or_partial = resolver.resolve_path_in_value_ns(db.upcast(), path.mod_path());
            if let Some(ResolveValueResult::ValueNs(ValueNs::StaticId(id))) = value_or_partial {
                if db.static_data(id).mutable {
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
            return walk_unsafe(unsafe_exprs, db, infer, def, body, *child, true);
        }
        _ => {}
    }

    expr.walk_child_exprs(|child| {
        walk_unsafe(unsafe_exprs, db, infer, def, body, child, inside_unsafe_block);
    });
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn missing_unsafe_diagnostic_with_raw_ptr() {
        check_diagnostics(
            r#"
fn main() {
    let x = &5 as *const usize;
    unsafe { let y = *x; }
    let z = *x;
}         //^^ This operation is unsafe and requires an unsafe function or block
"#,
        )
    }

    #[test]
    fn missing_unsafe_diagnostic_with_unsafe_call() {
        check_diagnostics(
            r#"
struct HasUnsafe;

impl HasUnsafe {
    unsafe fn unsafe_fn(&self) {
        let x = &5 as *const usize;
        let y = *x;
    }
}

unsafe fn unsafe_fn() {
    let x = &5 as *const usize;
    let y = *x;
}

fn main() {
    unsafe_fn();
  //^^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
    HasUnsafe.unsafe_fn();
  //^^^^^^^^^^^^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
    unsafe {
        unsafe_fn();
        HasUnsafe.unsafe_fn();
    }
}
"#,
        );
    }

    #[test]
    fn missing_unsafe_diagnostic_with_static_mut() {
        check_diagnostics(
            r#"
struct Ty {
    a: u8,
}

static mut static_mut: Ty = Ty { a: 0 };

fn main() {
    let x = static_mut.a;
          //^^^^^^^^^^ This operation is unsafe and requires an unsafe function or block
    unsafe {
        let x = static_mut.a;
    }
}
"#,
        );
    }
}
