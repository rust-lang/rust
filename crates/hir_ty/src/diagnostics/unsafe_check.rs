//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use hir_def::{
    body::Body,
    expr::{Expr, ExprId, UnaryOp},
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId,
};

use crate::{db::HirDatabase, InferenceResult, Interner, TyExt, TyKind};

pub fn missing_unsafe(db: &dyn HirDatabase, def: DefWithBodyId) -> Vec<ExprId> {
    let infer = db.infer(def);

    let is_unsafe = match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).is_unsafe(),
        DefWithBodyId::StaticId(_) | DefWithBodyId::ConstId(_) => false,
    };
    if is_unsafe {
        return Vec::new();
    }

    unsafe_expressions(db, &infer, def)
        .into_iter()
        .filter(|it| !it.inside_unsafe_block)
        .map(|it| it.expr)
        .collect()
}

struct UnsafeExpr {
    pub(crate) expr: ExprId,
    pub(crate) inside_unsafe_block: bool,
}

fn unsafe_expressions(
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
        &Expr::Call { callee, .. } => {
            if let Some(func) = infer[callee].as_fn_def(db) {
                if db.function_data(func).is_unsafe() {
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
                .map(|(func, _)| db.function_data(func).is_unsafe())
                .unwrap_or(false)
            {
                unsafe_exprs.push(UnsafeExpr { expr: current, inside_unsafe_block });
            }
        }
        Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
            if let TyKind::Raw(..) = &infer[*expr].kind(Interner) {
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
