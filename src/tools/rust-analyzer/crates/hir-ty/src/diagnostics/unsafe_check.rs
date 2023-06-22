//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use hir_def::{
    body::Body,
    hir::{Expr, ExprId, UnaryOp},
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    DefWithBodyId,
};

use crate::{
    db::HirDatabase, utils::is_fn_unsafe_to_call, InferenceResult, Interner, TyExt, TyKind,
};

pub fn missing_unsafe(db: &dyn HirDatabase, def: DefWithBodyId) -> Vec<ExprId> {
    let infer = db.infer(def);
    let mut res = Vec::new();

    let is_unsafe = match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).has_unsafe_kw(),
        DefWithBodyId::StaticId(_)
        | DefWithBodyId::ConstId(_)
        | DefWithBodyId::VariantId(_)
        | DefWithBodyId::InTypeConstId(_) => false,
    };
    if is_unsafe {
        return res;
    }

    let body = db.body(def);
    unsafe_expressions(db, &infer, def, &body, body.body_expr, &mut |expr| {
        if !expr.inside_unsafe_block {
            res.push(expr.expr);
        }
    });

    res
}

pub struct UnsafeExpr {
    pub expr: ExprId,
    pub inside_unsafe_block: bool,
}

// FIXME: Move this out, its not a diagnostic only thing anymore, and handle unsafe pattern accesses as well
pub fn unsafe_expressions(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
    body: &Body,
    current: ExprId,
    unsafe_expr_cb: &mut dyn FnMut(UnsafeExpr),
) {
    walk_unsafe(db, infer, def, body, current, false, unsafe_expr_cb)
}

fn walk_unsafe(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
    body: &Body,
    current: ExprId,
    inside_unsafe_block: bool,
    unsafe_expr_cb: &mut dyn FnMut(UnsafeExpr),
) {
    let expr = &body.exprs[current];
    match expr {
        &Expr::Call { callee, .. } => {
            if let Some(func) = infer[callee].as_fn_def(db) {
                if is_fn_unsafe_to_call(db, func) {
                    unsafe_expr_cb(UnsafeExpr { expr: current, inside_unsafe_block });
                }
            }
        }
        Expr::Path(path) => {
            let resolver = resolver_for_expr(db.upcast(), def, current);
            let value_or_partial = resolver.resolve_path_in_value_ns(db.upcast(), path);
            if let Some(ResolveValueResult::ValueNs(ValueNs::StaticId(id))) = value_or_partial {
                if db.static_data(id).mutable {
                    unsafe_expr_cb(UnsafeExpr { expr: current, inside_unsafe_block });
                }
            }
        }
        Expr::MethodCall { .. } => {
            if infer
                .method_resolution(current)
                .map(|(func, _)| is_fn_unsafe_to_call(db, func))
                .unwrap_or(false)
            {
                unsafe_expr_cb(UnsafeExpr { expr: current, inside_unsafe_block });
            }
        }
        Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
            if let TyKind::Raw(..) = &infer[*expr].kind(Interner) {
                unsafe_expr_cb(UnsafeExpr { expr: current, inside_unsafe_block });
            }
        }
        Expr::Unsafe { .. } => {
            return expr.walk_child_exprs(|child| {
                walk_unsafe(db, infer, def, body, child, true, unsafe_expr_cb);
            });
        }
        _ => {}
    }

    expr.walk_child_exprs(|child| {
        walk_unsafe(db, infer, def, body, child, inside_unsafe_block, unsafe_expr_cb);
    });
}
