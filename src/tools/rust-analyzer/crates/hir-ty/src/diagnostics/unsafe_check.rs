//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use hir_def::{
    body::Body,
    hir::{Expr, ExprId, UnaryOp},
    resolver::{resolver_for_expr, ResolveValueResult, Resolver, ValueNs},
    type_ref::Rawness,
    DefWithBodyId,
};

use crate::{
    db::HirDatabase, utils::is_fn_unsafe_to_call, InferenceResult, Interner, TyExt, TyKind,
};

/// Returns `(unsafe_exprs, fn_is_unsafe)`.
///
/// If `fn_is_unsafe` is false, `unsafe_exprs` are hard errors. If true, they're `unsafe_op_in_unsafe_fn`.
pub fn missing_unsafe(db: &dyn HirDatabase, def: DefWithBodyId) -> (Vec<ExprId>, bool) {
    let _p = tracing::info_span!("missing_unsafe").entered();

    let mut res = Vec::new();
    let is_unsafe = match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).is_unsafe(),
        DefWithBodyId::StaticId(_)
        | DefWithBodyId::ConstId(_)
        | DefWithBodyId::VariantId(_)
        | DefWithBodyId::InTypeConstId(_) => false,
    };

    let body = db.body(def);
    let infer = db.infer(def);
    unsafe_expressions(db, &infer, def, &body, body.body_expr, &mut |expr| {
        if !expr.inside_unsafe_block {
            res.push(expr.expr);
        }
    });

    (res, is_unsafe)
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
    walk_unsafe(
        db,
        infer,
        body,
        &mut resolver_for_expr(db.upcast(), def, current),
        def,
        current,
        false,
        unsafe_expr_cb,
    )
}

fn walk_unsafe(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    body: &Body,
    resolver: &mut Resolver,
    def: DefWithBodyId,
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
            let g = resolver.update_to_inner_scope(db.upcast(), def, current);
            let value_or_partial = resolver.resolve_path_in_value_ns(db.upcast(), path);
            if let Some(ResolveValueResult::ValueNs(ValueNs::StaticId(id), _)) = value_or_partial {
                let static_data = db.static_data(id);
                if static_data.mutable || (static_data.is_extern && !static_data.has_safe_kw) {
                    unsafe_expr_cb(UnsafeExpr { expr: current, inside_unsafe_block });
                }
            }
            resolver.reset_to_guard(g);
        }
        Expr::Ref { expr, rawness: Rawness::RawPtr, mutability: _ } => {
            if let Expr::Path(_) = body.exprs[*expr] {
                // Do not report unsafe for `addr_of[_mut]!(EXTERN_OR_MUT_STATIC)`,
                // see https://github.com/rust-lang/rust/pull/125834.
                return;
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
                walk_unsafe(db, infer, body, resolver, def, child, true, unsafe_expr_cb);
            });
        }
        _ => {}
    }

    expr.walk_child_exprs(|child| {
        walk_unsafe(db, infer, body, resolver, def, child, inside_unsafe_block, unsafe_expr_cb);
    });
}
