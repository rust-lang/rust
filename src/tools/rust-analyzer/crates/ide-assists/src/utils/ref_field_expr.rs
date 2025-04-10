//! This module contains a helper for converting a field access expression into a
//! path expression. This is used when destructuring a tuple or struct.
//!
//! It determines whether to deref the new expression and/or wrap it in parentheses,
//! based on the parent of the existing expression.
use syntax::{
    AstNode, T,
    ast::{self, FieldExpr, MethodCallExpr, make},
};

use crate::AssistContext;

/// Decides whether the new path expression needs to be dereferenced and/or wrapped in parens.
/// Returns the relevant parent expression to replace and the [RefData].
pub(crate) fn determine_ref_and_parens(
    ctx: &AssistContext<'_>,
    field_expr: &FieldExpr,
) -> (ast::Expr, RefData) {
    let s = field_expr.syntax();
    let mut ref_data = RefData { needs_deref: true, needs_parentheses: true };
    let mut target_node = field_expr.clone().into();

    let parent = match s.parent().map(ast::Expr::cast) {
        Some(Some(parent)) => parent,
        Some(None) => {
            ref_data.needs_parentheses = false;
            return (target_node, ref_data);
        }
        None => return (target_node, ref_data),
    };

    match parent {
        ast::Expr::ParenExpr(it) => {
            // already parens in place -> don't replace
            ref_data.needs_parentheses = false;
            // there might be a ref outside: `&(t.0)` -> can be removed
            if let Some(it) = it.syntax().parent().and_then(ast::RefExpr::cast) {
                ref_data.needs_deref = false;
                target_node = it.into();
            }
        }
        ast::Expr::RefExpr(it) => {
            // `&*` -> cancel each other out
            ref_data.needs_deref = false;
            ref_data.needs_parentheses = false;
            // might be surrounded by parens -> can be removed too
            match it.syntax().parent().and_then(ast::ParenExpr::cast) {
                Some(parent) => target_node = parent.into(),
                None => target_node = it.into(),
            };
        }
        // higher precedence than deref `*`
        // https://doc.rust-lang.org/reference/expressions.html#expression-precedence
        // -> requires parentheses
        ast::Expr::PathExpr(_it) => {}
        ast::Expr::MethodCallExpr(it) => {
            // `field_expr` is `self_param` (otherwise it would be in `ArgList`)

            // test if there's already auto-ref in place (`value` -> `&value`)
            // -> no method accepting `self`, but `&self` -> no need for deref
            //
            // other combinations (`&value` -> `value`, `&&value` -> `&value`, `&value` -> `&&value`) might or might not be able to auto-ref/deref,
            // but there might be trait implementations an added `&` might resolve to
            // -> ONLY handle auto-ref from `value` to `&value`
            fn is_auto_ref(ctx: &AssistContext<'_>, call_expr: &MethodCallExpr) -> bool {
                fn impl_(ctx: &AssistContext<'_>, call_expr: &MethodCallExpr) -> Option<bool> {
                    let rec = call_expr.receiver()?;
                    let rec_ty = ctx.sema.type_of_expr(&rec)?.original();
                    // input must be actual value
                    if rec_ty.is_reference() {
                        return Some(false);
                    }

                    // doesn't resolve trait impl
                    let f = ctx.sema.resolve_method_call(call_expr)?;
                    let self_param = f.self_param(ctx.db())?;
                    // self must be ref
                    match self_param.access(ctx.db()) {
                        hir::Access::Shared | hir::Access::Exclusive => Some(true),
                        hir::Access::Owned => Some(false),
                    }
                }
                impl_(ctx, call_expr).unwrap_or(false)
            }

            if is_auto_ref(ctx, &it) {
                ref_data.needs_deref = false;
                ref_data.needs_parentheses = false;
            }
        }
        ast::Expr::FieldExpr(_it) => {
            // `t.0.my_field`
            ref_data.needs_deref = false;
            ref_data.needs_parentheses = false;
        }
        ast::Expr::IndexExpr(_it) => {
            // `t.0[1]`
            ref_data.needs_deref = false;
            ref_data.needs_parentheses = false;
        }
        ast::Expr::TryExpr(_it) => {
            // `t.0?`
            // requires deref and parens: `(*_0)`
        }
        // lower precedence than deref `*` -> no parens
        _ => {
            ref_data.needs_parentheses = false;
        }
    };

    (target_node, ref_data)
}

/// Indicates whether to deref an expression or wrap it in parens
pub(crate) struct RefData {
    needs_deref: bool,
    needs_parentheses: bool,
}

impl RefData {
    /// Derefs `expr` and wraps it in parens if necessary
    pub(crate) fn wrap_expr(&self, mut expr: ast::Expr) -> ast::Expr {
        if self.needs_deref {
            expr = make::expr_prefix(T![*], expr).into();
        }

        if self.needs_parentheses {
            expr = make::expr_paren(expr).into();
        }

        expr
    }
}
