use crate::expr::{lit_ends_in_dot, rewrite_unary_prefix, rewrite_unary_suffix};
use crate::pairs::{PairParts, rewrite_pair};
use crate::rewrite::{RewriteContext, RewriteResult};
use crate::shape::Shape;

use rustc_ast::ast;

fn needs_space_before_range(context: &RewriteContext<'_>, lhs: &ast::Expr) -> bool {
    match lhs.kind {
        ast::ExprKind::Lit(token_lit) => lit_ends_in_dot(&token_lit, context),
        ast::ExprKind::Unary(_, ref expr) => needs_space_before_range(context, expr),
        ast::ExprKind::Binary(_, _, ref rhs_expr) => needs_space_before_range(context, rhs_expr),
        _ => false,
    }
}

fn needs_space_after_range(rhs: &ast::Expr) -> bool {
    // Don't format `.. ..` into `....`, which is invalid.
    //
    // This check is unnecessary for `lhs`, because a range
    // starting from another range needs parentheses as `(x ..) ..`
    // (`x .. ..` is a range from `x` to `..`).
    matches!(rhs.kind, ast::ExprKind::Range(None, _, _))
}

pub(crate) fn rewrite_range(
    context: &RewriteContext<'_>,
    shape: Shape,
    lhs: Option<&ast::Expr>,
    rhs: Option<&ast::Expr>,
    delim: &str,
) -> RewriteResult {
    let default_sp_delim = |lhs: Option<&ast::Expr>, rhs: Option<&ast::Expr>| {
        let space_if = |b: bool| if b { " " } else { "" };

        format!(
            "{}{}{}",
            lhs.map_or("", |lhs| space_if(needs_space_before_range(context, lhs))),
            delim,
            rhs.map_or("", |rhs| space_if(needs_space_after_range(rhs))),
        )
    };

    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            let sp_delim = if context.config.spaces_around_ranges() {
                format!(" {delim} ")
            } else {
                default_sp_delim(Some(lhs), Some(rhs))
            };
            rewrite_pair(
                lhs,
                rhs,
                PairParts::infix(&sp_delim),
                context,
                shape,
                context.config.binop_separator(),
            )
        }
        (None, Some(rhs)) => {
            let sp_delim = if context.config.spaces_around_ranges() {
                format!("{delim} ")
            } else {
                default_sp_delim(None, Some(rhs))
            };
            rewrite_unary_prefix(context, &sp_delim, rhs, shape)
        }
        (Some(lhs), None) => {
            let sp_delim = if context.config.spaces_around_ranges() {
                format!(" {delim}")
            } else {
                default_sp_delim(Some(lhs), None)
            };
            rewrite_unary_suffix(context, &sp_delim, lhs, shape)
        }
        (None, None) => Ok(delim.to_owned()),
    }
}
