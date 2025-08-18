use rustc_ast::ast;
use rustc_span::Span;
use thin_vec::thin_vec;
use tracing::debug;

use crate::attr::get_attrs_from_stmt;
use crate::config::StyleEdition;
use crate::config::lists::*;
use crate::expr::{block_contains_comment, is_simple_block, is_unsafe_block, rewrite_cond};
use crate::items::{span_hi_for_param, span_lo_for_param};
use crate::lists::{ListFormatting, Separator, definitive_tactic, itemize_list, write_list};
use crate::overflow::OverflowableItem;
use crate::rewrite::{Rewrite, RewriteContext, RewriteError, RewriteErrorExt, RewriteResult};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::types::rewrite_bound_params;
use crate::utils::{NodeIdExt, last_line_width, left_most_sub_expr, stmt_expr};

// This module is pretty messy because of the rules around closures and blocks:
// FIXME - the below is probably no longer true in full.
//   * if there is a return type, then there must be braces,
//   * given a closure with braces, whether that is parsed to give an inner block
//     or not depends on if there is a return type and if there are statements
//     in that block,
//   * if the first expression in the body ends with a block (i.e., is a
//     statement without needing a semi-colon), then adding or removing braces
//     can change whether it is treated as an expression or statement.

pub(crate) fn rewrite_closure(
    binder: &ast::ClosureBinder,
    constness: ast::Const,
    capture: ast::CaptureBy,
    coroutine_kind: &Option<ast::CoroutineKind>,
    movability: ast::Movability,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    debug!("rewrite_closure {:?}", body);

    let (prefix, extra_offset) = rewrite_closure_fn_decl(
        binder,
        constness,
        capture,
        coroutine_kind,
        movability,
        fn_decl,
        body,
        span,
        context,
        shape,
    )?;
    // 1 = space between `|...|` and body.
    let body_shape = shape
        .offset_left(extra_offset)
        .max_width_error(shape.width, span)?;

    if let ast::ExprKind::Block(ref block, _) = body.kind {
        // The body of the closure is an empty block.
        if block.stmts.is_empty() && !block_contains_comment(context, block) {
            return body
                .rewrite_result(context, shape)
                .map(|s| format!("{} {}", prefix, s));
        }

        let result = match fn_decl.output {
            ast::FnRetTy::Default(_) if !context.inside_macro() => {
                try_rewrite_without_block(body, &prefix, context, shape, body_shape)
            }
            _ => Err(RewriteError::Unknown),
        };

        result.or_else(|_| {
            // Either we require a block, or tried without and failed.
            rewrite_closure_block(block, &prefix, context, body_shape)
        })
    } else {
        rewrite_closure_expr(body, &prefix, context, body_shape).or_else(|_| {
            // The closure originally had a non-block expression, but we can't fit on
            // one line, so we'll insert a block.
            rewrite_closure_with_block(body, &prefix, context, body_shape)
        })
    }
}

fn try_rewrite_without_block(
    expr: &ast::Expr,
    prefix: &str,
    context: &RewriteContext<'_>,
    shape: Shape,
    body_shape: Shape,
) -> RewriteResult {
    let expr = get_inner_expr(expr, prefix, context);

    if is_block_closure_forced(context, expr) {
        rewrite_closure_with_block(expr, prefix, context, shape)
    } else {
        rewrite_closure_expr(expr, prefix, context, body_shape)
    }
}

fn get_inner_expr<'a>(
    expr: &'a ast::Expr,
    prefix: &str,
    context: &RewriteContext<'_>,
) -> &'a ast::Expr {
    if let ast::ExprKind::Block(ref block, _) = expr.kind {
        if !needs_block(block, prefix, context) {
            // block.stmts.len() == 1 except with `|| {{}}`;
            // https://github.com/rust-lang/rustfmt/issues/3844
            if let Some(expr) = block.stmts.first().and_then(stmt_expr) {
                return get_inner_expr(expr, prefix, context);
            }
        }
    }

    expr
}

// Figure out if a block is necessary.
fn needs_block(block: &ast::Block, prefix: &str, context: &RewriteContext<'_>) -> bool {
    let has_attributes = block.stmts.first().map_or(false, |first_stmt| {
        !get_attrs_from_stmt(first_stmt).is_empty()
    });

    is_unsafe_block(block)
        || block.stmts.len() > 1
        || has_attributes
        || block_contains_comment(context, block)
        || prefix.contains('\n')
}

fn veto_block(e: &ast::Expr) -> bool {
    match e.kind {
        ast::ExprKind::Call(..)
        | ast::ExprKind::Binary(..)
        | ast::ExprKind::Cast(..)
        | ast::ExprKind::Type(..)
        | ast::ExprKind::Assign(..)
        | ast::ExprKind::AssignOp(..)
        | ast::ExprKind::Field(..)
        | ast::ExprKind::Index(..)
        | ast::ExprKind::Range(..)
        | ast::ExprKind::Try(..) => true,
        _ => false,
    }
}

// Rewrite closure with a single expression wrapping its body with block.
// || { #[attr] foo() } -> Block { #[attr] foo() }
fn rewrite_closure_with_block(
    body: &ast::Expr,
    prefix: &str,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    let left_most = left_most_sub_expr(body);
    let veto_block = veto_block(body) && !expr_requires_semi_to_be_stmt(left_most);
    if veto_block {
        return Err(RewriteError::Unknown);
    }

    let block = ast::Block {
        stmts: thin_vec![ast::Stmt {
            id: ast::NodeId::root(),
            kind: ast::StmtKind::Expr(Box::new(body.clone())),
            span: body.span,
        }],
        id: ast::NodeId::root(),
        rules: ast::BlockCheckMode::Default,
        tokens: None,
        span: body
            .attrs
            .first()
            .map(|attr| attr.span.to(body.span))
            .unwrap_or(body.span),
    };
    let block = crate::expr::rewrite_block_with_visitor(
        context,
        "",
        &block,
        Some(&body.attrs),
        None,
        shape,
        false,
    )?;
    Ok(format!("{prefix} {block}"))
}

// Rewrite closure with a single expression without wrapping its body with block.
fn rewrite_closure_expr(
    expr: &ast::Expr,
    prefix: &str,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    fn allow_multi_line(expr: &ast::Expr) -> bool {
        match expr.kind {
            ast::ExprKind::Match(..)
            | ast::ExprKind::Gen(..)
            | ast::ExprKind::Block(..)
            | ast::ExprKind::TryBlock(..)
            | ast::ExprKind::Loop(..)
            | ast::ExprKind::Struct(..) => true,

            ast::ExprKind::AddrOf(_, _, ref expr)
            | ast::ExprKind::Try(ref expr)
            | ast::ExprKind::Unary(_, ref expr)
            | ast::ExprKind::Cast(ref expr, _) => allow_multi_line(expr),

            _ => false,
        }
    }

    // When rewriting closure's body without block, we require it to fit in a single line
    // unless it is a block-like expression or we are inside macro call.
    let veto_multiline = (!allow_multi_line(expr) && !context.inside_macro())
        || context.config.force_multiline_blocks();
    expr.rewrite_result(context, shape)
        .and_then(|rw| {
            if veto_multiline && rw.contains('\n') {
                Err(RewriteError::Unknown)
            } else {
                Ok(rw)
            }
        })
        .map(|rw| format!("{} {}", prefix, rw))
}

// Rewrite closure whose body is block.
fn rewrite_closure_block(
    block: &ast::Block,
    prefix: &str,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    Ok(format!(
        "{} {}",
        prefix,
        block.rewrite_result(context, shape)?
    ))
}

// Return type is (prefix, extra_offset)
fn rewrite_closure_fn_decl(
    binder: &ast::ClosureBinder,
    constness: ast::Const,
    capture: ast::CaptureBy,
    coroutine_kind: &Option<ast::CoroutineKind>,
    movability: ast::Movability,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Result<(String, usize), RewriteError> {
    let binder = match binder {
        ast::ClosureBinder::For { generic_params, .. } if generic_params.is_empty() => {
            "for<> ".to_owned()
        }
        ast::ClosureBinder::For { generic_params, .. } => {
            let lifetime_str =
                rewrite_bound_params(context, shape, generic_params).unknown_error()?;
            format!("for<{lifetime_str}> ")
        }
        ast::ClosureBinder::NotPresent => "".to_owned(),
    };

    let const_ = if matches!(constness, ast::Const::Yes(_)) {
        "const "
    } else {
        ""
    };

    let immovable = if movability == ast::Movability::Static {
        "static "
    } else {
        ""
    };
    let coro = match coroutine_kind {
        Some(ast::CoroutineKind::Async { .. }) => "async ",
        Some(ast::CoroutineKind::Gen { .. }) => "gen ",
        Some(ast::CoroutineKind::AsyncGen { .. }) => "async gen ",
        None => "",
    };
    let mover = if matches!(capture, ast::CaptureBy::Value { .. }) {
        "move "
    } else {
        ""
    };
    // 4 = "|| {".len(), which is overconservative when the closure consists of
    // a single expression.
    let nested_shape = shape
        .shrink_left(binder.len() + const_.len() + immovable.len() + coro.len() + mover.len())
        .and_then(|shape| shape.sub_width(4))
        .max_width_error(shape.width, span)?;

    // 1 = |
    let param_offset = nested_shape.indent + 1;
    let param_shape = nested_shape
        .offset_left(1)
        .max_width_error(nested_shape.width, span)?
        .visual_indent(0);
    let ret_str = fn_decl.output.rewrite_result(context, param_shape)?;

    let param_items = itemize_list(
        context.snippet_provider,
        fn_decl.inputs.iter(),
        "|",
        ",",
        |param| span_lo_for_param(param),
        |param| span_hi_for_param(context, param),
        |param| param.rewrite_result(context, param_shape),
        context.snippet_provider.span_after(span, "|"),
        body.span.lo(),
        false,
    );
    let item_vec = param_items.collect::<Vec<_>>();
    // 1 = space between parameters and return type.
    let horizontal_budget = nested_shape.width.saturating_sub(ret_str.len() + 1);
    let tactic = definitive_tactic(
        &item_vec,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        horizontal_budget,
    );
    let param_shape = match tactic {
        DefinitiveListTactic::Horizontal => param_shape
            .sub_width(ret_str.len() + 1)
            .max_width_error(param_shape.width, span)?,
        _ => param_shape,
    };

    let fmt = ListFormatting::new(param_shape, context.config)
        .tactic(tactic)
        .preserve_newline(true);
    let list_str = write_list(&item_vec, &fmt)?;
    let mut prefix = format!("{binder}{const_}{immovable}{coro}{mover}|{list_str}|");

    if !ret_str.is_empty() {
        if prefix.contains('\n') {
            prefix.push('\n');
            prefix.push_str(&param_offset.to_string(context.config));
        } else {
            prefix.push(' ');
        }
        prefix.push_str(&ret_str);
    }
    // 1 = space between `|...|` and body.
    let extra_offset = last_line_width(&prefix) + 1;

    Ok((prefix, extra_offset))
}

// Rewriting closure which is placed at the end of the function call's arg.
// Returns `None` if the reformatted closure 'looks bad'.
pub(crate) fn rewrite_last_closure(
    context: &RewriteContext<'_>,
    expr: &ast::Expr,
    shape: Shape,
) -> RewriteResult {
    if let ast::ExprKind::Closure(ref closure) = expr.kind {
        let ast::Closure {
            ref binder,
            constness,
            capture_clause,
            ref coroutine_kind,
            movability,
            ref fn_decl,
            ref body,
            fn_decl_span: _,
            fn_arg_span: _,
        } = **closure;
        let body = match body.kind {
            ast::ExprKind::Block(ref block, _)
                if !is_unsafe_block(block)
                    && !context.inside_macro()
                    && is_simple_block(context, block, Some(&body.attrs)) =>
            {
                stmt_expr(&block.stmts[0]).unwrap_or(body)
            }
            _ => body,
        };
        let (prefix, extra_offset) = rewrite_closure_fn_decl(
            binder,
            constness,
            capture_clause,
            coroutine_kind,
            movability,
            fn_decl,
            body,
            expr.span,
            context,
            shape,
        )?;
        // If the closure goes multi line before its body, do not overflow the closure.
        if prefix.contains('\n') {
            return Err(RewriteError::Unknown);
        }

        let body_shape = shape
            .offset_left(extra_offset)
            .max_width_error(shape.width, expr.span)?;

        // We force to use block for the body of the closure for certain kinds of expressions.
        if is_block_closure_forced(context, body) {
            return rewrite_closure_with_block(body, &prefix, context, body_shape).map(
                |body_str| {
                    match fn_decl.output {
                        ast::FnRetTy::Default(..) if body_str.lines().count() <= 7 => {
                            // If the expression can fit in a single line, we need not force block
                            // closure.  However, if the closure has a return type, then we must
                            // keep the blocks.
                            match rewrite_closure_expr(body, &prefix, context, shape) {
                                Ok(single_line_body_str)
                                    if !single_line_body_str.contains('\n') =>
                                {
                                    single_line_body_str
                                }
                                _ => body_str,
                            }
                        }
                        _ => body_str,
                    }
                },
            );
        }

        // When overflowing the closure which consists of a single control flow expression,
        // force to use block if its condition uses multi line.
        let is_multi_lined_cond = rewrite_cond(context, body, body_shape).map_or(false, |cond| {
            cond.contains('\n') || cond.len() > body_shape.width
        });
        if is_multi_lined_cond {
            return rewrite_closure_with_block(body, &prefix, context, body_shape);
        }

        // Seems fine, just format the closure in usual manner.
        return expr.rewrite_result(context, shape);
    }
    Err(RewriteError::Unknown)
}

/// Returns `true` if the given vector of arguments has more than one `ast::ExprKind::Closure`.
pub(crate) fn args_have_many_closure(args: &[OverflowableItem<'_>]) -> bool {
    args.iter()
        .filter_map(OverflowableItem::to_expr)
        .filter(|expr| matches!(expr.kind, ast::ExprKind::Closure(..)))
        .count()
        > 1
}

fn is_block_closure_forced(context: &RewriteContext<'_>, expr: &ast::Expr) -> bool {
    // If we are inside macro, we do not want to add or remove block from closure body.
    if context.inside_macro() {
        false
    } else {
        is_block_closure_forced_inner(expr, context.config.style_edition())
    }
}

fn is_block_closure_forced_inner(expr: &ast::Expr, style_edition: StyleEdition) -> bool {
    match expr.kind {
        ast::ExprKind::If(..) | ast::ExprKind::While(..) | ast::ExprKind::ForLoop { .. } => true,
        ast::ExprKind::Loop(..) if style_edition >= StyleEdition::Edition2024 => true,
        ast::ExprKind::AddrOf(_, _, ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => is_block_closure_forced_inner(expr, style_edition),
        _ => false,
    }
}

/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
// From https://github.com/rust-lang/rust/blob/master/src/libsyntax/parse/classify.rs.
fn expr_requires_semi_to_be_stmt(e: &ast::Expr) -> bool {
    match e.kind {
        ast::ExprKind::If(..)
        | ast::ExprKind::Match(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::ForLoop { .. }
        | ast::ExprKind::TryBlock(..) => false,
        _ => true,
    }
}
