use syntax::source_map::Span;
use syntax::{ast, ptr};

use crate::config::lists::*;
use crate::config::Version;
use crate::expr::{block_contains_comment, is_simple_block, is_unsafe_block, rewrite_cond};
use crate::items::{span_hi_for_arg, span_lo_for_arg};
use crate::lists::{definitive_tactic, itemize_list, write_list, ListFormatting, Separator};
use crate::overflow::OverflowableItem;
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::utils::{last_line_width, left_most_sub_expr, stmt_expr, NodeIdExt};

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
    capture: ast::CaptureBy,
    is_async: &ast::IsAsync,
    movability: ast::Movability,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    debug!("rewrite_closure {:?}", body);

    let (prefix, extra_offset) = rewrite_closure_fn_decl(
        capture, is_async, movability, fn_decl, body, span, context, shape,
    )?;
    // 1 = space between `|...|` and body.
    let body_shape = shape.offset_left(extra_offset)?;

    if let ast::ExprKind::Block(ref block, _) = body.node {
        // The body of the closure is an empty block.
        if block.stmts.is_empty() && !block_contains_comment(block, context.source_map) {
            return body
                .rewrite(context, shape)
                .map(|s| format!("{} {}", prefix, s));
        }

        let result = match fn_decl.output {
            ast::FunctionRetTy::Default(_) if !context.inside_macro() => {
                try_rewrite_without_block(body, &prefix, context, shape, body_shape)
            }
            _ => None,
        };

        result.or_else(|| {
            // Either we require a block, or tried without and failed.
            rewrite_closure_block(block, &prefix, context, body_shape)
        })
    } else {
        rewrite_closure_expr(body, &prefix, context, body_shape).or_else(|| {
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
) -> Option<String> {
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
    if let ast::ExprKind::Block(ref block, _) = expr.node {
        if !needs_block(block, prefix, context) {
            // block.stmts.len() == 1
            if let Some(expr) = stmt_expr(&block.stmts[0]) {
                return get_inner_expr(expr, prefix, context);
            }
        }
    }

    expr
}

// Figure out if a block is necessary.
fn needs_block(block: &ast::Block, prefix: &str, context: &RewriteContext<'_>) -> bool {
    is_unsafe_block(block)
        || block.stmts.len() > 1
        || block_contains_comment(block, context.source_map)
        || prefix.contains('\n')
}

fn veto_block(e: &ast::Expr) -> bool {
    match e.node {
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
fn rewrite_closure_with_block(
    body: &ast::Expr,
    prefix: &str,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    let left_most = left_most_sub_expr(body);
    let veto_block = veto_block(body) && !expr_requires_semi_to_be_stmt(left_most);
    if veto_block {
        return None;
    }

    let block = ast::Block {
        stmts: vec![ast::Stmt {
            id: ast::NodeId::root(),
            node: ast::StmtKind::Expr(ptr::P(body.clone())),
            span: body.span,
        }],
        id: ast::NodeId::root(),
        rules: ast::BlockCheckMode::Default,
        span: body.span,
    };
    let block =
        crate::expr::rewrite_block_with_visitor(context, "", &block, None, None, shape, false)?;
    Some(format!("{} {}", prefix, block))
}

// Rewrite closure with a single expression without wrapping its body with block.
fn rewrite_closure_expr(
    expr: &ast::Expr,
    prefix: &str,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    fn allow_multi_line(expr: &ast::Expr) -> bool {
        match expr.node {
            ast::ExprKind::Match(..)
            | ast::ExprKind::Block(..)
            | ast::ExprKind::TryBlock(..)
            | ast::ExprKind::Loop(..)
            | ast::ExprKind::Struct(..) => true,

            ast::ExprKind::AddrOf(_, ref expr)
            | ast::ExprKind::Box(ref expr)
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
    expr.rewrite(context, shape)
        .and_then(|rw| {
            if veto_multiline && rw.contains('\n') {
                None
            } else {
                Some(rw)
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
) -> Option<String> {
    Some(format!("{} {}", prefix, block.rewrite(context, shape)?))
}

// Return type is (prefix, extra_offset)
fn rewrite_closure_fn_decl(
    capture: ast::CaptureBy,
    asyncness: &ast::IsAsync,
    movability: ast::Movability,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<(String, usize)> {
    let is_async = if asyncness.is_async() { "async " } else { "" };
    let mover = if capture == ast::CaptureBy::Value {
        "move "
    } else {
        ""
    };
    let immovable = if movability == ast::Movability::Static {
        "static "
    } else {
        ""
    };
    // 4 = "|| {".len(), which is overconservative when the closure consists of
    // a single expression.
    let nested_shape = shape
        .shrink_left(is_async.len() + mover.len() + immovable.len())?
        .sub_width(4)?;

    // 1 = |
    let argument_offset = nested_shape.indent + 1;
    let arg_shape = nested_shape.offset_left(1)?.visual_indent(0);
    let ret_str = fn_decl.output.rewrite(context, arg_shape)?;

    let arg_items = itemize_list(
        context.snippet_provider,
        fn_decl.inputs.iter(),
        "|",
        ",",
        |arg| span_lo_for_arg(arg),
        |arg| span_hi_for_arg(context, arg),
        |arg| arg.rewrite(context, arg_shape),
        context.snippet_provider.span_after(span, "|"),
        body.span.lo(),
        false,
    );
    let item_vec = arg_items.collect::<Vec<_>>();
    // 1 = space between arguments and return type.
    let horizontal_budget = nested_shape.width.saturating_sub(ret_str.len() + 1);
    let tactic = definitive_tactic(
        &item_vec,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        horizontal_budget,
    );
    let arg_shape = match tactic {
        DefinitiveListTactic::Horizontal => arg_shape.sub_width(ret_str.len() + 1)?,
        _ => arg_shape,
    };

    let fmt = ListFormatting::new(arg_shape, context.config)
        .tactic(tactic)
        .preserve_newline(true);
    let list_str = write_list(&item_vec, &fmt)?;
    let mut prefix = format!("{}{}{}|{}|", is_async, immovable, mover, list_str);

    if !ret_str.is_empty() {
        if prefix.contains('\n') {
            prefix.push('\n');
            prefix.push_str(&argument_offset.to_string(context.config));
        } else {
            prefix.push(' ');
        }
        prefix.push_str(&ret_str);
    }
    // 1 = space between `|...|` and body.
    let extra_offset = last_line_width(&prefix) + 1;

    Some((prefix, extra_offset))
}

// Rewriting closure which is placed at the end of the function call's arg.
// Returns `None` if the reformatted closure 'looks bad'.
pub(crate) fn rewrite_last_closure(
    context: &RewriteContext<'_>,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    if let ast::ExprKind::Closure(capture, ref is_async, movability, ref fn_decl, ref body, _) =
        expr.node
    {
        let body = match body.node {
            ast::ExprKind::Block(ref block, _)
                if !is_unsafe_block(block)
                    && !context.inside_macro()
                    && is_simple_block(block, Some(&body.attrs), context.source_map) =>
            {
                stmt_expr(&block.stmts[0]).unwrap_or(body)
            }
            _ => body,
        };
        let (prefix, extra_offset) = rewrite_closure_fn_decl(
            capture, is_async, movability, fn_decl, body, expr.span, context, shape,
        )?;
        // If the closure goes multi line before its body, do not overflow the closure.
        if prefix.contains('\n') {
            return None;
        }

        let body_shape = shape.offset_left(extra_offset)?;

        // We force to use block for the body of the closure for certain kinds of expressions.
        if is_block_closure_forced(context, body) {
            return rewrite_closure_with_block(body, &prefix, context, body_shape).and_then(
                |body_str| {
                    // If the expression can fit in a single line, we need not force block closure.
                    if body_str.lines().count() <= 7 {
                        match rewrite_closure_expr(body, &prefix, context, shape) {
                            Some(ref single_line_body_str)
                                if !single_line_body_str.contains('\n') =>
                            {
                                Some(single_line_body_str.clone())
                            }
                            _ => Some(body_str),
                        }
                    } else {
                        Some(body_str)
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
        return expr.rewrite(context, shape);
    }
    None
}

/// Returns `true` if the given vector of arguments has more than one `ast::ExprKind::Closure`.
pub(crate) fn args_have_many_closure(args: &[OverflowableItem<'_>]) -> bool {
    args.iter()
        .filter_map(OverflowableItem::to_expr)
        .filter(|expr| match expr.node {
            ast::ExprKind::Closure(..) => true,
            _ => false,
        })
        .count()
        > 1
}

fn is_block_closure_forced(context: &RewriteContext<'_>, expr: &ast::Expr) -> bool {
    // If we are inside macro, we do not want to add or remove block from closure body.
    if context.inside_macro() {
        false
    } else {
        is_block_closure_forced_inner(expr, context.config.version())
    }
}

fn is_block_closure_forced_inner(expr: &ast::Expr, version: Version) -> bool {
    match expr.node {
        ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..)
        | ast::ExprKind::ForLoop(..) => true,
        ast::ExprKind::Loop(..) if version == Version::Two => true,
        ast::ExprKind::AddrOf(_, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => is_block_closure_forced_inner(expr, version),
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
    match e.node {
        ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::Match(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::TryBlock(..) => false,
        _ => true,
    }
}
