// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::{ast, ptr};
use syntax::codemap::Span;
use syntax::parse::classify;

use codemap::SpanUtils;
use expr::{block_contains_comment, is_simple_block, is_unsafe_block, rewrite_cond, ToExpr};
use items::{span_hi_for_arg, span_lo_for_arg};
use lists::{definitive_tactic, itemize_list, write_list, DefinitiveListTactic, ListFormatting,
            ListTactic, Separator, SeparatorPlace, SeparatorTactic};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use utils::{last_line_width, left_most_sub_expr, stmt_expr};

// This module is pretty messy because of the rules around closures and blocks:
// FIXME - the below is probably no longer true in full.
//   * if there is a return type, then there must be braces,
//   * given a closure with braces, whether that is parsed to give an inner block
//     or not depends on if there is a return type and if there are statements
//     in that block,
//   * if the first expression in the body ends with a block (i.e., is a
//     statement without needing a semi-colon), then adding or removing braces
//     can change whether it is treated as an expression or statement.


pub fn rewrite_closure(
    capture: ast::CaptureBy,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    debug!("rewrite_closure {:?}", body);

    let (prefix, extra_offset) =
        rewrite_closure_fn_decl(capture, fn_decl, body, span, context, shape)?;
    // 1 = space between `|...|` and body.
    let body_shape = shape.offset_left(extra_offset)?;

    if let ast::ExprKind::Block(ref block) = body.node {
        // The body of the closure is an empty block.
        if block.stmts.is_empty() && !block_contains_comment(block, context.codemap) {
            return Some(format!("{} {{}}", prefix));
        }

        let result = match fn_decl.output {
            ast::FunctionRetTy::Default(_) => {
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
    context: &RewriteContext,
    shape: Shape,
    body_shape: Shape,
) -> Option<String> {
    let expr = get_inner_expr(expr, prefix, context);

    if is_block_closure_forced(expr) {
        rewrite_closure_with_block(expr, prefix, context, shape)
    } else {
        rewrite_closure_expr(expr, prefix, context, body_shape)
    }
}

fn get_inner_expr<'a>(
    expr: &'a ast::Expr,
    prefix: &str,
    context: &RewriteContext,
) -> &'a ast::Expr {
    if let ast::ExprKind::Block(ref block) = expr.node {
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
fn needs_block(block: &ast::Block, prefix: &str, context: &RewriteContext) -> bool {
    is_unsafe_block(block) || block.stmts.len() > 1 || context.inside_macro
        || block_contains_comment(block, context.codemap) || prefix.contains('\n')
}

// Rewrite closure with a single expression wrapping its body with block.
fn rewrite_closure_with_block(
    body: &ast::Expr,
    prefix: &str,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let block = ast::Block {
        stmts: vec![
            ast::Stmt {
                id: ast::NodeId::new(0),
                node: ast::StmtKind::Expr(ptr::P(body.clone())),
                span: body.span,
            },
        ],
        id: ast::NodeId::new(0),
        rules: ast::BlockCheckMode::Default,
        span: body.span,
    };
    rewrite_closure_block(&block, prefix, context, shape)
}

// Rewrite closure with a single expression without wrapping its body with block.
fn rewrite_closure_expr(
    expr: &ast::Expr,
    prefix: &str,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let mut rewrite = expr.rewrite(context, shape);
    if classify::expr_requires_semi_to_be_stmt(left_most_sub_expr(expr)) {
        rewrite = and_one_line(rewrite);
    }
    rewrite = rewrite.and_then(|rw| {
        if context.config.multiline_closure_forces_block() && rw.contains('\n') {
            None
        } else {
            Some(rw)
        }
    });
    rewrite.map(|rw| format!("{} {}", prefix, rw))
}

// Rewrite closure whose body is block.
fn rewrite_closure_block(
    block: &ast::Block,
    prefix: &str,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let block_shape = shape.block();
    let block_str = block.rewrite(context, block_shape)?;
    Some(format!("{} {}", prefix, block_str))
}

// Return type is (prefix, extra_offset)
fn rewrite_closure_fn_decl(
    capture: ast::CaptureBy,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<(String, usize)> {
    let mover = if capture == ast::CaptureBy::Value {
        "move "
    } else {
        ""
    };
    // 4 = "|| {".len(), which is overconservative when the closure consists of
    // a single expression.
    let nested_shape = shape.shrink_left(mover.len())?.sub_width(4)?;

    // 1 = |
    let argument_offset = nested_shape.indent + 1;
    let arg_shape = nested_shape.offset_left(1)?.visual_indent(0);
    let ret_str = fn_decl.output.rewrite(context, arg_shape)?;

    let arg_items = itemize_list(
        context.codemap,
        fn_decl.inputs.iter(),
        "|",
        |arg| span_lo_for_arg(arg),
        |arg| span_hi_for_arg(context, arg),
        |arg| arg.rewrite(context, arg_shape),
        context.codemap.span_after(span, "|"),
        body.span.lo(),
        false,
    );
    let item_vec = arg_items.collect::<Vec<_>>();
    // 1 = space between arguments and return type.
    let horizontal_budget = nested_shape
        .width
        .checked_sub(ret_str.len() + 1)
        .unwrap_or(0);
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

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        separator_place: SeparatorPlace::Back,
        shape: arg_shape,
        ends_with_newline: false,
        preserve_newline: true,
        config: context.config,
    };
    let list_str = write_list(&item_vec, &fmt)?;
    let mut prefix = format!("{}|{}|", mover, list_str);

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
pub fn rewrite_last_closure(
    context: &RewriteContext,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    if let ast::ExprKind::Closure(capture, ref fn_decl, ref body, _) = expr.node {
        let body = match body.node {
            ast::ExprKind::Block(ref block) if is_simple_block(block, context.codemap) => {
                stmt_expr(&block.stmts[0]).unwrap_or(body)
            }
            _ => body,
        };
        let (prefix, extra_offset) =
            rewrite_closure_fn_decl(capture, fn_decl, body, expr.span, context, shape)?;
        // If the closure goes multi line before its body, do not overflow the closure.
        if prefix.contains('\n') {
            return None;
        }
        // If we are inside macro, we do not want to add or remove block from closure body.
        if context.inside_macro {
            return expr.rewrite(context, shape);
        }

        let body_shape = shape.offset_left(extra_offset)?;

        // We force to use block for the body of the closure for certain kinds of expressions.
        if is_block_closure_forced(body) {
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
        let is_multi_lined_cond = rewrite_cond(context, body, body_shape)
            .map(|cond| cond.contains('\n') || cond.len() > body_shape.width)
            .unwrap_or(false);
        if is_multi_lined_cond {
            return rewrite_closure_with_block(body, &prefix, context, body_shape);
        }

        // Seems fine, just format the closure in usual manner.
        return expr.rewrite(context, shape);
    }
    None
}

/// Returns true if the given vector of arguments has more than one `ast::ExprKind::Closure`.
pub fn args_have_many_closure<T>(args: &[&T]) -> bool
where
    T: ToExpr,
{
    args.iter()
        .filter(|arg| {
            arg.to_expr()
                .map(|e| match e.node {
                    ast::ExprKind::Closure(..) => true,
                    _ => false,
                })
                .unwrap_or(false)
        })
        .count() > 1
}

fn is_block_closure_forced(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..)
        | ast::ExprKind::ForLoop(..) => true,
        ast::ExprKind::AddrOf(_, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => is_block_closure_forced(expr),
        _ => false,
    }
}

fn and_one_line(x: Option<String>) -> Option<String> {
    x.and_then(|x| if x.contains('\n') { None } else { Some(x) })
}
