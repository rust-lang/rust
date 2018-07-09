// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Formatting of chained expressions, i.e. expressions which are chained by
//! dots: struct and enum field access, method calls, and try shorthand (?).
//!
//! Instead of walking these subexpressions one-by-one, as is our usual strategy
//! for expression formatting, we collect maximal sequences of these expressions
//! and handle them simultaneously.
//!
//! Whenever possible, the entire chain is put on a single line. If that fails,
//! we put each subexpression on a separate, much like the (default) function
//! argument function argument strategy.
//!
//! Depends on config options: `chain_indent` is the indent to use for
//! blocks in the parent/root/base of the chain (and the rest of the chain's
//! alignment).
//! E.g., `let foo = { aaaa; bbb; ccc }.bar.baz();`, we would layout for the
//! following values of `chain_indent`:
//! Block:
//!
//! ```ignore
//! let foo = {
//!     aaaa;
//!     bbb;
//!     ccc
//! }.bar
//!     .baz();
//! ```
//!
//! Visual:
//!
//! ```ignore
//! let foo = {
//!               aaaa;
//!               bbb;
//!               ccc
//!           }
//!           .bar
//!           .baz();
//! ```
//!
//! If the first item in the chain is a block expression, we align the dots with
//! the braces.
//! Block:
//!
//! ```ignore
//! let a = foo.bar
//!     .baz()
//!     .qux
//! ```
//!
//! Visual:
//!
//! ```ignore
//! let a = foo.bar
//!            .baz()
//!            .qux
//! ```

use config::IndentStyle;
use expr::rewrite_call;
use macros::convert_try_mac;
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use spanned::Spanned;
use utils::{
    first_line_width, last_line_extendable, last_line_width, mk_sp, trimmed_last_line_width,
    wrap_str,
};

use std::borrow::Cow;
use std::cmp::min;

use syntax::codemap::Span;
use syntax::{ast, ptr};

pub fn rewrite_chain(expr: &ast::Expr, context: &RewriteContext, shape: Shape) -> Option<String> {
    debug!("rewrite_chain {:?}", shape);
    let chain = Chain::from_ast(expr, context);
    if chain.children.is_empty() {
        return rewrite_try(&chain.parent.expr, chain.parent.tries, context, shape);
    }
    match context.config.indent_style() {
        IndentStyle::Block => rewrite_chain_block(chain, context, shape),
        IndentStyle::Visual => rewrite_chain_visual(chain, context, shape),
    }
}

// An expression plus trailing `?`s to be formatted together.
struct ChainItem {
    expr: ast::Expr,
    tries: usize,
}

struct Chain {
    parent: ChainItem,
    // TODO do we need to clone the exprs?
    children: Vec<ast::Expr>,
    span: Span,
}

impl Chain {
    fn from_ast(expr: &ast::Expr, context: &RewriteContext) -> Chain {
        let (parent, mut subexpr_list) = make_subexpr_list(expr, context);
        let tries = subexpr_list.iter().rev().take_while(|e| is_try(e)).count();
        let new_len = subexpr_list.len() - tries;
        subexpr_list.truncate(new_len);
        Chain {
            parent: ChainItem {
                expr: parent,
                tries,
            },
            children: subexpr_list,
            span: expr.span,
        }
    }
}

fn rewrite_chain_block(chain: Chain, context: &RewriteContext, shape: Shape) -> Option<String> {
    let (parent, subexpr_list) = (&chain.parent.expr, &chain.children);

    let suffix_try_num = subexpr_list.iter().take_while(|e| is_try(e)).count();

    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let parent_shape = shape;
    let parent_rewrite = parent
        .rewrite(context, parent_shape)
        .map(|parent_rw| parent_rw + &"?".repeat(chain.parent.tries))?;
    let parent_rewrite_contains_newline = parent_rewrite.contains('\n');
    let is_small_parent = shape.offset + parent_rewrite.len() <= context.config.tab_spaces();
    let parent_is_block = is_block_expr(context, &parent, &parent_rewrite);

    // Decide how to layout the rest of the chain. `extend` is true if we can
    // put the first non-parent item on the same line as the parent.
    let (nested_shape, extend) = if !parent_rewrite_contains_newline && is_continuable(&parent) {
        let shape = if parent_is_block {
            shape
        } else {
            chain_indent(context, shape.add_offset(parent_rewrite.len()))
        };
        (shape, is_small_parent)
    } else if parent_is_block {
        (parent_shape, true)
    } else {
        (
            chain_indent(context, shape.add_offset(parent_rewrite.len())),
            false,
        )
    };

    let other_child_shape = nested_shape.with_max_width(context.config);

    let first_child_shape = if extend {
        let offset = trimmed_last_line_width(&parent_rewrite) + chain.parent.tries;
        if parent_is_block {
            parent_shape.offset_left(offset)?
        } else {
            parent_shape
                .block_indent(context.config.tab_spaces())
                .offset_left(offset)?
        }
    } else {
        other_child_shape
    };
    debug!(
        "child_shapes {:?} {:?}",
        first_child_shape, other_child_shape
    );

    let last_subexpr = &subexpr_list[suffix_try_num];
    let subexpr_list = &subexpr_list[suffix_try_num..];

    let mut rewrites: Vec<String> = Vec::with_capacity(subexpr_list.len());
    let mut is_block_like = Vec::with_capacity(subexpr_list.len());
    is_block_like.push(true);
    for (i, expr) in subexpr_list.iter().skip(1).rev().enumerate() {
        let shape = if *is_block_like.last().unwrap() && !(extend && i == 0) {
            first_child_shape
        } else {
            other_child_shape
        };
        let rewrite = rewrite_chain_subexpr(expr, chain.span, context, shape)?;
        is_block_like.push(is_block_expr(context, expr, &rewrite));
        rewrites.push(rewrite);
    }

    // Total of all items excluding the last.
    let extend_last_subexpr = if is_small_parent {
        rewrites.len() == 1 && last_line_extendable(&rewrites[0])
    } else {
        rewrites.is_empty() && last_line_extendable(&parent_rewrite)
    };
    let almost_total = if extend_last_subexpr {
        last_line_width(&parent_rewrite)
    } else {
        rewrites.iter().fold(0, |a, b| a + b.len()) + parent_rewrite.len()
    } + suffix_try_num;
    let one_line_budget = if rewrites.is_empty() {
        shape.width
    } else {
        min(shape.width, context.config.width_heuristics().chain_width)
    };
    let all_in_one_line = !parent_rewrite_contains_newline
        && rewrites.iter().all(|s| !s.contains('\n'))
        && almost_total < one_line_budget;
    let last_shape = if is_block_like[rewrites.len()] {
        first_child_shape
    } else {
        other_child_shape
    }.sub_width(shape.rhs_overhead(context.config) + suffix_try_num)?;

    // Rewrite the last child. The last child of a chain requires special treatment. We need to
    // know whether 'overflowing' the last child make a better formatting:
    //
    // A chain with overflowing the last child:
    // ```
    // parent.child1.child2.last_child(
    //     a,
    //     b,
    //     c,
    // )
    // ```
    //
    // A chain without overflowing the last child (in vertical layout):
    // ```
    // parent
    //     .child1
    //     .child2
    //     .last_child(a, b, c)
    // ```
    //
    // In particular, overflowing is effective when the last child is a method with a multi-lined
    // block-like argument (e.g. closure):
    // ```
    // parent.child1.child2.last_child(|a, b, c| {
    //     let x = foo(a, b, c);
    //     let y = bar(a, b, c);
    //
    //     // ...
    //
    //     result
    // })
    // ```

    // `rewrite_last` rewrites the last child on its own line. We use a closure here instead of
    // directly calling `rewrite_chain_subexpr()` to avoid exponential blowup.
    let rewrite_last = || rewrite_chain_subexpr(last_subexpr, chain.span, context, last_shape);
    let (last_subexpr_str, fits_single_line) = if all_in_one_line || extend_last_subexpr {
        // First we try to 'overflow' the last child and see if it looks better than using
        // vertical layout.
        parent_shape.offset_left(almost_total).map(|shape| {
            if let Some(rw) = rewrite_chain_subexpr(last_subexpr, chain.span, context, shape) {
                // We allow overflowing here only if both of the following conditions match:
                // 1. The entire chain fits in a single line except the last child.
                // 2. `last_child_str.lines().count() >= 5`.
                let line_count = rw.lines().count();
                let fits_single_line = almost_total + first_line_width(&rw) <= one_line_budget;
                if fits_single_line && line_count >= 5 {
                    (Some(rw), true)
                } else {
                    // We could not know whether overflowing is better than using vertical layout,
                    // just by looking at the overflowed rewrite. Now we rewrite the last child
                    // on its own line, and compare two rewrites to choose which is better.
                    match rewrite_last() {
                        Some(ref new_rw) if !fits_single_line => (Some(new_rw.clone()), false),
                        Some(ref new_rw) if new_rw.lines().count() >= line_count => {
                            (Some(rw), fits_single_line)
                        }
                        new_rw @ Some(..) => (new_rw, false),
                        _ => (Some(rw), fits_single_line),
                    }
                }
            } else {
                (rewrite_last(), false)
            }
        })?
    } else {
        (rewrite_last(), false)
    };
    rewrites.push(last_subexpr_str?);
    // We should never look at this, since we only look at the block-ness of the
    // previous item in the chain.
    is_block_like.push(false);

    let connector = if fits_single_line && !parent_rewrite_contains_newline {
        // Yay, we can put everything on one line.
        Cow::from("")
    } else {
        // Use new lines.
        if *context.force_one_line_chain.borrow() {
            return None;
        }
        nested_shape.indent.to_string_with_newline(context.config)
    };

    let first_connector = if is_small_parent
        || fits_single_line
        || last_line_extendable(&parent_rewrite)
    {
        ""
    } else {
        &connector
    };

    let result = if is_small_parent && rewrites.len() > 1 {
        let second_connector = if fits_single_line
            || rewrites[1] == "?"
            || last_line_extendable(&rewrites[0])
        {
            ""
        } else {
            &connector
        };
        format!(
            "{}{}{}{}{}",
            parent_rewrite,
            first_connector,
            rewrites[0],
            second_connector,
            join_rewrites(&rewrites[1..], &is_block_like[2..], &connector),
        )
    } else {
        format!(
            "{}{}{}",
            parent_rewrite,
            first_connector,
            join_rewrites(&rewrites, &is_block_like[1..], &connector),
        )
    };
    let result = format!("{}{}", result, "?".repeat(suffix_try_num));
    Some(result)
}

fn rewrite_chain_visual(chain: Chain, context: &RewriteContext, shape: Shape) -> Option<String> {
    let parent = &chain.parent.expr;

    let suffix_try_num = chain.children.iter().take_while(|e| is_try(e)).count();

    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let parent_shape = if is_block_expr(context, &parent, "\n") {
        shape.visual_indent(0)
    } else {
        shape
    };
    let parent_rewrite = parent
        .rewrite(context, parent_shape)
        .map(|parent_rw| parent_rw + &"?".repeat(chain.parent.tries))?;
    let parent_rewrite_contains_newline = parent_rewrite.contains('\n');
    let is_small_parent = shape.offset + parent_rewrite.len() <= context.config.tab_spaces();

    let nested_shape = shape.visual_indent(0);
    let other_child_shape = nested_shape.with_max_width(context.config);

    // Decide how to layout the rest of the chain. `extend` is true if we can
    // put the first non-parent item on the same line as the parent.
    let extend = !parent_rewrite_contains_newline && is_continuable(&parent);

    let first_child_shape = if extend {
        let overhead = last_line_width(&parent_rewrite);
        parent_shape.offset_left(overhead)?
    } else {
        other_child_shape
    };
    debug!(
        "child_shapes {:?} {:?}",
        first_child_shape, other_child_shape
    );

    let last_subexpr = &chain.children[suffix_try_num];
    let subexpr_list = &chain.children[suffix_try_num..];

    let mut rewrites: Vec<String> = Vec::with_capacity(subexpr_list.len());
    for (i, expr) in subexpr_list.iter().skip(1).rev().enumerate() {
        let shape = if i == 0 {
            first_child_shape
        } else {
            other_child_shape
        };
        let rewrite = rewrite_chain_subexpr(expr, chain.span, context, shape)?;
        rewrites.push(rewrite);
    }

    // Total of all items excluding the last.
    let extend_last_subexpr = if is_small_parent {
        rewrites.len() == 1 && last_line_extendable(&rewrites[0])
    } else {
        rewrites.is_empty() && last_line_extendable(&parent_rewrite)
    };
    let almost_total = if extend_last_subexpr {
        last_line_width(&parent_rewrite)
    } else {
        rewrites.iter().fold(0, |a, b| a + b.len()) + parent_rewrite.len()
    } + suffix_try_num;
    let one_line_budget = if rewrites.is_empty() {
        shape.width
    } else {
        min(shape.width, context.config.width_heuristics().chain_width)
    };
    let all_in_one_line = !parent_rewrite_contains_newline
        && rewrites.iter().all(|s| !s.contains('\n'))
        && almost_total < one_line_budget;
    let last_shape =
        other_child_shape.sub_width(shape.rhs_overhead(context.config) + suffix_try_num)?;

    // Rewrite the last child. The last child of a chain requires special treatment. We need to
    // know whether 'overflowing' the last child make a better formatting:
    //
    // A chain with overflowing the last child:
    // ```
    // parent.child1.child2.last_child(
    //     a,
    //     b,
    //     c,
    // )
    // ```
    //
    // A chain without overflowing the last child (in vertical layout):
    // ```
    // parent
    //     .child1
    //     .child2
    //     .last_child(a, b, c)
    // ```
    //
    // In particular, overflowing is effective when the last child is a method with a multi-lined
    // block-like argument (e.g. closure):
    // ```
    // parent.child1.child2.last_child(|a, b, c| {
    //     let x = foo(a, b, c);
    //     let y = bar(a, b, c);
    //
    //     // ...
    //
    //     result
    // })
    // ```

    // `rewrite_last` rewrites the last child on its own line. We use a closure here instead of
    // directly calling `rewrite_chain_subexpr()` to avoid exponential blowup.
    let rewrite_last = || rewrite_chain_subexpr(last_subexpr, chain.span, context, last_shape);
    let (last_subexpr_str, fits_single_line) = if all_in_one_line || extend_last_subexpr {
        // First we try to 'overflow' the last child and see if it looks better than using
        // vertical layout.
        parent_shape.offset_left(almost_total).map(|shape| {
            if let Some(rw) = rewrite_chain_subexpr(last_subexpr, chain.span, context, shape) {
                // We allow overflowing here only if both of the following conditions match:
                // 1. The entire chain fits in a single line except the last child.
                // 2. `last_child_str.lines().count() >= 5`.
                let line_count = rw.lines().count();
                let fits_single_line = almost_total + first_line_width(&rw) <= one_line_budget;
                if fits_single_line && line_count >= 5 {
                    (Some(rw), true)
                } else {
                    // We could not know whether overflowing is better than using vertical layout,
                    // just by looking at the overflowed rewrite. Now we rewrite the last child
                    // on its own line, and compare two rewrites to choose which is better.
                    match rewrite_last() {
                        Some(ref new_rw) if !fits_single_line => (Some(new_rw.clone()), false),
                        Some(ref new_rw) if new_rw.lines().count() >= line_count => {
                            (Some(rw), fits_single_line)
                        }
                        new_rw @ Some(..) => (new_rw, false),
                        _ => (Some(rw), fits_single_line),
                    }
                }
            } else {
                (rewrite_last(), false)
            }
        })?
    } else {
        (rewrite_last(), false)
    };
    rewrites.push(last_subexpr_str?);

    let connector = if fits_single_line && !parent_rewrite_contains_newline {
        // Yay, we can put everything on one line.
        Cow::from("")
    } else {
        // Use new lines.
        if *context.force_one_line_chain.borrow() {
            return None;
        }
        nested_shape.indent.to_string_with_newline(context.config)
    };

    let result = if is_small_parent && rewrites.len() > 1 {
        format!(
            "{}{}{}",
            parent_rewrite,
            rewrites[0],
            join_rewrites_vis(&rewrites[1..], &connector),
        )
    } else {
        format!(
            "{}{}",
            parent_rewrite,
            join_rewrites_vis(&rewrites, &connector),
        )
    };
    let result = format!("{}{}", result, "?".repeat(suffix_try_num));
    wrap_str(result, context.config.max_width(), shape)
}

fn rewrite_try(
    expr: &ast::Expr,
    try_count: usize,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let sub_expr = expr.rewrite(context, shape.sub_width(try_count)?)?;
    Some(format!("{}{}", sub_expr, "?".repeat(try_count)))
}

fn join_rewrites(rewrites: &[String], is_block_like: &[bool], connector: &str) -> String {
    let mut rewrite_iter = rewrites.iter();
    let mut result = rewrite_iter.next().unwrap().clone();

    for (rewrite, prev_is_block_like) in rewrite_iter.zip(is_block_like.iter()) {
        if rewrite != "?" && !prev_is_block_like {
            result.push_str(connector);
        }
        result.push_str(&rewrite);
    }

    result
}

fn join_rewrites_vis(rewrites: &[String], connector: &str) -> String {
    let mut rewrite_iter = rewrites.iter();
    let mut result = rewrite_iter.next().unwrap().clone();

    for rewrite in rewrite_iter {
        if rewrite != "?" {
            result.push_str(connector);
        }
        result.push_str(&rewrite);
    }

    result
}

// States whether an expression's last line exclusively consists of closing
// parens, braces, and brackets in its idiomatic formatting.
fn is_block_expr(context: &RewriteContext, expr: &ast::Expr, repr: &str) -> bool {
    match expr.node {
        ast::ExprKind::Mac(..) | ast::ExprKind::Call(..) => {
            context.use_block_indent() && repr.contains('\n')
        }
        ast::ExprKind::Struct(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..)
        | ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Match(..) => repr.contains('\n'),
        ast::ExprKind::Paren(ref expr)
        | ast::ExprKind::Binary(_, _, ref expr)
        | ast::ExprKind::Index(_, ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Closure(_, _, _, _, ref expr, _) => is_block_expr(context, expr, repr),
        ast::ExprKind::MethodCall(_, ref exprs) => {
            // TODO maybe should be like Call
            is_block_expr(context, exprs.last().unwrap(), repr)
        }
        _ => false,
    }
}

fn chain_indent(context: &RewriteContext, shape: Shape) -> Shape {
    shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config)
}

// Returns the root of the chain and a Vec of the prefixes of the rest of the chain.
// E.g., for input `a.b.c` we return (`a`, [`a.b.c`, `a.b`])
fn make_subexpr_list(expr: &ast::Expr, context: &RewriteContext) -> (ast::Expr, Vec<ast::Expr>) {
    let mut subexpr_list = vec![expr.clone()];

    while let Some(subexpr) = pop_expr_chain(subexpr_list.last().unwrap(), context) {
        subexpr_list.push(subexpr.clone());
    }

    let parent = subexpr_list.pop().unwrap();
    (parent, subexpr_list)
}

// Returns the expression's subexpression, if it exists. When the subexpr
// is a try! macro, we'll convert it to shorthand when the option is set.
fn pop_expr_chain(expr: &ast::Expr, context: &RewriteContext) -> Option<ast::Expr> {
    match expr.node {
        ast::ExprKind::MethodCall(_, ref expressions) => {
            Some(convert_try(&expressions[0], context))
        }
        ast::ExprKind::Field(ref subexpr, _) | ast::ExprKind::Try(ref subexpr) => {
            Some(convert_try(subexpr, context))
        }
        _ => None,
    }
}

fn convert_try(expr: &ast::Expr, context: &RewriteContext) -> ast::Expr {
    match expr.node {
        ast::ExprKind::Mac(ref mac) if context.config.use_try_shorthand() => {
            if let Some(subexpr) = convert_try_mac(mac, context) {
                subexpr
            } else {
                expr.clone()
            }
        }
        _ => expr.clone(),
    }
}

// Rewrite the last element in the chain `expr`. E.g., given `a.b.c` we rewrite
// `.c`.
fn rewrite_chain_subexpr(
    expr: &ast::Expr,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let rewrite_element = |expr_str: String| {
        if expr_str.len() <= shape.width {
            Some(expr_str)
        } else {
            None
        }
    };

    match expr.node {
        ast::ExprKind::MethodCall(ref segment, ref expressions) => {
            let types = match segment.args {
                Some(ref params) => match **params {
                    ast::GenericArgs::AngleBracketed(ref data) => &data.args[..],
                    _ => &[],
                },
                _ => &[],
            };
            rewrite_method_call(segment.ident, types, expressions, span, context, shape)
        }
        ast::ExprKind::Field(ref nested, ref field) => {
            let space = if is_tup_field_access(expr) && is_tup_field_access(nested) {
                " "
            } else {
                ""
            };
            rewrite_element(format!("{}.{}", space, field.name))
        }
        ast::ExprKind::Try(_) => rewrite_element(String::from("?")),
        _ => unreachable!(),
    }
}

fn is_tup_field_access(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Field(_, ref field) => {
            field.name.to_string().chars().all(|c| c.is_digit(10))
        }
        _ => false,
    }
}

// Determines if we can continue formatting a given expression on the same line.
fn is_continuable(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Path(..) => true,
        _ => false,
    }
}

fn is_try(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Try(..) => true,
        _ => false,
    }
}

fn rewrite_method_call(
    method_name: ast::Ident,
    types: &[ast::GenericArg],
    args: &[ptr::P<ast::Expr>],
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let (lo, type_str) = if types.is_empty() {
        (args[0].span.hi(), String::new())
    } else {
        let type_list = types
            .iter()
            .map(|ty| ty.rewrite(context, shape))
            .collect::<Option<Vec<_>>>()?;

        let type_str = format!("::<{}>", type_list.join(", "));

        (types.last().unwrap().span().hi(), type_str)
    };

    let callee_str = format!(".{}{}", method_name, type_str);
    let span = mk_sp(lo, span.hi());

    rewrite_call(context, &callee_str, &args[1..], span, shape)
}
