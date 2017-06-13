// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Formatting of chained expressions, i.e. expressions which are chained by
/// dots: struct and enum field access, method calls, and try shorthand (?).
///
/// Instead of walking these subexpressions one-by-one, as is our usual strategy
/// for expression formatting, we collect maximal sequences of these expressions
/// and handle them simultaneously.
///
/// Whenever possible, the entire chain is put on a single line. If that fails,
/// we put each subexpression on a separate, much like the (default) function
/// argument function argument strategy.
///
/// Depends on config options: `chain_indent` is the indent to use for
/// blocks in the parent/root/base of the chain (and the rest of the chain's
/// alignment).
/// E.g., `let foo = { aaaa; bbb; ccc }.bar.baz();`, we would layout for the
/// following values of `chain_indent`:
/// Visual:
/// ```
/// let foo = {
///               aaaa;
///               bbb;
///               ccc
///           }
///           .bar
///           .baz();
/// ```
/// Inherit:
/// ```
/// let foo = {
///     aaaa;
///     bbb;
///     ccc
/// }
/// .bar
/// .baz();
/// ```
/// Tabbed:
/// ```
/// let foo = {
///         aaaa;
///         bbb;
///         ccc
///     }
///     .bar
///     .baz();
/// ```
///
/// If the first item in the chain is a block expression, we align the dots with
/// the braces.
/// Visual:
/// ```
/// let a = foo.bar
///            .baz()
///            .qux
/// ```
/// Inherit:
/// ```
/// let a = foo.bar
/// .baz()
/// .qux
/// ```
/// Tabbed:
/// ```
/// let a = foo.bar
///     .baz()
///     .qux
/// ```

use Shape;
use rewrite::{Rewrite, RewriteContext};
use utils::{wrap_str, first_line_width, last_line_width, mk_sp};
use expr::rewrite_call;
use config::IndentStyle;
use macros::convert_try_mac;

use std::cmp::min;
use std::iter;
use syntax::{ast, ptr};
use syntax::codemap::Span;

pub fn rewrite_chain(expr: &ast::Expr, context: &RewriteContext, shape: Shape) -> Option<String> {
    debug!("rewrite_chain {:?}", shape);
    let total_span = expr.span;
    let (parent, subexpr_list) = make_subexpr_list(expr, context);

    // Bail out if the chain is just try sugar, i.e., an expression followed by
    // any number of `?`s.
    if chain_only_try(&subexpr_list) {
        return rewrite_try(&parent, subexpr_list.len(), context, shape);
    }
    let trailing_try_num = subexpr_list
        .iter()
        .take_while(|e| match e.node {
            ast::ExprKind::Try(..) => true,
            _ => false,
        })
        .count();

    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let parent_shape = if is_block_expr(context, &parent, "\n") {
        match context.config.chain_indent() {
            IndentStyle::Visual => shape.visual_indent(0),
            IndentStyle::Block => shape.block(),
        }
    } else {
        shape
    };
    let parent_rewrite = try_opt!(parent.rewrite(context, parent_shape));
    let parent_rewrite_contains_newline = parent_rewrite.contains('\n');
    let is_small_parent = parent_rewrite.len() <= context.config.tab_spaces();

    // Decide how to layout the rest of the chain. `extend` is true if we can
    // put the first non-parent item on the same line as the parent.
    let first_subexpr_is_try = subexpr_list.last().map_or(false, is_try);
    let (nested_shape, extend) = if !parent_rewrite_contains_newline && is_continuable(&parent) {
        let nested_shape = if first_subexpr_is_try {
            parent_shape.block_indent(context.config.tab_spaces())
        } else {
            chain_indent(context, shape.add_offset(parent_rewrite.len()))
        };
        (
            nested_shape,
            context.config.chain_indent() == IndentStyle::Visual || is_small_parent,
        )
    } else if is_block_expr(context, &parent, &parent_rewrite) {
        match context.config.chain_indent() {
            // Try to put the first child on the same line with parent's last line
            IndentStyle::Block => (parent_shape.block_indent(context.config.tab_spaces()), true),
            // The parent is a block, so align the rest of the chain with the closing
            // brace.
            IndentStyle::Visual => (parent_shape, false),
        }
    } else if parent_rewrite_contains_newline {
        (chain_indent(context, parent_shape), false)
    } else {
        (shape.block_indent(context.config.tab_spaces()), false)
    };

    let other_child_shape = nested_shape.with_max_width(context.config);

    let first_child_shape = if extend {
        let overhead = last_line_width(&parent_rewrite);
        let offset = parent_rewrite.lines().rev().next().unwrap().trim().len();
        match context.config.chain_indent() {
            IndentStyle::Visual => try_opt!(parent_shape.offset_left(overhead)),
            IndentStyle::Block => try_opt!(parent_shape.block().offset_left(offset)),
        }
    } else {
        other_child_shape
    };
    debug!(
        "child_shapes {:?} {:?}",
        first_child_shape,
        other_child_shape
    );

    let child_shape_iter = Some(first_child_shape).into_iter().chain(
        ::std::iter::repeat(
            other_child_shape,
        ).take(
            subexpr_list.len() - 1,
        ),
    );
    let iter = subexpr_list.iter().rev().zip(child_shape_iter);
    let mut rewrites = try_opt!(
        iter.map(|(e, shape)| {
            rewrite_chain_subexpr(e, total_span, context, shape)
        }).collect::<Option<Vec<_>>>()
    );

    // Total of all items excluding the last.
    let last_non_try_index = rewrites.len() - (1 + trailing_try_num);
    let almost_total = rewrites[..last_non_try_index].iter().fold(0, |a, b| {
        a + first_line_width(b)
    }) + parent_rewrite.len();
    let one_line_len = rewrites.iter().fold(0, |a, r| a + first_line_width(r)) +
        parent_rewrite.len();

    let one_line_budget = min(shape.width, context.config.chain_one_line_max());
    let veto_single_line = if one_line_len > one_line_budget {
        if rewrites.len() > 1 {
            true
        } else if rewrites.len() == 1 {
            context.config.chain_split_single_child() || one_line_len > shape.width
        } else {
            false
        }
    } else if context.config.take_source_hints() && subexpr_list.len() > 1 {
        // Look at the source code. Unless all chain elements start on the same
        // line, we won't consider putting them on a single line either.
        let last_span = context.snippet(mk_sp(subexpr_list[1].span.hi, total_span.hi));
        let first_span = context.snippet(subexpr_list[1].span);
        let last_iter = last_span.chars().take_while(|c| c.is_whitespace());

        first_span.chars().chain(last_iter).any(|c| c == '\n')
    } else {
        false
    };

    let mut fits_single_line = !veto_single_line && almost_total <= shape.width;
    if fits_single_line {
        let len = rewrites.len();
        let (init, last) = rewrites.split_at_mut(len - (1 + trailing_try_num));
        fits_single_line = init.iter().all(|s| !s.contains('\n'));

        if fits_single_line {
            fits_single_line = match expr.node {
                ref e @ ast::ExprKind::MethodCall(..) => {
                    if rewrite_method_call_with_overflow(
                        e,
                        &mut last[0],
                        almost_total,
                        total_span,
                        context,
                        shape,
                    )
                    {
                        // If the first line of the last method does not fit into a single line
                        // after the others, allow new lines.
                        almost_total + first_line_width(&last[0]) < context.config.max_width()
                    } else {
                        false
                    }
                }
                _ => !last[0].contains('\n'),
            }
        }
    }

    // Try overflowing the last element if we are using block indent.
    if !fits_single_line && context.use_block_indent() {
        let (init, last) = rewrites.split_at_mut(last_non_try_index);
        let almost_single_line = init.iter().all(|s| !s.contains('\n'));
        if almost_single_line {
            let overflow_shape = Shape {
                width: one_line_budget,
                ..parent_shape
            };
            fits_single_line = rewrite_last_child_with_overflow(
                context,
                &subexpr_list[trailing_try_num],
                overflow_shape,
                total_span,
                almost_total,
                one_line_budget,
                &mut last[0],
            );
        }
    }

    let connector = if fits_single_line && !parent_rewrite_contains_newline {
        // Yay, we can put everything on one line.
        String::new()
    } else {
        // Use new lines.
        if context.force_one_line_chain {
            return None;
        }
        format!("\n{}", nested_shape.indent.to_string(context.config))
    };

    let first_connector = choose_first_connector(
        context,
        &parent_rewrite,
        &rewrites[0],
        &connector,
        &subexpr_list,
        extend,
    );

    if is_small_parent && rewrites.len() > 1 {
        let second_connector = choose_first_connector(
            context,
            &rewrites[0],
            &rewrites[1],
            &connector,
            &subexpr_list[0..subexpr_list.len() - 1],
            false,
        );
        wrap_str(
            format!(
                "{}{}{}{}{}",
                parent_rewrite,
                first_connector,
                rewrites[0],
                second_connector,
                join_rewrites(
                    &rewrites[1..],
                    &subexpr_list[0..subexpr_list.len() - 1],
                    &connector,
                )
            ),
            context.config.max_width(),
            shape,
        )
    } else {
        wrap_str(
            format!(
                "{}{}{}",
                parent_rewrite,
                first_connector,
                join_rewrites(&rewrites, &subexpr_list, &connector)
            ),
            context.config.max_width(),
            shape,
        )
    }
}

fn is_extendable_parent(context: &RewriteContext, parent_str: &str) -> bool {
    context.config.chain_indent() == IndentStyle::Block &&
        parent_str.lines().last().map_or(false, |s| {
            s.trim().chars().all(|c| {
                c == ')' || c == ']' || c == '}' || c == '?'
            })
        })
}

// True if the chain is only `?`s.
fn chain_only_try(exprs: &[ast::Expr]) -> bool {
    exprs.iter().all(|e| if let ast::ExprKind::Try(_) = e.node {
        true
    } else {
        false
    })
}

// Try to rewrite and replace the last non-try child. Return `true` if
// replacing succeeds.
fn rewrite_last_child_with_overflow(
    context: &RewriteContext,
    expr: &ast::Expr,
    shape: Shape,
    span: Span,
    almost_total: usize,
    one_line_budget: usize,
    last_child: &mut String,
) -> bool {
    if let Some(shape) = shape.shrink_left(almost_total) {
        if let Some(ref mut rw) = rewrite_chain_subexpr(expr, span, context, shape) {
            if almost_total + first_line_width(rw) <= one_line_budget {
                ::std::mem::swap(last_child, rw);
                return true;
            }
        }
    }
    false
}

pub fn rewrite_try(
    expr: &ast::Expr,
    try_count: usize,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let sub_expr = try_opt!(expr.rewrite(context, try_opt!(shape.sub_width(try_count))));
    Some(format!(
        "{}{}",
        sub_expr,
        iter::repeat("?").take(try_count).collect::<String>()
    ))
}

fn join_rewrites(rewrites: &[String], subexps: &[ast::Expr], connector: &str) -> String {
    let mut rewrite_iter = rewrites.iter();
    let mut result = rewrite_iter.next().unwrap().clone();
    let mut subexpr_iter = subexps.iter().rev();
    subexpr_iter.next();

    for (rewrite, expr) in rewrite_iter.zip(subexpr_iter) {
        match expr.node {
            ast::ExprKind::Try(_) => (),
            _ => result.push_str(connector),
        };
        result.push_str(&rewrite[..]);
    }

    result
}

// States whether an expression's last line exclusively consists of closing
// parens, braces, and brackets in its idiomatic formatting.
fn is_block_expr(context: &RewriteContext, expr: &ast::Expr, repr: &str) -> bool {
    match expr.node {
        ast::ExprKind::Mac(..) |
        ast::ExprKind::Call(..) => context.use_block_indent() && repr.contains('\n'),
        ast::ExprKind::Struct(..) |
        ast::ExprKind::While(..) |
        ast::ExprKind::WhileLet(..) |
        ast::ExprKind::If(..) |
        ast::ExprKind::IfLet(..) |
        ast::ExprKind::Block(..) |
        ast::ExprKind::Loop(..) |
        ast::ExprKind::ForLoop(..) |
        ast::ExprKind::Match(..) => repr.contains('\n'),
        ast::ExprKind::Paren(ref expr) |
        ast::ExprKind::Binary(_, _, ref expr) |
        ast::ExprKind::Index(_, ref expr) |
        ast::ExprKind::Unary(_, ref expr) => is_block_expr(context, expr, repr),
        _ => false,
    }
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

fn chain_indent(context: &RewriteContext, shape: Shape) -> Shape {
    match context.config.chain_indent() {
        IndentStyle::Visual => shape.visual_indent(0),
        IndentStyle::Block => shape.block_indent(context.config.tab_spaces()),
    }
}

fn rewrite_method_call_with_overflow(
    expr_kind: &ast::ExprKind,
    last: &mut String,
    almost_total: usize,
    total_span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> bool {
    if let &ast::ExprKind::MethodCall(ref method_name, ref types, ref expressions) = expr_kind {
        let shape = match shape.shrink_left(almost_total) {
            Some(b) => b,
            None => return false,
        };
        let mut last_rewrite = rewrite_method_call(
            method_name.node,
            types,
            expressions,
            total_span,
            context,
            shape,
        );

        if let Some(ref mut s) = last_rewrite {
            ::std::mem::swap(s, last);
            true
        } else {
            false
        }
    } else {
        unreachable!();
    }
}

// Returns the expression's subexpression, if it exists. When the subexpr
// is a try! macro, we'll convert it to shorthand when the option is set.
fn pop_expr_chain(expr: &ast::Expr, context: &RewriteContext) -> Option<ast::Expr> {
    match expr.node {
        ast::ExprKind::MethodCall(_, _, ref expressions) => {
            Some(convert_try(&expressions[0], context))
        }
        ast::ExprKind::TupField(ref subexpr, _) |
        ast::ExprKind::Field(ref subexpr, _) |
        ast::ExprKind::Try(ref subexpr) => Some(convert_try(subexpr, context)),
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
    let rewrite_element = |expr_str: String| if expr_str.len() <= shape.width {
        Some(expr_str)
    } else {
        None
    };

    match expr.node {
        ast::ExprKind::MethodCall(ref method_name, ref types, ref expressions) => {
            rewrite_method_call(method_name.node, types, expressions, span, context, shape)
        }
        ast::ExprKind::Field(_, ref field) => rewrite_element(format!(".{}", field.node)),
        ast::ExprKind::TupField(ref expr, ref field) => {
            let space = match expr.node {
                ast::ExprKind::TupField(..) => " ",
                _ => "",
            };
            rewrite_element(format!("{}.{}", space, field.node))
        }
        ast::ExprKind::Try(_) => rewrite_element(String::from("?")),
        _ => unreachable!(),
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

fn choose_first_connector<'a>(
    context: &RewriteContext,
    parent_str: &str,
    first_child_str: &str,
    connector: &'a str,
    subexpr_list: &[ast::Expr],
    extend: bool,
) -> &'a str {
    if subexpr_list.is_empty() {
        ""
    } else if extend || subexpr_list.last().map_or(false, is_try) ||
               is_extendable_parent(context, parent_str)
    {
        // 1 = ";", being conservative here.
        if last_line_width(parent_str) + first_line_width(first_child_str) + 1 <=
            context.config.max_width()
        {
            ""
        } else {
            connector
        }
    } else {
        connector
    }
}

fn rewrite_method_call(
    method_name: ast::Ident,
    types: &[ptr::P<ast::Ty>],
    args: &[ptr::P<ast::Expr>],
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let (lo, type_str) = if types.is_empty() {
        (args[0].span.hi, String::new())
    } else {
        let type_list: Vec<_> =
            try_opt!(types.iter().map(|ty| ty.rewrite(context, shape)).collect());

        let type_str = if context.config.spaces_within_angle_brackets() && type_list.len() > 0 {
            format!("::< {} >", type_list.join(", "))
        } else {
            format!("::<{}>", type_list.join(", "))
        };

        (types.last().unwrap().span.hi, type_str)
    };

    let callee_str = format!(".{}{}", method_name, type_str);
    let span = mk_sp(lo, span.hi);

    rewrite_call(context, &callee_str, &args[1..], span, shape)
}
