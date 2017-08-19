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
use config::IndentStyle;
use expr::rewrite_call;
use macros::convert_try_mac;
use rewrite::{Rewrite, RewriteContext};
use utils::{first_line_width, last_line_extendable, last_line_width, mk_sp, wrap_str};

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
    let suffix_try_num = subexpr_list.iter().take_while(|e| is_try(e)).count();
    let prefix_try_num = subexpr_list.iter().rev().take_while(|e| is_try(e)).count();

    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let parent_shape = if is_block_expr(context, &parent, "\n") {
        match context.config.chain_indent() {
            IndentStyle::Visual => shape.visual_indent(0),
            IndentStyle::Block => shape,
        }
    } else {
        shape
    };
    let parent_rewrite = try_opt!(
        parent
            .rewrite(context, parent_shape)
            .map(|parent_rw| parent_rw + &repeat_try(prefix_try_num))
    );
    let parent_rewrite_contains_newline = parent_rewrite.contains('\n');
    let is_small_parent = parent_rewrite.len() <= context.config.tab_spaces();

    // Decide how to layout the rest of the chain. `extend` is true if we can
    // put the first non-parent item on the same line as the parent.
    let (nested_shape, extend) = if !parent_rewrite_contains_newline && is_continuable(&parent) {
        (
            chain_indent(context, shape.add_offset(parent_rewrite.len())),
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
    } else {
        (
            chain_indent(context, shape.add_offset(parent_rewrite.len())),
            false,
        )
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

    let child_shape_iter = Some(first_child_shape)
        .into_iter()
        .chain(iter::repeat(other_child_shape));
    let subexpr_num = subexpr_list.len();
    let last_subexpr = &subexpr_list[suffix_try_num];
    let subexpr_list = &subexpr_list[suffix_try_num..subexpr_num - prefix_try_num];
    let iter = subexpr_list.iter().skip(1).rev().zip(child_shape_iter);
    let mut rewrites = try_opt!(
        iter.map(|(e, shape)| {
            rewrite_chain_subexpr(e, total_span, context, shape)
        }).collect::<Option<Vec<_>>>()
    );

    // Total of all items excluding the last.
    let extend_last_subexr = last_line_extendable(&parent_rewrite) && rewrites.is_empty();
    let almost_total = if extend_last_subexr {
        last_line_width(&parent_rewrite)
    } else {
        rewrites.iter().fold(0, |a, b| a + b.len()) + parent_rewrite.len()
    };
    let one_line_budget = if rewrites.is_empty() && !context.config.chain_split_single_child() {
        shape.width
    } else {
        min(shape.width, context.config.chain_one_line_max())
    };
    let all_in_one_line = !parent_rewrite_contains_newline &&
        rewrites.iter().all(|s| !s.contains('\n')) &&
        almost_total < one_line_budget;
    let rewrite_last = || rewrite_chain_subexpr(last_subexpr, total_span, context, nested_shape);
    let (last_subexpr_str, fits_single_line) = try_opt!(if all_in_one_line || extend_last_subexr {
        parent_shape.offset_left(almost_total).map(|shape| {
            if let Some(rw) = rewrite_chain_subexpr(last_subexpr, total_span, context, shape) {
                let line_count = rw.lines().count();
                let fits_single_line = almost_total + first_line_width(&rw) <= one_line_budget;
                if (line_count >= 5 && fits_single_line) || extend_last_subexr {
                    (Some(rw), true)
                } else {
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
        })
    } else {
        Some((rewrite_last(), false))
    });
    rewrites.push(try_opt!(last_subexpr_str));

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

    let first_connector = if is_small_parent || fits_single_line ||
        last_line_extendable(&parent_rewrite) ||
        context.config.chain_indent() == IndentStyle::Visual
    {
        ""
    } else {
        connector.as_str()
    };

    let subexpr_num = subexpr_list.len();
    let result = if is_small_parent && rewrites.len() > 1 {
        let second_connector = choose_first_connector(
            context,
            &rewrites[0],
            &rewrites[1],
            &connector,
            &subexpr_list[..subexpr_num - 1],
            false,
        );
        format!(
            "{}{}{}{}{}",
            parent_rewrite,
            first_connector,
            rewrites[0],
            second_connector,
            join_rewrites(&rewrites[1..], &subexpr_list[..subexpr_num - 1], &connector)
        )
    } else {
        format!(
            "{}{}{}",
            parent_rewrite,
            first_connector,
            join_rewrites(&rewrites, &subexpr_list, &connector)
        )
    };
    let result = format!("{}{}", result, repeat_try(suffix_try_num));
    wrap_str(result, context.config.max_width(), shape)
}

fn is_extendable_parent(context: &RewriteContext, parent_str: &str) -> bool {
    context.config.chain_indent() == IndentStyle::Block && last_line_extendable(parent_str)
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
fn repeat_try(try_count: usize) -> String {
    iter::repeat("?").take(try_count).collect::<String>()
}

fn rewrite_try(
    expr: &ast::Expr,
    try_count: usize,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let sub_expr = try_opt!(expr.rewrite(context, try_opt!(shape.sub_width(try_count))));
    Some(format!("{}{}", sub_expr, repeat_try(try_count)))
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
        ast::ExprKind::Mac(..) | ast::ExprKind::Call(..) => {
            context.use_block_indent() && repr.contains('\n')
        }
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
        IndentStyle::Block => shape
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config),
    }
}

// Returns the expression's subexpression, if it exists. When the subexpr
// is a try! macro, we'll convert it to shorthand when the option is set.
fn pop_expr_chain(expr: &ast::Expr, context: &RewriteContext) -> Option<ast::Expr> {
    match expr.node {
        ast::ExprKind::MethodCall(_, ref expressions) => {
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
        ast::ExprKind::MethodCall(ref segment, ref expressions) => {
            let types = match segment.parameters {
                Some(ref params) => match **params {
                    ast::PathParameters::AngleBracketed(ref data) => &data.types[..],
                    _ => &[],
                },
                _ => &[],
            };
            rewrite_method_call(segment.identifier, types, expressions, span, context, shape)
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
        (args[0].span.hi(), String::new())
    } else {
        let type_list: Vec<_> =
            try_opt!(types.iter().map(|ty| ty.rewrite(context, shape)).collect());

        let type_str = if context.config.spaces_within_angle_brackets() && type_list.len() > 0 {
            format!("::< {} >", type_list.join(", "))
        } else {
            format!("::<{}>", type_list.join(", "))
        };

        (types.last().unwrap().span.hi(), type_str)
    };

    let callee_str = format!(".{}{}", method_name, type_str);
    let span = mk_sp(lo, span.hi());

    rewrite_call(context, &callee_str, &args[1..], span, shape)
}
