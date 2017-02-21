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
/// Depends on config options: `chain_base_indent` is the indent to use for
/// blocks in the parent/root/base of the chain.
/// E.g., `let foo = { aaaa; bbb; ccc }.bar.baz();`, we would layout for the
/// following values of `chain_base_indent`:
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
/// `chain_indent` dictates how the rest of the chain is aligned.
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
/// `chains_overflow_last` applies only to chains where the last item is a
/// method call. Usually, any line break in a chain sub-expression causes the
/// whole chain to be split with newlines at each `.`. With `chains_overflow_last`
/// true, then we allow the last method call to spill over multiple lines without
/// forcing the rest of the chain to be split.

use {Indent, Shape};
use rewrite::{Rewrite, RewriteContext};
use utils::{wrap_str, first_line_width};
use expr::rewrite_call;
use config::BlockIndentStyle;
use macros::convert_try_mac;

use std::iter;
use syntax::{ast, ptr};
use syntax::codemap::{mk_sp, Span};

pub fn rewrite_chain(expr: &ast::Expr, context: &RewriteContext, shape: Shape) -> Option<String> {
    debug!("rewrite_chain {:?}", shape);
    let total_span = expr.span;
    let (parent, subexpr_list) = make_subexpr_list(expr, context);

    // Bail out if the chain is just try sugar, i.e., an expression followed by
    // any number of `?`s.
    if chain_only_try(&subexpr_list) {
        return rewrite_try(&parent, subexpr_list.len(), context, shape);
    }

    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let mut parent_shape = shape;
    if is_block_expr(&parent, "\n") {
        parent_shape = chain_base_indent(context, shape);
    }
    let parent_rewrite = try_opt!(parent.rewrite(context, parent_shape));

    // Decide how to layout the rest of the chain. `extend` is true if we can
    // put the first non-parent item on the same line as the parent.
    let (nested_shape, extend) = if !parent_rewrite.contains('\n') && is_continuable(&parent) {
        let nested_shape = if let ast::ExprKind::Try(..) = subexpr_list.last().unwrap().node {
            parent_shape.block_indent(context.config.tab_spaces)
        } else {
            chain_indent(context, shape.add_offset(parent_rewrite.len()))
        };
        (nested_shape, true)
    } else if is_block_expr(&parent, &parent_rewrite) {
        // The parent is a block, so align the rest of the chain with the closing
        // brace.
        (parent_shape, false)
    } else if parent_rewrite.contains('\n') {
        (chain_indent(context, parent_shape.block_indent(context.config.tab_spaces)), false)
    } else {
        (chain_indent_newline(context, shape.add_offset(parent_rewrite.len())), false)
    };

    let max_width = try_opt!((shape.width + shape.indent.width() + shape.offset).checked_sub(nested_shape.indent.width() + nested_shape.offset));
    // The alignement in the shape is only used if we start the item on a new
    // line, so we don't need to preserve the offset.
    let child_shape = Shape { width: max_width, ..nested_shape };
    debug!("child_shape {:?}", child_shape);
    let mut rewrites = try_opt!(subexpr_list.iter()
        .rev()
        .map(|e| rewrite_chain_subexpr(e, total_span, context, child_shape))
        .collect::<Option<Vec<_>>>());

    // Total of all items excluding the last.
    let almost_total = rewrites[..rewrites.len() - 1]
        .iter()
        .fold(0, |a, b| a + first_line_width(b)) + parent_rewrite.len();

    let veto_single_line = if context.config.take_source_hints && subexpr_list.len() > 1 {
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
        let (init, last) = rewrites.split_at_mut(len - 1);
        fits_single_line = init.iter().all(|s| !s.contains('\n'));

        if fits_single_line {
            fits_single_line = match expr.node {
                ref e @ ast::ExprKind::MethodCall(..) if context.config.chains_overflow_last => {
                    rewrite_method_call_with_overflow(e,
                                                      &mut last[0],
                                                      almost_total,
                                                      total_span,
                                                      context,
                                                      shape)
                }
                _ => !last[0].contains('\n'),
            }
        }
    }

    let connector = if fits_single_line && !parent_rewrite.contains('\n') {
        // Yay, we can put everything on one line.
        String::new()
    } else {
        // Use new lines.
        format!("\n{}", nested_shape.indent.to_string(context.config))
    };

    let first_connector = if extend || subexpr_list.is_empty() {
        ""
    } else if let ast::ExprKind::Try(_) = subexpr_list[0].node {
        ""
    } else {
        &*connector
    };

    wrap_str(format!("{}{}{}",
                     parent_rewrite,
                     first_connector,
                     join_rewrites(&rewrites, &subexpr_list, &connector)),
             context.config.max_width,
             shape)
}

// True if the chain is only `?`s.
fn chain_only_try(exprs: &[ast::Expr]) -> bool {
    exprs.iter().all(|e| if let ast::ExprKind::Try(_) = e.node {
        true
    } else {
        false
    })
}

pub fn rewrite_try(expr: &ast::Expr,
                   try_count: usize,
                   context: &RewriteContext,
                   shape: Shape)
                   -> Option<String> {
    let sub_expr = try_opt!(expr.rewrite(context, try_opt!(shape.sub_width(try_count))));
    Some(format!("{}{}",
                 sub_expr,
                 iter::repeat("?").take(try_count).collect::<String>()))
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
fn is_block_expr(expr: &ast::Expr, repr: &str) -> bool {
    match expr.node {
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
        ast::ExprKind::Unary(_, ref expr) => is_block_expr(expr, repr),
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

fn chain_base_indent(context: &RewriteContext, shape: Shape) -> Shape {
    match context.config.chain_base_indent {
        BlockIndentStyle::Visual => shape,
        BlockIndentStyle::Inherit => shape.block_indent(0),
        BlockIndentStyle::Tabbed => shape.block_indent(context.config.tab_spaces),
    }
}

fn chain_indent(context: &RewriteContext, shape: Shape) -> Shape {
    match context.config.chain_indent {
        BlockIndentStyle::Visual => shape,
        BlockIndentStyle::Inherit => shape.block_indent(0),
        BlockIndentStyle::Tabbed => shape.block_indent(context.config.tab_spaces),
    }
}

// Ignores visual indenting because this function should be called where it is
// not possible to use visual indentation because we are starting on a newline.
fn chain_indent_newline(context: &RewriteContext, shape: Shape) -> Shape {
    match context.config.chain_indent {
        BlockIndentStyle::Inherit => shape.block_indent(0),
        BlockIndentStyle::Visual | BlockIndentStyle::Tabbed => {
            shape.block_indent(context.config.tab_spaces)
        }
    }
}

fn rewrite_method_call_with_overflow(expr_kind: &ast::ExprKind,
                                     last: &mut String,
                                     almost_total: usize,
                                     total_span: Span,
                                     context: &RewriteContext,
                                     shape: Shape)
                                     -> bool {
    if let &ast::ExprKind::MethodCall(ref method_name, ref types, ref expressions) = expr_kind {
        let shape = match shape.shrink_left(almost_total) {
            Some(b) => b,
            None => return false,
        };
        let mut last_rewrite = rewrite_method_call(method_name.node,
                                                   types,
                                                   expressions,
                                                   total_span,
                                                   context,
                                                   shape);

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
        ast::ExprKind::Mac(ref mac) if context.config.use_try_shorthand => {
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
fn rewrite_chain_subexpr(expr: &ast::Expr,
                         span: Span,
                         context: &RewriteContext,
                         shape: Shape)
                         -> Option<String> {
    match expr.node {
        ast::ExprKind::MethodCall(ref method_name, ref types, ref expressions) => {
            rewrite_method_call(method_name.node, types, expressions, span, context, shape)
        }
        ast::ExprKind::Field(_, ref field) => {
            let s = format!(".{}", field.node);
            if s.len() <= shape.width {
                Some(s)
            } else {
                None
            }
        }
        ast::ExprKind::TupField(_, ref field) => {
            let s = format!(".{}", field.node);
            if s.len() <= shape.width {
                Some(s)
            } else {
                None
            }
        }
        ast::ExprKind::Try(_) => {
            if shape.width >= 1 {
                Some("?".into())
            } else {
                None
            }
        }
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

fn rewrite_method_call(method_name: ast::Ident,
                       types: &[ptr::P<ast::Ty>],
                       args: &[ptr::P<ast::Expr>],
                       span: Span,
                       context: &RewriteContext,
                       shape: Shape)
                       -> Option<String> {
    let (lo, type_str) = if types.is_empty() {
        (args[0].span.hi, String::new())
    } else {
        let type_list: Vec<_> = try_opt!(types.iter()
            .map(|ty| ty.rewrite(context, shape))
            .collect());

        let type_str = if context.config.spaces_within_angle_brackets && type_list.len() > 0 {
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
