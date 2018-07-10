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
use utils::{first_line_width, last_line_extendable, last_line_width, mk_sp, wrap_str};

use std::borrow::Cow;
use std::cmp::min;

use syntax::codemap::Span;
use syntax::{ast, ptr};

pub fn rewrite_chain(expr: &ast::Expr, context: &RewriteContext, shape: Shape) -> Option<String> {
    debug!("rewrite_chain {:?}", shape);
    let chain = Chain::from_ast(expr, context);
    // If this is just an expression with some `?`s, then format it trivially and
    // return early.
    if chain.children.is_empty() {
        let rewrite = chain.parent.expr.rewrite(context, shape.sub_width(chain.parent.tries)?)?;
        return Some(format!("{}{}", rewrite, "?".repeat(chain.parent.tries)));
    }

    match context.config.indent_style() {
        IndentStyle::Block => rewrite_chain_block(chain, context, shape),
        IndentStyle::Visual => rewrite_chain_visual(chain, context, shape),
    }
}

// An expression plus trailing `?`s to be formatted together.
#[derive(Debug)]
struct ChainItem {
    expr: ast::Expr,
    tries: usize,
}

#[derive(Debug)]
struct Chain {
    parent: ChainItem,
    // TODO do we need to clone the exprs?
    children: Vec<ChainItem>,
}

impl Chain {
    fn from_ast(expr: &ast::Expr, context: &RewriteContext) -> Chain {
        let mut subexpr_list = make_subexpr_list(expr, context);

        // Un-parse the expression tree into ChainItems
        let mut children = vec![];
        let mut sub_tries = 0;
        loop {
            if subexpr_list.is_empty() {
                break;
            }

            let subexpr = subexpr_list.pop().unwrap();
            match subexpr.node {
                ast::ExprKind::Try(_) => sub_tries += 1,
                _ => {
                    children.push(ChainItem {
                        expr: subexpr.clone(),
                        tries: sub_tries,
                    });
                    sub_tries = 0;
                }
            }
        }

        Chain {
            parent: children.remove(0),
            children,
        }
    }
}

fn rewrite_chain_block(chain: Chain, context: &RewriteContext, shape: Shape) -> Option<String> {
    debug!("rewrite_chain_block {:?} {:?}", chain, shape);
    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    // Root is the parent plus any other chain items placed on the first line to
    // avoid an orphan. E.g.,
    // ```
    // foo.bar
    //     .baz()
    // ```
    // If `bar` were not part of the root, then baz would be orphaned and 'float'.
    let mut root_rewrite = chain.parent.expr
        .rewrite(context, shape)
        .map(|parent_rw| parent_rw + &"?".repeat(chain.parent.tries))?;

    let mut children: &[_] = &chain.children;
    let mut root_ends_with_block = is_block_expr(context, &chain.parent.expr, &root_rewrite);
    let tab_width = context.config.tab_spaces().saturating_sub(shape.offset);

    while root_rewrite.len() <= tab_width && !root_rewrite.contains('\n') {
        let item = &children[0];
        let shape = shape.offset_left(root_rewrite.len())?;
        match rewrite_chain_subexpr(&item.expr, context, shape) {
            Some(rewrite) => {
                root_rewrite.push_str(&rewrite);
                root_rewrite.push_str(&"?".repeat(item.tries));
            }
            None => break,
        }

        root_ends_with_block = is_block_expr(context, &item.expr, &root_rewrite);

        children = &children[1..];
        if children.is_empty() {
            return Some(root_rewrite);
        }
    }

    // Separate out the last item in the chain for special treatment below.
    let last = &children[children.len() - 1];
    children = &children[..children.len() - 1];

    // Decide how to layout the rest of the chain.
    let child_shape = if root_ends_with_block {
        shape
    } else {
        shape.block_indent(context.config.tab_spaces())
    }.with_max_width(context.config);

    let mut rewrites: Vec<String> = Vec::with_capacity(children.len() + 2);
    rewrites.push(root_rewrite);
    let mut is_block_like = Vec::with_capacity(children.len() + 2);
    is_block_like.push(root_ends_with_block);
    for item in children {
        let rewrite = rewrite_chain_subexpr(&item.expr, context, child_shape)?;
        is_block_like.push(is_block_expr(context, &item.expr, &rewrite));
        rewrites.push(format!("{}{}", rewrite, "?".repeat(item.tries)));
    }

    // Total of all items excluding the last.
    let extend_last_subexpr = last_line_extendable(&rewrites[rewrites.len() - 1]);
    let almost_total = if extend_last_subexpr {
        last_line_width(&rewrites[rewrites.len() - 1])
    } else {
        rewrites.iter().fold(0, |a, b| a + b.len())
    } + last.tries;
    let one_line_budget = if rewrites.len() == 1 {
        shape.width
    } else {
        min(shape.width, context.config.width_heuristics().chain_width)
    };
    let all_in_one_line = rewrites.iter().all(|s| !s.contains('\n'))
        && almost_total < one_line_budget;
    let last_shape = if all_in_one_line {
        shape.sub_width(last.tries)?
    } else {
        child_shape.sub_width(shape.rhs_overhead(context.config) + last.tries)?
    };

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

    let mut last_subexpr_str = None;
    let mut fits_single_line = false;
    if all_in_one_line || extend_last_subexpr {
        // First we try to 'overflow' the last child and see if it looks better than using
        // vertical layout.
        if let Some(shape) = last_shape.offset_left(almost_total) {
            if let Some(rw) = rewrite_chain_subexpr(&last.expr, context, shape) {
                // We allow overflowing here only if both of the following conditions match:
                // 1. The entire chain fits in a single line except the last child.
                // 2. `last_child_str.lines().count() >= 5`.
                let line_count = rw.lines().count();
                let could_fit_single_line = almost_total + first_line_width(&rw) <= one_line_budget;
                if fits_single_line && line_count >= 5 {
                    last_subexpr_str = Some(rw);
                    fits_single_line = true;
                } else {
                    // We could not know whether overflowing is better than using vertical layout,
                    // just by looking at the overflowed rewrite. Now we rewrite the last child
                    // on its own line, and compare two rewrites to choose which is better.
                    match rewrite_chain_subexpr(&last.expr, context, last_shape) {
                        Some(ref new_rw) if !could_fit_single_line => {
                            last_subexpr_str = Some(new_rw.clone());
                        }
                        Some(ref new_rw) if new_rw.lines().count() >= line_count => {
                            last_subexpr_str = Some(rw);
                            fits_single_line = could_fit_single_line;
                        }
                        new_rw @ Some(..) => {
                            last_subexpr_str = new_rw;
                        }
                        _ => {
                            last_subexpr_str = Some(rw);
                            fits_single_line = could_fit_single_line;
                        }
                    }
                }
            }
        }
    }

    last_subexpr_str = last_subexpr_str.or_else(|| rewrite_chain_subexpr(&last.expr, context, last_shape));
    rewrites.push(format!("{}{}", last_subexpr_str?, "?".repeat(last.tries)));

    // We should never look at this, since we only look at the block-ness of the
    // previous item in the chain.
    is_block_like.push(false);

    let connector = if fits_single_line && all_in_one_line {
        // Yay, we can put everything on one line.
        Cow::from("")
    } else {
        // Use new lines.
        if *context.force_one_line_chain.borrow() {
            return None;
        }
        child_shape.indent.to_string_with_newline(context.config)
    };

    let result = join_rewrites(&rewrites, &is_block_like, &connector);
    Some(result)
}

fn rewrite_chain_visual(chain: Chain, context: &RewriteContext, shape: Shape) -> Option<String> {
    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let parent_shape = if is_block_expr(context, &chain.parent.expr, "\n") {
        shape.visual_indent(0)
    } else {
        shape
    };
    let mut children: &[_] = &chain.children;
    let mut root_rewrite = chain.parent.expr
        .rewrite(context, parent_shape)
        .map(|parent_rw| parent_rw + &"?".repeat(chain.parent.tries))?;

    if !root_rewrite.contains('\n') && is_continuable(&chain.parent.expr) {
        let item = &children[0];
        let overhead = last_line_width(&root_rewrite);
        let shape = parent_shape.offset_left(overhead)?;
        let rewrite = rewrite_chain_subexpr(&item.expr, context, shape)?;
        root_rewrite.push_str(&rewrite);
        root_rewrite.push_str(&"?".repeat(item.tries));
        children = &children[1..];
        if children.is_empty() {
            return Some(root_rewrite);
        }
    }

    let last = &children[children.len() - 1];
    children = &children[..children.len() - 1];

    let child_shape = shape.visual_indent(0).with_max_width(context.config);

    let mut rewrites: Vec<String> = Vec::with_capacity(children.len() + 2);
    rewrites.push(root_rewrite);
    for item in chain.children.iter() {
        let rewrite = rewrite_chain_subexpr(&item.expr, context, child_shape)?;
        rewrites.push(format!("{}{}", rewrite, "?".repeat(item.tries)));
    }

    // Total of all items excluding the last.
    let almost_total = rewrites.iter().fold(0, |a, b| a + b.len()) + last.tries;
    let one_line_budget = if rewrites.len() == 1 {
        shape.width
    } else {
        min(shape.width, context.config.width_heuristics().chain_width)
    };
    let all_in_one_line = rewrites.iter().all(|s| !s.contains('\n'))
        && almost_total < one_line_budget;
    let last_shape = child_shape.sub_width(shape.rhs_overhead(context.config) + last.tries)?;

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

    let mut last_subexpr_str = None;
    let mut fits_single_line = false;
    if all_in_one_line {
        // First we try to 'overflow' the last child and see if it looks better than using
        // vertical layout.
        if let Some(shape) = parent_shape.offset_left(almost_total) {
            if let Some(rw) = rewrite_chain_subexpr(&last.expr, context, shape) {
                // We allow overflowing here only if both of the following conditions match:
                // 1. The entire chain fits in a single line except the last child.
                // 2. `last_child_str.lines().count() >= 5`.
                let line_count = rw.lines().count();
                let could_fit_single_line = almost_total + first_line_width(&rw) <= one_line_budget;
                if could_fit_single_line && line_count >= 5 {
                    last_subexpr_str = Some(rw);
                    fits_single_line = true;
                } else {
                    // We could not know whether overflowing is better than using vertical layout,
                    // just by looking at the overflowed rewrite. Now we rewrite the last child
                    // on its own line, and compare two rewrites to choose which is better.
                    match rewrite_chain_subexpr(&last.expr, context, last_shape) {
                        Some(ref new_rw) if !could_fit_single_line => {
                            last_subexpr_str = Some(new_rw.clone());
                        }
                        Some(ref new_rw) if new_rw.lines().count() >= line_count => {
                            last_subexpr_str = Some(rw);
                            fits_single_line = could_fit_single_line;
                        }
                        new_rw @ Some(..) => {
                            last_subexpr_str = new_rw;
                        }
                        _ => {
                            last_subexpr_str = Some(rw);
                            fits_single_line = could_fit_single_line;
                        }
                    }
                }
            }
        }
    } 

    last_subexpr_str = last_subexpr_str.or_else(|| rewrite_chain_subexpr(&last.expr, context, last_shape));
    rewrites.push(last_subexpr_str?);

    let connector = if fits_single_line && all_in_one_line {
        // Yay, we can put everything on one line.
        Cow::from("")
    } else {
        // Use new lines.
        if *context.force_one_line_chain.borrow() {
            return None;
        }
        child_shape.indent.to_string_with_newline(context.config)
    };

    let result = format!("{}{}",
        join_rewrites_vis(&rewrites, &connector),
        "?".repeat(last.tries),
    );
    wrap_str(result, context.config.max_width(), shape)
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
        ast::ExprKind::Mac(..)
        | ast::ExprKind::Call(..)
        | ast::ExprKind::MethodCall(..) => {
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
        | ast::ExprKind::Closure(_, _, _, _, ref expr, _) 
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Yield(Some(ref expr)) => is_block_expr(context, expr, repr),
        _ => false,
    }
}

// Returns a Vec of the prefixes of the chain.
// E.g., for input `a.b.c` we return [`a.b.c`, `a.b`, 'a']
fn make_subexpr_list(expr: &ast::Expr, context: &RewriteContext) -> Vec<ast::Expr> {
    let mut subexpr_list = vec![expr.clone()];

    while let Some(subexpr) = pop_expr_chain(subexpr_list.last().unwrap(), context) {
        subexpr_list.push(subexpr.clone());
    }

    subexpr_list
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
            rewrite_method_call(segment.ident, types, expressions, expr.span, context, shape)
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
