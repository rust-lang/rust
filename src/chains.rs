// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Formatting of chained expressions, i.e. expressions which are chained by
// dots: struct and enum field access and method calls.
//
// Instead of walking these subexpressions one-by-one, as is our usual strategy
// for expression formatting, we collect maximal sequences of these expressions
// and handle them simultaneously.
//
// Whenever possible, the entire chain is put on a single line. If that fails,
// we put each subexpression on a separate, much like the (default) function
// argument function argument strategy.

use rewrite::{Rewrite, RewriteContext};
use utils::make_indent;
use expr::rewrite_call;

use syntax::{ast, ptr};
use syntax::codemap::{mk_sp, Span};
use syntax::print::pprust;

pub fn rewrite_chain(mut expr: &ast::Expr,
                     context: &RewriteContext,
                     width: usize,
                     offset: usize)
                     -> Option<String> {
    let total_span = expr.span;
    let mut subexpr_list = vec![expr];

    while let Some(subexpr) = pop_expr_chain(expr) {
        subexpr_list.push(subexpr);
        expr = subexpr;
    }

    let parent = subexpr_list.pop().unwrap();
    let parent_rewrite = try_opt!(expr.rewrite(context, width, offset));
    let (extra_indent, extend) = if !parent_rewrite.contains('\n') && is_continuable(parent) ||
                                    parent_rewrite.len() <= context.config.tab_spaces {
        (parent_rewrite.len(), true)
    } else {
        (context.config.tab_spaces, false)
    };
    let indent = offset + extra_indent;

    let max_width = try_opt!(width.checked_sub(extra_indent));
    let rewrites = try_opt!(subexpr_list.into_iter()
                                        .rev()
                                        .map(|e| {
                                            rewrite_chain_expr(e,
                                                               total_span,
                                                               context,
                                                               max_width,
                                                               indent)
                                        })
                                        .collect::<Option<Vec<_>>>());

    let total_width = rewrites.iter().fold(0, |a, b| a + b.len()) + parent_rewrite.len();
    let fits_single_line = total_width <= width && rewrites.iter().all(|s| !s.contains('\n'));

    let connector = if fits_single_line {
        String::new()
    } else {
        format!("\n{}", make_indent(indent))
    };

    let first_connector = if extend {
        ""
    } else {
        &connector[..]
    };

    Some(format!("{}{}{}", parent_rewrite, first_connector, rewrites.join(&connector)))
}

fn pop_expr_chain<'a>(expr: &'a ast::Expr) -> Option<&'a ast::Expr> {
    match expr.node {
        ast::Expr_::ExprMethodCall(_, _, ref expressions) => {
            Some(&expressions[0])
        }
        ast::Expr_::ExprTupField(ref subexpr, _) |
        ast::Expr_::ExprField(ref subexpr, _) => {
            Some(subexpr)
        }
        _ => None,
    }
}

fn rewrite_chain_expr(expr: &ast::Expr,
                      span: Span,
                      context: &RewriteContext,
                      width: usize,
                      offset: usize)
                      -> Option<String> {
    match expr.node {
        ast::Expr_::ExprMethodCall(ref method_name, ref types, ref expressions) => {
            rewrite_method_call(method_name.node, types, expressions, span, context, width, offset)
        }
        ast::Expr_::ExprField(_, ref field) => {
            Some(format!(".{}", field.node))
        }
        ast::Expr_::ExprTupField(_, ref field) => {
            Some(format!(".{}", field.node))
        }
        _ => unreachable!(),
    }
}

// Determines we can continue formatting a given expression on the same line.
fn is_continuable(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::Expr_::ExprPath(..) => true,
        _ => false,
    }
}

fn rewrite_method_call(method_name: ast::Ident,
                       types: &[ptr::P<ast::Ty>],
                       args: &[ptr::P<ast::Expr>],
                       span: Span,
                       context: &RewriteContext,
                       width: usize,
                       offset: usize)
                       -> Option<String> {
    let type_str = if types.is_empty() {
        String::new()
    } else {
        let type_list = types.iter().map(|ty| pprust::ty_to_string(ty)).collect::<Vec<_>>();
        format!("::<{}>", type_list.join(", "))
    };

    let callee_str = format!(".{}{}", method_name, type_str);
    let inner_context = &RewriteContext {
        block_indent: offset,
        ..*context
    };

    rewrite_call(inner_context, &callee_str, args, span, width, offset)
}
