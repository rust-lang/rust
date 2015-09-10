// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rewrite::{Rewrite, RewriteContext};
use utils::{make_indent, extra_offset};
use expr::rewrite_call;

use syntax::{ast, ptr};
use syntax::codemap::{mk_sp, Span};
use syntax::print::pprust;

pub fn rewrite_chain(orig_expr: &ast::Expr,
                     context: &RewriteContext,
                     width: usize,
                     offset: usize)
                     -> Option<String> {
    let mut expr = orig_expr;
    let mut rewrites = Vec::new();
    let indent = offset + context.config.tab_spaces;
    let max_width = try_opt!(context.config.max_width.checked_sub(indent));

    while let Some(pair) = pop_expr_chain(expr, orig_expr.span, context, max_width, indent) {
        let (rewrite, parent_expr) = pair;

        rewrites.push(try_opt!(rewrite));
        expr = parent_expr;
    }

    let parent_rewrite = try_opt!(expr.rewrite(context, width, offset));

    if rewrites.len() == 1 {
        let extra_offset = extra_offset(&parent_rewrite, offset);
        let offset = offset + extra_offset;
        let max_width = try_opt!(width.checked_sub(extra_offset));

        let rerewrite = pop_expr_chain(orig_expr, orig_expr.span, context, max_width, offset)
                            .unwrap()
                            .0;

        return Some(format!("{}{}", parent_rewrite, try_opt!(rerewrite)));
    }

    let total_width = rewrites.iter().fold(0, |a, b| a + b.len()) + parent_rewrite.len();

    let connector = if total_width <= width && rewrites.iter().all(|s| !s.contains('\n')) {
        String::new()
    } else {
        format!("\n{}", make_indent(indent))
    };

    // FIXME: don't do this. There's a more efficient way. VecDeque?
    rewrites.reverse();

    // Put the first link on the same line as parent, if it fits.
    let first_connector = if parent_rewrite.len() + rewrites[0].len() <= width &&
                             is_continuable(expr) &&
                             !rewrites[0].contains('\n') ||
                             parent_rewrite.len() <= context.config.tab_spaces {
        ""
    } else {
        &connector[..]
    };

    Some(format!("{}{}{}", parent_rewrite, first_connector, rewrites.join(&connector)))
}

// Returns None when the expression is not a chainable. Otherwise, rewrites the
// outermost chain element and returns the remaining chain.
fn pop_expr_chain<'a>(expr: &'a ast::Expr,
                      span: Span,
                      context: &RewriteContext,
                      width: usize,
                      offset: usize)
                      -> Option<(Option<String>, &'a ast::Expr)> {
    match expr.node {
        ast::Expr_::ExprMethodCall(ref method_name, ref types, ref expressions) => {
            Some((rewrite_method_call(method_name.node,
                                      types,
                                      expressions,
                                      span,
                                      context,
                                      width,
                                      offset),
                  &expressions[0]))
        }
        ast::Expr_::ExprField(ref subexpr, ref field) => {
            Some((Some(format!(".{}", field.node)), subexpr))
        }
        ast::Expr_::ExprTupField(ref subexpr, ref field) => {
            Some((Some(format!(".{}", field.node)), subexpr))
        }
        _ => None,
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
