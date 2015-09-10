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
use utils::{span_after, make_indent, extra_offset};
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
    let indent = context.block_indent + context.config.tab_spaces;
    let max_width = context.config.max_width - context.config.tab_spaces;

    loop {
        match expr.node {
            ast::Expr_::ExprMethodCall(ref method_name, ref types, ref expressions) => {
                // FIXME: a lot of duplication between this and the
                // rewrite_method_call in expr.rs.
                let new_span = mk_sp(expressions[0].span.hi, expr.span.hi);
                let lo = span_after(new_span, "(", context.codemap);
                let new_span = mk_sp(lo, expr.span.hi);

                let rewrite = rewrite_method_call(method_name.node,
                                                  types,
                                                  &expressions[1..],
                                                  new_span,
                                                  context,
                                                  max_width,
                                                  indent);
                rewrites.push(try_opt!(rewrite));
                expr = &expressions[0];
            }
            ast::Expr_::ExprField(ref subexpr, ref field) => {
                expr = subexpr;
                rewrites.push(format!(".{}", field.node));
            }
            ast::Expr_::ExprTupField(ref subexpr, ref field) => {
                expr = subexpr;
                rewrites.push(format!(".{}", field.node));
            }
            _ => break,
        }
    }

    let parent_rewrite = try_opt!(expr.rewrite(context, width, offset));

    // TODO: add exception for when rewrites.len() == 1
    if rewrites.len() == 1 {
        let extra_offset = extra_offset(&parent_rewrite, offset);
        let max_width = try_opt!(width.checked_sub(extra_offset));
        // FIXME: massive duplication
        let rerewrite = match orig_expr.node {
            ast::Expr_::ExprMethodCall(ref method_name, ref types, ref expressions) => {
                let new_span = mk_sp(expressions[0].span.hi, orig_expr.span.hi);
                let lo = span_after(new_span, "(", context.codemap);
                let new_span = mk_sp(lo, orig_expr.span.hi);

                rewrite_method_call(method_name.node,
                                    types,
                                    &expressions[1..],
                                    new_span,
                                    context,
                                    max_width,
                                    offset + extra_offset)
            }
            ast::Expr_::ExprField(_, ref field) => {
                Some(format!(".{}", field.node))
            }
            ast::Expr_::ExprTupField(_, ref field) => {
                Some(format!(".{}", field.node))
            }
            _ => unreachable!(),
        };

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
                             !rewrites[0].contains('\n') {
        ""
    } else {
        &connector[..]
    };

    Some(format!("{}{}{}", parent_rewrite, first_connector, rewrites.join(&connector)))
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

    rewrite_call(context, &callee_str, args, span, width, offset)
}
