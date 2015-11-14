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

use Indent;
use rewrite::{Rewrite, RewriteContext};
use utils::first_line_width;
use expr::rewrite_call;
use config::BlockIndentStyle;

use syntax::{ast, ptr};
use syntax::codemap::{mk_sp, Span};
use syntax::print::pprust;

pub fn rewrite_chain(mut expr: &ast::Expr,
                     context: &RewriteContext,
                     width: usize,
                     offset: Indent)
                     -> Option<String> {
    let total_span = expr.span;
    let mut subexpr_list = vec![expr];

    while let Some(subexpr) = pop_expr_chain(expr) {
        subexpr_list.push(subexpr);
        expr = subexpr;
    }

    let parent_block_indent = match context.config.chain_base_indent {
        BlockIndentStyle::Visual => offset,
        BlockIndentStyle::Inherit => context.block_indent,
        BlockIndentStyle::Tabbed => context.block_indent.block_indent(context.config),
    };
    let parent_context = &RewriteContext { block_indent: parent_block_indent, ..*context };
    let parent = subexpr_list.pop().unwrap();
    let parent_rewrite = try_opt!(expr.rewrite(parent_context, width, offset));
    let (indent, extend) = if !parent_rewrite.contains('\n') && is_continuable(parent) ||
                              parent_rewrite.len() <= context.config.tab_spaces {
        (offset + Indent::new(0, parent_rewrite.len()), true)
    } else if is_block_expr(parent, &parent_rewrite) {
        (parent_block_indent, false)
    } else {
        match context.config.chain_indent {
            BlockIndentStyle::Inherit => (context.block_indent, false),
            BlockIndentStyle::Tabbed => {
                (context.block_indent.block_indent(context.config), false)
            }
            BlockIndentStyle::Visual => {
                (offset + Indent::new(context.config.tab_spaces, 0), false)
            }
        }
    };

    let max_width = try_opt!((width + offset.width()).checked_sub(indent.width()));
    let mut rewrites = try_opt!(subexpr_list.iter()
                                            .rev()
                                            .map(|e| {
                                                rewrite_chain_expr(e,
                                                                   total_span,
                                                                   context,
                                                                   max_width,
                                                                   indent)
                                            })
                                            .collect::<Option<Vec<_>>>());

    // Total of all items excluding the last.
    let almost_total = rewrites[..rewrites.len() - 1]
                           .iter()
                           .fold(0, |a, b| a + first_line_width(b)) +
                       parent_rewrite.len();
    let total_width = almost_total + first_line_width(rewrites.last().unwrap());
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

    let fits_single_line = !veto_single_line &&
                           match subexpr_list[0].node {
        ast::Expr_::ExprMethodCall(ref method_name, ref types, ref expressions)
            if context.config.chains_overflow_last => {
            let len = rewrites.len();
            let (init, last) = rewrites.split_at_mut(len - 1);
            let last = &mut last[0];

            if init.iter().all(|s| !s.contains('\n')) && total_width <= width {
                let last_rewrite = width.checked_sub(almost_total)
                                        .and_then(|inner_width| {
                                            rewrite_method_call(method_name.node,
                                                                types,
                                                                expressions,
                                                                total_span,
                                                                context,
                                                                inner_width,
                                                                offset + almost_total)
                                        });
                match last_rewrite {
                    Some(mut string) => {
                        ::std::mem::swap(&mut string, last);
                        true
                    }
                    None => false,
                }
            } else {
                false
            }
        }
        _ => total_width <= width && rewrites.iter().all(|s| !s.contains('\n')),
    };

    let connector = if fits_single_line && !parent_rewrite.contains('\n') {
        String::new()
    } else {
        format!("\n{}", indent.to_string(context.config))
    };

    let first_connector = if extend {
        ""
    } else {
        &connector[..]
    };

    Some(format!("{}{}{}",
                 parent_rewrite,
                 first_connector,
                 rewrites.join(&connector)))
}

// States whether an expression's last line exclusively consists of closing
// parens, braces and brackets in its idiomatic formatting.
fn is_block_expr(expr: &ast::Expr, repr: &str) -> bool {
    match expr.node {
        ast::Expr_::ExprStruct(..) |
        ast::Expr_::ExprWhile(..) |
        ast::Expr_::ExprWhileLet(..) |
        ast::Expr_::ExprIf(..) |
        ast::Expr_::ExprIfLet(..) |
        ast::Expr_::ExprBlock(..) |
        ast::Expr_::ExprLoop(..) |
        ast::Expr_::ExprForLoop(..) |
        ast::Expr_::ExprMatch(..) => repr.contains('\n'),
        ast::Expr_::ExprParen(ref expr) |
        ast::Expr_::ExprBinary(_, _, ref expr) |
        ast::Expr_::ExprIndex(_, ref expr) |
        ast::Expr_::ExprUnary(_, ref expr) => is_block_expr(expr, repr),
        _ => false,
    }
}

fn pop_expr_chain<'a>(expr: &'a ast::Expr) -> Option<&'a ast::Expr> {
    match expr.node {
        ast::Expr_::ExprMethodCall(_, _, ref expressions) => Some(&expressions[0]),
        ast::Expr_::ExprTupField(ref subexpr, _) |
        ast::Expr_::ExprField(ref subexpr, _) => Some(subexpr),
        _ => None,
    }
}

fn rewrite_chain_expr(expr: &ast::Expr,
                      span: Span,
                      context: &RewriteContext,
                      width: usize,
                      offset: Indent)
                      -> Option<String> {
    match expr.node {
        ast::Expr_::ExprMethodCall(ref method_name, ref types, ref expressions) => {
            let inner = &RewriteContext { block_indent: offset, ..*context };
            rewrite_method_call(method_name.node,
                                types,
                                expressions,
                                span,
                                inner,
                                width,
                                offset)
        }
        ast::Expr_::ExprField(_, ref field) => Some(format!(".{}", field.node)),
        ast::Expr_::ExprTupField(_, ref field) => Some(format!(".{}", field.node)),
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
                       offset: Indent)
                       -> Option<String> {
    let (lo, type_str) = if types.is_empty() {
        (args[0].span.hi, String::new())
    } else {
        let type_list = types.iter().map(|ty| pprust::ty_to_string(ty)).collect::<Vec<_>>();

        (types.last().unwrap().span.hi,
         format!("::<{}>", type_list.join(", ")))
    };

    let callee_str = format!(".{}{}", method_name, type_str);
    let span = mk_sp(lo, span.hi);

    rewrite_call(context, &callee_str, &args[1..], span, width, offset)
}
