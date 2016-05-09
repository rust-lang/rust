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

use Indent;
use rewrite::{Rewrite, RewriteContext};
use utils::{wrap_str, first_line_width};
use expr::rewrite_call;
use config::BlockIndentStyle;

use syntax::{ast, ptr};
use syntax::codemap::{mk_sp, Span};


pub fn rewrite_chain(expr: &ast::Expr,
                     context: &RewriteContext,
                     width: usize,
                     offset: Indent)
                     -> Option<String> {
    let total_span = expr.span;
    let (parent, subexpr_list) = make_subexpr_list(expr);

    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    let parent_block_indent = chain_base_indent(context, offset);
    let parent_context = &RewriteContext { block_indent: parent_block_indent, ..*context };
    let parent_rewrite = try_opt!(parent.rewrite(parent_context, width, offset));

    // Decide how to layout the rest of the chain. `extend` is true if we can
    // put the first non-parent item on the same line as the parent.
    let (indent, extend) = if !parent_rewrite.contains('\n') && is_continuable(parent) ||
                              parent_rewrite.len() <= context.config.tab_spaces {
// <<<<<<< HEAD
//         // Try and put at least the first two items on the same line.
//         (chain_indent(context, offset + Indent::new(0, parent_rewrite.len())), true)
// =======
        let indent = if let ast::ExprKind::Try(..) = subexpr_list.last().unwrap().node {
            parent_block_indent.block_indent(context.config)
        } else {
            offset + Indent::new(0, parent_rewrite.len())
        };
        (indent, true)
    } else if is_block_expr(parent, &parent_rewrite) {
        // The parent is a block, so align the rest of the chain with the closing
        // brace.
        (parent_block_indent, false)
    } else if parent_rewrite.contains('\n') {
        (chain_indent(context, parent_block_indent.block_indent(context.config)), false)
    } else {
        (chain_indent_newline(context, offset + Indent::new(0, parent_rewrite.len())), false)
    };

    let max_width = try_opt!((width + offset.width()).checked_sub(indent.width()));
    let mut rewrites = try_opt!(subexpr_list.iter()
        .rev()
        .map(|e| rewrite_chain_subexpr(e, total_span, context, max_width, indent))
        .collect::<Option<Vec<_>>>());

    // Total of all items excluding the last.
    let almost_total = rewrites[..rewrites.len() - 1]
        .iter()
        .fold(0, |a, b| a + first_line_width(b)) + parent_rewrite.len();
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

    let mut fits_single_line = !veto_single_line && total_width <= width;
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
                                                      width,
                                                      total_span,
                                                      context,
                                                      offset)
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
        format!("\n{}", indent.to_string(context.config))
    };

    let first_connector = if extend {
        ""
    } else {
        &*connector
    };

    wrap_str(format!("{}{}{}",
                     parent_rewrite,
                     first_connector,
                     join_rewrites(&rewrites, &subexpr_list, &connector)),
             context.config.max_width,
             width,
             offset)
}

fn join_rewrites(rewrites: &[String], subexps: &[&ast::Expr], connector: &str) -> String {
    let mut rewrite_iter = rewrites.iter();
    let mut result = rewrite_iter.next().unwrap().clone();

    for (rewrite, expr) in rewrite_iter.zip(subexps.iter()) {
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
fn make_subexpr_list(mut expr: &ast::Expr) -> (&ast::Expr, Vec<&ast::Expr>) {
    fn pop_expr_chain(expr: &ast::Expr) -> Option<&ast::Expr> {
        match expr.node {
            ast::ExprKind::MethodCall(_, _, ref expressions) => Some(&expressions[0]),
            ast::ExprKind::TupField(ref subexpr, _) |
            ast::ExprKind::Field(ref subexpr, _) => Some(subexpr),
            _ => None,
        }
    }

    let mut subexpr_list = vec![expr];

    while let Some(subexpr) = pop_expr_chain(expr) {
        subexpr_list.push(subexpr);
        expr = subexpr;
    }

    let parent = subexpr_list.pop().unwrap();
    (parent, subexpr_list)
}

fn chain_base_indent(context: &RewriteContext, offset: Indent) -> Indent {
    match context.config.chain_base_indent {
        BlockIndentStyle::Visual => offset,
        BlockIndentStyle::Inherit => context.block_indent,
        BlockIndentStyle::Tabbed => context.block_indent.block_indent(context.config),
    }
}

fn chain_indent(context: &RewriteContext, offset: Indent) -> Indent {
    match context.config.chain_indent {
        BlockIndentStyle::Visual => offset,
        BlockIndentStyle::Inherit => context.block_indent,
        BlockIndentStyle::Tabbed => context.block_indent.block_indent(context.config),
    }
}

// Ignores visual indenting because this function should be called where it is
// not possible to use visual indentation because we are starting on a newline.
fn chain_indent_newline(context: &RewriteContext, _offset: Indent) -> Indent {
    match context.config.chain_indent {
        BlockIndentStyle::Inherit => context.block_indent,
        BlockIndentStyle::Visual | BlockIndentStyle::Tabbed => {
            context.block_indent.block_indent(context.config)
        }
    }
}

fn rewrite_method_call_with_overflow(expr_kind: &ast::ExprKind,
                                     last: &mut String,
                                     almost_total: usize,
                                     width: usize,
                                     total_span: Span,
                                     context: &RewriteContext,
                                     offset: Indent)
                                     -> bool {
    if let &ast::ExprKind::MethodCall(ref method_name, ref types, ref expressions) = expr_kind {
        let budget = match width.checked_sub(almost_total) {
            Some(b) => b,
            None => return false,
        };
        let mut last_rewrite = rewrite_method_call(method_name.node,
                                                   types,
                                                   expressions,
                                                   total_span,
                                                   context,
                                                   budget,
                                                   offset + almost_total);

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

fn pop_expr_chain(expr: &ast::Expr) -> Option<&ast::Expr> {
    match expr.node {
        ast::ExprKind::MethodCall(_, _, ref expressions) => Some(&expressions[0]),
        ast::ExprKind::TupField(ref subexpr, _) |
        ast::ExprKind::Field(ref subexpr, _) |
        ast::ExprKind::Try(ref subexpr) => Some(subexpr),
        _ => None,
    }
}

// Rewrite the last element in the chain `expr`. E.g., given `a.b.c` we rewrite
// `.c`.
fn rewrite_chain_subexpr(expr: &ast::Expr,
                         span: Span,
                         context: &RewriteContext,
                         width: usize,
                         offset: Indent)
                         -> Option<String> {
    match expr.node {
        ast::ExprKind::MethodCall(ref method_name, ref types, ref expressions) => {
            let inner = &RewriteContext { block_indent: offset, ..*context };
            rewrite_method_call(method_name.node,
                                types,
                                expressions,
                                span,
                                inner,
                                width,
                                offset)
        }
        ast::ExprKind::Field(_, ref field) => {
            let s = format!(".{}", field.node);
            if s.len() <= width {
                Some(s)
            } else {
                None
            }
        }
        ast::ExprKind::TupField(_, ref field) => {
            let s = format!(".{}", field.node);
            if s.len() <= width {
                Some(s)
            } else {
                None
            }
        }
        ast::ExprKind::Try(_) => {
            if width >= 1 {
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
                       width: usize,
                       offset: Indent)
                       -> Option<String> {
    let (lo, type_str) = if types.is_empty() {
        (args[0].span.hi, String::new())
    } else {
        let type_list: Vec<_> = try_opt!(types.iter()
            .map(|ty| ty.rewrite(context, width, offset))
            .collect());

        (types.last().unwrap().span.hi, format!("::<{}>", type_list.join(", ")))
    };

    let callee_str = format!(".{}{}", method_name, type_str);
    let span = mk_sp(lo, span.hi);

    rewrite_call(context, &callee_str, &args[1..], span, width, offset)
}
