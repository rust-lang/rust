// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::{Ordering, min};
use std::ops::Deref;
use std::iter::ExactSizeIterator;
use std::fmt::Write;

use {Indent, Shape, Spanned};
use codemap::SpanUtils;
use rewrite::{Rewrite, RewriteContext};
use lists::{write_list, itemize_list, ListFormatting, SeparatorTactic, ListTactic,
            DefinitiveListTactic, definitive_tactic, ListItem, format_item_list, struct_lit_shape,
            struct_lit_tactic, shape_for_tactic, struct_lit_formatting};
use string::{StringFormat, rewrite_string};
use utils::{extra_offset, last_line_width, wrap_str, binary_search, first_line_width,
            semicolon_for_stmt, trimmed_last_line_width, left_most_sub_expr, stmt_expr,
            colon_spaces, contains_skip, mk_sp};
use visitor::FmtVisitor;
use config::{Config, IndentStyle, MultilineStyle, ControlBraceStyle, Style};
use comment::{FindUncommented, rewrite_comment, contains_comment, recover_comment_removed};
use types::{rewrite_path, PathContext};
use items::{span_lo_for_arg, span_hi_for_arg};
use chains::rewrite_chain;
use macros::{rewrite_macro, MacroPosition};

use syntax::{ast, ptr};
use syntax::codemap::{CodeMap, Span, BytePos};
use syntax::parse::classify;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        format_expr(self, ExprType::SubExpression, context, shape)
    }
}

#[derive(PartialEq)]
enum ExprType {
    Statement,
    SubExpression,
}

fn combine_attr_and_expr(
    context: &RewriteContext,
    shape: Shape,
    attr_str: &str,
    expr_str: &str,
) -> String {
    let separator = if attr_str.is_empty() {
        String::new()
    } else {
        if expr_str.contains('\n') || attr_str.contains('\n') ||
            attr_str.len() + expr_str.len() > shape.width
        {
            format!("\n{}", shape.indent.to_string(context.config))
        } else {
            String::from(" ")
        }
    };
    format!("{}{}{}", attr_str, separator, expr_str)
}

fn format_expr(
    expr: &ast::Expr,
    expr_type: ExprType,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let attr_rw = (&*expr.attrs).rewrite(context, shape);
    if contains_skip(&*expr.attrs) {
        if let Some(attr_str) = attr_rw {
            return Some(combine_attr_and_expr(
                context,
                shape,
                &attr_str,
                &context.snippet(expr.span),
            ));
        } else {
            return Some(context.snippet(expr.span));
        }
    }
    let expr_rw = match expr.node {
        ast::ExprKind::Array(ref expr_vec) => {
            rewrite_array(
                expr_vec.iter().map(|e| &**e),
                mk_sp(context.codemap.span_after(expr.span, "["), expr.span.hi),
                context,
                shape,
            )
        }
        ast::ExprKind::Lit(ref l) => {
            match l.node {
                ast::LitKind::Str(_, ast::StrStyle::Cooked) => {
                    rewrite_string_lit(context, l.span, shape)
                }
                _ => {
                    wrap_str(
                        context.snippet(expr.span),
                        context.config.max_width(),
                        shape,
                    )
                }
            }
        }
        ast::ExprKind::Call(ref callee, ref args) => {
            let inner_span = mk_sp(callee.span.hi, expr.span.hi);
            rewrite_call_with_binary_search(context, &**callee, args, inner_span, shape)
        }
        ast::ExprKind::Paren(ref subexpr) => rewrite_paren(context, subexpr, shape),
        ast::ExprKind::Binary(ref op, ref lhs, ref rhs) => {
            // FIXME: format comments between operands and operator
            rewrite_pair(
                &**lhs,
                &**rhs,
                "",
                &format!(" {} ", context.snippet(op.span)),
                "",
                context,
                shape,
            )
        }
        ast::ExprKind::Unary(ref op, ref subexpr) => rewrite_unary_op(context, op, subexpr, shape),
        ast::ExprKind::Struct(ref path, ref fields, ref base) => {
            rewrite_struct_lit(
                context,
                path,
                fields,
                base.as_ref().map(|e| &**e),
                expr.span,
                shape,
            )
        }
        ast::ExprKind::Tup(ref items) => rewrite_tuple(context, items, expr.span, shape),
        ast::ExprKind::While(ref cond, ref block, label) => {
            ControlFlow::new_while(None, cond, block, label, expr.span).rewrite(context, shape)
        }
        ast::ExprKind::WhileLet(ref pat, ref cond, ref block, label) => {
            ControlFlow::new_while(Some(pat), cond, block, label, expr.span).rewrite(context, shape)
        }
        ast::ExprKind::ForLoop(ref pat, ref cond, ref block, label) => {
            ControlFlow::new_for(pat, cond, block, label, expr.span).rewrite(context, shape)
        }
        ast::ExprKind::Loop(ref block, label) => {
            ControlFlow::new_loop(block, label, expr.span).rewrite(context, shape)
        }
        ast::ExprKind::Block(ref block) => block.rewrite(context, shape),
        ast::ExprKind::If(ref cond, ref if_block, ref else_block) => {
            ControlFlow::new_if(
                cond,
                None,
                if_block,
                else_block.as_ref().map(|e| &**e),
                expr_type == ExprType::SubExpression,
                false,
                expr.span,
            ).rewrite(context, shape)
        }
        ast::ExprKind::IfLet(ref pat, ref cond, ref if_block, ref else_block) => {
            ControlFlow::new_if(
                cond,
                Some(pat),
                if_block,
                else_block.as_ref().map(|e| &**e),
                expr_type == ExprType::SubExpression,
                false,
                expr.span,
            ).rewrite(context, shape)
        }
        ast::ExprKind::Match(ref cond, ref arms) => {
            rewrite_match(context, cond, arms, shape, expr.span)
        }
        ast::ExprKind::Path(ref qself, ref path) => {
            rewrite_path(context, PathContext::Expr, qself.as_ref(), path, shape)
        }
        ast::ExprKind::Assign(ref lhs, ref rhs) => {
            rewrite_assignment(context, lhs, rhs, None, shape)
        }
        ast::ExprKind::AssignOp(ref op, ref lhs, ref rhs) => {
            rewrite_assignment(context, lhs, rhs, Some(op), shape)
        }
        ast::ExprKind::Continue(ref opt_ident) => {
            let id_str = match *opt_ident {
                Some(ident) => format!(" {}", ident.node),
                None => String::new(),
            };
            wrap_str(
                format!("continue{}", id_str),
                context.config.max_width(),
                shape,
            )
        }
        ast::ExprKind::Break(ref opt_ident, ref opt_expr) => {
            let id_str = match *opt_ident {
                Some(ident) => format!(" {}", ident.node),
                None => String::new(),
            };

            if let Some(ref expr) = *opt_expr {
                rewrite_unary_prefix(context, &format!("break{} ", id_str), &**expr, shape)
            } else {
                wrap_str(
                    format!("break{}", id_str),
                    context.config.max_width(),
                    shape,
                )
            }
        }
        ast::ExprKind::Closure(capture, ref fn_decl, ref body, _) => {
            rewrite_closure(capture, fn_decl, body, expr.span, context, shape)
        }
        ast::ExprKind::Try(..) |
        ast::ExprKind::Field(..) |
        ast::ExprKind::TupField(..) |
        ast::ExprKind::MethodCall(..) => rewrite_chain(expr, context, shape),
        ast::ExprKind::Mac(ref mac) => {
            // Failure to rewrite a marco should not imply failure to
            // rewrite the expression.
            rewrite_macro(mac, None, context, shape, MacroPosition::Expression).or_else(|| {
                wrap_str(
                    context.snippet(expr.span),
                    context.config.max_width(),
                    shape,
                )
            })
        }
        ast::ExprKind::Ret(None) => {
            wrap_str("return".to_owned(), context.config.max_width(), shape)
        }
        ast::ExprKind::Ret(Some(ref expr)) => {
            rewrite_unary_prefix(context, "return ", &**expr, shape)
        }
        ast::ExprKind::Box(ref expr) => rewrite_unary_prefix(context, "box ", &**expr, shape),
        ast::ExprKind::AddrOf(mutability, ref expr) => {
            rewrite_expr_addrof(context, mutability, expr, shape)
        }
        ast::ExprKind::Cast(ref expr, ref ty) => {
            rewrite_pair(&**expr, &**ty, "", " as ", "", context, shape)
        }
        ast::ExprKind::Type(ref expr, ref ty) => {
            rewrite_pair(&**expr, &**ty, "", ": ", "", context, shape)
        }
        ast::ExprKind::Index(ref expr, ref index) => {
            rewrite_index(&**expr, &**index, context, shape)
        }
        ast::ExprKind::Repeat(ref expr, ref repeats) => {
            let (lbr, rbr) = if context.config.spaces_within_square_brackets() {
                ("[ ", " ]")
            } else {
                ("[", "]")
            };
            rewrite_pair(&**expr, &**repeats, lbr, "; ", rbr, context, shape)
        }
        ast::ExprKind::Range(ref lhs, ref rhs, limits) => {
            let delim = match limits {
                ast::RangeLimits::HalfOpen => "..",
                ast::RangeLimits::Closed => "...",
            };

            match (lhs.as_ref().map(|x| &**x), rhs.as_ref().map(|x| &**x)) {
                (Some(ref lhs), Some(ref rhs)) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!(" {} ", delim)
                    } else {
                        delim.into()
                    };
                    rewrite_pair(&**lhs, &**rhs, "", &sp_delim, "", context, shape)
                }
                (None, Some(ref rhs)) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!("{} ", delim)
                    } else {
                        delim.into()
                    };
                    rewrite_unary_prefix(context, &sp_delim, &**rhs, shape)
                }
                (Some(ref lhs), None) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!(" {}", delim)
                    } else {
                        delim.into()
                    };
                    rewrite_unary_suffix(context, &sp_delim, &**lhs, shape)
                }
                (None, None) => wrap_str(delim.into(), context.config.max_width(), shape),
            }
        }
        // We do not format these expressions yet, but they should still
        // satisfy our width restrictions.
        ast::ExprKind::InPlace(..) |
        ast::ExprKind::InlineAsm(..) => {
            wrap_str(
                context.snippet(expr.span),
                context.config.max_width(),
                shape,
            )
        }
        ast::ExprKind::Catch(ref block) => {
            if let rewrite @ Some(_) = try_one_line_block(context, shape, "do catch ", block) {
                return rewrite;
            }
            // 9 = `do catch `
            let budget = shape.width.checked_sub(9).unwrap_or(0);
            Some(format!(
                "{}{}",
                "do catch ",
                try_opt!(
                    block.rewrite(&context, Shape::legacy(budget, shape.indent))
                )
            ))
        }
    };
    match (attr_rw, expr_rw) {
        (Some(attr_str), Some(expr_str)) => {
            recover_comment_removed(
                combine_attr_and_expr(context, shape, &attr_str, &expr_str),
                expr.span,
                context,
                shape,
            )
        }
        _ => None,
    }
}

fn try_one_line_block(
    context: &RewriteContext,
    shape: Shape,
    prefix: &str,
    block: &ast::Block,
) -> Option<String> {
    if is_simple_block(block, context.codemap) {
        let expr_shape = Shape::legacy(shape.width - prefix.len(), shape.indent);
        let expr_str = try_opt!(block.stmts[0].rewrite(context, expr_shape));
        let result = format!("{}{{ {} }}", prefix, expr_str);
        if result.len() <= shape.width && !result.contains('\n') {
            return Some(result);
        }
    }
    None
}

pub fn rewrite_pair<LHS, RHS>(
    lhs: &LHS,
    rhs: &RHS,
    prefix: &str,
    infix: &str,
    suffix: &str,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String>
where
    LHS: Rewrite,
    RHS: Rewrite,
{
    // Get "full width" rhs and see if it fits on the current line. This
    // usually works fairly well since it tends to place operands of
    // operations with high precendence close together.
    // Note that this is non-conservative, but its just to see if it's even
    // worth trying to put everything on one line.
    let rhs_shape = try_opt!(shape.sub_width(suffix.len()));
    let rhs_result = rhs.rewrite(context, rhs_shape);

    if let Some(rhs_result) = rhs_result {
        // This is needed in case of line break not caused by a
        // shortage of space, but by end-of-line comments, for example.
        if !rhs_result.contains('\n') {
            let lhs_shape = try_opt!(try_opt!(shape.offset_left(prefix.len())).sub_width(
                infix.len(),
            ));
            let lhs_result = lhs.rewrite(context, lhs_shape);
            if let Some(lhs_result) = lhs_result {
                let mut result = format!("{}{}{}", prefix, lhs_result, infix);

                let remaining_width = shape
                    .width
                    .checked_sub(last_line_width(&result) + suffix.len())
                    .unwrap_or(0);

                if rhs_result.len() <= remaining_width {
                    result.push_str(&rhs_result);
                    result.push_str(suffix);
                    return Some(result);
                }

                // Try rewriting the rhs into the remaining space.
                let rhs_shape = shape.shrink_left(last_line_width(&result) + suffix.len());
                if let Some(rhs_shape) = rhs_shape {
                    if let Some(rhs_result) = rhs.rewrite(context, rhs_shape) {
                        // FIXME this should always hold.
                        if rhs_result.len() <= remaining_width {
                            result.push_str(&rhs_result);
                            result.push_str(suffix);
                            return Some(result);
                        }
                    }
                }
            }
        }
    }

    // We have to use multiple lines.

    // Re-evaluate the rhs because we have more space now:
    let infix = infix.trim_right();
    let rhs_shape = match context.config.control_style() {
        Style::Legacy => {
            try_opt!(shape.sub_width(suffix.len() + prefix.len())).visual_indent(prefix.len())
        }
        Style::Rfc => {
            // Try to calculate the initial constraint on the right hand side.
            let rhs_overhead = context
                .config
                .max_width()
                .checked_sub(shape.used_width() + shape.width)
                .unwrap_or(0);
            try_opt!(
                Shape::indented(shape.indent.block_indent(context.config), context.config)
                    .sub_width(rhs_overhead)
            )
        }
    };
    let rhs_result = try_opt!(rhs.rewrite(context, rhs_shape));
    let lhs_overhead = shape.used_width() + prefix.len() + infix.len();
    let lhs_shape = Shape {
        width: try_opt!(context.config.max_width().checked_sub(lhs_overhead)),
        ..shape
    };
    let lhs_result = try_opt!(lhs.rewrite(context, lhs_shape));
    Some(format!(
        "{}{}{}\n{}{}{}",
        prefix,
        lhs_result,
        infix,
        rhs_shape.indent.to_string(context.config),
        rhs_result,
        suffix
    ))
}

pub fn rewrite_array<'a, I>(
    expr_iter: I,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String>
where
    I: Iterator<Item = &'a ast::Expr>,
{
    let bracket_size = if context.config.spaces_within_square_brackets() {
        2 // "[ "
    } else {
        1 // "["
    };

    let nested_shape = match context.config.array_layout() {
        IndentStyle::Block => shape.block().block_indent(context.config.tab_spaces()),
        IndentStyle::Visual => {
            try_opt!(shape.visual_indent(bracket_size).sub_width(
                bracket_size * 2,
            ))
        }
    };

    let items = itemize_list(
        context.codemap,
        expr_iter,
        "]",
        |item| item.span.lo,
        |item| item.span.hi,
        |item| item.rewrite(context, nested_shape),
        span.lo,
        span.hi,
    ).collect::<Vec<_>>();

    if items.is_empty() {
        if context.config.spaces_within_square_brackets() {
            return Some("[ ]".to_string());
        } else {
            return Some("[]".to_string());
        }
    }

    let has_long_item = items.iter().any(|li| {
        li.item.as_ref().map(|s| s.len() > 10).unwrap_or(false)
    });

    let tactic = match context.config.array_layout() {
        IndentStyle::Block => {
            // FIXME wrong shape in one-line case
            match shape.width.checked_sub(2 * bracket_size) {
                Some(width) => {
                    let tactic =
                        ListTactic::LimitedHorizontalVertical(context.config.array_width());
                    definitive_tactic(&items, tactic, width)
                }
                None => DefinitiveListTactic::Vertical,
            }
        }
        IndentStyle::Visual => {
            if has_long_item || items.iter().any(ListItem::is_multiline) {
                definitive_tactic(
                    &items,
                    ListTactic::LimitedHorizontalVertical(context.config.array_width()),
                    nested_shape.width,
                )
            } else {
                DefinitiveListTactic::Mixed
            }
        }
    };

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        shape: nested_shape,
        ends_with_newline: false,
        config: context.config,
    };
    let list_str = try_opt!(write_list(&items, &fmt));

    let result = if context.config.array_layout() == IndentStyle::Visual ||
        tactic != DefinitiveListTactic::Vertical
    {
        if context.config.spaces_within_square_brackets() && list_str.len() > 0 {
            format!("[ {} ]", list_str)
        } else {
            format!("[{}]", list_str)
        }
    } else {
        format!(
            "[\n{}{},\n{}]",
            nested_shape.indent.to_string(context.config),
            list_str,
            shape.block().indent.to_string(context.config)
        )
    };

    Some(result)
}

// This functions is pretty messy because of the rules around closures and blocks:
// FIXME - the below is probably no longer true in full.
//   * if there is a return type, then there must be braces,
//   * given a closure with braces, whether that is parsed to give an inner block
//     or not depends on if there is a return type and if there are statements
//     in that block,
//   * if the first expression in the body ends with a block (i.e., is a
//     statement without needing a semi-colon), then adding or removing braces
//     can change whether it is treated as an expression or statement.
fn rewrite_closure(
    capture: ast::CaptureBy,
    fn_decl: &ast::FnDecl,
    body: &ast::Expr,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let mover = if capture == ast::CaptureBy::Value {
        "move "
    } else {
        ""
    };
    // 4 = "|| {".len(), which is overconservative when the closure consists of
    // a single expression.
    let nested_shape = try_opt!(try_opt!(shape.shrink_left(mover.len())).sub_width(4));

    // 1 = |
    let argument_offset = nested_shape.indent + 1;
    let arg_shape = try_opt!(nested_shape.shrink_left(1)).visual_indent(0);
    let ret_str = try_opt!(fn_decl.output.rewrite(context, arg_shape));

    let arg_items = itemize_list(
        context.codemap,
        fn_decl.inputs.iter(),
        "|",
        |arg| span_lo_for_arg(arg),
        |arg| span_hi_for_arg(arg),
        |arg| arg.rewrite(context, arg_shape),
        context.codemap.span_after(span, "|"),
        body.span.lo,
    );
    let item_vec = arg_items.collect::<Vec<_>>();
    // 1 = space between arguments and return type.
    let horizontal_budget = nested_shape
        .width
        .checked_sub(ret_str.len() + 1)
        .unwrap_or(0);
    let tactic = definitive_tactic(&item_vec, ListTactic::HorizontalVertical, horizontal_budget);
    let arg_shape = match tactic {
        DefinitiveListTactic::Horizontal => try_opt!(arg_shape.sub_width(ret_str.len() + 1)),
        _ => arg_shape,
    };

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        shape: arg_shape,
        ends_with_newline: false,
        config: context.config,
    };
    let list_str = try_opt!(write_list(&item_vec, &fmt));
    let mut prefix = format!("{}|{}|", mover, list_str);

    if !ret_str.is_empty() {
        if prefix.contains('\n') {
            prefix.push('\n');
            prefix.push_str(&argument_offset.to_string(context.config));
        } else {
            prefix.push(' ');
        }
        prefix.push_str(&ret_str);
    }

    // 1 = space between `|...|` and body.
    let extra_offset = extra_offset(&prefix, shape) + 1;
    let body_shape = try_opt!(shape.offset_left(extra_offset));

    if let ast::ExprKind::Block(ref block) = body.node {
        // The body of the closure is an empty block.
        if block.stmts.is_empty() && !block_contains_comment(block, context.codemap) {
            return Some(format!("{} {{}}", prefix));
        }

        // Figure out if the block is necessary.
        let needs_block = block.rules != ast::BlockCheckMode::Default ||
            block.stmts.len() > 1 || context.inside_macro ||
            block_contains_comment(block, context.codemap) ||
            prefix.contains('\n');

        if ret_str.is_empty() && !needs_block {
            // lock.stmts.len() == 1
            if let Some(ref expr) = stmt_expr(&block.stmts[0]) {
                if let Some(rw) = rewrite_closure_expr(expr, &prefix, context, body_shape) {
                    return Some(rw);
                }
            }
        }

        if !needs_block {
            // We need braces, but we might still prefer a one-liner.
            let stmt = &block.stmts[0];
            // 4 = braces and spaces.
            if let Some(body_shape) = body_shape.sub_width(4) {
                // Checks if rewrite succeeded and fits on a single line.
                if let Some(rewrite) = and_one_line(stmt.rewrite(context, body_shape)) {
                    return Some(format!("{} {{ {} }}", prefix, rewrite));
                }
            }
        }

        // Either we require a block, or tried without and failed.
        return rewrite_closure_block(&block, prefix, context, body_shape);
    }

    if let Some(rw) = rewrite_closure_expr(body, &prefix, context, body_shape) {
        return Some(rw);
    }

    // The closure originally had a non-block expression, but we can't fit on
    // one line, so we'll insert a block.
    let block = ast::Block {
        stmts: vec![
            ast::Stmt {
                id: ast::NodeId::new(0),
                node: ast::StmtKind::Expr(ptr::P(body.clone())),
                span: body.span,
            },
        ],
        id: ast::NodeId::new(0),
        rules: ast::BlockCheckMode::Default,
        span: body.span,
    };
    return rewrite_closure_block(&block, prefix, context, body_shape);

    fn rewrite_closure_expr(
        expr: &ast::Expr,
        prefix: &str,
        context: &RewriteContext,
        shape: Shape,
    ) -> Option<String> {
        let mut rewrite = expr.rewrite(context, shape);
        if classify::expr_requires_semi_to_be_stmt(left_most_sub_expr(expr)) {
            rewrite = and_one_line(rewrite);
        }
        rewrite.map(|rw| format!("{} {}", prefix, rw))
    }

    fn rewrite_closure_block(
        block: &ast::Block,
        prefix: String,
        context: &RewriteContext,
        shape: Shape,
    ) -> Option<String> {
        // Start with visual indent, then fall back to block indent if the
        // closure is large.
        let block_threshold = context.config.closure_block_indent_threshold();
        if block_threshold >= 0 {
            if let Some(block_str) = block.rewrite(&context, shape) {
                if block_str.matches('\n').count() <= block_threshold as usize &&
                    !need_block_indent(&block_str, shape)
                {
                    if let Some(block_str) = block_str.rewrite(context, shape) {
                        return Some(format!("{} {}", prefix, block_str));
                    }
                }
            }
        }

        // The body of the closure is big enough to be block indented, that
        // means we must re-format.
        let block_shape = shape.block().with_max_width(context.config);
        let block_str = try_opt!(block.rewrite(&context, block_shape));
        Some(format!("{} {}", prefix, block_str))
    }
}

fn and_one_line(x: Option<String>) -> Option<String> {
    x.and_then(|x| if x.contains('\n') { None } else { Some(x) })
}

fn nop_block_collapse(block_str: Option<String>, budget: usize) -> Option<String> {
    debug!("nop_block_collapse {:?} {}", block_str, budget);
    block_str.map(|block_str| if block_str.starts_with('{') && budget >= 2 &&
        (block_str[1..]
             .find(|c: char| !c.is_whitespace())
             .unwrap() == block_str.len() - 2)
    {
        "{}".to_owned()
    } else {
        block_str.to_owned()
    })
}

impl Rewrite for ast::Block {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        // shape.width is used only for the single line case: either the empty block `{}`,
        // or an unsafe expression `unsafe { e }`.

        if self.stmts.is_empty() && !block_contains_comment(self, context.codemap) &&
            shape.width >= 2
        {
            return Some("{}".to_owned());
        }

        // If a block contains only a single-line comment, then leave it on one line.
        let user_str = context.snippet(self.span);
        let user_str = user_str.trim();
        if user_str.starts_with('{') && user_str.ends_with('}') {
            let comment_str = user_str[1..user_str.len() - 1].trim();
            if self.stmts.is_empty() && !comment_str.contains('\n') &&
                !comment_str.starts_with("//") &&
                comment_str.len() + 4 <= shape.width
            {
                return Some(format!("{{ {} }}", comment_str));
            }
        }

        let mut visitor = FmtVisitor::from_codemap(context.parse_session, context.config);
        visitor.block_indent = shape.indent;
        visitor.is_if_else_block = context.is_if_else_block;

        let prefix = match self.rules {
            ast::BlockCheckMode::Unsafe(..) => {
                let snippet = context.snippet(self.span);
                let open_pos = try_opt!(snippet.find_uncommented("{"));
                visitor.last_pos = self.span.lo + BytePos(open_pos as u32);

                // Extract comment between unsafe and block start.
                let trimmed = &snippet[6..open_pos].trim();

                let prefix = if !trimmed.is_empty() {
                    // 9 = "unsafe  {".len(), 7 = "unsafe ".len()
                    let budget = try_opt!(shape.width.checked_sub(9));
                    format!(
                        "unsafe {} ",
                        try_opt!(rewrite_comment(
                            trimmed,
                            true,
                            Shape::legacy(budget, shape.indent + 7),
                            context.config,
                        ))
                    )
                } else {
                    "unsafe ".to_owned()
                };
                if let result @ Some(_) = try_one_line_block(context, shape, &prefix, self) {
                    return result;
                }
                prefix
            }
            ast::BlockCheckMode::Default => {
                visitor.last_pos = self.span.lo;
                String::new()
            }
        };

        visitor.visit_block(self);
        if visitor.failed && shape.indent.alignment != 0 {
            self.rewrite(
                context,
                Shape::indented(shape.indent.block_only(), context.config),
            )
        } else {
            Some(format!("{}{}", prefix, visitor.buffer))
        }
    }
}

impl Rewrite for ast::Stmt {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let result = match self.node {
            ast::StmtKind::Local(ref local) => local.rewrite(context, shape),
            ast::StmtKind::Expr(ref ex) |
            ast::StmtKind::Semi(ref ex) => {
                let suffix = if semicolon_for_stmt(self) { ";" } else { "" };

                format_expr(
                    ex,
                    match self.node {
                        ast::StmtKind::Expr(_) => ExprType::SubExpression,
                        ast::StmtKind::Semi(_) => ExprType::Statement,
                        _ => unreachable!(),
                    },
                    context,
                    try_opt!(shape.sub_width(suffix.len())),
                ).map(|s| s + suffix)
            }
            ast::StmtKind::Mac(..) |
            ast::StmtKind::Item(..) => None,
        };
        result.and_then(|res| {
            recover_comment_removed(res, self.span, context, shape)
        })
    }
}

// Abstraction over control flow expressions
#[derive(Debug)]
struct ControlFlow<'a> {
    cond: Option<&'a ast::Expr>,
    block: &'a ast::Block,
    else_block: Option<&'a ast::Expr>,
    label: Option<ast::SpannedIdent>,
    pat: Option<&'a ast::Pat>,
    keyword: &'a str,
    matcher: &'a str,
    connector: &'a str,
    allow_single_line: bool,
    // True if this is an `if` expression in an `else if` :-( hacky
    nested_if: bool,
    span: Span,
}

impl<'a> ControlFlow<'a> {
    fn new_if(
        cond: &'a ast::Expr,
        pat: Option<&'a ast::Pat>,
        block: &'a ast::Block,
        else_block: Option<&'a ast::Expr>,
        allow_single_line: bool,
        nested_if: bool,
        span: Span,
    ) -> ControlFlow<'a> {
        ControlFlow {
            cond: Some(cond),
            block: block,
            else_block: else_block,
            label: None,
            pat: pat,
            keyword: "if",
            matcher: match pat {
                Some(..) => "let",
                None => "",
            },
            connector: " =",
            allow_single_line: allow_single_line,
            nested_if: nested_if,
            span: span,
        }
    }

    fn new_loop(
        block: &'a ast::Block,
        label: Option<ast::SpannedIdent>,
        span: Span,
    ) -> ControlFlow<'a> {
        ControlFlow {
            cond: None,
            block: block,
            else_block: None,
            label: label,
            pat: None,
            keyword: "loop",
            matcher: "",
            connector: "",
            allow_single_line: false,
            nested_if: false,
            span: span,
        }
    }

    fn new_while(
        pat: Option<&'a ast::Pat>,
        cond: &'a ast::Expr,
        block: &'a ast::Block,
        label: Option<ast::SpannedIdent>,
        span: Span,
    ) -> ControlFlow<'a> {
        ControlFlow {
            cond: Some(cond),
            block: block,
            else_block: None,
            label: label,
            pat: pat,
            keyword: "while",
            matcher: match pat {
                Some(..) => "let",
                None => "",
            },
            connector: " =",
            allow_single_line: false,
            nested_if: false,
            span: span,
        }
    }

    fn new_for(
        pat: &'a ast::Pat,
        cond: &'a ast::Expr,
        block: &'a ast::Block,
        label: Option<ast::SpannedIdent>,
        span: Span,
    ) -> ControlFlow<'a> {
        ControlFlow {
            cond: Some(cond),
            block: block,
            else_block: None,
            label: label,
            pat: Some(pat),
            keyword: "for",
            matcher: "",
            connector: " in",
            allow_single_line: false,
            nested_if: false,
            span: span,
        }
    }

    fn rewrite_single_line(
        &self,
        pat_expr_str: &str,
        context: &RewriteContext,
        width: usize,
    ) -> Option<String> {
        assert!(self.allow_single_line);
        let else_block = try_opt!(self.else_block);
        let fixed_cost = self.keyword.len() + "  {  } else {  }".len();

        if let ast::ExprKind::Block(ref else_node) = else_block.node {
            if !is_simple_block(self.block, context.codemap) ||
                !is_simple_block(else_node, context.codemap) ||
                pat_expr_str.contains('\n')
            {
                return None;
            }

            let new_width = try_opt!(width.checked_sub(pat_expr_str.len() + fixed_cost));
            let expr = &self.block.stmts[0];
            let if_str = try_opt!(expr.rewrite(
                context,
                Shape::legacy(new_width, Indent::empty()),
            ));

            let new_width = try_opt!(new_width.checked_sub(if_str.len()));
            let else_expr = &else_node.stmts[0];
            let else_str = try_opt!(else_expr.rewrite(
                context,
                Shape::legacy(new_width, Indent::empty()),
            ));

            if if_str.contains('\n') || else_str.contains('\n') {
                return None;
            }

            let result = format!(
                "{} {} {{ {} }} else {{ {} }}",
                self.keyword,
                pat_expr_str,
                if_str,
                else_str
            );

            if result.len() <= width {
                return Some(result);
            }
        }

        None
    }
}

impl<'a> Rewrite for ControlFlow<'a> {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        debug!("ControlFlow::rewrite {:?} {:?}", self, shape);
        let constr_shape = if self.nested_if {
            // We are part of an if-elseif-else chain. Our constraints are tightened.
            // 7 = "} else " .len()
            try_opt!(shape.shrink_left(7))
        } else {
            shape
        };

        let label_string = rewrite_label(self.label);
        // 1 = space after keyword.
        let offset = self.keyword.len() + label_string.len() + 1;

        let pat_expr_string = match self.cond {
            Some(cond) => {
                let mut cond_shape = match context.config.control_style() {
                    Style::Legacy => try_opt!(constr_shape.shrink_left(offset)),
                    Style::Rfc => try_opt!(constr_shape.offset_left(offset)),
                };
                if context.config.control_brace_style() != ControlBraceStyle::AlwaysNextLine {
                    // 2 = " {".len()
                    cond_shape = try_opt!(cond_shape.sub_width(2));
                }

                try_opt!(rewrite_pat_expr(
                    context,
                    self.pat,
                    cond,
                    self.matcher,
                    self.connector,
                    self.keyword,
                    cond_shape,
                ))
            }
            None => String::new(),
        };

        let force_newline_brace = context.config.control_style() == Style::Rfc &&
            pat_expr_string.contains('\n');

        // Try to format if-else on single line.
        if self.allow_single_line && context.config.single_line_if_else_max_width() > 0 {
            let trial = self.rewrite_single_line(&pat_expr_string, context, shape.width);

            if trial.is_some() &&
                trial.as_ref().unwrap().len() <= context.config.single_line_if_else_max_width()
            {
                return trial;
            }
        }

        let used_width = if pat_expr_string.contains('\n') {
            last_line_width(&pat_expr_string)
        } else {
            // 2 = spaces after keyword and condition.
            label_string.len() + self.keyword.len() + pat_expr_string.len() + 2
        };

        let block_width = shape.width.checked_sub(used_width).unwrap_or(0);
        // This is used only for the empty block case: `{}`. So, we use 1 if we know
        // we should avoid the single line case.
        let block_width = if self.else_block.is_some() || self.nested_if {
            min(1, block_width)
        } else {
            block_width
        };

        let block_shape = Shape {
            width: block_width,
            ..shape
        };
        let mut block_context = context.clone();
        block_context.is_if_else_block = self.else_block.is_some();

        let block_str = try_opt!(self.block.rewrite(&block_context, block_shape));

        let cond_span = if let Some(cond) = self.cond {
            cond.span
        } else {
            mk_sp(self.block.span.lo, self.block.span.lo)
        };

        // for event in event
        let between_kwd_cond = mk_sp(
            context.codemap.span_after(self.span, self.keyword.trim()),
            self.pat.map_or(
                cond_span.lo,
                |p| if self.matcher.is_empty() {
                    p.span.lo
                } else {
                    context.codemap.span_before(self.span, self.matcher.trim())
                },
            ),
        );

        let between_kwd_cond_comment = extract_comment(between_kwd_cond, context, shape);

        let after_cond_comment =
            extract_comment(mk_sp(cond_span.hi, self.block.span.lo), context, shape);

        let alt_block_sep = String::from("\n") +
            &shape.indent.block_only().to_string(context.config);
        let block_sep = if self.cond.is_none() && between_kwd_cond_comment.is_some() {
            ""
        } else if context.config.control_brace_style() == ControlBraceStyle::AlwaysNextLine ||
                   force_newline_brace
        {
            alt_block_sep.as_str()
        } else {
            " "
        };

        let mut result =
            format!("{}{}{}{}{}{}",
                                 label_string,
                                 self.keyword,
                                 between_kwd_cond_comment
                                     .as_ref()
                                     .map_or(if pat_expr_string.is_empty() ||
                                                pat_expr_string.starts_with('\n') {
                                                 ""
                                             } else {
                                                 " "
                                             },
                                             |s| &**s),
                                 pat_expr_string,
                                 after_cond_comment.as_ref().map_or(block_sep, |s| &**s),
                                 block_str);

        if let Some(else_block) = self.else_block {
            let shape = Shape::indented(shape.indent, context.config);
            let mut last_in_chain = false;
            let rewrite = match else_block.node {
                // If the else expression is another if-else expression, prevent it
                // from being formatted on a single line.
                // Note how we're passing the original shape, as the
                // cost of "else" should not cascade.
                ast::ExprKind::IfLet(ref pat, ref cond, ref if_block, ref next_else_block) => {
                    ControlFlow::new_if(
                        cond,
                        Some(pat),
                        if_block,
                        next_else_block.as_ref().map(|e| &**e),
                        false,
                        true,
                        mk_sp(else_block.span.lo, self.span.hi),
                    ).rewrite(context, shape)
                }
                ast::ExprKind::If(ref cond, ref if_block, ref next_else_block) => {
                    ControlFlow::new_if(
                        cond,
                        None,
                        if_block,
                        next_else_block.as_ref().map(|e| &**e),
                        false,
                        true,
                        mk_sp(else_block.span.lo, self.span.hi),
                    ).rewrite(context, shape)
                }
                _ => {
                    last_in_chain = true;
                    // When rewriting a block, the width is only used for single line
                    // blocks, passing 1 lets us avoid that.
                    let else_shape = Shape {
                        width: min(1, shape.width),
                        ..shape
                    };
                    else_block.rewrite(context, else_shape)
                }
            };

            let between_kwd_else_block =
                mk_sp(
                    self.block.span.hi,
                    context.codemap.span_before(
                        mk_sp(self.block.span.hi, else_block.span.lo),
                        "else",
                    ),
                );
            let between_kwd_else_block_comment =
                extract_comment(between_kwd_else_block, context, shape);

            let after_else = mk_sp(
                context.codemap.span_after(
                    mk_sp(self.block.span.hi, else_block.span.lo),
                    "else",
                ),
                else_block.span.lo,
            );
            let after_else_comment = extract_comment(after_else, context, shape);

            let between_sep = match context.config.control_brace_style() {
                ControlBraceStyle::AlwaysNextLine |
                ControlBraceStyle::ClosingNextLine => &*alt_block_sep,
                ControlBraceStyle::AlwaysSameLine => " ",
            };
            let after_sep = match context.config.control_brace_style() {
                ControlBraceStyle::AlwaysNextLine if last_in_chain => &*alt_block_sep,
                _ => " ",
            };
            try_opt!(
                write!(
                    &mut result,
                    "{}else{}",
                    between_kwd_else_block_comment.as_ref().map_or(
                        between_sep,
                        |s| &**s,
                    ),
                    after_else_comment.as_ref().map_or(after_sep, |s| &**s)
                ).ok()
            );
            result.push_str(&try_opt!(rewrite));
        }

        Some(result)
    }
}

fn rewrite_label(label: Option<ast::SpannedIdent>) -> String {
    match label {
        Some(ident) => format!("{}: ", ident.node),
        None => "".to_owned(),
    }
}

fn extract_comment(span: Span, context: &RewriteContext, shape: Shape) -> Option<String> {
    let comment_str = context.snippet(span);
    if contains_comment(&comment_str) {
        let comment = try_opt!(rewrite_comment(
            comment_str.trim(),
            false,
            shape,
            context.config,
        ));
        Some(format!(
            "\n{indent}{}\n{indent}",
            comment,
            indent = shape.indent.to_string(context.config)
        ))
    } else {
        None
    }
}

fn block_contains_comment(block: &ast::Block, codemap: &CodeMap) -> bool {
    let snippet = codemap.span_to_snippet(block.span).unwrap();
    contains_comment(&snippet)
}

// Checks that a block contains no statements, an expression and no comments.
// FIXME: incorrectly returns false when comment is contained completely within
// the expression.
pub fn is_simple_block(block: &ast::Block, codemap: &CodeMap) -> bool {
    (block.stmts.len() == 1 && stmt_is_expr(&block.stmts[0]) &&
         !block_contains_comment(block, codemap))
}

/// Checks whether a block contains at most one statement or expression, and no comments.
pub fn is_simple_block_stmt(block: &ast::Block, codemap: &CodeMap) -> bool {
    block.stmts.len() <= 1 && !block_contains_comment(block, codemap)
}

/// Checks whether a block contains no statements, expressions, or comments.
pub fn is_empty_block(block: &ast::Block, codemap: &CodeMap) -> bool {
    block.stmts.is_empty() && !block_contains_comment(block, codemap)
}

pub fn stmt_is_expr(stmt: &ast::Stmt) -> bool {
    match stmt.node {
        ast::StmtKind::Expr(..) => true,
        _ => false,
    }
}

fn is_unsafe_block(block: &ast::Block) -> bool {
    if let ast::BlockCheckMode::Unsafe(..) = block.rules {
        true
    } else {
        false
    }
}

// inter-match-arm-comment-rules:
//  - all comments following a match arm before the start of the next arm
//    are about the second arm
fn rewrite_match_arm_comment(
    context: &RewriteContext,
    missed_str: &str,
    shape: Shape,
    arm_indent_str: &str,
) -> Option<String> {
    // The leading "," is not part of the arm-comment
    let missed_str = match missed_str.find_uncommented(",") {
        Some(n) => &missed_str[n + 1..],
        None => &missed_str[..],
    };

    let mut result = String::new();
    // any text not preceeded by a newline is pushed unmodified to the block
    let first_brk = missed_str.find(|c: char| c == '\n').unwrap_or(0);
    result.push_str(&missed_str[..first_brk]);
    let missed_str = &missed_str[first_brk..]; // If missed_str had one newline, it starts with it

    let first = missed_str.find(|c: char| !c.is_whitespace()).unwrap_or(
        missed_str
            .len(),
    );
    if missed_str[..first].chars().filter(|c| c == &'\n').count() >= 2 {
        // Excessive vertical whitespace before comment should be preserved
        // FIXME handle vertical whitespace better
        result.push('\n');
    }
    let missed_str = missed_str[first..].trim();
    if !missed_str.is_empty() {
        let comment = try_opt!(rewrite_comment(&missed_str, false, shape, context.config));
        result.push('\n');
        result.push_str(arm_indent_str);
        result.push_str(&comment);
    }

    Some(result)
}

fn rewrite_match(
    context: &RewriteContext,
    cond: &ast::Expr,
    arms: &[ast::Arm],
    shape: Shape,
    span: Span,
) -> Option<String> {
    if arms.is_empty() {
        return None;
    }

    // `match `cond` {`
    let cond_shape = match context.config.control_style() {
        Style::Legacy => try_opt!(shape.shrink_left(6).and_then(|s| s.sub_width(2))),
        Style::Rfc => try_opt!(shape.offset_left(8)),
    };
    let cond_str = try_opt!(cond.rewrite(context, cond_shape));
    let alt_block_sep = String::from("\n") + &shape.indent.block_only().to_string(context.config);
    let block_sep = match context.config.control_brace_style() {
        ControlBraceStyle::AlwaysSameLine => " ",
        _ => alt_block_sep.as_str(),
    };
    let mut result = format!("match {}{}{{", cond_str, block_sep);

    let arm_shape = if context.config.indent_match_arms() {
        shape.block_indent(context.config.tab_spaces())
    } else {
        shape.block_indent(0)
    };

    let arm_indent_str = arm_shape.indent.to_string(context.config);

    let open_brace_pos = context.codemap.span_after(
        mk_sp(cond.span.hi, arm_start_pos(&arms[0])),
        "{",
    );

    for (i, arm) in arms.iter().enumerate() {
        // Make sure we get the stuff between arms.
        let missed_str = if i == 0 {
            context.snippet(mk_sp(open_brace_pos, arm_start_pos(arm)))
        } else {
            context.snippet(mk_sp(arm_end_pos(&arms[i - 1]), arm_start_pos(arm)))
        };
        let comment = try_opt!(rewrite_match_arm_comment(
            context,
            &missed_str,
            arm_shape,
            &arm_indent_str,
        ));
        result.push_str(&comment);
        result.push('\n');
        result.push_str(&arm_indent_str);

        let arm_str = arm.rewrite(&context, arm_shape.with_max_width(context.config));
        if let Some(ref arm_str) = arm_str {
            result.push_str(arm_str);
        } else {
            // We couldn't format the arm, just reproduce the source.
            let snippet = context.snippet(mk_sp(arm_start_pos(arm), arm_end_pos(arm)));
            result.push_str(&snippet);
            result.push_str(arm_comma(context.config, &arm.body));
        }
    }
    // BytePos(1) = closing match brace.
    let last_span = mk_sp(arm_end_pos(&arms[arms.len() - 1]), span.hi - BytePos(1));
    let last_comment = context.snippet(last_span);
    let comment = try_opt!(rewrite_match_arm_comment(
        context,
        &last_comment,
        arm_shape,
        &arm_indent_str,
    ));
    result.push_str(&comment);
    result.push('\n');
    result.push_str(&shape.indent.to_string(context.config));
    result.push('}');
    Some(result)
}

fn arm_start_pos(arm: &ast::Arm) -> BytePos {
    let &ast::Arm {
        ref attrs,
        ref pats,
        ..
    } = arm;
    if !attrs.is_empty() {
        return attrs[0].span.lo;
    }

    pats[0].span.lo
}

fn arm_end_pos(arm: &ast::Arm) -> BytePos {
    arm.body.span.hi
}

fn arm_comma(config: &Config, body: &ast::Expr) -> &'static str {
    if config.match_block_trailing_comma() {
        ","
    } else if let ast::ExprKind::Block(ref block) = body.node {
        if let ast::BlockCheckMode::Default = block.rules {
            ""
        } else {
            ","
        }
    } else {
        ","
    }
}

// Match arms.
impl Rewrite for ast::Arm {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        debug!("Arm::rewrite {:?} {:?}", self, shape);
        let &ast::Arm {
            ref attrs,
            ref pats,
            ref guard,
            ref body,
        } = self;

        let attr_str = if !attrs.is_empty() {
            if contains_skip(attrs) {
                return None;
            }
            format!(
                "{}\n{}",
                try_opt!(attrs.rewrite(context, shape)),
                shape.indent.to_string(context.config)
            )
        } else {
            String::new()
        };

        // Patterns
        // 5 = ` => {`
        let pat_shape = try_opt!(shape.sub_width(5));

        let pat_strs = try_opt!(
            pats.iter()
                .map(|p| p.rewrite(context, pat_shape))
                .collect::<Option<Vec<_>>>()
        );

        let all_simple = pat_strs.iter().all(|p| pat_is_simple(p));
        let items: Vec<_> = pat_strs.into_iter().map(ListItem::from_str).collect();
        let fmt = ListFormatting {
            tactic: if all_simple {
                DefinitiveListTactic::Mixed
            } else {
                DefinitiveListTactic::Vertical
            },
            separator: " |",
            trailing_separator: SeparatorTactic::Never,
            shape: pat_shape,
            ends_with_newline: false,
            config: context.config,
        };
        let pats_str = try_opt!(write_list(items, &fmt));

        let guard_shape = if pats_str.contains('\n') {
            shape.with_max_width(context.config)
        } else {
            shape
        };

        let guard_str = try_opt!(rewrite_guard(
            context,
            guard,
            guard_shape,
            trimmed_last_line_width(&pats_str),
        ));

        let pats_str = format!("{}{}", pats_str, guard_str);

        let (mut extend, body) = match body.node {
            ast::ExprKind::Block(ref block)
                if !is_unsafe_block(block) && is_simple_block(block, context.codemap) &&
                       context.config.wrap_match_arms() => {
                if let ast::StmtKind::Expr(ref expr) = block.stmts[0].node {
                    (false, &**expr)
                } else {
                    (false, &**body)
                }
            }
            ast::ExprKind::Call(_, ref args) => (args.len() == 1, &**body),
            ast::ExprKind::Closure(..) |
            ast::ExprKind::Struct(..) |
            ast::ExprKind::Tup(..) => (true, &**body),
            _ => (false, &**body),
        };
        extend &= context.use_block_indent();

        let comma = arm_comma(&context.config, body);
        let alt_block_sep = String::from("\n") +
            &shape.indent.block_only().to_string(context.config);

        let pat_width = extra_offset(&pats_str, shape);
        // Let's try and get the arm body on the same line as the condition.
        // 4 = ` => `.len()
        if shape.width > pat_width + comma.len() + 4 {
            let arm_shape = shape
                .offset_left(pat_width + 4)
                .unwrap()
                .sub_width(comma.len())
                .unwrap();
            let rewrite = nop_block_collapse(body.rewrite(context, arm_shape), arm_shape.width);
            let is_block = if let ast::ExprKind::Block(..) = body.node {
                true
            } else {
                false
            };

            match rewrite {
                Some(ref body_str)
                    if (!body_str.contains('\n') && body_str.len() <= arm_shape.width) ||
                           !context.config.wrap_match_arms() ||
                           (extend && first_line_width(body_str) <= arm_shape.width) ||
                           is_block => {
                    let block_sep = match context.config.control_brace_style() {
                        ControlBraceStyle::AlwaysNextLine if is_block => alt_block_sep.as_str(),
                        _ => " ",
                    };

                    return Some(format!(
                        "{}{} =>{}{}{}",
                        attr_str.trim_left(),
                        pats_str,
                        block_sep,
                        body_str,
                        comma
                    ));
                }
                _ => {}
            }
        }

        // FIXME: we're doing a second rewrite of the expr; This may not be
        // necessary.
        let body_shape = try_opt!(shape.block_left(context.config.tab_spaces()));
        let next_line_body = try_opt!(nop_block_collapse(
            body.rewrite(context, body_shape),
            body_shape.width,
        ));
        let indent_str = shape.indent.block_indent(context.config).to_string(
            context.config,
        );
        let (body_prefix, body_suffix) = if context.config.wrap_match_arms() {
            if context.config.match_block_trailing_comma() {
                ("{", "},")
            } else {
                ("{", "}")
            }
        } else {
            ("", ",")
        };


        let block_sep = match context.config.control_brace_style() {
            ControlBraceStyle::AlwaysNextLine => alt_block_sep + body_prefix + "\n",
            _ if body_prefix.is_empty() => "\n".to_owned(),
            _ => " ".to_owned() + body_prefix + "\n",
        };

        if context.config.wrap_match_arms() {
            Some(format!(
                "{}{} =>{}{}{}\n{}{}",
                attr_str.trim_left(),
                pats_str,
                block_sep,
                indent_str,
                next_line_body,
                shape.indent.to_string(context.config),
                body_suffix
            ))
        } else {
            Some(format!(
                "{}{} =>{}{}{}{}",
                attr_str.trim_left(),
                pats_str,
                block_sep,
                indent_str,
                next_line_body,
                body_suffix
            ))
        }
    }
}

// A pattern is simple if it is very short or it is short-ish and just a path.
// E.g. `Foo::Bar` is simple, but `Foo(..)` is not.
fn pat_is_simple(pat_str: &str) -> bool {
    pat_str.len() <= 16 ||
        (pat_str.len() <= 24 && pat_str.chars().all(|c| c.is_alphabetic() || c == ':'))
}

// The `if ...` guard on a match arm.
fn rewrite_guard(
    context: &RewriteContext,
    guard: &Option<ptr::P<ast::Expr>>,
    shape: Shape,
    // The amount of space used up on this line for the pattern in
    // the arm (excludes offset).
    pattern_width: usize,
) -> Option<String> {
    if let Some(ref guard) = *guard {
        // First try to fit the guard string on the same line as the pattern.
        // 4 = ` if `, 5 = ` => {`
        if let Some(cond_shape) = shape.shrink_left(pattern_width + 4).and_then(
            |s| s.sub_width(5),
        )
        {
            if let Some(cond_str) = guard.rewrite(context, cond_shape).and_then(|s| {
                s.rewrite(context, cond_shape)
            })
            {
                if !cond_str.contains('\n') {
                    return Some(format!(" if {}", cond_str));
                }
            }
        }

        // Not enough space to put the guard after the pattern, try a newline.
        // 3 == `if `
        if let Some(cond_shape) = Shape::indented(
            shape.indent.block_indent(context.config) + 3,
            context.config,
        ).sub_width(3)
        {
            if let Some(cond_str) = guard.rewrite(context, cond_shape) {
                return Some(format!(
                    "\n{}if {}",
                    shape.indent.block_indent(context.config).to_string(
                        context.config,
                    ),
                    cond_str
                ));
            }
        }

        None
    } else {
        Some(String::new())
    }
}

fn rewrite_pat_expr(
    context: &RewriteContext,
    pat: Option<&ast::Pat>,
    expr: &ast::Expr,
    matcher: &str,
    // Connecting piece between pattern and expression,
    // *without* trailing space.
    connector: &str,
    keyword: &str,
    shape: Shape,
) -> Option<String> {
    debug!("rewrite_pat_expr {:?} {:?} {:?}", shape, pat, expr);
    let mut pat_string = String::new();
    let mut result = match pat {
        Some(pat) => {
            let matcher = if matcher.is_empty() {
                matcher.to_owned()
            } else {
                format!("{} ", matcher)
            };
            let pat_shape = try_opt!(try_opt!(shape.offset_left(matcher.len())).sub_width(
                connector.len(),
            ));
            pat_string = try_opt!(pat.rewrite(context, pat_shape));
            format!("{}{}{}", matcher, pat_string, connector)
        }
        None => String::new(),
    };

    // Consider only the last line of the pat string.
    let extra_offset = extra_offset(&result, shape);

    // The expression may (partially) fit on the current line.
    if shape.width > extra_offset + 1 {
        let spacer = if pat.is_some() { " " } else { "" };

        let expr_shape = try_opt!(shape.offset_left(extra_offset + spacer.len()));
        let expr_rewrite = expr.rewrite(context, expr_shape);

        if let Some(expr_string) = expr_rewrite {
            if pat.is_none() || pat_is_simple(&pat_string) || !expr_string.contains('\n') {
                result.push_str(spacer);
                result.push_str(&expr_string);
                return Some(result);
            }
        }
    }

    if pat.is_none() && keyword == "if" {
        return None;
    }

    let nested_indent = shape.indent.block_only().block_indent(context.config);

    // The expression won't fit on the current line, jump to next.
    result.push('\n');
    result.push_str(&nested_indent.to_string(context.config));

    let expr_rewrite = expr.rewrite(&context, Shape::indented(nested_indent, context.config));
    result.push_str(&try_opt!(expr_rewrite));

    Some(result)
}

fn rewrite_string_lit(context: &RewriteContext, span: Span, shape: Shape) -> Option<String> {
    let string_lit = context.snippet(span);

    if !context.config.format_strings() && !context.config.force_format_strings() {
        return Some(string_lit);
    }

    if !context.config.force_format_strings() &&
        !string_requires_rewrite(context, span, &string_lit, shape)
    {
        return Some(string_lit);
    }

    let fmt = StringFormat {
        opener: "\"",
        closer: "\"",
        line_start: " ",
        line_end: "\\",
        shape: shape,
        trim_end: false,
        config: context.config,
    };

    // Remove the quote characters.
    let str_lit = &string_lit[1..string_lit.len() - 1];

    rewrite_string(str_lit, &fmt)
}

fn string_requires_rewrite(
    context: &RewriteContext,
    span: Span,
    string: &str,
    shape: Shape,
) -> bool {
    if context.codemap.lookup_char_pos(span.lo).col.0 != shape.indent.width() {
        return true;
    }

    for (i, line) in string.lines().enumerate() {
        if i == 0 {
            if line.len() > shape.width {
                return true;
            }
        } else {
            if line.len() > shape.width + shape.indent.width() {
                return true;
            }
        }
    }

    false
}

pub fn rewrite_call_with_binary_search<R>(
    context: &RewriteContext,
    callee: &R,
    args: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
) -> Option<String>
where
    R: Rewrite,
{
    let closure = |callee_max_width| {
        // FIXME using byte lens instead of char lens (and probably all over the
        // place too)
        let callee_shape = Shape {
            width: callee_max_width,
            ..shape
        };
        let callee_str = callee.rewrite(context, callee_shape).ok_or(
            Ordering::Greater,
        )?;

        rewrite_call_inner(context, &callee_str, args, span, shape, false)
    };

    binary_search(1, shape.width, closure)
}

pub fn rewrite_call(
    context: &RewriteContext,
    callee: &str,
    args: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
) -> Option<String> {
    rewrite_call_inner(context, &callee, args, span, shape, false).ok()
}

fn rewrite_call_inner(
    context: &RewriteContext,
    callee_str: &str,
    args: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
    force_trailing_comma: bool,
) -> Result<String, Ordering> {
    // 2 = `( `, 1 = `(`
    let paren_overhead = if context.config.spaces_within_parens() {
        2
    } else {
        1
    };
    let used_width = extra_offset(&callee_str, shape);
    let one_line_width = shape
        .width
        .checked_sub(used_width + 2 * paren_overhead)
        .ok_or(Ordering::Greater)?;

    let nested_shape = shape_from_fn_call_style(
        context,
        shape,
        used_width + 2 * paren_overhead,
        used_width + paren_overhead,
    ).ok_or(Ordering::Greater)?;

    let span_lo = context.codemap.span_after(span, "(");
    let args_span = mk_sp(span_lo, span.hi);

    let (extendable, list_str) = rewrite_call_args(
        context,
        args,
        args_span,
        nested_shape,
        one_line_width,
        force_trailing_comma,
    ).or_else(|| if context.use_block_indent() {
        rewrite_call_args(
            context,
            args,
            args_span,
            Shape::indented(
                shape.block().indent.block_indent(context.config),
                context.config,
            ),
            0,
            force_trailing_comma,
        )
    } else {
        None
    })
        .ok_or(Ordering::Less)?;

    if !context.use_block_indent() && need_block_indent(&list_str, nested_shape) && !extendable {
        let mut new_context = context.clone();
        new_context.use_block = true;
        return rewrite_call_inner(
            &new_context,
            callee_str,
            args,
            span,
            shape,
            force_trailing_comma,
        );
    }

    let args_shape = shape.sub_width(last_line_width(&callee_str)).ok_or(
        Ordering::Less,
    )?;
    Ok(format!(
        "{}{}",
        callee_str,
        wrap_args_with_parens(
            context,
            &list_str,
            extendable,
            args_shape,
            nested_shape,
        )
    ))
}

fn need_block_indent(s: &str, shape: Shape) -> bool {
    s.lines().skip(1).any(|s| {
        s.find(|c| !char::is_whitespace(c)).map_or(false, |w| {
            w + 1 < shape.indent.width()
        })
    })
}

fn rewrite_call_args(
    context: &RewriteContext,
    args: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
    one_line_width: usize,
    force_trailing_comma: bool,
) -> Option<(bool, String)> {
    let mut item_context = context.clone();
    item_context.inside_macro = false;
    let items = itemize_list(
        context.codemap,
        args.iter(),
        ")",
        |item| item.span.lo,
        |item| item.span.hi,
        |item| item.rewrite(&item_context, shape),
        span.lo,
        span.hi,
    );
    let mut item_vec: Vec<_> = items.collect();

    // Try letting the last argument overflow to the next line with block
    // indentation. If its first line fits on one line with the other arguments,
    // we format the function arguments horizontally.
    let tactic = try_overflow_last_arg(&item_context, &mut item_vec, args, shape, one_line_width);

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if force_trailing_comma {
            SeparatorTactic::Always
        } else if context.inside_macro || !context.use_block_indent() {
            SeparatorTactic::Never
        } else {
            context.config.trailing_comma()
        },
        shape: shape,
        ends_with_newline: false,
        config: context.config,
    };

    write_list(&item_vec, &fmt).map(|args_str| {
        (tactic != DefinitiveListTactic::Vertical, args_str)
    })
}

fn try_overflow_last_arg(
    context: &RewriteContext,
    item_vec: &mut Vec<ListItem>,
    args: &[ptr::P<ast::Expr>],
    shape: Shape,
    one_line_width: usize,
) -> DefinitiveListTactic {
    let overflow_last = can_be_overflowed(&context, args);

    // Replace the last item with its first line to see if it fits with
    // first arguments.
    let (orig_last, placeholder) = if overflow_last {
        let mut context = context.clone();
        match args[args.len() - 1].node {
            ast::ExprKind::MethodCall(..) => context.force_one_line_chain = true,
            _ => (),
        }
        last_arg_shape(&context, &item_vec, shape).map_or((None, None), |arg_shape| {
            rewrite_last_arg_with_overflow(
                &context,
                &args[args.len() - 1],
                &mut item_vec[args.len() - 1],
                arg_shape,
            )
        })
    } else {
        (None, None)
    };

    let tactic = definitive_tactic(
        &*item_vec,
        ListTactic::LimitedHorizontalVertical(context.config.fn_call_width()),
        one_line_width,
    );

    // Replace the stub with the full overflowing last argument if the rewrite
    // succeeded and its first line fits with the other arguments.
    match (overflow_last, tactic, placeholder) {
        (true, DefinitiveListTactic::Horizontal, placeholder @ Some(..)) => {
            item_vec[args.len() - 1].item = placeholder;
        }
        (true, _, _) => {
            item_vec[args.len() - 1].item = orig_last;
        }
        (false, _, _) => {}
    }

    tactic
}

fn last_arg_shape(context: &RewriteContext, items: &Vec<ListItem>, shape: Shape) -> Option<Shape> {
    let overhead = items.iter().rev().skip(1).fold(0, |acc, i| {
        acc + i.item.as_ref().map_or(0, |s| first_line_width(&s))
    });
    let max_width = min(context.config.fn_call_width(), shape.width);
    let arg_indent = if context.use_block_indent() {
        shape.block().indent.block_unindent(context.config)
    } else {
        shape.block().indent
    };
    Some(Shape {
        width: try_opt!(max_width.checked_sub(overhead)),
        indent: arg_indent,
        offset: 0,
    })
}

fn rewrite_last_arg_with_overflow(
    context: &RewriteContext,
    last_arg: &ptr::P<ast::Expr>,
    last_item: &mut ListItem,
    shape: Shape,
) -> (Option<String>, Option<String>) {
    let rewrite = last_arg.rewrite(context, shape);
    let orig_last = last_item.item.clone();

    if let Some(rewrite) = rewrite {
        let rewrite_first_line = Some(rewrite[..first_line_width(&rewrite)].to_owned());
        last_item.item = rewrite_first_line;
        (orig_last, Some(rewrite))
    } else {
        (orig_last, None)
    }
}

fn can_be_overflowed(context: &RewriteContext, args: &[ptr::P<ast::Expr>]) -> bool {
    args.last().map_or(false, |x| {
        can_be_overflowed_expr(context, &x, args.len())
    })
}

fn can_be_overflowed_expr(context: &RewriteContext, expr: &ast::Expr, args_len: usize) -> bool {
    match expr.node {
        ast::ExprKind::Match(..) => {
            (context.use_block_indent() && args_len == 1) ||
                (context.config.fn_call_style() == IndentStyle::Visual && args_len > 1)
        }
        ast::ExprKind::If(..) |
        ast::ExprKind::IfLet(..) |
        ast::ExprKind::ForLoop(..) |
        ast::ExprKind::Loop(..) |
        ast::ExprKind::While(..) |
        ast::ExprKind::WhileLet(..) => {
            context.config.combine_control_expr() && context.use_block_indent() && args_len == 1
        }
        ast::ExprKind::Block(..) |
        ast::ExprKind::Closure(..) => {
            context.use_block_indent() ||
                context.config.fn_call_style() == IndentStyle::Visual && args_len > 1
        }
        ast::ExprKind::Call(..) |
        ast::ExprKind::MethodCall(..) |
        ast::ExprKind::Mac(..) |
        ast::ExprKind::Struct(..) => context.use_block_indent() && args_len == 1,
        ast::ExprKind::Tup(..) => context.use_block_indent(),
        ast::ExprKind::AddrOf(_, ref expr) |
        ast::ExprKind::Box(ref expr) |
        ast::ExprKind::Try(ref expr) |
        ast::ExprKind::Unary(_, ref expr) |
        ast::ExprKind::Cast(ref expr, _) => can_be_overflowed_expr(context, expr, args_len),
        _ => false,
    }
}

fn paren_overhead(context: &RewriteContext) -> usize {
    if context.config.spaces_within_parens() {
        4
    } else {
        2
    }
}

fn wrap_args_with_parens(
    context: &RewriteContext,
    args_str: &str,
    is_extendable: bool,
    shape: Shape,
    nested_shape: Shape,
) -> String {
    if !context.use_block_indent() ||
        (context.inside_macro && !args_str.contains('\n') &&
             args_str.len() + paren_overhead(context) <= shape.width) || is_extendable
    {
        if context.config.spaces_within_parens() && args_str.len() > 0 {
            format!("( {} )", args_str)
        } else {
            format!("({})", args_str)
        }
    } else {
        format!(
            "(\n{}{}\n{})",
            nested_shape.indent.to_string(context.config),
            args_str,
            shape.block().indent.to_string(context.config)
        )
    }
}

fn rewrite_paren(context: &RewriteContext, subexpr: &ast::Expr, shape: Shape) -> Option<String> {
    debug!("rewrite_paren, shape: {:?}", shape);
    let paren_overhead = paren_overhead(context);
    let sub_shape = try_opt!(shape.sub_width(paren_overhead / 2)).visual_indent(paren_overhead / 2);

    let paren_wrapper = |s: &str| if context.config.spaces_within_parens() && s.len() > 0 {
        format!("( {} )", s)
    } else {
        format!("({})", s)
    };

    let subexpr_str = try_opt!(subexpr.rewrite(context, sub_shape));
    debug!("rewrite_paren, subexpr_str: `{:?}`", subexpr_str);

    if subexpr_str.contains('\n') {
        Some(paren_wrapper(&subexpr_str))
    } else {
        if subexpr_str.len() + paren_overhead <= shape.width {
            Some(paren_wrapper(&subexpr_str))
        } else {
            let sub_shape = try_opt!(shape.offset_left(2));
            let subexpr_str = try_opt!(subexpr.rewrite(context, sub_shape));
            Some(paren_wrapper(&subexpr_str))
        }
    }
}

fn rewrite_index(
    expr: &ast::Expr,
    index: &ast::Expr,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let expr_str = try_opt!(expr.rewrite(context, shape));

    let (lbr, rbr) = if context.config.spaces_within_square_brackets() {
        ("[ ", " ]")
    } else {
        ("[", "]")
    };

    let offset = expr_str.len() + lbr.len();
    if let Some(index_shape) = shape.visual_indent(offset).sub_width(offset + rbr.len()) {
        if let Some(index_str) = index.rewrite(context, index_shape) {
            return Some(format!("{}{}{}{}", expr_str, lbr, index_str, rbr));
        }
    }

    let indent = shape.indent.block_indent(&context.config);
    let indent = indent.to_string(&context.config);
    // FIXME this is not right, since we don't take into account that shape.width
    // might be reduced from max_width by something on the right.
    let budget = try_opt!(context.config.max_width().checked_sub(
        indent.len() + lbr.len() +
            rbr.len(),
    ));
    let index_str = try_opt!(index.rewrite(context, Shape::legacy(budget, shape.indent)));
    Some(format!(
        "{}\n{}{}{}{}",
        expr_str,
        indent,
        lbr,
        index_str,
        rbr
    ))
}

fn rewrite_struct_lit<'a>(
    context: &RewriteContext,
    path: &ast::Path,
    fields: &'a [ast::Field],
    base: Option<&'a ast::Expr>,
    span: Span,
    shape: Shape,
) -> Option<String> {
    debug!("rewrite_struct_lit: shape {:?}", shape);

    enum StructLitField<'a> {
        Regular(&'a ast::Field),
        Base(&'a ast::Expr),
    }

    // 2 = " {".len()
    let path_shape = try_opt!(shape.sub_width(2));
    let path_str = try_opt!(rewrite_path(
        context,
        PathContext::Expr,
        None,
        path,
        path_shape,
    ));

    if fields.len() == 0 && base.is_none() {
        return Some(format!("{} {{}}", path_str));
    }

    let field_iter = fields.into_iter().map(StructLitField::Regular).chain(
        base.into_iter().map(StructLitField::Base),
    );

    // Foo { a: Foo } - indent is +3, width is -5.
    let (h_shape, v_shape) = try_opt!(struct_lit_shape(shape, context, path_str.len() + 3, 2));

    let span_lo = |item: &StructLitField| match *item {
        StructLitField::Regular(field) => field.span.lo,
        StructLitField::Base(expr) => {
            let last_field_hi = fields.last().map_or(span.lo, |field| field.span.hi);
            let snippet = context.snippet(mk_sp(last_field_hi, expr.span.lo));
            let pos = snippet.find_uncommented("..").unwrap();
            last_field_hi + BytePos(pos as u32)
        }
    };
    let span_hi = |item: &StructLitField| match *item {
        StructLitField::Regular(field) => field.span.hi,
        StructLitField::Base(expr) => expr.span.hi,
    };
    let rewrite = |item: &StructLitField| match *item {
        StructLitField::Regular(field) => {
            // The 1 taken from the v_budget is for the comma.
            rewrite_field(context, field, try_opt!(v_shape.sub_width(1)))
        }
        StructLitField::Base(expr) => {
            // 2 = ..
            expr.rewrite(context, try_opt!(v_shape.shrink_left(2)))
                .map(|s| format!("..{}", s))
        }
    };

    let items = itemize_list(
        context.codemap,
        field_iter,
        "}",
        span_lo,
        span_hi,
        rewrite,
        context.codemap.span_after(span, "{"),
        span.hi,
    );
    let item_vec = items.collect::<Vec<_>>();

    let tactic = struct_lit_tactic(h_shape, context, &item_vec);
    let nested_shape = shape_for_tactic(tactic, h_shape, v_shape);
    let fmt = struct_lit_formatting(nested_shape, tactic, context, base.is_some());

    let fields_str = try_opt!(write_list(&item_vec, &fmt));
    let fields_str = if context.config.struct_lit_style() == IndentStyle::Block &&
        (fields_str.contains('\n') ||
             context.config.struct_lit_multiline_style() == MultilineStyle::ForceMulti ||
             fields_str.len() > h_shape.map(|s| s.width).unwrap_or(0))
    {
        format!(
            "\n{}{}\n{}",
            v_shape.indent.to_string(context.config),
            fields_str,
            shape.indent.to_string(context.config)
        )
    } else {
        // One liner or visual indent.
        format!(" {} ", fields_str)
    };

    Some(format!("{} {{{}}}", path_str, fields_str))

    // FIXME if context.config.struct_lit_style() == Visual, but we run out
    // of space, we should fall back to BlockIndent.
}

pub fn struct_lit_field_separator(config: &Config) -> &str {
    colon_spaces(
        config.space_before_struct_lit_field_colon(),
        config.space_after_struct_lit_field_colon(),
    )
}

fn rewrite_field(context: &RewriteContext, field: &ast::Field, shape: Shape) -> Option<String> {
    let name = &field.ident.node.to_string();
    if field.is_shorthand {
        Some(name.to_string())
    } else {
        let separator = struct_lit_field_separator(context.config);
        let overhead = name.len() + separator.len();
        let mut expr_shape = try_opt!(shape.sub_width(overhead));
        expr_shape.offset += overhead;
        let expr = field.expr.rewrite(context, expr_shape);

        let mut attrs_str = try_opt!((*field.attrs).rewrite(context, shape));
        if !attrs_str.is_empty() {
            attrs_str.push_str(&format!("\n{}", shape.indent.to_string(context.config)));
        };

        match expr {
            Some(e) => Some(format!("{}{}{}{}", attrs_str, name, separator, e)),
            None => {
                let expr_offset = shape.indent.block_indent(context.config);
                let expr = field.expr.rewrite(
                    context,
                    Shape::indented(expr_offset, context.config),
                );
                expr.map(|s| {
                    format!(
                        "{}{}:\n{}{}",
                        attrs_str,
                        name,
                        expr_offset.to_string(&context.config),
                        s
                    )
                })
            }
        }
    }
}

fn shape_from_fn_call_style(
    context: &RewriteContext,
    shape: Shape,
    overhead: usize,
    offset: usize,
) -> Option<Shape> {
    if context.use_block_indent() {
        Some(shape.block().block_indent(context.config.tab_spaces()))
    } else {
        shape.visual_indent(offset).sub_width(overhead)
    }
}

pub fn rewrite_tuple_type<'a, I>(
    context: &RewriteContext,
    mut items: I,
    span: Span,
    shape: Shape,
) -> Option<String>
where
    I: ExactSizeIterator,
    <I as Iterator>::Item: Deref,
    <I::Item as Deref>::Target: Rewrite + Spanned + 'a,
{
    // In case of length 1, need a trailing comma
    debug!("rewrite_tuple_type {:?}", shape);
    if items.len() == 1 {
        // 3 = "(" + ",)"
        let nested_shape = try_opt!(shape.sub_width(3)).visual_indent(1);
        return items.next().unwrap().rewrite(context, nested_shape).map(
            |s| {
                if context.config.spaces_within_parens() {
                    format!("( {}, )", s)
                } else {
                    format!("({},)", s)
                }
            },
        );
    }

    let list_lo = context.codemap.span_after(span, "(");
    let nested_shape = try_opt!(shape.sub_width(2)).visual_indent(1);
    let items = itemize_list(
        context.codemap,
        items,
        ")",
        |item| item.span().lo,
        |item| item.span().hi,
        |item| item.rewrite(context, nested_shape),
        list_lo,
        span.hi - BytePos(1),
    );
    let list_str = try_opt!(format_item_list(items, nested_shape, context.config));

    if context.config.spaces_within_parens() && list_str.len() > 0 {
        Some(format!("( {} )", list_str))
    } else {
        Some(format!("({})", list_str))
    }
}

pub fn rewrite_tuple(
    context: &RewriteContext,
    items: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
) -> Option<String> {
    debug!("rewrite_tuple {:?}", shape);
    // Use old `rewrite_tuple`
    if context.config.fn_call_style() == IndentStyle::Visual {
        return rewrite_tuple_type(context, items.iter().map(|x| &**x), span, shape);
    }

    // We use the same rule as funcation call for rewriting tuple.
    // 1 = ","
    rewrite_call_inner(
        context,
        &String::new(),
        items,
        span,
        shape,
        items.len() == 1,
    ).ok()
}

pub fn rewrite_unary_prefix<R: Rewrite>(
    context: &RewriteContext,
    prefix: &str,
    rewrite: &R,
    shape: Shape,
) -> Option<String> {
    rewrite
        .rewrite(context, try_opt!(shape.offset_left(prefix.len())))
        .map(|r| format!("{}{}", prefix, r))
}

// FIXME: this is probably not correct for multi-line Rewrites. we should
// subtract suffix.len() from the last line budget, not the first!
pub fn rewrite_unary_suffix<R: Rewrite>(
    context: &RewriteContext,
    suffix: &str,
    rewrite: &R,
    shape: Shape,
) -> Option<String> {
    rewrite
        .rewrite(context, try_opt!(shape.sub_width(suffix.len())))
        .map(|mut r| {
            r.push_str(suffix);
            r
        })
}

fn rewrite_unary_op(
    context: &RewriteContext,
    op: &ast::UnOp,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    // For some reason, an UnOp is not spanned like BinOp!
    let operator_str = match *op {
        ast::UnOp::Deref => "*",
        ast::UnOp::Not => "!",
        ast::UnOp::Neg => "-",
    };
    rewrite_unary_prefix(context, operator_str, expr, shape)
}

fn rewrite_assignment(
    context: &RewriteContext,
    lhs: &ast::Expr,
    rhs: &ast::Expr,
    op: Option<&ast::BinOp>,
    shape: Shape,
) -> Option<String> {
    let operator_str = match op {
        Some(op) => context.snippet(op.span),
        None => "=".to_owned(),
    };

    // 1 = space between lhs and operator.
    let lhs_shape = try_opt!(shape.sub_width(operator_str.len() + 1));
    let lhs_str = format!(
        "{} {}",
        try_opt!(lhs.rewrite(context, lhs_shape)),
        operator_str
    );

    rewrite_assign_rhs(context, lhs_str, rhs, shape)
}

// The left hand side must contain everything up to, and including, the
// assignment operator.
pub fn rewrite_assign_rhs<S: Into<String>>(
    context: &RewriteContext,
    lhs: S,
    ex: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    let mut result = lhs.into();
    let last_line_width = last_line_width(&result) -
        if result.contains('\n') {
            shape.indent.width()
        } else {
            0
        };
    // 1 = space between operator and rhs.
    let orig_shape = try_opt!(shape.block_indent(0).offset_left(last_line_width + 1));
    let rhs = match ex.node {
        ast::ExprKind::Mac(ref mac) => {
            match rewrite_macro(mac, None, context, orig_shape, MacroPosition::Expression) {
                None if !context.snippet(ex.span).contains("\n") => {
                    context.snippet(ex.span).rewrite(context, orig_shape)
                }
                rhs @ _ => rhs,
            }
        }
        _ => ex.rewrite(context, orig_shape),
    };

    fn count_line_breaks(src: &str) -> usize {
        src.chars().filter(|&x| x == '\n').count()
    }

    match rhs {
        Some(ref new_str) if count_line_breaks(new_str) < 2 => {
            result.push(' ');
            result.push_str(new_str);
        }
        _ => {
            // Expression did not fit on the same line as the identifier or is
            // at least three lines big. Try splitting the line and see
            // if that works better.
            let new_shape = try_opt!(shape.block_left(context.config.tab_spaces()));
            let new_rhs = ex.rewrite(context, new_shape);

            // FIXME: DRY!
            match (rhs, new_rhs) {
                (Some(ref orig_rhs), Some(ref replacement_rhs))
                    if count_line_breaks(orig_rhs) > count_line_breaks(replacement_rhs) + 1 ||
                           (orig_rhs.rewrite(context, shape).is_none() &&
                                replacement_rhs.rewrite(context, new_shape).is_some()) => {
                    result.push_str(&format!("\n{}", new_shape.indent.to_string(context.config)));
                    result.push_str(replacement_rhs);
                }
                (None, Some(ref final_rhs)) => {
                    result.push_str(&format!("\n{}", new_shape.indent.to_string(context.config)));
                    result.push_str(final_rhs);
                }
                (None, None) => return None,
                (Some(ref orig_rhs), _) => {
                    result.push(' ');
                    result.push_str(orig_rhs);
                }
            }
        }
    }

    Some(result)
}

fn rewrite_expr_addrof(
    context: &RewriteContext,
    mutability: ast::Mutability,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    let operator_str = match mutability {
        ast::Mutability::Immutable => "&",
        ast::Mutability::Mutable => "&mut ",
    };
    rewrite_unary_prefix(context, operator_str, expr, shape)
}
