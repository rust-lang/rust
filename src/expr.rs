// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::Cow;
use std::cmp::min;
use std::iter::repeat;

use syntax::{ast, ptr};
use syntax::codemap::{BytePos, CodeMap, Span};

use chains::rewrite_chain;
use closures;
use codemap::{LineRangeUtils, SpanUtils};
use comment::{combine_strs_with_missing_comments, contains_comment, recover_comment_removed,
              rewrite_comment, rewrite_missing_comment, FindUncommented};
use config::{Config, ControlBraceStyle, IndentStyle};
use lists::{definitive_tactic, itemize_list, shape_for_tactic, struct_lit_formatting,
            struct_lit_shape, struct_lit_tactic, write_list, DefinitiveListTactic, ListFormatting,
            ListItem, ListTactic, Separator, SeparatorPlace, SeparatorTactic};
use macros::{rewrite_macro, MacroArg, MacroPosition};
use patterns::{can_be_overflowed_pat, TuplePatField};
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use spanned::Spanned;
use string::{rewrite_string, StringFormat};
use types::{can_be_overflowed_type, rewrite_path, PathContext};
use utils::{colon_spaces, contains_skip, count_newlines, extra_offset, first_line_width,
            inner_attributes, last_line_extendable, last_line_width, mk_sp, outer_attributes,
            paren_overhead, ptr_vec_to_ref_vec, semicolon_for_stmt, trimmed_last_line_width,
            wrap_str};
use vertical::rewrite_with_alignment;
use visitor::FmtVisitor;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        format_expr(self, ExprType::SubExpression, context, shape)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum ExprType {
    Statement,
    SubExpression,
}

pub fn format_expr(
    expr: &ast::Expr,
    expr_type: ExprType,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    skip_out_of_file_lines_range!(context, expr.span);

    if contains_skip(&*expr.attrs) {
        return Some(context.snippet(expr.span()).to_owned());
    }

    let expr_rw = match expr.node {
        ast::ExprKind::Array(ref expr_vec) => rewrite_array(
            &ptr_vec_to_ref_vec(expr_vec),
            mk_sp(context.codemap.span_after(expr.span, "["), expr.span.hi()),
            context,
            shape,
            false,
        ),
        ast::ExprKind::Lit(ref l) => rewrite_literal(context, l, shape),
        ast::ExprKind::Call(ref callee, ref args) => {
            let inner_span = mk_sp(callee.span.hi(), expr.span.hi());
            let callee_str = callee.rewrite(context, shape)?;
            rewrite_call(context, &callee_str, args, inner_span, shape)
        }
        ast::ExprKind::Paren(ref subexpr) => rewrite_paren(context, subexpr, shape),
        ast::ExprKind::Binary(ref op, ref lhs, ref rhs) => {
            // FIXME: format comments between operands and operator
            rewrite_pair(
                &**lhs,
                &**rhs,
                PairParts::new("", &format!(" {} ", context.snippet(op.span)), ""),
                context,
                shape,
                context.config.binop_separator(),
            )
        }
        ast::ExprKind::Unary(ref op, ref subexpr) => rewrite_unary_op(context, op, subexpr, shape),
        ast::ExprKind::Struct(ref path, ref fields, ref base) => rewrite_struct_lit(
            context,
            path,
            fields,
            base.as_ref().map(|e| &**e),
            expr.span,
            shape,
        ),
        ast::ExprKind::Tup(ref items) => {
            rewrite_tuple(context, &ptr_vec_to_ref_vec(items), expr.span, shape)
        }
        ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..) => to_control_flow(expr, expr_type)
            .and_then(|control_flow| control_flow.rewrite(context, shape)),
        ast::ExprKind::Block(ref block) => {
            match expr_type {
                ExprType::Statement => {
                    if is_unsafe_block(block) {
                        block.rewrite(context, shape)
                    } else if let rw @ Some(_) = rewrite_empty_block(context, block, shape) {
                        // Rewrite block without trying to put it in a single line.
                        rw
                    } else {
                        let prefix = block_prefix(context, block, shape)?;
                        rewrite_block_with_visitor(context, &prefix, block, shape, true)
                    }
                }
                ExprType::SubExpression => block.rewrite(context, shape),
            }
        }
        ast::ExprKind::Match(ref cond, ref arms) => {
            rewrite_match(context, cond, arms, shape, expr.span, &expr.attrs)
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
            Some(format!("continue{}", id_str))
        }
        ast::ExprKind::Break(ref opt_ident, ref opt_expr) => {
            let id_str = match *opt_ident {
                Some(ident) => format!(" {}", ident.node),
                None => String::new(),
            };

            if let Some(ref expr) = *opt_expr {
                rewrite_unary_prefix(context, &format!("break{} ", id_str), &**expr, shape)
            } else {
                Some(format!("break{}", id_str))
            }
        }
        ast::ExprKind::Yield(ref opt_expr) => if let Some(ref expr) = *opt_expr {
            rewrite_unary_prefix(context, "yield ", &**expr, shape)
        } else {
            Some("yield".to_string())
        },
        ast::ExprKind::Closure(capture, ref fn_decl, ref body, _) => {
            closures::rewrite_closure(capture, fn_decl, body, expr.span, context, shape)
        }
        ast::ExprKind::Try(..)
        | ast::ExprKind::Field(..)
        | ast::ExprKind::TupField(..)
        | ast::ExprKind::MethodCall(..) => rewrite_chain(expr, context, shape),
        ast::ExprKind::Mac(ref mac) => {
            rewrite_macro(mac, None, context, shape, MacroPosition::Expression).or_else(|| {
                wrap_str(
                    context.snippet(expr.span).to_owned(),
                    context.config.max_width(),
                    shape,
                )
            })
        }
        ast::ExprKind::Ret(None) => Some("return".to_owned()),
        ast::ExprKind::Ret(Some(ref expr)) => {
            rewrite_unary_prefix(context, "return ", &**expr, shape)
        }
        ast::ExprKind::Box(ref expr) => rewrite_unary_prefix(context, "box ", &**expr, shape),
        ast::ExprKind::AddrOf(mutability, ref expr) => {
            rewrite_expr_addrof(context, mutability, expr, shape)
        }
        ast::ExprKind::Cast(ref expr, ref ty) => rewrite_pair(
            &**expr,
            &**ty,
            PairParts::new("", " as ", ""),
            context,
            shape,
            SeparatorPlace::Front,
        ),
        ast::ExprKind::Type(ref expr, ref ty) => rewrite_pair(
            &**expr,
            &**ty,
            PairParts::new("", ": ", ""),
            context,
            shape,
            SeparatorPlace::Back,
        ),
        ast::ExprKind::Index(ref expr, ref index) => {
            rewrite_index(&**expr, &**index, context, shape)
        }
        ast::ExprKind::Repeat(ref expr, ref repeats) => {
            let (lbr, rbr) = if context.config.spaces_within_parens_and_brackets() {
                ("[ ", " ]")
            } else {
                ("[", "]")
            };
            rewrite_pair(
                &**expr,
                &**repeats,
                PairParts::new(lbr, "; ", rbr),
                context,
                shape,
                SeparatorPlace::Back,
            )
        }
        ast::ExprKind::Range(ref lhs, ref rhs, limits) => {
            let delim = match limits {
                ast::RangeLimits::HalfOpen => "..",
                ast::RangeLimits::Closed => "..=",
            };

            fn needs_space_before_range(context: &RewriteContext, lhs: &ast::Expr) -> bool {
                match lhs.node {
                    ast::ExprKind::Lit(ref lit) => match lit.node {
                        ast::LitKind::FloatUnsuffixed(..) => {
                            context.snippet(lit.span).ends_with('.')
                        }
                        _ => false,
                    },
                    _ => false,
                }
            }

            match (lhs.as_ref().map(|x| &**x), rhs.as_ref().map(|x| &**x)) {
                (Some(lhs), Some(rhs)) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!(" {} ", delim)
                    } else if needs_space_before_range(context, lhs) {
                        format!(" {}", delim)
                    } else {
                        delim.to_owned()
                    };
                    rewrite_pair(
                        &*lhs,
                        &*rhs,
                        PairParts::new("", &sp_delim, ""),
                        context,
                        shape,
                        context.config.binop_separator(),
                    )
                }
                (None, Some(rhs)) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!("{} ", delim)
                    } else {
                        delim.to_owned()
                    };
                    rewrite_unary_prefix(context, &sp_delim, &*rhs, shape)
                }
                (Some(lhs), None) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!(" {}", delim)
                    } else {
                        delim.to_owned()
                    };
                    rewrite_unary_suffix(context, &sp_delim, &*lhs, shape)
                }
                (None, None) => Some(delim.to_owned()),
            }
        }
        // We do not format these expressions yet, but they should still
        // satisfy our width restrictions.
        ast::ExprKind::InPlace(..) | ast::ExprKind::InlineAsm(..) => {
            Some(context.snippet(expr.span).to_owned())
        }
        ast::ExprKind::Catch(ref block) => {
            if let rw @ Some(_) = rewrite_single_line_block(context, "do catch ", block, shape) {
                rw
            } else {
                // 9 = `do catch `
                let budget = shape.width.checked_sub(9).unwrap_or(0);
                Some(format!(
                    "{}{}",
                    "do catch ",
                    block.rewrite(context, Shape::legacy(budget, shape.indent))?
                ))
            }
        }
    };

    expr_rw
        .and_then(|expr_str| recover_comment_removed(expr_str, expr.span, context))
        .and_then(|expr_str| {
            let attrs = outer_attributes(&expr.attrs);
            let attrs_str = attrs.rewrite(context, shape)?;
            let span = mk_sp(
                attrs.last().map_or(expr.span.lo(), |attr| attr.span.hi()),
                expr.span.lo(),
            );
            combine_strs_with_missing_comments(context, &attrs_str, &expr_str, span, shape, false)
        })
}

#[derive(new, Clone, Copy)]
pub struct PairParts<'a> {
    prefix: &'a str,
    infix: &'a str,
    suffix: &'a str,
}

pub fn rewrite_pair<LHS, RHS>(
    lhs: &LHS,
    rhs: &RHS,
    pp: PairParts,
    context: &RewriteContext,
    shape: Shape,
    separator_place: SeparatorPlace,
) -> Option<String>
where
    LHS: Rewrite,
    RHS: Rewrite,
{
    let lhs_overhead = match separator_place {
        SeparatorPlace::Back => shape.used_width() + pp.prefix.len() + pp.infix.trim_right().len(),
        SeparatorPlace::Front => shape.used_width(),
    };
    let lhs_shape = Shape {
        width: context.budget(lhs_overhead),
        ..shape
    };
    let lhs_result = lhs.rewrite(context, lhs_shape)
        .map(|lhs_str| format!("{}{}", pp.prefix, lhs_str))?;

    // Try to the both lhs and rhs on the same line.
    let rhs_orig_result = shape
        .offset_left(last_line_width(&lhs_result) + pp.infix.len())
        .and_then(|s| s.sub_width(pp.suffix.len()))
        .and_then(|rhs_shape| rhs.rewrite(context, rhs_shape));
    if let Some(ref rhs_result) = rhs_orig_result {
        // If the rhs looks like block expression, we allow it to stay on the same line
        // with the lhs even if it is multi-lined.
        let allow_same_line = rhs_result
            .lines()
            .next()
            .map(|first_line| first_line.ends_with('{'))
            .unwrap_or(false);
        if !rhs_result.contains('\n') || allow_same_line {
            let one_line_width = last_line_width(&lhs_result) + pp.infix.len()
                + first_line_width(rhs_result) + pp.suffix.len();
            if one_line_width <= shape.width {
                return Some(format!(
                    "{}{}{}{}",
                    lhs_result, pp.infix, rhs_result, pp.suffix
                ));
            }
        }
    }

    // We have to use multiple lines.
    // Re-evaluate the rhs because we have more space now:
    let mut rhs_shape = match context.config.indent_style() {
        IndentStyle::Visual => shape
            .sub_width(pp.suffix.len() + pp.prefix.len())?
            .visual_indent(pp.prefix.len()),
        IndentStyle::Block => {
            // Try to calculate the initial constraint on the right hand side.
            let rhs_overhead = shape.rhs_overhead(context.config);
            Shape::indented(shape.indent.block_indent(context.config), context.config)
                .sub_width(rhs_overhead)?
        }
    };
    let infix = match separator_place {
        SeparatorPlace::Back => pp.infix.trim_right(),
        SeparatorPlace::Front => pp.infix.trim_left(),
    };
    if separator_place == SeparatorPlace::Front {
        rhs_shape = rhs_shape.offset_left(infix.len())?;
    }
    let rhs_result = rhs.rewrite(context, rhs_shape)?;
    let indent_str = rhs_shape.indent.to_string(context.config);
    let infix_with_sep = match separator_place {
        SeparatorPlace::Back => format!("{}\n{}", infix, indent_str),
        SeparatorPlace::Front => format!("\n{}{}", indent_str, infix),
    };
    Some(format!(
        "{}{}{}{}",
        lhs_result, infix_with_sep, rhs_result, pp.suffix
    ))
}

pub fn rewrite_array<T: Rewrite + Spanned + ToExpr>(
    exprs: &[&T],
    span: Span,
    context: &RewriteContext,
    shape: Shape,
    trailing_comma: bool,
) -> Option<String> {
    let bracket_size = if context.config.spaces_within_parens_and_brackets() {
        2 // "[ "
    } else {
        1 // "["
    };

    let nested_shape = match context.config.indent_style() {
        IndentStyle::Block => shape
            .block()
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config)
            .sub_width(1)?,
        IndentStyle::Visual => shape
            .visual_indent(bracket_size)
            .sub_width(bracket_size * 2)?,
    };

    let items = itemize_list(
        context.codemap,
        exprs.iter(),
        "]",
        ",",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| item.rewrite(context, nested_shape),
        span.lo(),
        span.hi(),
        false,
    ).collect::<Vec<_>>();

    if items.is_empty() {
        if context.config.spaces_within_parens_and_brackets() {
            return Some("[ ]".to_string());
        } else {
            return Some("[]".to_string());
        }
    }

    let tactic = array_tactic(context, shape, nested_shape, exprs, &items, bracket_size);
    let ends_with_newline = tactic.ends_with_newline(context.config.indent_style());

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if trailing_comma {
            SeparatorTactic::Always
        } else if context.inside_macro && !exprs.is_empty() {
            let ends_with_bracket = context.snippet(span).ends_with(']');
            let bracket_offset = if ends_with_bracket { 1 } else { 0 };
            let snippet = context.snippet(mk_sp(span.lo(), span.hi() - BytePos(bracket_offset)));
            let last_char_index = snippet.rfind(|c: char| !c.is_whitespace())?;
            if &snippet[last_char_index..last_char_index + 1] == "," {
                SeparatorTactic::Always
            } else {
                SeparatorTactic::Never
            }
        } else if context.config.indent_style() == IndentStyle::Visual {
            SeparatorTactic::Never
        } else {
            SeparatorTactic::Vertical
        },
        separator_place: SeparatorPlace::Back,
        shape: nested_shape,
        ends_with_newline: ends_with_newline,
        preserve_newline: false,
        config: context.config,
    };
    let list_str = write_list(&items, &fmt)?;

    let result = if context.config.indent_style() == IndentStyle::Visual
        || tactic == DefinitiveListTactic::Horizontal
    {
        if context.config.spaces_within_parens_and_brackets() && !list_str.is_empty() {
            format!("[ {} ]", list_str)
        } else {
            format!("[{}]", list_str)
        }
    } else {
        format!(
            "[\n{}{}\n{}]",
            nested_shape.indent.to_string(context.config),
            list_str,
            shape.block().indent.to_string(context.config)
        )
    };

    Some(result)
}

fn array_tactic<T: Rewrite + Spanned + ToExpr>(
    context: &RewriteContext,
    shape: Shape,
    nested_shape: Shape,
    exprs: &[&T],
    items: &[ListItem],
    bracket_size: usize,
) -> DefinitiveListTactic {
    let has_long_item = items
        .iter()
        .any(|li| li.item.as_ref().map(|s| s.len() > 10).unwrap_or(false));

    match context.config.indent_style() {
        IndentStyle::Block => {
            let tactic = match shape.width.checked_sub(2 * bracket_size) {
                Some(width) => {
                    let tactic = ListTactic::LimitedHorizontalVertical(
                        context.config.width_heuristics().array_width,
                    );
                    definitive_tactic(items, tactic, Separator::Comma, width)
                }
                None => DefinitiveListTactic::Vertical,
            };
            if tactic == DefinitiveListTactic::Vertical && !has_long_item
                && is_every_args_simple(exprs)
            {
                DefinitiveListTactic::Mixed
            } else {
                tactic
            }
        }
        IndentStyle::Visual => {
            if has_long_item || items.iter().any(ListItem::is_multiline) {
                definitive_tactic(
                    items,
                    ListTactic::LimitedHorizontalVertical(
                        context.config.width_heuristics().array_width,
                    ),
                    Separator::Comma,
                    nested_shape.width,
                )
            } else {
                DefinitiveListTactic::Mixed
            }
        }
    }
}

fn nop_block_collapse(block_str: Option<String>, budget: usize) -> Option<String> {
    debug!("nop_block_collapse {:?} {}", block_str, budget);
    block_str.map(|block_str| {
        if block_str.starts_with('{') && budget >= 2
            && (block_str[1..].find(|c: char| !c.is_whitespace()).unwrap() == block_str.len() - 2)
        {
            "{}".to_owned()
        } else {
            block_str.to_owned()
        }
    })
}

fn rewrite_empty_block(
    context: &RewriteContext,
    block: &ast::Block,
    shape: Shape,
) -> Option<String> {
    if block.stmts.is_empty() && !block_contains_comment(block, context.codemap) && shape.width >= 2
    {
        return Some("{}".to_owned());
    }

    // If a block contains only a single-line comment, then leave it on one line.
    let user_str = context.snippet(block.span);
    let user_str = user_str.trim();
    if user_str.starts_with('{') && user_str.ends_with('}') {
        let comment_str = user_str[1..user_str.len() - 1].trim();
        if block.stmts.is_empty() && !comment_str.contains('\n') && !comment_str.starts_with("//")
            && comment_str.len() + 4 <= shape.width
        {
            return Some(format!("{{ {} }}", comment_str));
        }
    }

    None
}

fn block_prefix(context: &RewriteContext, block: &ast::Block, shape: Shape) -> Option<String> {
    Some(match block.rules {
        ast::BlockCheckMode::Unsafe(..) => {
            let snippet = context.snippet(block.span);
            let open_pos = snippet.find_uncommented("{")?;
            // Extract comment between unsafe and block start.
            let trimmed = &snippet[6..open_pos].trim();

            if !trimmed.is_empty() {
                // 9 = "unsafe  {".len(), 7 = "unsafe ".len()
                let budget = shape.width.checked_sub(9)?;
                format!(
                    "unsafe {} ",
                    rewrite_comment(
                        trimmed,
                        true,
                        Shape::legacy(budget, shape.indent + 7),
                        context.config,
                    )?
                )
            } else {
                "unsafe ".to_owned()
            }
        }
        ast::BlockCheckMode::Default => String::new(),
    })
}

fn rewrite_single_line_block(
    context: &RewriteContext,
    prefix: &str,
    block: &ast::Block,
    shape: Shape,
) -> Option<String> {
    if is_simple_block(block, context.codemap) {
        let expr_shape = shape.offset_left(last_line_width(prefix))?;
        let expr_str = block.stmts[0].rewrite(context, expr_shape)?;
        let result = format!("{}{{ {} }}", prefix, expr_str);
        if result.len() <= shape.width && !result.contains('\n') {
            return Some(result);
        }
    }
    None
}

pub fn rewrite_block_with_visitor(
    context: &RewriteContext,
    prefix: &str,
    block: &ast::Block,
    shape: Shape,
    has_braces: bool,
) -> Option<String> {
    if let rw @ Some(_) = rewrite_empty_block(context, block, shape) {
        return rw;
    }

    let mut visitor = FmtVisitor::from_context(context);
    visitor.block_indent = shape.indent;
    visitor.is_if_else_block = context.is_if_else_block;
    match block.rules {
        ast::BlockCheckMode::Unsafe(..) => {
            let snippet = context.snippet(block.span);
            let open_pos = snippet.find_uncommented("{")?;
            visitor.last_pos = block.span.lo() + BytePos(open_pos as u32)
        }
        ast::BlockCheckMode::Default => visitor.last_pos = block.span.lo(),
    }

    visitor.visit_block(block, None, has_braces);
    Some(format!("{}{}", prefix, visitor.buffer))
}

impl Rewrite for ast::Block {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        // shape.width is used only for the single line case: either the empty block `{}`,
        // or an unsafe expression `unsafe { e }`.
        if let rw @ Some(_) = rewrite_empty_block(context, self, shape) {
            return rw;
        }

        let prefix = block_prefix(context, self, shape)?;

        let result = rewrite_block_with_visitor(context, &prefix, self, shape, true);
        if let Some(ref result_str) = result {
            if result_str.lines().count() <= 3 {
                if let rw @ Some(_) = rewrite_single_line_block(context, &prefix, self, shape) {
                    return rw;
                }
            }
        }

        result
    }
}

impl Rewrite for ast::Stmt {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        skip_out_of_file_lines_range!(context, self.span());

        let result = match self.node {
            ast::StmtKind::Local(ref local) => local.rewrite(context, shape),
            ast::StmtKind::Expr(ref ex) | ast::StmtKind::Semi(ref ex) => {
                let suffix = if semicolon_for_stmt(context, self) {
                    ";"
                } else {
                    ""
                };

                let shape = shape.sub_width(suffix.len())?;
                format_expr(ex, ExprType::Statement, context, shape).map(|s| s + suffix)
            }
            ast::StmtKind::Mac(..) | ast::StmtKind::Item(..) => None,
        };
        result.and_then(|res| recover_comment_removed(res, self.span(), context))
    }
}

// Rewrite condition if the given expression has one.
pub fn rewrite_cond(context: &RewriteContext, expr: &ast::Expr, shape: Shape) -> Option<String> {
    match expr.node {
        ast::ExprKind::Match(ref cond, _) => {
            // `match `cond` {`
            let cond_shape = match context.config.indent_style() {
                IndentStyle::Visual => shape.shrink_left(6).and_then(|s| s.sub_width(2))?,
                IndentStyle::Block => shape.offset_left(8)?,
            };
            cond.rewrite(context, cond_shape)
        }
        _ => to_control_flow(expr, ExprType::SubExpression).and_then(|control_flow| {
            let alt_block_sep =
                String::from("\n") + &shape.indent.block_only().to_string(context.config);
            control_flow
                .rewrite_cond(context, shape, &alt_block_sep)
                .and_then(|rw| Some(rw.0))
        }),
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

fn to_control_flow(expr: &ast::Expr, expr_type: ExprType) -> Option<ControlFlow> {
    match expr.node {
        ast::ExprKind::If(ref cond, ref if_block, ref else_block) => Some(ControlFlow::new_if(
            cond,
            None,
            if_block,
            else_block.as_ref().map(|e| &**e),
            expr_type == ExprType::SubExpression,
            false,
            expr.span,
        )),
        ast::ExprKind::IfLet(ref pat, ref cond, ref if_block, ref else_block) => {
            Some(ControlFlow::new_if(
                cond,
                Some(pat),
                if_block,
                else_block.as_ref().map(|e| &**e),
                expr_type == ExprType::SubExpression,
                false,
                expr.span,
            ))
        }
        ast::ExprKind::ForLoop(ref pat, ref cond, ref block, label) => {
            Some(ControlFlow::new_for(pat, cond, block, label, expr.span))
        }
        ast::ExprKind::Loop(ref block, label) => {
            Some(ControlFlow::new_loop(block, label, expr.span))
        }
        ast::ExprKind::While(ref cond, ref block, label) => {
            Some(ControlFlow::new_while(None, cond, block, label, expr.span))
        }
        ast::ExprKind::WhileLet(ref pat, ref cond, ref block, label) => Some(
            ControlFlow::new_while(Some(pat), cond, block, label, expr.span),
        ),
        _ => None,
    }
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
        let else_block = self.else_block?;
        let fixed_cost = self.keyword.len() + "  {  } else {  }".len();

        if let ast::ExprKind::Block(ref else_node) = else_block.node {
            if !is_simple_block(self.block, context.codemap)
                || !is_simple_block(else_node, context.codemap)
                || pat_expr_str.contains('\n')
            {
                return None;
            }

            let new_width = width.checked_sub(pat_expr_str.len() + fixed_cost)?;
            let expr = &self.block.stmts[0];
            let if_str = expr.rewrite(context, Shape::legacy(new_width, Indent::empty()))?;

            let new_width = new_width.checked_sub(if_str.len())?;
            let else_expr = &else_node.stmts[0];
            let else_str = else_expr.rewrite(context, Shape::legacy(new_width, Indent::empty()))?;

            if if_str.contains('\n') || else_str.contains('\n') {
                return None;
            }

            let result = format!(
                "{} {} {{ {} }} else {{ {} }}",
                self.keyword, pat_expr_str, if_str, else_str
            );

            if result.len() <= width {
                return Some(result);
            }
        }

        None
    }
}

impl<'a> ControlFlow<'a> {
    fn rewrite_cond(
        &self,
        context: &RewriteContext,
        shape: Shape,
        alt_block_sep: &str,
    ) -> Option<(String, usize)> {
        // Do not take the rhs overhead from the upper expressions into account
        // when rewriting pattern.
        let new_width = context
            .config
            .max_width()
            .checked_sub(shape.used_width())
            .unwrap_or(0);
        let fresh_shape = Shape {
            width: new_width,
            ..shape
        };
        let constr_shape = if self.nested_if {
            // We are part of an if-elseif-else chain. Our constraints are tightened.
            // 7 = "} else " .len()
            fresh_shape.offset_left(7)?
        } else {
            fresh_shape
        };

        let label_string = rewrite_label(self.label);
        // 1 = space after keyword.
        let offset = self.keyword.len() + label_string.len() + 1;

        let pat_expr_string = match self.cond {
            Some(cond) => rewrite_pat_expr(
                context,
                self.pat,
                cond,
                self.matcher,
                self.connector,
                self.keyword,
                constr_shape,
                offset,
            )?,
            None => String::new(),
        };

        let brace_overhead =
            if context.config.control_brace_style() != ControlBraceStyle::AlwaysNextLine {
                // 2 = ` {`
                2
            } else {
                0
            };
        let one_line_budget = context
            .config
            .max_width()
            .checked_sub(constr_shape.used_width() + offset + brace_overhead)
            .unwrap_or(0);
        let force_newline_brace = (pat_expr_string.contains('\n')
            || pat_expr_string.len() > one_line_budget)
            && !last_line_extendable(&pat_expr_string);

        // Try to format if-else on single line.
        if self.allow_single_line
            && context
                .config
                .width_heuristics()
                .single_line_if_else_max_width > 0
        {
            let trial = self.rewrite_single_line(&pat_expr_string, context, shape.width);

            if let Some(cond_str) = trial {
                if cond_str.len()
                    <= context
                        .config
                        .width_heuristics()
                        .single_line_if_else_max_width
                {
                    return Some((cond_str, 0));
                }
            }
        }

        let cond_span = if let Some(cond) = self.cond {
            cond.span
        } else {
            mk_sp(self.block.span.lo(), self.block.span.lo())
        };

        // `for event in event`
        // Do not include label in the span.
        let lo = self.label.map_or(self.span.lo(), |label| label.span.hi());
        let between_kwd_cond = mk_sp(
            context
                .codemap
                .span_after(mk_sp(lo, self.span.hi()), self.keyword.trim()),
            self.pat.map_or(cond_span.lo(), |p| {
                if self.matcher.is_empty() {
                    p.span.lo()
                } else {
                    context.codemap.span_before(self.span, self.matcher.trim())
                }
            }),
        );

        let between_kwd_cond_comment = extract_comment(between_kwd_cond, context, shape);

        let after_cond_comment =
            extract_comment(mk_sp(cond_span.hi(), self.block.span.lo()), context, shape);

        let block_sep = if self.cond.is_none() && between_kwd_cond_comment.is_some() {
            ""
        } else if context.config.control_brace_style() == ControlBraceStyle::AlwaysNextLine
            || force_newline_brace
        {
            alt_block_sep
        } else {
            " "
        };

        let used_width = if pat_expr_string.contains('\n') {
            last_line_width(&pat_expr_string)
        } else {
            // 2 = spaces after keyword and condition.
            label_string.len() + self.keyword.len() + pat_expr_string.len() + 2
        };

        Some((
            format!(
                "{}{}{}{}{}",
                label_string,
                self.keyword,
                between_kwd_cond_comment.as_ref().map_or(
                    if pat_expr_string.is_empty() || pat_expr_string.starts_with('\n') {
                        ""
                    } else {
                        " "
                    },
                    |s| &**s,
                ),
                pat_expr_string,
                after_cond_comment.as_ref().map_or(block_sep, |s| &**s)
            ),
            used_width,
        ))
    }
}

impl<'a> Rewrite for ControlFlow<'a> {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        debug!("ControlFlow::rewrite {:?} {:?}", self, shape);

        let alt_block_sep = String::from("\n") + &shape.indent.to_string(context.config);
        let (cond_str, used_width) = self.rewrite_cond(context, shape, &alt_block_sep)?;
        // If `used_width` is 0, it indicates that whole control flow is written in a single line.
        if used_width == 0 {
            return Some(cond_str);
        }

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
        let block_str =
            rewrite_block_with_visitor(&block_context, "", self.block, block_shape, true)?;

        let mut result = format!("{}{}", cond_str, block_str);

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
                        mk_sp(else_block.span.lo(), self.span.hi()),
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
                        mk_sp(else_block.span.lo(), self.span.hi()),
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
                    format_expr(else_block, ExprType::Statement, context, else_shape)
                }
            };

            let between_kwd_else_block = mk_sp(
                self.block.span.hi(),
                context
                    .codemap
                    .span_before(mk_sp(self.block.span.hi(), else_block.span.lo()), "else"),
            );
            let between_kwd_else_block_comment =
                extract_comment(between_kwd_else_block, context, shape);

            let after_else = mk_sp(
                context
                    .codemap
                    .span_after(mk_sp(self.block.span.hi(), else_block.span.lo()), "else"),
                else_block.span.lo(),
            );
            let after_else_comment = extract_comment(after_else, context, shape);

            let between_sep = match context.config.control_brace_style() {
                ControlBraceStyle::AlwaysNextLine | ControlBraceStyle::ClosingNextLine => {
                    &*alt_block_sep
                }
                ControlBraceStyle::AlwaysSameLine => " ",
            };
            let after_sep = match context.config.control_brace_style() {
                ControlBraceStyle::AlwaysNextLine if last_in_chain => &*alt_block_sep,
                _ => " ",
            };

            result.push_str(&format!(
                "{}else{}",
                between_kwd_else_block_comment
                    .as_ref()
                    .map_or(between_sep, |s| &**s),
                after_else_comment.as_ref().map_or(after_sep, |s| &**s),
            ));
            result.push_str(&rewrite?);
        }

        Some(result)
    }
}

fn rewrite_label(label: Option<ast::SpannedIdent>) -> Cow<'static, str> {
    match label {
        Some(ident) => Cow::from(format!("{}: ", ident.node)),
        None => Cow::from(""),
    }
}

fn extract_comment(span: Span, context: &RewriteContext, shape: Shape) -> Option<String> {
    match rewrite_missing_comment(span, shape, context) {
        Some(ref comment) if !comment.is_empty() => Some(format!(
            "\n{indent}{}\n{indent}",
            comment,
            indent = shape.indent.to_string(context.config)
        )),
        _ => None,
    }
}

pub fn block_contains_comment(block: &ast::Block, codemap: &CodeMap) -> bool {
    let snippet = codemap.span_to_snippet(block.span).unwrap();
    contains_comment(&snippet)
}

// Checks that a block contains no statements, an expression and no comments.
// FIXME: incorrectly returns false when comment is contained completely within
// the expression.
pub fn is_simple_block(block: &ast::Block, codemap: &CodeMap) -> bool {
    (block.stmts.len() == 1 && stmt_is_expr(&block.stmts[0])
        && !block_contains_comment(block, codemap))
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

pub fn is_unsafe_block(block: &ast::Block) -> bool {
    if let ast::BlockCheckMode::Unsafe(..) = block.rules {
        true
    } else {
        false
    }
}

// A simple wrapper type against ast::Arm. Used inside write_list().
struct ArmWrapper<'a> {
    pub arm: &'a ast::Arm,
    // True if the arm is the last one in match expression. Used to decide on whether we should add
    // trailing comma to the match arm when `config.trailing_comma() == Never`.
    pub is_last: bool,
}

impl<'a> ArmWrapper<'a> {
    pub fn new(arm: &'a ast::Arm, is_last: bool) -> ArmWrapper<'a> {
        ArmWrapper { arm, is_last }
    }
}

impl<'a> Rewrite for ArmWrapper<'a> {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        rewrite_match_arm(context, self.arm, shape, self.is_last)
    }
}

fn rewrite_match(
    context: &RewriteContext,
    cond: &ast::Expr,
    arms: &[ast::Arm],
    shape: Shape,
    span: Span,
    attrs: &[ast::Attribute],
) -> Option<String> {
    // Do not take the rhs overhead from the upper expressions into account
    // when rewriting match condition.
    let cond_shape = Shape {
        width: context.budget(shape.used_width()),
        ..shape
    };
    // 6 = `match `
    let cond_shape = match context.config.indent_style() {
        IndentStyle::Visual => cond_shape.shrink_left(6)?,
        IndentStyle::Block => cond_shape.offset_left(6)?,
    };
    let cond_str = cond.rewrite(context, cond_shape)?;
    let alt_block_sep = String::from("\n") + &shape.indent.to_string(context.config);
    let block_sep = match context.config.control_brace_style() {
        ControlBraceStyle::AlwaysNextLine => &alt_block_sep,
        _ if last_line_extendable(&cond_str) => " ",
        // 2 = ` {`
        _ if cond_str.contains('\n') || cond_str.len() + 2 > cond_shape.width => &alt_block_sep,
        _ => " ",
    };

    let nested_indent_str = shape
        .indent
        .block_indent(context.config)
        .to_string(context.config);
    // Inner attributes.
    let inner_attrs = &inner_attributes(attrs);
    let inner_attrs_str = if inner_attrs.is_empty() {
        String::new()
    } else {
        inner_attrs
            .rewrite(context, shape)
            .map(|s| format!("{}{}\n", nested_indent_str, s))?
    };

    let open_brace_pos = if inner_attrs.is_empty() {
        let hi = if arms.is_empty() {
            span.hi()
        } else {
            arms[0].span().lo()
        };
        context.codemap.span_after(mk_sp(cond.span.hi(), hi), "{")
    } else {
        inner_attrs[inner_attrs.len() - 1].span().hi()
    };

    if arms.is_empty() {
        let snippet = context.snippet(mk_sp(open_brace_pos, span.hi() - BytePos(1)));
        if snippet.trim().is_empty() {
            Some(format!("match {} {{}}", cond_str))
        } else {
            // Empty match with comments or inner attributes? We are not going to bother, sorry ;)
            Some(context.snippet(span).to_owned())
        }
    } else {
        Some(format!(
            "match {}{}{{\n{}{}{}\n{}}}",
            cond_str,
            block_sep,
            inner_attrs_str,
            nested_indent_str,
            rewrite_match_arms(context, arms, shape, span, open_brace_pos)?,
            shape.indent.to_string(context.config),
        ))
    }
}

fn arm_comma(config: &Config, body: &ast::Expr, is_last: bool) -> &'static str {
    if is_last && config.trailing_comma() == SeparatorTactic::Never {
        ""
    } else if config.match_block_trailing_comma() {
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

fn rewrite_match_arms(
    context: &RewriteContext,
    arms: &[ast::Arm],
    shape: Shape,
    span: Span,
    open_brace_pos: BytePos,
) -> Option<String> {
    let arm_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);

    let arm_len = arms.len();
    let is_last_iter = repeat(false)
        .take(arm_len.checked_sub(1).unwrap_or(0))
        .chain(repeat(true));
    let items = itemize_list(
        context.codemap,
        arms.iter()
            .zip(is_last_iter)
            .map(|(arm, is_last)| ArmWrapper::new(arm, is_last)),
        "}",
        "|",
        |arm| arm.arm.span().lo(),
        |arm| arm.arm.span().hi(),
        |arm| arm.rewrite(context, arm_shape),
        open_brace_pos,
        span.hi(),
        false,
    );
    let arms_vec: Vec<_> = items.collect();
    let fmt = ListFormatting {
        tactic: DefinitiveListTactic::Vertical,
        // We will add/remove commas inside `arm.rewrite()`, and hence no separator here.
        separator: "",
        trailing_separator: SeparatorTactic::Never,
        separator_place: SeparatorPlace::Back,
        shape: arm_shape,
        ends_with_newline: true,
        preserve_newline: true,
        config: context.config,
    };

    write_list(&arms_vec, &fmt)
}

fn rewrite_match_arm(
    context: &RewriteContext,
    arm: &ast::Arm,
    shape: Shape,
    is_last: bool,
) -> Option<String> {
    let (missing_span, attrs_str) = if !arm.attrs.is_empty() {
        if contains_skip(&arm.attrs) {
            let (_, body) = flatten_arm_body(context, &arm.body);
            // `arm.span()` does not include trailing comma, add it manually.
            return Some(format!(
                "{}{}",
                context.snippet(arm.span()),
                arm_comma(context.config, body, is_last),
            ));
        }
        let missing_span = mk_sp(
            arm.attrs[arm.attrs.len() - 1].span.hi(),
            arm.pats[0].span.lo(),
        );
        (missing_span, arm.attrs.rewrite(context, shape)?)
    } else {
        (mk_sp(arm.span().lo(), arm.span().lo()), String::new())
    };
    let pats_str =
        rewrite_match_pattern(context, &arm.pats, &arm.guard, shape).and_then(|pats_str| {
            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                &pats_str,
                missing_span,
                shape,
                false,
            )
        })?;
    rewrite_match_body(
        context,
        &arm.body,
        &pats_str,
        shape,
        arm.guard.is_some(),
        is_last,
    )
}

/// Returns true if the given pattern is short. A short pattern is defined by the following grammer:
///
/// [small, ntp]:
///     - single token
///     - `&[single-line, ntp]`
///
/// [small]:
///     - `[small, ntp]`
///     - unary tuple constructor `([small, ntp])`
///     - `&[small]`
fn is_short_pattern(pat: &ast::Pat, pat_str: &str) -> bool {
    // We also require that the pattern is reasonably 'small' with its literal width.
    pat_str.len() <= 20 && !pat_str.contains('\n') && is_short_pattern_inner(pat)
}

fn is_short_pattern_inner(pat: &ast::Pat) -> bool {
    match pat.node {
        ast::PatKind::Wild | ast::PatKind::Lit(_) => true,
        ast::PatKind::Ident(_, _, ref pat) => pat.is_none(),
        ast::PatKind::Struct(..)
        | ast::PatKind::Mac(..)
        | ast::PatKind::Slice(..)
        | ast::PatKind::Path(..)
        | ast::PatKind::Range(..) => false,
        ast::PatKind::Tuple(ref subpats, _) => subpats.len() <= 1,
        ast::PatKind::TupleStruct(ref path, ref subpats, _) => {
            path.segments.len() <= 1 && subpats.len() <= 1
        }
        ast::PatKind::Box(ref p) | ast::PatKind::Ref(ref p, _) => is_short_pattern_inner(&*p),
    }
}

fn rewrite_match_pattern(
    context: &RewriteContext,
    pats: &[ptr::P<ast::Pat>],
    guard: &Option<ptr::P<ast::Expr>>,
    shape: Shape,
) -> Option<String> {
    // Patterns
    // 5 = ` => {`
    let pat_shape = shape.sub_width(5)?;

    let pat_strs = pats.iter()
        .map(|p| p.rewrite(context, pat_shape))
        .collect::<Option<Vec<_>>>()?;

    let use_mixed_layout = pats.iter()
        .zip(pat_strs.iter())
        .all(|(pat, pat_str)| is_short_pattern(pat, pat_str));
    let items: Vec<_> = pat_strs.into_iter().map(ListItem::from_str).collect();
    let tactic = if use_mixed_layout {
        DefinitiveListTactic::Mixed
    } else {
        definitive_tactic(
            &items,
            ListTactic::HorizontalVertical,
            Separator::VerticalBar,
            pat_shape.width,
        )
    };
    let fmt = ListFormatting {
        tactic: tactic,
        separator: " |",
        trailing_separator: SeparatorTactic::Never,
        separator_place: context.config.binop_separator(),
        shape: pat_shape,
        ends_with_newline: false,
        preserve_newline: false,
        config: context.config,
    };
    let pats_str = write_list(&items, &fmt)?;

    // Guard
    let guard_str = rewrite_guard(context, guard, shape, trimmed_last_line_width(&pats_str))?;

    Some(format!("{}{}", pats_str, guard_str))
}

// (extend, body)
// @extend: true if the arm body can be put next to `=>`
// @body: flattened body, if the body is block with a single expression
fn flatten_arm_body<'a>(context: &'a RewriteContext, body: &'a ast::Expr) -> (bool, &'a ast::Expr) {
    match body.node {
        ast::ExprKind::Block(ref block)
            if !is_unsafe_block(block) && is_simple_block(block, context.codemap) =>
        {
            if let ast::StmtKind::Expr(ref expr) = block.stmts[0].node {
                (
                    !context.config.force_multiline_blocks() && can_extend_match_arm_body(expr),
                    &*expr,
                )
            } else {
                (false, &*body)
            }
        }
        _ => (
            !context.config.force_multiline_blocks() && body.can_be_overflowed(context, 1),
            &*body,
        ),
    }
}

fn rewrite_match_body(
    context: &RewriteContext,
    body: &ptr::P<ast::Expr>,
    pats_str: &str,
    shape: Shape,
    has_guard: bool,
    is_last: bool,
) -> Option<String> {
    let (extend, body) = flatten_arm_body(context, body);
    let (is_block, is_empty_block) = if let ast::ExprKind::Block(ref block) = body.node {
        (true, is_empty_block(block, context.codemap))
    } else {
        (false, false)
    };

    let comma = arm_comma(context.config, body, is_last);
    let alt_block_sep = String::from("\n") + &shape.indent.to_string(context.config);
    let alt_block_sep = alt_block_sep.as_str();

    let combine_orig_body = |body_str: &str| {
        let block_sep = match context.config.control_brace_style() {
            ControlBraceStyle::AlwaysNextLine if is_block => alt_block_sep,
            _ => " ",
        };

        Some(format!("{} =>{}{}{}", pats_str, block_sep, body_str, comma))
    };

    let forbid_same_line = has_guard && pats_str.contains('\n') && !is_empty_block;
    let next_line_indent = if !is_block || is_empty_block {
        shape.indent.block_indent(context.config)
    } else {
        shape.indent
    };
    let combine_next_line_body = |body_str: &str| {
        if is_block {
            return Some(format!(
                "{} =>\n{}{}",
                pats_str,
                next_line_indent.to_string(context.config),
                body_str
            ));
        }

        let indent_str = shape.indent.to_string(context.config);
        let nested_indent_str = next_line_indent.to_string(context.config);
        let (body_prefix, body_suffix) = if context.config.match_arm_blocks() {
            let comma = if context.config.match_block_trailing_comma() {
                ","
            } else {
                ""
            };
            ("{", format!("\n{}}}{}", indent_str, comma))
        } else {
            ("", String::from(","))
        };

        let block_sep = match context.config.control_brace_style() {
            ControlBraceStyle::AlwaysNextLine => format!("{}{}\n", alt_block_sep, body_prefix),
            _ if body_prefix.is_empty() => "\n".to_owned(),
            _ if forbid_same_line => format!("{}{}\n", alt_block_sep, body_prefix),
            _ => format!(" {}\n", body_prefix),
        } + &nested_indent_str;

        Some(format!(
            "{} =>{}{}{}",
            pats_str, block_sep, body_str, body_suffix
        ))
    };

    // Let's try and get the arm body on the same line as the condition.
    // 4 = ` => `.len()
    let orig_body_shape = shape
        .offset_left(extra_offset(pats_str, shape) + 4)
        .and_then(|shape| shape.sub_width(comma.len()));
    let orig_body = if let Some(body_shape) = orig_body_shape {
        let rewrite = nop_block_collapse(
            format_expr(body, ExprType::Statement, context, body_shape),
            body_shape.width,
        );

        match rewrite {
            Some(ref body_str)
                if !forbid_same_line
                    && (is_block
                        || (!body_str.contains('\n') && body_str.len() <= body_shape.width)) =>
            {
                return combine_orig_body(body_str);
            }
            _ => rewrite,
        }
    } else {
        None
    };
    let orig_budget = orig_body_shape.map_or(0, |shape| shape.width);

    // Try putting body on the next line and see if it looks better.
    let next_line_body_shape = Shape::indented(next_line_indent, context.config);
    let next_line_body = nop_block_collapse(
        format_expr(body, ExprType::Statement, context, next_line_body_shape),
        next_line_body_shape.width,
    );
    match (orig_body, next_line_body) {
        (Some(ref orig_str), Some(ref next_line_str))
            if forbid_same_line || prefer_next_line(orig_str, next_line_str) =>
        {
            combine_next_line_body(next_line_str)
        }
        (Some(ref orig_str), _) if extend && first_line_width(orig_str) <= orig_budget => {
            combine_orig_body(orig_str)
        }
        (Some(ref orig_str), Some(ref next_line_str)) if orig_str.contains('\n') => {
            combine_next_line_body(next_line_str)
        }
        (None, Some(ref next_line_str)) => combine_next_line_body(next_line_str),
        (None, None) => None,
        (Some(ref orig_str), _) => combine_orig_body(orig_str),
    }
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
        let cond_shape = shape
            .offset_left(pattern_width + 4)
            .and_then(|s| s.sub_width(5));
        if let Some(cond_shape) = cond_shape {
            if let Some(cond_str) = guard.rewrite(context, cond_shape) {
                if !cond_str.contains('\n') || pattern_width <= context.config.tab_spaces() {
                    return Some(format!(" if {}", cond_str));
                }
            }
        }

        // Not enough space to put the guard after the pattern, try a newline.
        // 3 = `if `, 5 = ` => {`
        let cond_shape = Shape::indented(shape.indent.block_indent(context.config), context.config)
            .offset_left(3)
            .and_then(|s| s.sub_width(5));
        if let Some(cond_shape) = cond_shape {
            if let Some(cond_str) = guard.rewrite(context, cond_shape) {
                return Some(format!(
                    "\n{}if {}",
                    cond_shape.indent.to_string(context.config),
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
    offset: usize,
) -> Option<String> {
    debug!("rewrite_pat_expr {:?} {:?} {:?}", shape, pat, expr);
    let cond_shape = shape.offset_left(offset)?;
    if let Some(pat) = pat {
        let matcher = if matcher.is_empty() {
            matcher.to_owned()
        } else {
            format!("{} ", matcher)
        };
        let pat_shape = cond_shape
            .offset_left(matcher.len())?
            .sub_width(connector.len())?;
        let pat_string = pat.rewrite(context, pat_shape)?;
        let result = format!("{}{}{}", matcher, pat_string, connector);
        return rewrite_assign_rhs(context, result, expr, cond_shape);
    }

    let expr_rw = expr.rewrite(context, cond_shape);
    // The expression may (partially) fit on the current line.
    // We do not allow splitting between `if` and condition.
    if keyword == "if" || expr_rw.is_some() {
        return expr_rw;
    }

    // The expression won't fit on the current line, jump to next.
    let nested_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);
    let nested_indent_str = nested_shape.indent.to_string(context.config);
    expr.rewrite(context, nested_shape)
        .map(|expr_rw| format!("\n{}{}", nested_indent_str, expr_rw))
}

fn can_extend_match_arm_body(body: &ast::Expr) -> bool {
    match body.node {
        // We do not allow `if` to stay on the same line, since we could easily mistake
        // `pat => if cond { ... }` and `pat if cond => { ... }`.
        ast::ExprKind::If(..) | ast::ExprKind::IfLet(..) => false,
        ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..)
        | ast::ExprKind::Match(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::Closure(..)
        | ast::ExprKind::Array(..)
        | ast::ExprKind::Call(..)
        | ast::ExprKind::MethodCall(..)
        | ast::ExprKind::Mac(..)
        | ast::ExprKind::Struct(..)
        | ast::ExprKind::Tup(..) => true,
        ast::ExprKind::AddrOf(_, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => can_extend_match_arm_body(expr),
        _ => false,
    }
}

pub fn rewrite_literal(context: &RewriteContext, l: &ast::Lit, shape: Shape) -> Option<String> {
    match l.node {
        ast::LitKind::Str(_, ast::StrStyle::Cooked) => rewrite_string_lit(context, l.span, shape),
        _ => wrap_str(
            context.snippet(l.span).to_owned(),
            context.config.max_width(),
            shape,
        ),
    }
}

fn rewrite_string_lit(context: &RewriteContext, span: Span, shape: Shape) -> Option<String> {
    let string_lit = context.snippet(span);

    if !context.config.format_strings() {
        if string_lit
            .lines()
            .rev()
            .skip(1)
            .all(|line| line.ends_with('\\'))
        {
            let new_indent = shape.visual_indent(1).indent;
            let indented_string_lit = String::from(
                string_lit
                    .lines()
                    .map(|line| {
                        format!(
                            "{}{}",
                            new_indent.to_string(context.config),
                            line.trim_left()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim_left(),
            );
            return wrap_str(indented_string_lit, context.config.max_width(), shape);
        } else {
            return wrap_str(string_lit.to_owned(), context.config.max_width(), shape);
        }
    }

    // Remove the quote characters.
    let str_lit = &string_lit[1..string_lit.len() - 1];

    rewrite_string(
        str_lit,
        &StringFormat::new(shape.visual_indent(0), context.config),
        None,
    )
}

/// A list of `format!`-like macros, that take a long format string and a list of arguments to
/// format.
///
/// Organized as a list of `(&str, usize)` tuples, giving the name of the macro and the number of
/// arguments before the format string (none for `format!("format", ...)`, one for `assert!(result,
/// "format", ...)`, two for `assert_eq!(left, right, "format", ...)`).
const SPECIAL_MACRO_WHITELIST: &[(&str, usize)] = &[
    // format! like macros
    // From the Rust Standard Library.
    ("eprint!", 0),
    ("eprintln!", 0),
    ("format!", 0),
    ("format_args!", 0),
    ("print!", 0),
    ("println!", 0),
    ("panic!", 0),
    ("unreachable!", 0),
    // From the `log` crate.
    ("debug!", 0),
    ("error!", 0),
    ("info!", 0),
    ("warn!", 0),
    // write! like macros
    ("assert!", 1),
    ("debug_assert!", 1),
    ("write!", 1),
    ("writeln!", 1),
    // assert_eq! like macros
    ("assert_eq!", 2),
    ("assert_ne!", 2),
    ("debug_assert_eq!", 2),
    ("debug_assert_ne!", 2),
];

pub fn rewrite_call(
    context: &RewriteContext,
    callee: &str,
    args: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
) -> Option<String> {
    let force_trailing_comma = if context.inside_macro {
        span_ends_with_comma(context, span)
    } else {
        false
    };
    rewrite_call_inner(
        context,
        callee,
        &ptr_vec_to_ref_vec(args),
        span,
        shape,
        context.config.width_heuristics().fn_call_width,
        force_trailing_comma,
    )
}

pub fn rewrite_call_inner<'a, T>(
    context: &RewriteContext,
    callee_str: &str,
    args: &[&T],
    span: Span,
    shape: Shape,
    args_max_width: usize,
    force_trailing_comma: bool,
) -> Option<String>
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    // 2 = `( `, 1 = `(`
    let paren_overhead = if context.config.spaces_within_parens_and_brackets() {
        2
    } else {
        1
    };
    let used_width = extra_offset(callee_str, shape);
    let one_line_width = shape.width.checked_sub(used_width + 2 * paren_overhead)?;

    // 1 = "(" or ")"
    let one_line_shape = shape
        .offset_left(last_line_width(callee_str) + 1)?
        .sub_width(1)?;
    let nested_shape = shape_from_indent_style(
        context,
        shape,
        used_width + 2 * paren_overhead,
        used_width + paren_overhead,
    )?;

    let span_lo = context.codemap.span_after(span, "(");
    let args_span = mk_sp(span_lo, span.hi());

    let (extendable, list_str) = rewrite_call_args(
        context,
        args,
        args_span,
        one_line_shape,
        nested_shape,
        one_line_width,
        args_max_width,
        force_trailing_comma,
        callee_str,
    )?;

    if !context.use_block_indent() && need_block_indent(&list_str, nested_shape) && !extendable {
        let mut new_context = context.clone();
        new_context.use_block = true;
        return rewrite_call_inner(
            &new_context,
            callee_str,
            args,
            span,
            shape,
            args_max_width,
            force_trailing_comma,
        );
    }

    let args_shape = shape.sub_width(last_line_width(callee_str))?;
    Some(format!(
        "{}{}",
        callee_str,
        wrap_args_with_parens(context, &list_str, extendable, args_shape, nested_shape)
    ))
}

fn need_block_indent(s: &str, shape: Shape) -> bool {
    s.lines().skip(1).any(|s| {
        s.find(|c| !char::is_whitespace(c))
            .map_or(false, |w| w + 1 < shape.indent.width())
    })
}

fn rewrite_call_args<'a, T>(
    context: &RewriteContext,
    args: &[&T],
    span: Span,
    one_line_shape: Shape,
    nested_shape: Shape,
    one_line_width: usize,
    args_max_width: usize,
    force_trailing_comma: bool,
    callee_str: &str,
) -> Option<(bool, String)>
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    let items = itemize_list(
        context.codemap,
        args.iter(),
        ")",
        ",",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| item.rewrite(context, nested_shape),
        span.lo(),
        span.hi(),
        true,
    );
    let mut item_vec: Vec<_> = items.collect();

    // Try letting the last argument overflow to the next line with block
    // indentation. If its first line fits on one line with the other arguments,
    // we format the function arguments horizontally.
    let tactic = try_overflow_last_arg(
        context,
        &mut item_vec,
        &args[..],
        one_line_shape,
        nested_shape,
        one_line_width,
        args_max_width,
        callee_str,
    );

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
        separator_place: SeparatorPlace::Back,
        shape: nested_shape,
        ends_with_newline: context.use_block_indent() && tactic == DefinitiveListTactic::Vertical,
        preserve_newline: false,
        config: context.config,
    };

    write_list(&item_vec, &fmt)
        .map(|args_str| (tactic == DefinitiveListTactic::Horizontal, args_str))
}

fn try_overflow_last_arg<'a, T>(
    context: &RewriteContext,
    item_vec: &mut Vec<ListItem>,
    args: &[&T],
    one_line_shape: Shape,
    nested_shape: Shape,
    one_line_width: usize,
    args_max_width: usize,
    callee_str: &str,
) -> DefinitiveListTactic
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    // 1 = "("
    let combine_arg_with_callee =
        callee_str.len() + 1 <= context.config.tab_spaces() && args.len() == 1;
    let overflow_last = combine_arg_with_callee || can_be_overflowed(context, args);

    // Replace the last item with its first line to see if it fits with
    // first arguments.
    let placeholder = if overflow_last {
        let mut context = context.clone();
        if !combine_arg_with_callee {
            if let Some(expr) = args[args.len() - 1].to_expr() {
                if let ast::ExprKind::MethodCall(..) = expr.node {
                    context.force_one_line_chain = true;
                }
            }
        }
        last_arg_shape(args, item_vec, one_line_shape, args_max_width).and_then(|arg_shape| {
            rewrite_last_arg_with_overflow(&context, args, &mut item_vec[args.len() - 1], arg_shape)
        })
    } else {
        None
    };

    let mut tactic = definitive_tactic(
        &*item_vec,
        ListTactic::LimitedHorizontalVertical(args_max_width),
        Separator::Comma,
        one_line_width,
    );

    // Replace the stub with the full overflowing last argument if the rewrite
    // succeeded and its first line fits with the other arguments.
    match (overflow_last, tactic, placeholder) {
        (true, DefinitiveListTactic::Horizontal, Some(ref overflowed)) if args.len() == 1 => {
            // When we are rewriting a nested function call, we restrict the
            // bugdet for the inner function to avoid them being deeply nested.
            // However, when the inner function has a prefix or a suffix
            // (e.g. `foo() as u32`), this budget reduction may produce poorly
            // formatted code, where a prefix or a suffix being left on its own
            // line. Here we explicitlly check those cases.
            if count_newlines(overflowed) == 1 {
                let rw = args.last()
                    .and_then(|last_arg| last_arg.rewrite(context, nested_shape));
                let no_newline = rw.as_ref().map_or(false, |s| !s.contains('\n'));
                if no_newline {
                    item_vec[args.len() - 1].item = rw;
                } else {
                    item_vec[args.len() - 1].item = Some(overflowed.to_owned());
                }
            } else {
                item_vec[args.len() - 1].item = Some(overflowed.to_owned());
            }
        }
        (true, DefinitiveListTactic::Horizontal, placeholder @ Some(..)) => {
            item_vec[args.len() - 1].item = placeholder;
        }
        _ if args.len() >= 1 => {
            item_vec[args.len() - 1].item = args.last()
                .and_then(|last_arg| last_arg.rewrite(context, nested_shape));

            let default_tactic = || {
                definitive_tactic(
                    &*item_vec,
                    ListTactic::LimitedHorizontalVertical(args_max_width),
                    Separator::Comma,
                    one_line_width,
                )
            };

            // Use horizontal layout for a function with a single argument as long as
            // everything fits in a single line.
            if args.len() == 1
                && args_max_width != 0 // Vertical layout is forced.
                && !item_vec[0].has_comment()
                && !item_vec[0].inner_as_ref().contains('\n')
                && ::lists::total_item_width(&item_vec[0]) <= one_line_width
            {
                tactic = DefinitiveListTactic::Horizontal;
            } else {
                tactic = default_tactic();

                if tactic == DefinitiveListTactic::Vertical {
                    if let Some((all_simple, num_args_before)) =
                        maybe_get_args_offset(callee_str, args)
                    {
                        let one_line = all_simple
                            && definitive_tactic(
                                &item_vec[..num_args_before],
                                ListTactic::HorizontalVertical,
                                Separator::Comma,
                                nested_shape.width,
                            ) == DefinitiveListTactic::Horizontal
                            && definitive_tactic(
                                &item_vec[num_args_before + 1..],
                                ListTactic::HorizontalVertical,
                                Separator::Comma,
                                nested_shape.width,
                            ) == DefinitiveListTactic::Horizontal;

                        if one_line {
                            tactic = DefinitiveListTactic::SpecialMacro(num_args_before);
                        };
                    }
                }
            }
        }
        _ => (),
    }

    tactic
}

fn is_simple_arg(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Lit(..) => true,
        ast::ExprKind::Path(ref qself, ref path) => qself.is_none() && path.segments.len() <= 1,
        ast::ExprKind::AddrOf(_, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Cast(ref expr, _)
        | ast::ExprKind::Field(ref expr, _)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::TupField(ref expr, _)
        | ast::ExprKind::Unary(_, ref expr) => is_simple_arg(expr),
        ast::ExprKind::Index(ref lhs, ref rhs) | ast::ExprKind::Repeat(ref lhs, ref rhs) => {
            is_simple_arg(lhs) && is_simple_arg(rhs)
        }
        _ => false,
    }
}

fn is_every_args_simple<T: ToExpr>(lists: &[&T]) -> bool {
    lists
        .iter()
        .all(|arg| arg.to_expr().map_or(false, is_simple_arg))
}

/// In case special-case style is required, returns an offset from which we start horizontal layout.
fn maybe_get_args_offset<T: ToExpr>(callee_str: &str, args: &[&T]) -> Option<(bool, usize)> {
    if let Some(&(_, num_args_before)) = SPECIAL_MACRO_WHITELIST
        .iter()
        .find(|&&(s, _)| s == callee_str)
    {
        let all_simple = args.len() > num_args_before && is_every_args_simple(args);

        Some((all_simple, num_args_before))
    } else {
        None
    }
}

/// Returns a shape for the last argument which is going to be overflowed.
fn last_arg_shape<T>(
    lists: &[&T],
    items: &[ListItem],
    shape: Shape,
    args_max_width: usize,
) -> Option<Shape>
where
    T: Rewrite + Spanned + ToExpr,
{
    let is_nested_call = lists
        .iter()
        .next()
        .and_then(|item| item.to_expr())
        .map_or(false, is_nested_call);
    if items.len() == 1 && !is_nested_call {
        return Some(shape);
    }
    let offset = items.iter().rev().skip(1).fold(0, |acc, i| {
        // 2 = ", "
        acc + 2 + i.inner_as_ref().len()
    });
    Shape {
        width: min(args_max_width, shape.width),
        ..shape
    }.offset_left(offset)
}

fn rewrite_last_arg_with_overflow<'a, T>(
    context: &RewriteContext,
    args: &[&T],
    last_item: &mut ListItem,
    shape: Shape,
) -> Option<String>
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    let last_arg = args[args.len() - 1];
    let rewrite = if let Some(expr) = last_arg.to_expr() {
        match expr.node {
            // When overflowing the closure which consists of a single control flow expression,
            // force to use block if its condition uses multi line.
            ast::ExprKind::Closure(..) => {
                // If the argument consists of multiple closures, we do not overflow
                // the last closure.
                if closures::args_have_many_closure(args) {
                    None
                } else {
                    closures::rewrite_last_closure(context, expr, shape)
                }
            }
            _ => expr.rewrite(context, shape),
        }
    } else {
        last_arg.rewrite(context, shape)
    };

    if let Some(rewrite) = rewrite {
        let rewrite_first_line = Some(rewrite[..first_line_width(&rewrite)].to_owned());
        last_item.item = rewrite_first_line;
        Some(rewrite)
    } else {
        None
    }
}

fn can_be_overflowed<'a, T>(context: &RewriteContext, args: &[&T]) -> bool
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    args.last()
        .map_or(false, |x| x.can_be_overflowed(context, args.len()))
}

pub fn can_be_overflowed_expr(context: &RewriteContext, expr: &ast::Expr, args_len: usize) -> bool {
    match expr.node {
        ast::ExprKind::Match(..) => {
            (context.use_block_indent() && args_len == 1)
                || (context.config.indent_style() == IndentStyle::Visual && args_len > 1)
        }
        ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..) => {
            context.config.combine_control_expr() && context.use_block_indent() && args_len == 1
        }
        ast::ExprKind::Block(..) | ast::ExprKind::Closure(..) => {
            context.use_block_indent()
                || context.config.indent_style() == IndentStyle::Visual && args_len > 1
        }
        ast::ExprKind::Array(..)
        | ast::ExprKind::Call(..)
        | ast::ExprKind::Mac(..)
        | ast::ExprKind::MethodCall(..)
        | ast::ExprKind::Struct(..)
        | ast::ExprKind::Tup(..) => context.use_block_indent() && args_len == 1,
        ast::ExprKind::AddrOf(_, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => can_be_overflowed_expr(context, expr, args_len),
        _ => false,
    }
}

fn is_nested_call(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Call(..) | ast::ExprKind::Mac(..) => true,
        ast::ExprKind::AddrOf(_, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => is_nested_call(expr),
        _ => false,
    }
}

pub fn wrap_args_with_parens(
    context: &RewriteContext,
    args_str: &str,
    is_extendable: bool,
    shape: Shape,
    nested_shape: Shape,
) -> String {
    if !context.use_block_indent()
        || (context.inside_macro && !args_str.contains('\n')
            && args_str.len() + paren_overhead(context) <= shape.width) || is_extendable
    {
        if context.config.spaces_within_parens_and_brackets() && !args_str.is_empty() {
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

/// Return true if a function call or a method call represented by the given span ends with a
/// trailing comma. This function is used when rewriting macro, as adding or removing a trailing
/// comma from macro can potentially break the code.
fn span_ends_with_comma(context: &RewriteContext, span: Span) -> bool {
    let mut encountered_closing_paren = false;
    for c in context.snippet(span).chars().rev() {
        match c {
            ',' => return true,
            ')' => if encountered_closing_paren {
                return false;
            } else {
                encountered_closing_paren = true;
            },
            _ if c.is_whitespace() => continue,
            _ => return false,
        }
    }
    false
}

fn rewrite_paren(context: &RewriteContext, subexpr: &ast::Expr, shape: Shape) -> Option<String> {
    debug!("rewrite_paren, shape: {:?}", shape);
    let total_paren_overhead = paren_overhead(context);
    let paren_overhead = total_paren_overhead / 2;
    let sub_shape = shape
        .offset_left(paren_overhead)
        .and_then(|s| s.sub_width(paren_overhead))?;

    let paren_wrapper = |s: &str| {
        if context.config.spaces_within_parens_and_brackets() && !s.is_empty() {
            format!("( {} )", s)
        } else {
            format!("({})", s)
        }
    };

    let subexpr_str = subexpr.rewrite(context, sub_shape)?;
    debug!("rewrite_paren, subexpr_str: `{:?}`", subexpr_str);

    if subexpr_str.contains('\n')
        || first_line_width(&subexpr_str) + total_paren_overhead <= shape.width
    {
        Some(paren_wrapper(&subexpr_str))
    } else {
        None
    }
}

fn rewrite_index(
    expr: &ast::Expr,
    index: &ast::Expr,
    context: &RewriteContext,
    shape: Shape,
) -> Option<String> {
    let expr_str = expr.rewrite(context, shape)?;

    let (lbr, rbr) = if context.config.spaces_within_parens_and_brackets() {
        ("[ ", " ]")
    } else {
        ("[", "]")
    };

    let offset = last_line_width(&expr_str) + lbr.len();
    let rhs_overhead = shape.rhs_overhead(context.config);
    let index_shape = if expr_str.contains('\n') {
        Shape::legacy(context.config.max_width(), shape.indent)
            .offset_left(offset)
            .and_then(|shape| shape.sub_width(rbr.len() + rhs_overhead))
    } else {
        shape.visual_indent(offset).sub_width(offset + rbr.len())
    };
    let orig_index_rw = index_shape.and_then(|s| index.rewrite(context, s));

    // Return if index fits in a single line.
    match orig_index_rw {
        Some(ref index_str) if !index_str.contains('\n') => {
            return Some(format!("{}{}{}{}", expr_str, lbr, index_str, rbr));
        }
        _ => (),
    }

    // Try putting index on the next line and see if it fits in a single line.
    let indent = shape.indent.block_indent(context.config);
    let index_shape = Shape::indented(indent, context.config).offset_left(lbr.len())?;
    let index_shape = index_shape.sub_width(rbr.len() + rhs_overhead)?;
    let new_index_rw = index.rewrite(context, index_shape);
    match (orig_index_rw, new_index_rw) {
        (_, Some(ref new_index_str)) if !new_index_str.contains('\n') => Some(format!(
            "{}\n{}{}{}{}",
            expr_str,
            indent.to_string(context.config),
            lbr,
            new_index_str,
            rbr
        )),
        (None, Some(ref new_index_str)) => Some(format!(
            "{}\n{}{}{}{}",
            expr_str,
            indent.to_string(context.config),
            lbr,
            new_index_str,
            rbr
        )),
        (Some(ref index_str), _) => Some(format!("{}{}{}{}", expr_str, lbr, index_str, rbr)),
        _ => None,
    }
}

fn struct_lit_can_be_aligned(fields: &[ast::Field], base: &Option<&ast::Expr>) -> bool {
    if base.is_some() {
        return false;
    }

    fields.iter().all(|field| !field.is_shorthand)
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
    let path_shape = shape.sub_width(2)?;
    let path_str = rewrite_path(context, PathContext::Expr, None, path, path_shape)?;

    if fields.is_empty() && base.is_none() {
        return Some(format!("{} {{}}", path_str));
    }

    // Foo { a: Foo } - indent is +3, width is -5.
    let (h_shape, v_shape) = struct_lit_shape(shape, context, path_str.len() + 3, 2)?;

    let one_line_width = h_shape.map_or(0, |shape| shape.width);
    let body_lo = context.codemap.span_after(span, "{");
    let fields_str = if struct_lit_can_be_aligned(fields, &base)
        && context.config.struct_field_align_threshold() > 0
    {
        rewrite_with_alignment(
            fields,
            context,
            shape,
            mk_sp(body_lo, span.hi()),
            one_line_width,
        )?
    } else {
        let field_iter = fields
            .into_iter()
            .map(StructLitField::Regular)
            .chain(base.into_iter().map(StructLitField::Base));

        let span_lo = |item: &StructLitField| match *item {
            StructLitField::Regular(field) => field.span().lo(),
            StructLitField::Base(expr) => {
                let last_field_hi = fields.last().map_or(span.lo(), |field| field.span.hi());
                let snippet = context.snippet(mk_sp(last_field_hi, expr.span.lo()));
                let pos = snippet.find_uncommented("..").unwrap();
                last_field_hi + BytePos(pos as u32)
            }
        };
        let span_hi = |item: &StructLitField| match *item {
            StructLitField::Regular(field) => field.span().hi(),
            StructLitField::Base(expr) => expr.span.hi(),
        };
        let rewrite = |item: &StructLitField| match *item {
            StructLitField::Regular(field) => {
                // The 1 taken from the v_budget is for the comma.
                rewrite_field(context, field, v_shape.sub_width(1)?, 0)
            }
            StructLitField::Base(expr) => {
                // 2 = ..
                expr.rewrite(context, v_shape.offset_left(2)?)
                    .map(|s| format!("..{}", s))
            }
        };

        let items = itemize_list(
            context.codemap,
            field_iter,
            "}",
            ",",
            span_lo,
            span_hi,
            rewrite,
            body_lo,
            span.hi(),
            false,
        );
        let item_vec = items.collect::<Vec<_>>();

        let tactic = struct_lit_tactic(h_shape, context, &item_vec);
        let nested_shape = shape_for_tactic(tactic, h_shape, v_shape);
        let fmt = struct_lit_formatting(nested_shape, tactic, context, base.is_some());

        write_list(&item_vec, &fmt)?
    };

    let fields_str = wrap_struct_field(context, &fields_str, shape, v_shape, one_line_width);
    Some(format!("{} {{{}}}", path_str, fields_str))

    // FIXME if context.config.indent_style() == Visual, but we run out
    // of space, we should fall back to BlockIndent.
}

pub fn wrap_struct_field(
    context: &RewriteContext,
    fields_str: &str,
    shape: Shape,
    nested_shape: Shape,
    one_line_width: usize,
) -> String {
    if context.config.indent_style() == IndentStyle::Block
        && (fields_str.contains('\n') || !context.config.struct_lit_single_line()
            || fields_str.len() > one_line_width)
    {
        format!(
            "\n{}{}\n{}",
            nested_shape.indent.to_string(context.config),
            fields_str,
            shape.indent.to_string(context.config)
        )
    } else {
        // One liner or visual indent.
        format!(" {} ", fields_str)
    }
}

pub fn struct_lit_field_separator(config: &Config) -> &str {
    colon_spaces(config.space_before_colon(), config.space_after_colon())
}

pub fn rewrite_field(
    context: &RewriteContext,
    field: &ast::Field,
    shape: Shape,
    prefix_max_width: usize,
) -> Option<String> {
    if contains_skip(&field.attrs) {
        return Some(context.snippet(field.span()).to_owned());
    }
    let name = &field.ident.node.to_string();
    if field.is_shorthand {
        Some(name.to_string())
    } else {
        let mut separator = String::from(struct_lit_field_separator(context.config));
        for _ in 0..prefix_max_width.checked_sub(name.len()).unwrap_or(0) {
            separator.push(' ');
        }
        let overhead = name.len() + separator.len();
        let expr_shape = shape.offset_left(overhead)?;
        let expr = field.expr.rewrite(context, expr_shape);

        let mut attrs_str = field.attrs.rewrite(context, shape)?;
        if !attrs_str.is_empty() {
            attrs_str.push_str(&format!("\n{}", shape.indent.to_string(context.config)));
        };

        match expr {
            Some(e) => Some(format!("{}{}{}{}", attrs_str, name, separator, e)),
            None => {
                let expr_offset = shape.indent.block_indent(context.config);
                let expr = field
                    .expr
                    .rewrite(context, Shape::indented(expr_offset, context.config));
                expr.map(|s| {
                    format!(
                        "{}{}:\n{}{}",
                        attrs_str,
                        name,
                        expr_offset.to_string(context.config),
                        s
                    )
                })
            }
        }
    }
}

fn shape_from_indent_style(
    context: &RewriteContext,
    shape: Shape,
    overhead: usize,
    offset: usize,
) -> Option<Shape> {
    if context.use_block_indent() {
        // 1 = ","
        shape
            .block()
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config)
            .sub_width(1)
    } else {
        shape.visual_indent(offset).sub_width(overhead)
    }
}

fn rewrite_tuple_in_visual_indent_style<'a, T>(
    context: &RewriteContext,
    items: &[&T],
    span: Span,
    shape: Shape,
) -> Option<String>
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    let mut items = items.iter();
    // In case of length 1, need a trailing comma
    debug!("rewrite_tuple_in_visual_indent_style {:?}", shape);
    if items.len() == 1 {
        // 3 = "(" + ",)"
        let nested_shape = shape.sub_width(3)?.visual_indent(1);
        return items
            .next()
            .unwrap()
            .rewrite(context, nested_shape)
            .map(|s| {
                if context.config.spaces_within_parens_and_brackets() {
                    format!("( {}, )", s)
                } else {
                    format!("({},)", s)
                }
            });
    }

    let list_lo = context.codemap.span_after(span, "(");
    let nested_shape = shape.sub_width(2)?.visual_indent(1);
    let items = itemize_list(
        context.codemap,
        items,
        ")",
        ",",
        |item| item.span().lo(),
        |item| item.span().hi(),
        |item| item.rewrite(context, nested_shape),
        list_lo,
        span.hi() - BytePos(1),
        false,
    );
    let item_vec: Vec<_> = items.collect();
    let tactic = definitive_tactic(
        &item_vec,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        nested_shape.width,
    );
    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        separator_place: SeparatorPlace::Back,
        shape: shape,
        ends_with_newline: false,
        preserve_newline: false,
        config: context.config,
    };
    let list_str = write_list(&item_vec, &fmt)?;

    if context.config.spaces_within_parens_and_brackets() && !list_str.is_empty() {
        Some(format!("( {} )", list_str))
    } else {
        Some(format!("({})", list_str))
    }
}

pub fn rewrite_tuple<'a, T>(
    context: &RewriteContext,
    items: &[&T],
    span: Span,
    shape: Shape,
) -> Option<String>
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    debug!("rewrite_tuple {:?}", shape);
    if context.use_block_indent() {
        // We use the same rule as function calls for rewriting tuples.
        let force_trailing_comma = if context.inside_macro {
            span_ends_with_comma(context, span)
        } else {
            items.len() == 1
        };
        rewrite_call_inner(
            context,
            &String::new(),
            items,
            span,
            shape,
            context.config.width_heuristics().fn_call_width,
            force_trailing_comma,
        )
    } else {
        rewrite_tuple_in_visual_indent_style(context, items, span, shape)
    }
}

pub fn rewrite_unary_prefix<R: Rewrite>(
    context: &RewriteContext,
    prefix: &str,
    rewrite: &R,
    shape: Shape,
) -> Option<String> {
    rewrite
        .rewrite(context, shape.offset_left(prefix.len())?)
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
        .rewrite(context, shape.sub_width(suffix.len())?)
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
        None => "=",
    };

    // 1 = space between lhs and operator.
    let lhs_shape = shape.sub_width(operator_str.len() + 1)?;
    let lhs_str = format!("{} {}", lhs.rewrite(context, lhs_shape)?, operator_str);

    rewrite_assign_rhs(context, lhs_str, rhs, shape)
}

// The left hand side must contain everything up to, and including, the
// assignment operator.
pub fn rewrite_assign_rhs<S: Into<String>, R: Rewrite>(
    context: &RewriteContext,
    lhs: S,
    ex: &R,
    shape: Shape,
) -> Option<String> {
    let lhs = lhs.into();
    let last_line_width = last_line_width(&lhs)
        .checked_sub(if lhs.contains('\n') {
            shape.indent.width()
        } else {
            0
        })
        .unwrap_or(0);
    // 1 = space between operator and rhs.
    let orig_shape = shape.offset_left(last_line_width + 1).unwrap_or(Shape {
        width: 0,
        offset: shape.offset + last_line_width + 1,
        ..shape
    });
    let rhs = choose_rhs(context, ex, orig_shape, ex.rewrite(context, orig_shape))?;
    Some(lhs + &rhs)
}

pub fn choose_rhs<R: Rewrite>(
    context: &RewriteContext,
    expr: &R,
    shape: Shape,
    orig_rhs: Option<String>,
) -> Option<String> {
    match orig_rhs {
        Some(ref new_str) if !new_str.contains('\n') && new_str.len() <= shape.width => {
            Some(format!(" {}", new_str))
        }
        _ => {
            // Expression did not fit on the same line as the identifier.
            // Try splitting the line and see if that works better.
            let new_shape =
                Shape::indented(shape.indent.block_indent(context.config), context.config)
                    .sub_width(shape.rhs_overhead(context.config))?;
            let new_rhs = expr.rewrite(context, new_shape);
            let new_indent_str = &new_shape.indent.to_string(context.config);

            match (orig_rhs, new_rhs) {
                (Some(ref orig_rhs), Some(ref new_rhs))
                    if wrap_str(new_rhs.clone(), context.config.max_width(), new_shape)
                        .is_none() =>
                {
                    Some(format!(" {}", orig_rhs))
                }
                (Some(ref orig_rhs), Some(ref new_rhs)) if prefer_next_line(orig_rhs, new_rhs) => {
                    Some(format!("\n{}{}", new_indent_str, new_rhs))
                }
                (None, Some(ref new_rhs)) => Some(format!("\n{}{}", new_indent_str, new_rhs)),
                (None, None) => None,
                (Some(ref orig_rhs), _) => Some(format!(" {}", orig_rhs)),
            }
        }
    }
}

fn prefer_next_line(orig_rhs: &str, next_line_rhs: &str) -> bool {
    !next_line_rhs.contains('\n') || count_newlines(orig_rhs) > count_newlines(next_line_rhs) + 1
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

pub trait ToExpr {
    fn to_expr(&self) -> Option<&ast::Expr>;
    fn can_be_overflowed(&self, context: &RewriteContext, len: usize) -> bool;
}

impl ToExpr for ast::Expr {
    fn to_expr(&self) -> Option<&ast::Expr> {
        Some(self)
    }

    fn can_be_overflowed(&self, context: &RewriteContext, len: usize) -> bool {
        can_be_overflowed_expr(context, self, len)
    }
}

impl ToExpr for ast::Ty {
    fn to_expr(&self) -> Option<&ast::Expr> {
        None
    }

    fn can_be_overflowed(&self, context: &RewriteContext, len: usize) -> bool {
        can_be_overflowed_type(context, self, len)
    }
}

impl<'a> ToExpr for TuplePatField<'a> {
    fn to_expr(&self) -> Option<&ast::Expr> {
        None
    }

    fn can_be_overflowed(&self, context: &RewriteContext, len: usize) -> bool {
        can_be_overflowed_pat(context, self, len)
    }
}

impl<'a> ToExpr for ast::StructField {
    fn to_expr(&self) -> Option<&ast::Expr> {
        None
    }

    fn can_be_overflowed(&self, _: &RewriteContext, _: usize) -> bool {
        false
    }
}

impl<'a> ToExpr for MacroArg {
    fn to_expr(&self) -> Option<&ast::Expr> {
        match *self {
            MacroArg::Expr(ref expr) => Some(expr),
            _ => None,
        }
    }

    fn can_be_overflowed(&self, context: &RewriteContext, len: usize) -> bool {
        match *self {
            MacroArg::Expr(ref expr) => can_be_overflowed_expr(context, expr, len),
            MacroArg::Ty(ref ty) => can_be_overflowed_type(context, ty, len),
            MacroArg::Pat(..) => false,
        }
    }
}
