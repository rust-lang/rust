//! Format match expression.

use std::iter::repeat;

use rustc_ast::{ast, ptr};
use rustc_span::{BytePos, Span};

use crate::comment::{combine_strs_with_missing_comments, rewrite_comment};
use crate::config::lists::*;
use crate::config::{Config, ControlBraceStyle, IndentStyle, MatchArmLeadingPipe, Version};
use crate::expr::{
    format_expr, is_empty_block, is_simple_block, is_unsafe_block, prefer_next_line, rewrite_cond,
    ExprType, RhsTactics,
};
use crate::lists::{itemize_list, write_list, ListFormatting};
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::spanned::Spanned;
use crate::utils::{
    contains_skip, extra_offset, first_line_width, inner_attributes, last_line_extendable, mk_sp,
    semicolon_for_expr, trimmed_last_line_width, unicode_str_width,
};

/// A simple wrapper type against `ast::Arm`. Used inside `write_list()`.
struct ArmWrapper<'a> {
    arm: &'a ast::Arm,
    /// `true` if the arm is the last one in match expression. Used to decide on whether we should
    /// add trailing comma to the match arm when `config.trailing_comma() == Never`.
    is_last: bool,
    /// Holds a byte position of `|` at the beginning of the arm pattern, if available.
    beginning_vert: Option<BytePos>,
}

impl<'a> ArmWrapper<'a> {
    fn new(arm: &'a ast::Arm, is_last: bool, beginning_vert: Option<BytePos>) -> ArmWrapper<'a> {
        ArmWrapper {
            arm,
            is_last,
            beginning_vert,
        }
    }
}

impl<'a> Spanned for ArmWrapper<'a> {
    fn span(&self) -> Span {
        if let Some(lo) = self.beginning_vert {
            let lo = std::cmp::min(lo, self.arm.span().lo());
            mk_sp(lo, self.arm.span().hi())
        } else {
            self.arm.span()
        }
    }
}

impl<'a> Rewrite for ArmWrapper<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        rewrite_match_arm(
            context,
            self.arm,
            shape,
            self.is_last,
            self.beginning_vert.is_some(),
        )
    }
}

pub(crate) fn rewrite_match(
    context: &RewriteContext<'_>,
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
    let alt_block_sep = &shape.indent.to_string_with_newline(context.config);
    let block_sep = match context.config.control_brace_style() {
        ControlBraceStyle::AlwaysNextLine => alt_block_sep,
        _ if last_line_extendable(&cond_str) => " ",
        // 2 = ` {`
        _ if cond_str.contains('\n') || cond_str.len() + 2 > cond_shape.width => alt_block_sep,
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
        context
            .snippet_provider
            .span_after(mk_sp(cond.span.hi(), hi), "{")
    } else {
        inner_attrs[inner_attrs.len() - 1].span.hi()
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
        let span_after_cond = mk_sp(cond.span.hi(), span.hi());
        Some(format!(
            "match {}{}{{\n{}{}{}\n{}}}",
            cond_str,
            block_sep,
            inner_attrs_str,
            nested_indent_str,
            rewrite_match_arms(context, arms, shape, span_after_cond, open_brace_pos)?,
            shape.indent.to_string(context.config),
        ))
    }
}

fn arm_comma(config: &Config, body: &ast::Expr, is_last: bool) -> &'static str {
    if is_last && config.trailing_comma() == SeparatorTactic::Never {
        ""
    } else if config.match_block_trailing_comma() {
        ","
    } else if let ast::ExprKind::Block(ref block, _) = body.kind {
        if let ast::BlockCheckMode::Default = block.rules {
            ""
        } else {
            ","
        }
    } else {
        ","
    }
}

/// Collect a byte position of the beginning `|` for each arm, if available.
fn collect_beginning_verts(
    context: &RewriteContext<'_>,
    arms: &[ast::Arm],
) -> Vec<Option<BytePos>> {
    arms.iter()
        .map(|a| {
            context
                .snippet(a.pat.span)
                .starts_with('|')
                .then(|| a.pat.span().lo())
        })
        .collect()
}

fn rewrite_match_arms(
    context: &RewriteContext<'_>,
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
        .take(arm_len.saturating_sub(1))
        .chain(repeat(true));
    let beginning_verts = collect_beginning_verts(context, arms);
    let items = itemize_list(
        context.snippet_provider,
        arms.iter()
            .zip(is_last_iter)
            .zip(beginning_verts.into_iter())
            .map(|((arm, is_last), beginning_vert)| ArmWrapper::new(arm, is_last, beginning_vert)),
        "}",
        "|",
        |arm| arm.span().lo(),
        |arm| arm.span().hi(),
        |arm| arm.rewrite(context, arm_shape),
        open_brace_pos,
        span.hi(),
        false,
    );
    let arms_vec: Vec<_> = items.collect();
    // We will add/remove commas inside `arm.rewrite()`, and hence no separator here.
    let fmt = ListFormatting::new(arm_shape, context.config)
        .separator("")
        .preserve_newline(true);

    write_list(&arms_vec, &fmt)
}

fn rewrite_match_arm(
    context: &RewriteContext<'_>,
    arm: &ast::Arm,
    shape: Shape,
    is_last: bool,
    has_leading_pipe: bool,
) -> Option<String> {
    let (missing_span, attrs_str) = if !arm.attrs.is_empty() {
        if contains_skip(&arm.attrs) {
            let (_, body) = flatten_arm_body(context, &arm.body, None);
            // `arm.span()` does not include trailing comma, add it manually.
            return Some(format!(
                "{}{}",
                context.snippet(arm.span()),
                arm_comma(context.config, body, is_last),
            ));
        }
        let missing_span = mk_sp(arm.attrs[arm.attrs.len() - 1].span.hi(), arm.pat.span.lo());
        (missing_span, arm.attrs.rewrite(context, shape)?)
    } else {
        (mk_sp(arm.span().lo(), arm.span().lo()), String::new())
    };

    // Leading pipe offset
    // 2 = `| `
    let (pipe_offset, pipe_str) = match context.config.match_arm_leading_pipes() {
        MatchArmLeadingPipe::Never => (0, ""),
        MatchArmLeadingPipe::Preserve if !has_leading_pipe => (0, ""),
        MatchArmLeadingPipe::Preserve | MatchArmLeadingPipe::Always => (2, "| "),
    };

    // Patterns
    // 5 = ` => {`
    let pat_shape = shape.sub_width(5)?.offset_left(pipe_offset)?;
    let pats_str = arm.pat.rewrite(context, pat_shape)?;

    // Guard
    let block_like_pat = trimmed_last_line_width(&pats_str) <= context.config.tab_spaces();
    let new_line_guard = pats_str.contains('\n') && !block_like_pat;
    let guard_str = rewrite_guard(
        context,
        &arm.guard,
        shape,
        trimmed_last_line_width(&pats_str),
        new_line_guard,
    )?;

    let lhs_str = combine_strs_with_missing_comments(
        context,
        &attrs_str,
        &format!("{}{}{}", pipe_str, pats_str, guard_str),
        missing_span,
        shape,
        false,
    )?;

    let arrow_span = mk_sp(arm.pat.span.hi(), arm.body.span().lo());
    rewrite_match_body(
        context,
        &arm.body,
        &lhs_str,
        shape,
        guard_str.contains('\n'),
        arrow_span,
        is_last,
    )
}

fn stmt_is_expr_mac(stmt: &ast::Stmt) -> bool {
    if let ast::StmtKind::Expr(expr) = &stmt.kind {
        if let ast::ExprKind::MacCall(_) = &expr.kind {
            return true;
        }
    }
    false
}

fn block_can_be_flattened<'a>(
    context: &RewriteContext<'_>,
    expr: &'a ast::Expr,
) -> Option<&'a ast::Block> {
    match expr.kind {
        ast::ExprKind::Block(ref block, _)
            if !is_unsafe_block(block)
                && !context.inside_macro()
                && is_simple_block(context, block, Some(&expr.attrs))
                && !stmt_is_expr_mac(&block.stmts[0]) =>
        {
            Some(&*block)
        }
        _ => None,
    }
}

// (extend, body)
// @extend: true if the arm body can be put next to `=>`
// @body: flattened body, if the body is block with a single expression
fn flatten_arm_body<'a>(
    context: &'a RewriteContext<'_>,
    body: &'a ast::Expr,
    opt_shape: Option<Shape>,
) -> (bool, &'a ast::Expr) {
    let can_extend =
        |expr| !context.config.force_multiline_blocks() && can_flatten_block_around_this(expr);

    if let Some(block) = block_can_be_flattened(context, body) {
        if let ast::StmtKind::Expr(ref expr) = block.stmts[0].kind {
            if let ast::ExprKind::Block(..) = expr.kind {
                if expr.attrs.is_empty() {
                    flatten_arm_body(context, expr, None)
                } else {
                    (true, body)
                }
            } else {
                let cond_becomes_muti_line = opt_shape
                    .and_then(|shape| rewrite_cond(context, expr, shape))
                    .map_or(false, |cond| cond.contains('\n'));
                if cond_becomes_muti_line {
                    (false, &*body)
                } else {
                    (can_extend(expr), &*expr)
                }
            }
        } else {
            (false, &*body)
        }
    } else {
        (can_extend(body), &*body)
    }
}

fn rewrite_match_body(
    context: &RewriteContext<'_>,
    body: &ptr::P<ast::Expr>,
    pats_str: &str,
    shape: Shape,
    has_guard: bool,
    arrow_span: Span,
    is_last: bool,
) -> Option<String> {
    let (extend, body) = flatten_arm_body(
        context,
        body,
        shape.offset_left(extra_offset(pats_str, shape) + 4),
    );
    let (is_block, is_empty_block) = if let ast::ExprKind::Block(ref block, _) = body.kind {
        (true, is_empty_block(context, block, Some(&body.attrs)))
    } else {
        (false, false)
    };

    let comma = arm_comma(context.config, body, is_last);
    let alt_block_sep = &shape.indent.to_string_with_newline(context.config);

    let combine_orig_body = |body_str: &str| {
        let block_sep = match context.config.control_brace_style() {
            ControlBraceStyle::AlwaysNextLine if is_block => alt_block_sep,
            _ => " ",
        };

        Some(format!("{} =>{}{}{}", pats_str, block_sep, body_str, comma))
    };

    let next_line_indent = if !is_block || is_empty_block {
        shape.indent.block_indent(context.config)
    } else {
        shape.indent
    };

    let forbid_same_line =
        (has_guard && pats_str.contains('\n') && !is_empty_block) || !body.attrs.is_empty();

    // Look for comments between `=>` and the start of the body.
    let arrow_comment = {
        let arrow_snippet = context.snippet(arrow_span).trim();
        // search for the arrow starting from the end of the snippet since there may be a match
        // expression within the guard
        let arrow_index = arrow_snippet.rfind("=>").unwrap();
        // 2 = `=>`
        let comment_str = arrow_snippet[arrow_index + 2..].trim();
        if comment_str.is_empty() {
            String::new()
        } else {
            rewrite_comment(comment_str, false, shape, context.config)?
        }
    };

    let combine_next_line_body = |body_str: &str| {
        let nested_indent_str = next_line_indent.to_string_with_newline(context.config);

        if is_block {
            let mut result = pats_str.to_owned();
            result.push_str(" =>");
            if !arrow_comment.is_empty() {
                result.push_str(&nested_indent_str);
                result.push_str(&arrow_comment);
            }
            result.push_str(&nested_indent_str);
            result.push_str(body_str);
            result.push_str(comma);
            return Some(result);
        }

        let indent_str = shape.indent.to_string_with_newline(context.config);
        let (body_prefix, body_suffix) =
            if context.config.match_arm_blocks() && !context.inside_macro() {
                let comma = if context.config.match_block_trailing_comma() {
                    ","
                } else {
                    ""
                };
                let semicolon = if context.config.version() == Version::One {
                    ""
                } else {
                    if semicolon_for_expr(context, body) {
                        ";"
                    } else {
                        ""
                    }
                };
                ("{", format!("{}{}}}{}", semicolon, indent_str, comma))
            } else {
                ("", String::from(","))
            };

        let block_sep = match context.config.control_brace_style() {
            ControlBraceStyle::AlwaysNextLine => format!("{}{}", alt_block_sep, body_prefix),
            _ if body_prefix.is_empty() => "".to_owned(),
            _ if forbid_same_line || !arrow_comment.is_empty() => {
                format!("{}{}", alt_block_sep, body_prefix)
            }
            _ => format!(" {}", body_prefix),
        } + &nested_indent_str;

        let mut result = pats_str.to_owned();
        result.push_str(" =>");
        if !arrow_comment.is_empty() {
            result.push_str(&indent_str);
            result.push_str(&arrow_comment);
        }
        result.push_str(&block_sep);
        result.push_str(body_str);
        result.push_str(&body_suffix);
        Some(result)
    };

    // Let's try and get the arm body on the same line as the condition.
    // 4 = ` => `.len()
    let orig_body_shape = shape
        .offset_left(extra_offset(pats_str, shape) + 4)
        .and_then(|shape| shape.sub_width(comma.len()));
    let orig_body = if forbid_same_line || !arrow_comment.is_empty() {
        None
    } else if let Some(body_shape) = orig_body_shape {
        let rewrite = nop_block_collapse(
            format_expr(body, ExprType::Statement, context, body_shape),
            body_shape.width,
        );

        match rewrite {
            Some(ref body_str)
                if is_block
                    || (!body_str.contains('\n')
                        && unicode_str_width(body_str) <= body_shape.width) =>
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
            if prefer_next_line(orig_str, next_line_str, RhsTactics::Default) =>
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
    context: &RewriteContext<'_>,
    guard: &Option<ptr::P<ast::Expr>>,
    shape: Shape,
    // The amount of space used up on this line for the pattern in
    // the arm (excludes offset).
    pattern_width: usize,
    multiline_pattern: bool,
) -> Option<String> {
    if let Some(ref guard) = *guard {
        // First try to fit the guard string on the same line as the pattern.
        // 4 = ` if `, 5 = ` => {`
        let cond_shape = shape
            .offset_left(pattern_width + 4)
            .and_then(|s| s.sub_width(5));
        if !multiline_pattern {
            if let Some(cond_shape) = cond_shape {
                if let Some(cond_str) = guard.rewrite(context, cond_shape) {
                    if !cond_str.contains('\n') || pattern_width <= context.config.tab_spaces() {
                        return Some(format!(" if {}", cond_str));
                    }
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
                    "{}if {}",
                    cond_shape.indent.to_string_with_newline(context.config),
                    cond_str
                ));
            }
        }

        None
    } else {
        Some(String::new())
    }
}

fn nop_block_collapse(block_str: Option<String>, budget: usize) -> Option<String> {
    debug!("nop_block_collapse {:?} {}", block_str, budget);
    block_str.map(|block_str| {
        if block_str.starts_with('{')
            && budget >= 2
            && (block_str[1..].find(|c: char| !c.is_whitespace()).unwrap() == block_str.len() - 2)
        {
            String::from("{}")
        } else {
            block_str
        }
    })
}

fn can_flatten_block_around_this(body: &ast::Expr) -> bool {
    match body.kind {
        // We do not allow `if` to stay on the same line, since we could easily mistake
        // `pat => if cond { ... }` and `pat if cond => { ... }`.
        ast::ExprKind::If(..) => false,
        // We do not allow collapsing a block around expression with condition
        // to avoid it being cluttered with match arm.
        ast::ExprKind::ForLoop(..) | ast::ExprKind::While(..) => false,
        ast::ExprKind::Loop(..)
        | ast::ExprKind::Match(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::Closure(..)
        | ast::ExprKind::Array(..)
        | ast::ExprKind::Call(..)
        | ast::ExprKind::MethodCall(..)
        | ast::ExprKind::MacCall(..)
        | ast::ExprKind::Struct(..)
        | ast::ExprKind::Tup(..) => true,
        ast::ExprKind::AddrOf(_, _, ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Index(ref expr, _)
        | ast::ExprKind::Cast(ref expr, _) => can_flatten_block_around_this(expr),
        _ => false,
    }
}
