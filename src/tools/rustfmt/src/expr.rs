use std::borrow::Cow;
use std::cmp::min;

use itertools::Itertools;
use rustc_ast::token::{DelimToken, LitKind};
use rustc_ast::{ast, ptr};
use rustc_span::{BytePos, Span};

use crate::chains::rewrite_chain;
use crate::closures;
use crate::comment::{
    combine_strs_with_missing_comments, contains_comment, recover_comment_removed, rewrite_comment,
    rewrite_missing_comment, CharClasses, FindUncommented,
};
use crate::config::lists::*;
use crate::config::{Config, ControlBraceStyle, HexLiteralCase, IndentStyle, Version};
use crate::lists::{
    definitive_tactic, itemize_list, shape_for_tactic, struct_lit_formatting, struct_lit_shape,
    struct_lit_tactic, write_list, ListFormatting, Separator,
};
use crate::macros::{rewrite_macro, MacroPosition};
use crate::matches::rewrite_match;
use crate::overflow::{self, IntoOverflowableItem, OverflowableItem};
use crate::pairs::{rewrite_all_pairs, rewrite_pair, PairParts};
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::{Indent, Shape};
use crate::source_map::{LineRangeUtils, SpanUtils};
use crate::spanned::Spanned;
use crate::string::{rewrite_string, StringFormat};
use crate::types::{rewrite_path, PathContext};
use crate::utils::{
    colon_spaces, contains_skip, count_newlines, first_line_ends_with, inner_attributes,
    last_line_extendable, last_line_width, mk_sp, outer_attributes, semicolon_for_expr,
    unicode_str_width, wrap_str,
};
use crate::vertical::rewrite_with_alignment;
use crate::visitor::FmtVisitor;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        format_expr(self, ExprType::SubExpression, context, shape)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum ExprType {
    Statement,
    SubExpression,
}

pub(crate) fn format_expr(
    expr: &ast::Expr,
    expr_type: ExprType,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    skip_out_of_file_lines_range!(context, expr.span);

    if contains_skip(&*expr.attrs) {
        return Some(context.snippet(expr.span()).to_owned());
    }
    let shape = if expr_type == ExprType::Statement && semicolon_for_expr(context, expr) {
        shape.sub_width(1)?
    } else {
        shape
    };

    let expr_rw = match expr.kind {
        ast::ExprKind::Array(ref expr_vec) => rewrite_array(
            "",
            expr_vec.iter(),
            expr.span,
            context,
            shape,
            choose_separator_tactic(context, expr.span),
            None,
        ),
        ast::ExprKind::Lit(ref l) => {
            if let Some(expr_rw) = rewrite_literal(context, l, shape) {
                Some(expr_rw)
            } else {
                if let LitKind::StrRaw(_) = l.token.kind {
                    Some(context.snippet(l.span).trim().into())
                } else {
                    None
                }
            }
        }
        ast::ExprKind::Call(ref callee, ref args) => {
            let inner_span = mk_sp(callee.span.hi(), expr.span.hi());
            let callee_str = callee.rewrite(context, shape)?;
            rewrite_call(context, &callee_str, args, inner_span, shape)
        }
        ast::ExprKind::Paren(ref subexpr) => rewrite_paren(context, subexpr, shape, expr.span),
        ast::ExprKind::Binary(op, ref lhs, ref rhs) => {
            // FIXME: format comments between operands and operator
            rewrite_all_pairs(expr, shape, context).or_else(|| {
                rewrite_pair(
                    &**lhs,
                    &**rhs,
                    PairParts::infix(&format!(" {} ", context.snippet(op.span))),
                    context,
                    shape,
                    context.config.binop_separator(),
                )
            })
        }
        ast::ExprKind::Unary(op, ref subexpr) => rewrite_unary_op(context, op, subexpr, shape),
        ast::ExprKind::Struct(ref struct_expr) => {
            let ast::StructExpr {
                qself,
                fields,
                path,
                rest,
            } = &**struct_expr;
            rewrite_struct_lit(
                context,
                path,
                qself.as_ref(),
                fields,
                rest,
                &expr.attrs,
                expr.span,
                shape,
            )
        }
        ast::ExprKind::Tup(ref items) => {
            rewrite_tuple(context, items.iter(), expr.span, shape, items.len() == 1)
        }
        ast::ExprKind::Let(..) => None,
        ast::ExprKind::If(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::While(..) => to_control_flow(expr, expr_type)
            .and_then(|control_flow| control_flow.rewrite(context, shape)),
        ast::ExprKind::ConstBlock(ref anon_const) => {
            Some(format!("const {}", anon_const.rewrite(context, shape)?))
        }
        ast::ExprKind::Block(ref block, opt_label) => {
            match expr_type {
                ExprType::Statement => {
                    if is_unsafe_block(block) {
                        rewrite_block(block, Some(&expr.attrs), opt_label, context, shape)
                    } else if let rw @ Some(_) =
                        rewrite_empty_block(context, block, Some(&expr.attrs), opt_label, "", shape)
                    {
                        // Rewrite block without trying to put it in a single line.
                        rw
                    } else {
                        let prefix = block_prefix(context, block, shape)?;

                        rewrite_block_with_visitor(
                            context,
                            &prefix,
                            block,
                            Some(&expr.attrs),
                            opt_label,
                            shape,
                            true,
                        )
                    }
                }
                ExprType::SubExpression => {
                    rewrite_block(block, Some(&expr.attrs), opt_label, context, shape)
                }
            }
        }
        ast::ExprKind::Match(ref cond, ref arms) => {
            rewrite_match(context, cond, arms, shape, expr.span, &expr.attrs)
        }
        ast::ExprKind::Path(ref qself, ref path) => {
            rewrite_path(context, PathContext::Expr, qself.as_ref(), path, shape)
        }
        ast::ExprKind::Assign(ref lhs, ref rhs, _) => {
            rewrite_assignment(context, lhs, rhs, None, shape)
        }
        ast::ExprKind::AssignOp(ref op, ref lhs, ref rhs) => {
            rewrite_assignment(context, lhs, rhs, Some(op), shape)
        }
        ast::ExprKind::Continue(ref opt_label) => {
            let id_str = match *opt_label {
                Some(label) => format!(" {}", label.ident),
                None => String::new(),
            };
            Some(format!("continue{}", id_str))
        }
        ast::ExprKind::Break(ref opt_label, ref opt_expr) => {
            let id_str = match *opt_label {
                Some(label) => format!(" {}", label.ident),
                None => String::new(),
            };

            if let Some(ref expr) = *opt_expr {
                rewrite_unary_prefix(context, &format!("break{} ", id_str), &**expr, shape)
            } else {
                Some(format!("break{}", id_str))
            }
        }
        ast::ExprKind::Yield(ref opt_expr) => {
            if let Some(ref expr) = *opt_expr {
                rewrite_unary_prefix(context, "yield ", &**expr, shape)
            } else {
                Some("yield".to_string())
            }
        }
        ast::ExprKind::Closure(capture, ref is_async, movability, ref fn_decl, ref body, _) => {
            closures::rewrite_closure(
                capture, is_async, movability, fn_decl, body, expr.span, context, shape,
            )
        }
        ast::ExprKind::Try(..)
        | ast::ExprKind::Field(..)
        | ast::ExprKind::MethodCall(..)
        | ast::ExprKind::Await(_) => rewrite_chain(expr, context, shape),
        ast::ExprKind::MacCall(ref mac) => {
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
        ast::ExprKind::AddrOf(borrow_kind, mutability, ref expr) => {
            rewrite_expr_addrof(context, borrow_kind, mutability, expr, shape)
        }
        ast::ExprKind::Cast(ref expr, ref ty) => rewrite_pair(
            &**expr,
            &**ty,
            PairParts::infix(" as "),
            context,
            shape,
            SeparatorPlace::Front,
        ),
        ast::ExprKind::Type(ref expr, ref ty) => rewrite_pair(
            &**expr,
            &**ty,
            PairParts::infix(": "),
            context,
            shape,
            SeparatorPlace::Back,
        ),
        ast::ExprKind::Index(ref expr, ref index) => {
            rewrite_index(&**expr, &**index, context, shape)
        }
        ast::ExprKind::Repeat(ref expr, ref repeats) => rewrite_pair(
            &**expr,
            &*repeats.value,
            PairParts::new("[", "; ", "]"),
            context,
            shape,
            SeparatorPlace::Back,
        ),
        ast::ExprKind::Range(ref lhs, ref rhs, limits) => {
            let delim = match limits {
                ast::RangeLimits::HalfOpen => "..",
                ast::RangeLimits::Closed => "..=",
            };

            fn needs_space_before_range(context: &RewriteContext<'_>, lhs: &ast::Expr) -> bool {
                match lhs.kind {
                    ast::ExprKind::Lit(ref lit) => match lit.kind {
                        ast::LitKind::Float(_, ast::LitFloatType::Unsuffixed) => {
                            context.snippet(lit.span).ends_with('.')
                        }
                        _ => false,
                    },
                    ast::ExprKind::Unary(_, ref expr) => needs_space_before_range(context, expr),
                    _ => false,
                }
            }

            fn needs_space_after_range(rhs: &ast::Expr) -> bool {
                // Don't format `.. ..` into `....`, which is invalid.
                //
                // This check is unnecessary for `lhs`, because a range
                // starting from another range needs parentheses as `(x ..) ..`
                // (`x .. ..` is a range from `x` to `..`).
                matches!(rhs.kind, ast::ExprKind::Range(None, _, _))
            }

            let default_sp_delim = |lhs: Option<&ast::Expr>, rhs: Option<&ast::Expr>| {
                let space_if = |b: bool| if b { " " } else { "" };

                format!(
                    "{}{}{}",
                    lhs.map_or("", |lhs| space_if(needs_space_before_range(context, lhs))),
                    delim,
                    rhs.map_or("", |rhs| space_if(needs_space_after_range(rhs))),
                )
            };

            match (lhs.as_ref().map(|x| &**x), rhs.as_ref().map(|x| &**x)) {
                (Some(lhs), Some(rhs)) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!(" {} ", delim)
                    } else {
                        default_sp_delim(Some(lhs), Some(rhs))
                    };
                    rewrite_pair(
                        &*lhs,
                        &*rhs,
                        PairParts::infix(&sp_delim),
                        context,
                        shape,
                        context.config.binop_separator(),
                    )
                }
                (None, Some(rhs)) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!("{} ", delim)
                    } else {
                        default_sp_delim(None, Some(rhs))
                    };
                    rewrite_unary_prefix(context, &sp_delim, &*rhs, shape)
                }
                (Some(lhs), None) => {
                    let sp_delim = if context.config.spaces_around_ranges() {
                        format!(" {}", delim)
                    } else {
                        default_sp_delim(Some(lhs), None)
                    };
                    rewrite_unary_suffix(context, &sp_delim, &*lhs, shape)
                }
                (None, None) => Some(delim.to_owned()),
            }
        }
        // We do not format these expressions yet, but they should still
        // satisfy our width restrictions.
        // Style Guide RFC for InlineAsm variant pending
        // https://github.com/rust-dev-tools/fmt-rfcs/issues/152
        ast::ExprKind::LlvmInlineAsm(..) | ast::ExprKind::InlineAsm(..) => {
            Some(context.snippet(expr.span).to_owned())
        }
        ast::ExprKind::TryBlock(ref block) => {
            if let rw @ Some(_) =
                rewrite_single_line_block(context, "try ", block, Some(&expr.attrs), None, shape)
            {
                rw
            } else {
                // 9 = `try `
                let budget = shape.width.saturating_sub(9);
                Some(format!(
                    "{}{}",
                    "try ",
                    rewrite_block(
                        block,
                        Some(&expr.attrs),
                        None,
                        context,
                        Shape::legacy(budget, shape.indent)
                    )?
                ))
            }
        }
        ast::ExprKind::Async(capture_by, _node_id, ref block) => {
            let mover = if capture_by == ast::CaptureBy::Value {
                "move "
            } else {
                ""
            };
            if let rw @ Some(_) = rewrite_single_line_block(
                context,
                format!("{}{}", "async ", mover).as_str(),
                block,
                Some(&expr.attrs),
                None,
                shape,
            ) {
                rw
            } else {
                // 6 = `async `
                let budget = shape.width.saturating_sub(6);
                Some(format!(
                    "{}{}{}",
                    "async ",
                    mover,
                    rewrite_block(
                        block,
                        Some(&expr.attrs),
                        None,
                        context,
                        Shape::legacy(budget, shape.indent)
                    )?
                ))
            }
        }
        ast::ExprKind::Underscore => Some("_".to_owned()),
        ast::ExprKind::Err => None,
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

pub(crate) fn rewrite_array<'a, T: 'a + IntoOverflowableItem<'a>>(
    name: &'a str,
    exprs: impl Iterator<Item = &'a T>,
    span: Span,
    context: &'a RewriteContext<'_>,
    shape: Shape,
    force_separator_tactic: Option<SeparatorTactic>,
    delim_token: Option<DelimToken>,
) -> Option<String> {
    overflow::rewrite_with_square_brackets(
        context,
        name,
        exprs,
        shape,
        span,
        force_separator_tactic,
        delim_token,
    )
}

fn rewrite_empty_block(
    context: &RewriteContext<'_>,
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
    label: Option<ast::Label>,
    prefix: &str,
    shape: Shape,
) -> Option<String> {
    if block_has_statements(block) {
        return None;
    }

    let label_str = rewrite_label(label);
    if attrs.map_or(false, |a| !inner_attributes(a).is_empty()) {
        return None;
    }

    if !block_contains_comment(context, block) && shape.width >= 2 {
        return Some(format!("{}{}{{}}", prefix, label_str));
    }

    // If a block contains only a single-line comment, then leave it on one line.
    let user_str = context.snippet(block.span);
    let user_str = user_str.trim();
    if user_str.starts_with('{') && user_str.ends_with('}') {
        let comment_str = user_str[1..user_str.len() - 1].trim();
        if block.stmts.is_empty()
            && !comment_str.contains('\n')
            && !comment_str.starts_with("//")
            && comment_str.len() + 4 <= shape.width
        {
            return Some(format!("{}{}{{ {} }}", prefix, label_str, comment_str));
        }
    }

    None
}

fn block_prefix(context: &RewriteContext<'_>, block: &ast::Block, shape: Shape) -> Option<String> {
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
    context: &RewriteContext<'_>,
    prefix: &str,
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
    label: Option<ast::Label>,
    shape: Shape,
) -> Option<String> {
    if is_simple_block(context, block, attrs) {
        let expr_shape = shape.offset_left(last_line_width(prefix))?;
        let expr_str = block.stmts[0].rewrite(context, expr_shape)?;
        let label_str = rewrite_label(label);
        let result = format!("{}{}{{ {} }}", prefix, label_str, expr_str);
        if result.len() <= shape.width && !result.contains('\n') {
            return Some(result);
        }
    }
    None
}

pub(crate) fn rewrite_block_with_visitor(
    context: &RewriteContext<'_>,
    prefix: &str,
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
    label: Option<ast::Label>,
    shape: Shape,
    has_braces: bool,
) -> Option<String> {
    if let rw @ Some(_) = rewrite_empty_block(context, block, attrs, label, prefix, shape) {
        return rw;
    }

    let mut visitor = FmtVisitor::from_context(context);
    visitor.block_indent = shape.indent;
    visitor.is_if_else_block = context.is_if_else_block();
    match (block.rules, label) {
        (ast::BlockCheckMode::Unsafe(..), _) | (ast::BlockCheckMode::Default, Some(_)) => {
            let snippet = context.snippet(block.span);
            let open_pos = snippet.find_uncommented("{")?;
            visitor.last_pos = block.span.lo() + BytePos(open_pos as u32)
        }
        (ast::BlockCheckMode::Default, None) => visitor.last_pos = block.span.lo(),
    }

    let inner_attrs = attrs.map(inner_attributes);
    let label_str = rewrite_label(label);
    visitor.visit_block(block, inner_attrs.as_deref(), has_braces);
    let visitor_context = visitor.get_context();
    context
        .skipped_range
        .borrow_mut()
        .append(&mut visitor_context.skipped_range.borrow_mut());
    Some(format!("{}{}{}", prefix, label_str, visitor.buffer))
}

impl Rewrite for ast::Block {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        rewrite_block(self, None, None, context, shape)
    }
}

fn rewrite_block(
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
    label: Option<ast::Label>,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    let prefix = block_prefix(context, block, shape)?;

    // shape.width is used only for the single line case: either the empty block `{}`,
    // or an unsafe expression `unsafe { e }`.
    if let rw @ Some(_) = rewrite_empty_block(context, block, attrs, label, &prefix, shape) {
        return rw;
    }

    let result = rewrite_block_with_visitor(context, &prefix, block, attrs, label, shape, true);
    if let Some(ref result_str) = result {
        if result_str.lines().count() <= 3 {
            if let rw @ Some(_) =
                rewrite_single_line_block(context, &prefix, block, attrs, label, shape)
            {
                return rw;
            }
        }
    }

    result
}

// Rewrite condition if the given expression has one.
pub(crate) fn rewrite_cond(
    context: &RewriteContext<'_>,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    match expr.kind {
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
                .map(|rw| rw.0)
        }),
    }
}

// Abstraction over control flow expressions
#[derive(Debug)]
struct ControlFlow<'a> {
    cond: Option<&'a ast::Expr>,
    block: &'a ast::Block,
    else_block: Option<&'a ast::Expr>,
    label: Option<ast::Label>,
    pat: Option<&'a ast::Pat>,
    keyword: &'a str,
    matcher: &'a str,
    connector: &'a str,
    allow_single_line: bool,
    // HACK: `true` if this is an `if` expression in an `else if`.
    nested_if: bool,
    span: Span,
}

fn extract_pats_and_cond(expr: &ast::Expr) -> (Option<&ast::Pat>, &ast::Expr) {
    match expr.kind {
        ast::ExprKind::Let(ref pat, ref cond, _) => (Some(pat), cond),
        _ => (None, expr),
    }
}

// FIXME: Refactor this.
fn to_control_flow(expr: &ast::Expr, expr_type: ExprType) -> Option<ControlFlow<'_>> {
    match expr.kind {
        ast::ExprKind::If(ref cond, ref if_block, ref else_block) => {
            let (pat, cond) = extract_pats_and_cond(cond);
            Some(ControlFlow::new_if(
                cond,
                pat,
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
            let (pat, cond) = extract_pats_and_cond(cond);
            Some(ControlFlow::new_while(pat, cond, block, label, expr.span))
        }
        _ => None,
    }
}

fn choose_matcher(pat: Option<&ast::Pat>) -> &'static str {
    pat.map_or("", |_| "let")
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
        let matcher = choose_matcher(pat);
        ControlFlow {
            cond: Some(cond),
            block,
            else_block,
            label: None,
            pat,
            keyword: "if",
            matcher,
            connector: " =",
            allow_single_line,
            nested_if,
            span,
        }
    }

    fn new_loop(block: &'a ast::Block, label: Option<ast::Label>, span: Span) -> ControlFlow<'a> {
        ControlFlow {
            cond: None,
            block,
            else_block: None,
            label,
            pat: None,
            keyword: "loop",
            matcher: "",
            connector: "",
            allow_single_line: false,
            nested_if: false,
            span,
        }
    }

    fn new_while(
        pat: Option<&'a ast::Pat>,
        cond: &'a ast::Expr,
        block: &'a ast::Block,
        label: Option<ast::Label>,
        span: Span,
    ) -> ControlFlow<'a> {
        let matcher = choose_matcher(pat);
        ControlFlow {
            cond: Some(cond),
            block,
            else_block: None,
            label,
            pat,
            keyword: "while",
            matcher,
            connector: " =",
            allow_single_line: false,
            nested_if: false,
            span,
        }
    }

    fn new_for(
        pat: &'a ast::Pat,
        cond: &'a ast::Expr,
        block: &'a ast::Block,
        label: Option<ast::Label>,
        span: Span,
    ) -> ControlFlow<'a> {
        ControlFlow {
            cond: Some(cond),
            block,
            else_block: None,
            label,
            pat: Some(pat),
            keyword: "for",
            matcher: "",
            connector: " in",
            allow_single_line: false,
            nested_if: false,
            span,
        }
    }

    fn rewrite_single_line(
        &self,
        pat_expr_str: &str,
        context: &RewriteContext<'_>,
        width: usize,
    ) -> Option<String> {
        assert!(self.allow_single_line);
        let else_block = self.else_block?;
        let fixed_cost = self.keyword.len() + "  {  } else {  }".len();

        if let ast::ExprKind::Block(ref else_node, _) = else_block.kind {
            if !is_simple_block(context, self.block, None)
                || !is_simple_block(context, else_node, None)
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

/// Returns `true` if the last line of pat_str has leading whitespace and it is wider than the
/// shape's indent.
fn last_line_offsetted(start_column: usize, pat_str: &str) -> bool {
    let mut leading_whitespaces = 0;
    for c in pat_str.chars().rev() {
        match c {
            '\n' => break,
            _ if c.is_whitespace() => leading_whitespaces += 1,
            _ => leading_whitespaces = 0,
        }
    }
    leading_whitespaces > start_column
}

impl<'a> ControlFlow<'a> {
    fn rewrite_pat_expr(
        &self,
        context: &RewriteContext<'_>,
        expr: &ast::Expr,
        shape: Shape,
        offset: usize,
    ) -> Option<String> {
        debug!("rewrite_pat_expr {:?} {:?} {:?}", shape, self.pat, expr);

        let cond_shape = shape.offset_left(offset)?;
        if let Some(pat) = self.pat {
            let matcher = if self.matcher.is_empty() {
                self.matcher.to_owned()
            } else {
                format!("{} ", self.matcher)
            };
            let pat_shape = cond_shape
                .offset_left(matcher.len())?
                .sub_width(self.connector.len())?;
            let pat_string = pat.rewrite(context, pat_shape)?;
            let comments_lo = context
                .snippet_provider
                .span_after(self.span.with_lo(pat.span.hi()), self.connector.trim());
            let comments_span = mk_sp(comments_lo, expr.span.lo());
            return rewrite_assign_rhs_with_comments(
                context,
                &format!("{}{}{}", matcher, pat_string, self.connector),
                expr,
                cond_shape,
                &RhsAssignKind::Expr(&expr.kind, expr.span),
                RhsTactics::Default,
                comments_span,
                true,
            );
        }

        let expr_rw = expr.rewrite(context, cond_shape);
        // The expression may (partially) fit on the current line.
        // We do not allow splitting between `if` and condition.
        if self.keyword == "if" || expr_rw.is_some() {
            return expr_rw;
        }

        // The expression won't fit on the current line, jump to next.
        let nested_shape = shape
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config);
        let nested_indent_str = nested_shape.indent.to_string_with_newline(context.config);
        expr.rewrite(context, nested_shape)
            .map(|expr_rw| format!("{}{}", nested_indent_str, expr_rw))
    }

    fn rewrite_cond(
        &self,
        context: &RewriteContext<'_>,
        shape: Shape,
        alt_block_sep: &str,
    ) -> Option<(String, usize)> {
        // Do not take the rhs overhead from the upper expressions into account
        // when rewriting pattern.
        let new_width = context.budget(shape.used_width());
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
            Some(cond) => self.rewrite_pat_expr(context, cond, constr_shape, offset)?,
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
            .saturating_sub(constr_shape.used_width() + offset + brace_overhead);
        let force_newline_brace = (pat_expr_string.contains('\n')
            || pat_expr_string.len() > one_line_budget)
            && (!last_line_extendable(&pat_expr_string)
                || last_line_offsetted(shape.used_width(), &pat_expr_string));

        // Try to format if-else on single line.
        if self.allow_single_line && context.config.single_line_if_else_max_width() > 0 {
            let trial = self.rewrite_single_line(&pat_expr_string, context, shape.width);

            if let Some(cond_str) = trial {
                if cond_str.len() <= context.config.single_line_if_else_max_width() {
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
        let lo = self
            .label
            .map_or(self.span.lo(), |label| label.ident.span.hi());
        let between_kwd_cond = mk_sp(
            context
                .snippet_provider
                .span_after(mk_sp(lo, self.span.hi()), self.keyword.trim()),
            if self.pat.is_none() {
                cond_span.lo()
            } else if self.matcher.is_empty() {
                self.pat.unwrap().span.lo()
            } else {
                context
                    .snippet_provider
                    .span_before(self.span, self.matcher.trim())
            },
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
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        debug!("ControlFlow::rewrite {:?} {:?}", self, shape);

        let alt_block_sep = &shape.indent.to_string_with_newline(context.config);
        let (cond_str, used_width) = self.rewrite_cond(context, shape, alt_block_sep)?;
        // If `used_width` is 0, it indicates that whole control flow is written in a single line.
        if used_width == 0 {
            return Some(cond_str);
        }

        let block_width = shape.width.saturating_sub(used_width);
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
        let block_str = {
            let old_val = context.is_if_else_block.replace(self.else_block.is_some());
            let result =
                rewrite_block_with_visitor(context, "", self.block, None, None, block_shape, true);
            context.is_if_else_block.replace(old_val);
            result?
        };

        let mut result = format!("{}{}", cond_str, block_str);

        if let Some(else_block) = self.else_block {
            let shape = Shape::indented(shape.indent, context.config);
            let mut last_in_chain = false;
            let rewrite = match else_block.kind {
                // If the else expression is another if-else expression, prevent it
                // from being formatted on a single line.
                // Note how we're passing the original shape, as the
                // cost of "else" should not cascade.
                ast::ExprKind::If(ref cond, ref if_block, ref next_else_block) => {
                    let (pats, cond) = extract_pats_and_cond(cond);
                    ControlFlow::new_if(
                        cond,
                        pats,
                        if_block,
                        next_else_block.as_ref().map(|e| &**e),
                        false,
                        true,
                        mk_sp(else_block.span.lo(), self.span.hi()),
                    )
                    .rewrite(context, shape)
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
                    .snippet_provider
                    .span_before(mk_sp(self.block.span.hi(), else_block.span.lo()), "else"),
            );
            let between_kwd_else_block_comment =
                extract_comment(between_kwd_else_block, context, shape);

            let after_else = mk_sp(
                context
                    .snippet_provider
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

fn rewrite_label(opt_label: Option<ast::Label>) -> Cow<'static, str> {
    match opt_label {
        Some(label) => Cow::from(format!("{}: ", label.ident)),
        None => Cow::from(""),
    }
}

fn extract_comment(span: Span, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
    match rewrite_missing_comment(span, shape, context) {
        Some(ref comment) if !comment.is_empty() => Some(format!(
            "{indent}{}{indent}",
            comment,
            indent = shape.indent.to_string_with_newline(context.config)
        )),
        _ => None,
    }
}

pub(crate) fn block_contains_comment(context: &RewriteContext<'_>, block: &ast::Block) -> bool {
    contains_comment(context.snippet(block.span))
}

// Checks that a block contains no statements, an expression and no comments or
// attributes.
// FIXME: incorrectly returns false when comment is contained completely within
// the expression.
pub(crate) fn is_simple_block(
    context: &RewriteContext<'_>,
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
) -> bool {
    block.stmts.len() == 1
        && stmt_is_expr(&block.stmts[0])
        && !block_contains_comment(context, block)
        && attrs.map_or(true, |a| a.is_empty())
}

/// Checks whether a block contains at most one statement or expression, and no
/// comments or attributes.
pub(crate) fn is_simple_block_stmt(
    context: &RewriteContext<'_>,
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
) -> bool {
    block.stmts.len() <= 1
        && !block_contains_comment(context, block)
        && attrs.map_or(true, |a| a.is_empty())
}

fn block_has_statements(block: &ast::Block) -> bool {
    block
        .stmts
        .iter()
        .any(|stmt| !matches!(stmt.kind, ast::StmtKind::Empty))
}

/// Checks whether a block contains no statements, expressions, comments, or
/// inner attributes.
pub(crate) fn is_empty_block(
    context: &RewriteContext<'_>,
    block: &ast::Block,
    attrs: Option<&[ast::Attribute]>,
) -> bool {
    !block_has_statements(block)
        && !block_contains_comment(context, block)
        && attrs.map_or(true, |a| inner_attributes(a).is_empty())
}

pub(crate) fn stmt_is_expr(stmt: &ast::Stmt) -> bool {
    matches!(stmt.kind, ast::StmtKind::Expr(..))
}

pub(crate) fn is_unsafe_block(block: &ast::Block) -> bool {
    matches!(block.rules, ast::BlockCheckMode::Unsafe(..))
}

pub(crate) fn rewrite_literal(
    context: &RewriteContext<'_>,
    l: &ast::Lit,
    shape: Shape,
) -> Option<String> {
    match l.kind {
        ast::LitKind::Str(_, ast::StrStyle::Cooked) => rewrite_string_lit(context, l.span, shape),
        ast::LitKind::Int(..) => rewrite_int_lit(context, l, shape),
        _ => wrap_str(
            context.snippet(l.span).to_owned(),
            context.config.max_width(),
            shape,
        ),
    }
}

fn rewrite_string_lit(context: &RewriteContext<'_>, span: Span, shape: Shape) -> Option<String> {
    let string_lit = context.snippet(span);

    if !context.config.format_strings() {
        if string_lit
            .lines()
            .dropping_back(1)
            .all(|line| line.ends_with('\\'))
            && context.config.version() == Version::Two
        {
            return Some(string_lit.to_owned());
        } else {
            return wrap_str(string_lit.to_owned(), context.config.max_width(), shape);
        }
    }

    // Remove the quote characters.
    let str_lit = &string_lit[1..string_lit.len() - 1];

    rewrite_string(
        str_lit,
        &StringFormat::new(shape.visual_indent(0), context.config),
        shape.width.saturating_sub(2),
    )
}

fn rewrite_int_lit(context: &RewriteContext<'_>, lit: &ast::Lit, shape: Shape) -> Option<String> {
    let span = lit.span;
    let symbol = lit.token.symbol.as_str();

    if let Some(symbol_stripped) = symbol.strip_prefix("0x") {
        let hex_lit = match context.config.hex_literal_case() {
            HexLiteralCase::Preserve => None,
            HexLiteralCase::Upper => Some(symbol_stripped.to_ascii_uppercase()),
            HexLiteralCase::Lower => Some(symbol_stripped.to_ascii_lowercase()),
        };
        if let Some(hex_lit) = hex_lit {
            return wrap_str(
                format!(
                    "0x{}{}",
                    hex_lit,
                    lit.token.suffix.map_or(String::new(), |s| s.to_string())
                ),
                context.config.max_width(),
                shape,
            );
        }
    }

    wrap_str(
        context.snippet(span).to_owned(),
        context.config.max_width(),
        shape,
    )
}

fn choose_separator_tactic(context: &RewriteContext<'_>, span: Span) -> Option<SeparatorTactic> {
    if context.inside_macro() {
        if span_ends_with_comma(context, span) {
            Some(SeparatorTactic::Always)
        } else {
            Some(SeparatorTactic::Never)
        }
    } else {
        None
    }
}

pub(crate) fn rewrite_call(
    context: &RewriteContext<'_>,
    callee: &str,
    args: &[ptr::P<ast::Expr>],
    span: Span,
    shape: Shape,
) -> Option<String> {
    overflow::rewrite_with_parens(
        context,
        callee,
        args.iter(),
        shape,
        span,
        context.config.fn_call_width(),
        choose_separator_tactic(context, span),
    )
}

pub(crate) fn is_simple_expr(expr: &ast::Expr) -> bool {
    match expr.kind {
        ast::ExprKind::Lit(..) => true,
        ast::ExprKind::Path(ref qself, ref path) => qself.is_none() && path.segments.len() <= 1,
        ast::ExprKind::AddrOf(_, _, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Cast(ref expr, _)
        | ast::ExprKind::Field(ref expr, _)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr) => is_simple_expr(expr),
        ast::ExprKind::Index(ref lhs, ref rhs) => is_simple_expr(lhs) && is_simple_expr(rhs),
        ast::ExprKind::Repeat(ref lhs, ref rhs) => {
            is_simple_expr(lhs) && is_simple_expr(&*rhs.value)
        }
        _ => false,
    }
}

pub(crate) fn is_every_expr_simple(lists: &[OverflowableItem<'_>]) -> bool {
    lists.iter().all(OverflowableItem::is_simple)
}

pub(crate) fn can_be_overflowed_expr(
    context: &RewriteContext<'_>,
    expr: &ast::Expr,
    args_len: usize,
) -> bool {
    match expr.kind {
        _ if !expr.attrs.is_empty() => false,
        ast::ExprKind::Match(..) => {
            (context.use_block_indent() && args_len == 1)
                || (context.config.indent_style() == IndentStyle::Visual && args_len > 1)
                || context.config.overflow_delimited_expr()
        }
        ast::ExprKind::If(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::While(..) => {
            context.config.combine_control_expr() && context.use_block_indent() && args_len == 1
        }

        // Handle always block-like expressions
        ast::ExprKind::Async(..) | ast::ExprKind::Block(..) | ast::ExprKind::Closure(..) => true,

        // Handle `[]` and `{}`-like expressions
        ast::ExprKind::Array(..) | ast::ExprKind::Struct(..) => {
            context.config.overflow_delimited_expr()
                || (context.use_block_indent() && args_len == 1)
        }
        ast::ExprKind::MacCall(ref mac) => {
            match (
                rustc_ast::ast::MacDelimiter::from_token(mac.args.delim()),
                context.config.overflow_delimited_expr(),
            ) {
                (Some(ast::MacDelimiter::Bracket), true)
                | (Some(ast::MacDelimiter::Brace), true) => true,
                _ => context.use_block_indent() && args_len == 1,
            }
        }

        // Handle parenthetical expressions
        ast::ExprKind::Call(..) | ast::ExprKind::MethodCall(..) | ast::ExprKind::Tup(..) => {
            context.use_block_indent() && args_len == 1
        }

        // Handle unary-like expressions
        ast::ExprKind::AddrOf(_, _, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => can_be_overflowed_expr(context, expr, args_len),
        _ => false,
    }
}

pub(crate) fn is_nested_call(expr: &ast::Expr) -> bool {
    match expr.kind {
        ast::ExprKind::Call(..) | ast::ExprKind::MacCall(..) => true,
        ast::ExprKind::AddrOf(_, _, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Cast(ref expr, _) => is_nested_call(expr),
        _ => false,
    }
}

/// Returns `true` if a function call or a method call represented by the given span ends with a
/// trailing comma. This function is used when rewriting macro, as adding or removing a trailing
/// comma from macro can potentially break the code.
pub(crate) fn span_ends_with_comma(context: &RewriteContext<'_>, span: Span) -> bool {
    let mut result: bool = Default::default();
    let mut prev_char: char = Default::default();
    let closing_delimiters = &[')', '}', ']'];

    for (kind, c) in CharClasses::new(context.snippet(span).chars()) {
        match c {
            _ if kind.is_comment() || c.is_whitespace() => continue,
            c if closing_delimiters.contains(&c) => {
                result &= !closing_delimiters.contains(&prev_char);
            }
            ',' => result = true,
            _ => result = false,
        }
        prev_char = c;
    }

    result
}

fn rewrite_paren(
    context: &RewriteContext<'_>,
    mut subexpr: &ast::Expr,
    shape: Shape,
    mut span: Span,
) -> Option<String> {
    debug!("rewrite_paren, shape: {:?}", shape);

    // Extract comments within parens.
    let mut pre_span;
    let mut post_span;
    let mut pre_comment;
    let mut post_comment;
    let remove_nested_parens = context.config.remove_nested_parens();
    loop {
        // 1 = "(" or ")"
        pre_span = mk_sp(span.lo() + BytePos(1), subexpr.span.lo());
        post_span = mk_sp(subexpr.span.hi(), span.hi() - BytePos(1));
        pre_comment = rewrite_missing_comment(pre_span, shape, context)?;
        post_comment = rewrite_missing_comment(post_span, shape, context)?;

        // Remove nested parens if there are no comments.
        if let ast::ExprKind::Paren(ref subsubexpr) = subexpr.kind {
            if remove_nested_parens && pre_comment.is_empty() && post_comment.is_empty() {
                span = subexpr.span;
                subexpr = subsubexpr;
                continue;
            }
        }

        break;
    }

    // 1 = `(` and `)`
    let sub_shape = shape.offset_left(1)?.sub_width(1)?;
    let subexpr_str = subexpr.rewrite(context, sub_shape)?;
    let fits_single_line = !pre_comment.contains("//") && !post_comment.contains("//");
    if fits_single_line {
        Some(format!("({}{}{})", pre_comment, subexpr_str, post_comment))
    } else {
        rewrite_paren_in_multi_line(context, subexpr, shape, pre_span, post_span)
    }
}

fn rewrite_paren_in_multi_line(
    context: &RewriteContext<'_>,
    subexpr: &ast::Expr,
    shape: Shape,
    pre_span: Span,
    post_span: Span,
) -> Option<String> {
    let nested_indent = shape.indent.block_indent(context.config);
    let nested_shape = Shape::indented(nested_indent, context.config);
    let pre_comment = rewrite_missing_comment(pre_span, nested_shape, context)?;
    let post_comment = rewrite_missing_comment(post_span, nested_shape, context)?;
    let subexpr_str = subexpr.rewrite(context, nested_shape)?;

    let mut result = String::with_capacity(subexpr_str.len() * 2);
    result.push('(');
    if !pre_comment.is_empty() {
        result.push_str(&nested_indent.to_string_with_newline(context.config));
        result.push_str(&pre_comment);
    }
    result.push_str(&nested_indent.to_string_with_newline(context.config));
    result.push_str(&subexpr_str);
    if !post_comment.is_empty() {
        result.push_str(&nested_indent.to_string_with_newline(context.config));
        result.push_str(&post_comment);
    }
    result.push_str(&shape.indent.to_string_with_newline(context.config));
    result.push(')');

    Some(result)
}

fn rewrite_index(
    expr: &ast::Expr,
    index: &ast::Expr,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Option<String> {
    let expr_str = expr.rewrite(context, shape)?;

    let offset = last_line_width(&expr_str) + 1;
    let rhs_overhead = shape.rhs_overhead(context.config);
    let index_shape = if expr_str.contains('\n') {
        Shape::legacy(context.config.max_width(), shape.indent)
            .offset_left(offset)
            .and_then(|shape| shape.sub_width(1 + rhs_overhead))
    } else {
        match context.config.indent_style() {
            IndentStyle::Block => shape
                .offset_left(offset)
                .and_then(|shape| shape.sub_width(1)),
            IndentStyle::Visual => shape.visual_indent(offset).sub_width(offset + 1),
        }
    };
    let orig_index_rw = index_shape.and_then(|s| index.rewrite(context, s));

    // Return if index fits in a single line.
    match orig_index_rw {
        Some(ref index_str) if !index_str.contains('\n') => {
            return Some(format!("{}[{}]", expr_str, index_str));
        }
        _ => (),
    }

    // Try putting index on the next line and see if it fits in a single line.
    let indent = shape.indent.block_indent(context.config);
    let index_shape = Shape::indented(indent, context.config).offset_left(1)?;
    let index_shape = index_shape.sub_width(1 + rhs_overhead)?;
    let new_index_rw = index.rewrite(context, index_shape);
    match (orig_index_rw, new_index_rw) {
        (_, Some(ref new_index_str)) if !new_index_str.contains('\n') => Some(format!(
            "{}{}[{}]",
            expr_str,
            indent.to_string_with_newline(context.config),
            new_index_str,
        )),
        (None, Some(ref new_index_str)) => Some(format!(
            "{}{}[{}]",
            expr_str,
            indent.to_string_with_newline(context.config),
            new_index_str,
        )),
        (Some(ref index_str), _) => Some(format!("{}[{}]", expr_str, index_str)),
        _ => None,
    }
}

fn struct_lit_can_be_aligned(fields: &[ast::ExprField], has_base: bool) -> bool {
    !has_base && fields.iter().all(|field| !field.is_shorthand)
}

fn rewrite_struct_lit<'a>(
    context: &RewriteContext<'_>,
    path: &ast::Path,
    qself: Option<&ast::QSelf>,
    fields: &'a [ast::ExprField],
    struct_rest: &ast::StructRest,
    attrs: &[ast::Attribute],
    span: Span,
    shape: Shape,
) -> Option<String> {
    debug!("rewrite_struct_lit: shape {:?}", shape);

    enum StructLitField<'a> {
        Regular(&'a ast::ExprField),
        Base(&'a ast::Expr),
        Rest(&'a Span),
    }

    // 2 = " {".len()
    let path_shape = shape.sub_width(2)?;
    let path_str = rewrite_path(context, PathContext::Expr, qself, path, path_shape)?;

    let has_base_or_rest = match struct_rest {
        ast::StructRest::None if fields.is_empty() => return Some(format!("{} {{}}", path_str)),
        ast::StructRest::Rest(_) if fields.is_empty() => {
            return Some(format!("{} {{ .. }}", path_str));
        }
        ast::StructRest::Rest(_) | ast::StructRest::Base(_) => true,
        _ => false,
    };

    // Foo { a: Foo } - indent is +3, width is -5.
    let (h_shape, v_shape) = struct_lit_shape(shape, context, path_str.len() + 3, 2)?;

    let one_line_width = h_shape.map_or(0, |shape| shape.width);
    let body_lo = context.snippet_provider.span_after(span, "{");
    let fields_str = if struct_lit_can_be_aligned(fields, has_base_or_rest)
        && context.config.struct_field_align_threshold() > 0
    {
        rewrite_with_alignment(
            fields,
            context,
            v_shape,
            mk_sp(body_lo, span.hi()),
            one_line_width,
        )?
    } else {
        let field_iter = fields.iter().map(StructLitField::Regular).chain(
            match struct_rest {
                ast::StructRest::Base(expr) => Some(StructLitField::Base(&**expr)),
                ast::StructRest::Rest(span) => Some(StructLitField::Rest(span)),
                ast::StructRest::None => None,
            }
            .into_iter(),
        );

        let span_lo = |item: &StructLitField<'_>| match *item {
            StructLitField::Regular(field) => field.span().lo(),
            StructLitField::Base(expr) => {
                let last_field_hi = fields.last().map_or(span.lo(), |field| field.span.hi());
                let snippet = context.snippet(mk_sp(last_field_hi, expr.span.lo()));
                let pos = snippet.find_uncommented("..").unwrap();
                last_field_hi + BytePos(pos as u32)
            }
            StructLitField::Rest(span) => span.lo(),
        };
        let span_hi = |item: &StructLitField<'_>| match *item {
            StructLitField::Regular(field) => field.span().hi(),
            StructLitField::Base(expr) => expr.span.hi(),
            StructLitField::Rest(span) => span.hi(),
        };
        let rewrite = |item: &StructLitField<'_>| match *item {
            StructLitField::Regular(field) => {
                // The 1 taken from the v_budget is for the comma.
                rewrite_field(context, field, v_shape.sub_width(1)?, 0)
            }
            StructLitField::Base(expr) => {
                // 2 = ..
                expr.rewrite(context, v_shape.offset_left(2)?)
                    .map(|s| format!("..{}", s))
            }
            StructLitField::Rest(_) => Some("..".to_owned()),
        };

        let items = itemize_list(
            context.snippet_provider,
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

        let ends_with_comma = span_ends_with_comma(context, span);
        let force_no_trailing_comma = context.inside_macro() && !ends_with_comma;

        let fmt = struct_lit_formatting(
            nested_shape,
            tactic,
            context,
            force_no_trailing_comma || has_base_or_rest || !context.use_block_indent(),
        );

        write_list(&item_vec, &fmt)?
    };

    let fields_str =
        wrap_struct_field(context, attrs, &fields_str, shape, v_shape, one_line_width)?;
    Some(format!("{} {{{}}}", path_str, fields_str))

    // FIXME if context.config.indent_style() == Visual, but we run out
    // of space, we should fall back to BlockIndent.
}

pub(crate) fn wrap_struct_field(
    context: &RewriteContext<'_>,
    attrs: &[ast::Attribute],
    fields_str: &str,
    shape: Shape,
    nested_shape: Shape,
    one_line_width: usize,
) -> Option<String> {
    let should_vertical = context.config.indent_style() == IndentStyle::Block
        && (fields_str.contains('\n')
            || !context.config.struct_lit_single_line()
            || fields_str.len() > one_line_width);

    let inner_attrs = &inner_attributes(attrs);
    if inner_attrs.is_empty() {
        if should_vertical {
            Some(format!(
                "{}{}{}",
                nested_shape.indent.to_string_with_newline(context.config),
                fields_str,
                shape.indent.to_string_with_newline(context.config)
            ))
        } else {
            // One liner or visual indent.
            Some(format!(" {} ", fields_str))
        }
    } else {
        Some(format!(
            "{}{}{}{}{}",
            nested_shape.indent.to_string_with_newline(context.config),
            inner_attrs.rewrite(context, shape)?,
            nested_shape.indent.to_string_with_newline(context.config),
            fields_str,
            shape.indent.to_string_with_newline(context.config)
        ))
    }
}

pub(crate) fn struct_lit_field_separator(config: &Config) -> &str {
    colon_spaces(config)
}

pub(crate) fn rewrite_field(
    context: &RewriteContext<'_>,
    field: &ast::ExprField,
    shape: Shape,
    prefix_max_width: usize,
) -> Option<String> {
    if contains_skip(&field.attrs) {
        return Some(context.snippet(field.span()).to_owned());
    }
    let mut attrs_str = field.attrs.rewrite(context, shape)?;
    if !attrs_str.is_empty() {
        attrs_str.push_str(&shape.indent.to_string_with_newline(context.config));
    };
    let name = context.snippet(field.ident.span);
    if field.is_shorthand {
        Some(attrs_str + name)
    } else {
        let mut separator = String::from(struct_lit_field_separator(context.config));
        for _ in 0..prefix_max_width.saturating_sub(name.len()) {
            separator.push(' ');
        }
        let overhead = name.len() + separator.len();
        let expr_shape = shape.offset_left(overhead)?;
        let expr = field.expr.rewrite(context, expr_shape);

        match expr {
            Some(ref e) if e.as_str() == name && context.config.use_field_init_shorthand() => {
                Some(attrs_str + name)
            }
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

fn rewrite_tuple_in_visual_indent_style<'a, T: 'a + IntoOverflowableItem<'a>>(
    context: &RewriteContext<'_>,
    mut items: impl Iterator<Item = &'a T>,
    span: Span,
    shape: Shape,
    is_singleton_tuple: bool,
) -> Option<String> {
    // In case of length 1, need a trailing comma
    debug!("rewrite_tuple_in_visual_indent_style {:?}", shape);
    if is_singleton_tuple {
        // 3 = "(" + ",)"
        let nested_shape = shape.sub_width(3)?.visual_indent(1);
        return items
            .next()
            .unwrap()
            .rewrite(context, nested_shape)
            .map(|s| format!("({},)", s));
    }

    let list_lo = context.snippet_provider.span_after(span, "(");
    let nested_shape = shape.sub_width(2)?.visual_indent(1);
    let items = itemize_list(
        context.snippet_provider,
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
    let fmt = ListFormatting::new(nested_shape, context.config)
        .tactic(tactic)
        .ends_with_newline(false);
    let list_str = write_list(&item_vec, &fmt)?;

    Some(format!("({})", list_str))
}

pub(crate) fn rewrite_tuple<'a, T: 'a + IntoOverflowableItem<'a>>(
    context: &'a RewriteContext<'_>,
    items: impl Iterator<Item = &'a T>,
    span: Span,
    shape: Shape,
    is_singleton_tuple: bool,
) -> Option<String> {
    debug!("rewrite_tuple {:?}", shape);
    if context.use_block_indent() {
        // We use the same rule as function calls for rewriting tuples.
        let force_tactic = if context.inside_macro() {
            if span_ends_with_comma(context, span) {
                Some(SeparatorTactic::Always)
            } else {
                Some(SeparatorTactic::Never)
            }
        } else if is_singleton_tuple {
            Some(SeparatorTactic::Always)
        } else {
            None
        };
        overflow::rewrite_with_parens(
            context,
            "",
            items,
            shape,
            span,
            context.config.fn_call_width(),
            force_tactic,
        )
    } else {
        rewrite_tuple_in_visual_indent_style(context, items, span, shape, is_singleton_tuple)
    }
}

pub(crate) fn rewrite_unary_prefix<R: Rewrite>(
    context: &RewriteContext<'_>,
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
pub(crate) fn rewrite_unary_suffix<R: Rewrite>(
    context: &RewriteContext<'_>,
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
    context: &RewriteContext<'_>,
    op: ast::UnOp,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    // For some reason, an UnOp is not spanned like BinOp!
    rewrite_unary_prefix(context, ast::UnOp::to_string(op), expr, shape)
}

pub(crate) enum RhsAssignKind<'ast> {
    Expr(&'ast ast::ExprKind, Span),
    Bounds,
    Ty,
}

impl<'ast> RhsAssignKind<'ast> {
    // TODO(calebcartwright)
    // Preemptive addition for handling RHS with chains, not yet utilized.
    // It may make more sense to construct the chain first and then check
    // whether there are actually chain elements.
    #[allow(dead_code)]
    fn is_chain(&self) -> bool {
        match self {
            RhsAssignKind::Expr(kind, _) => {
                matches!(
                    kind,
                    ast::ExprKind::Try(..)
                        | ast::ExprKind::Field(..)
                        | ast::ExprKind::MethodCall(..)
                        | ast::ExprKind::Await(_)
                )
            }
            _ => false,
        }
    }
}

fn rewrite_assignment(
    context: &RewriteContext<'_>,
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

    rewrite_assign_rhs(
        context,
        lhs_str,
        rhs,
        &RhsAssignKind::Expr(&rhs.kind, rhs.span),
        shape,
    )
}

/// Controls where to put the rhs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum RhsTactics {
    /// Use heuristics.
    Default,
    /// Put the rhs on the next line if it uses multiple line, without extra indentation.
    ForceNextLineWithoutIndent,
    /// Allow overflowing max width if neither `Default` nor `ForceNextLineWithoutIndent`
    /// did not work.
    AllowOverflow,
}

// The left hand side must contain everything up to, and including, the
// assignment operator.
pub(crate) fn rewrite_assign_rhs<S: Into<String>, R: Rewrite>(
    context: &RewriteContext<'_>,
    lhs: S,
    ex: &R,
    rhs_kind: &RhsAssignKind<'_>,
    shape: Shape,
) -> Option<String> {
    rewrite_assign_rhs_with(context, lhs, ex, shape, rhs_kind, RhsTactics::Default)
}

pub(crate) fn rewrite_assign_rhs_expr<R: Rewrite>(
    context: &RewriteContext<'_>,
    lhs: &str,
    ex: &R,
    shape: Shape,
    rhs_kind: &RhsAssignKind<'_>,
    rhs_tactics: RhsTactics,
) -> Option<String> {
    let last_line_width = last_line_width(lhs).saturating_sub(if lhs.contains('\n') {
        shape.indent.width()
    } else {
        0
    });
    // 1 = space between operator and rhs.
    let orig_shape = shape.offset_left(last_line_width + 1).unwrap_or(Shape {
        width: 0,
        offset: shape.offset + last_line_width + 1,
        ..shape
    });
    let has_rhs_comment = if let Some(offset) = lhs.find_last_uncommented("=") {
        lhs.trim_end().len() > offset + 1
    } else {
        false
    };

    choose_rhs(
        context,
        ex,
        orig_shape,
        ex.rewrite(context, orig_shape),
        rhs_kind,
        rhs_tactics,
        has_rhs_comment,
    )
}

pub(crate) fn rewrite_assign_rhs_with<S: Into<String>, R: Rewrite>(
    context: &RewriteContext<'_>,
    lhs: S,
    ex: &R,
    shape: Shape,
    rhs_kind: &RhsAssignKind<'_>,
    rhs_tactics: RhsTactics,
) -> Option<String> {
    let lhs = lhs.into();
    let rhs = rewrite_assign_rhs_expr(context, &lhs, ex, shape, rhs_kind, rhs_tactics)?;
    Some(lhs + &rhs)
}

pub(crate) fn rewrite_assign_rhs_with_comments<S: Into<String>, R: Rewrite>(
    context: &RewriteContext<'_>,
    lhs: S,
    ex: &R,
    shape: Shape,
    rhs_kind: &RhsAssignKind<'_>,
    rhs_tactics: RhsTactics,
    between_span: Span,
    allow_extend: bool,
) -> Option<String> {
    let lhs = lhs.into();
    let contains_comment = contains_comment(context.snippet(between_span));
    let shape = if contains_comment {
        shape.block_left(context.config.tab_spaces())?
    } else {
        shape
    };
    let rhs = rewrite_assign_rhs_expr(context, &lhs, ex, shape, rhs_kind, rhs_tactics)?;

    if contains_comment {
        let rhs = rhs.trim_start();
        combine_strs_with_missing_comments(context, &lhs, rhs, between_span, shape, allow_extend)
    } else {
        Some(lhs + &rhs)
    }
}

fn choose_rhs<R: Rewrite>(
    context: &RewriteContext<'_>,
    expr: &R,
    shape: Shape,
    orig_rhs: Option<String>,
    _rhs_kind: &RhsAssignKind<'_>,
    rhs_tactics: RhsTactics,
    has_rhs_comment: bool,
) -> Option<String> {
    match orig_rhs {
        Some(ref new_str) if new_str.is_empty() => Some(String::new()),
        Some(ref new_str)
            if !new_str.contains('\n') && unicode_str_width(new_str) <= shape.width =>
        {
            Some(format!(" {}", new_str))
        }
        _ => {
            // Expression did not fit on the same line as the identifier.
            // Try splitting the line and see if that works better.
            let new_shape = shape_from_rhs_tactic(context, shape, rhs_tactics)?;
            let new_rhs = expr.rewrite(context, new_shape);
            let new_indent_str = &shape
                .indent
                .block_indent(context.config)
                .to_string_with_newline(context.config);
            let before_space_str = if has_rhs_comment { "" } else { " " };

            match (orig_rhs, new_rhs) {
                (Some(ref orig_rhs), Some(ref new_rhs))
                    if wrap_str(new_rhs.clone(), context.config.max_width(), new_shape)
                        .is_none() =>
                {
                    Some(format!("{}{}", before_space_str, orig_rhs))
                }
                (Some(ref orig_rhs), Some(ref new_rhs))
                    if prefer_next_line(orig_rhs, new_rhs, rhs_tactics) =>
                {
                    Some(format!("{}{}", new_indent_str, new_rhs))
                }
                (None, Some(ref new_rhs)) => Some(format!("{}{}", new_indent_str, new_rhs)),
                (None, None) if rhs_tactics == RhsTactics::AllowOverflow => {
                    let shape = shape.infinite_width();
                    expr.rewrite(context, shape)
                        .map(|s| format!("{}{}", before_space_str, s))
                }
                (None, None) => None,
                (Some(orig_rhs), _) => Some(format!("{}{}", before_space_str, orig_rhs)),
            }
        }
    }
}

fn shape_from_rhs_tactic(
    context: &RewriteContext<'_>,
    shape: Shape,
    rhs_tactic: RhsTactics,
) -> Option<Shape> {
    match rhs_tactic {
        RhsTactics::ForceNextLineWithoutIndent => shape
            .with_max_width(context.config)
            .sub_width(shape.indent.width()),
        RhsTactics::Default | RhsTactics::AllowOverflow => {
            Shape::indented(shape.indent.block_indent(context.config), context.config)
                .sub_width(shape.rhs_overhead(context.config))
        }
    }
}

/// Returns true if formatting next_line_rhs is better on a new line when compared to the
/// original's line formatting.
///
/// It is considered better if:
/// 1. the tactic is ForceNextLineWithoutIndent
/// 2. next_line_rhs doesn't have newlines
/// 3. the original line has more newlines than next_line_rhs
/// 4. the original formatting of the first line ends with `(`, `{`, or `[` and next_line_rhs
///    doesn't
pub(crate) fn prefer_next_line(
    orig_rhs: &str,
    next_line_rhs: &str,
    rhs_tactics: RhsTactics,
) -> bool {
    rhs_tactics == RhsTactics::ForceNextLineWithoutIndent
        || !next_line_rhs.contains('\n')
        || count_newlines(orig_rhs) > count_newlines(next_line_rhs) + 1
        || first_line_ends_with(orig_rhs, '(') && !first_line_ends_with(next_line_rhs, '(')
        || first_line_ends_with(orig_rhs, '{') && !first_line_ends_with(next_line_rhs, '{')
        || first_line_ends_with(orig_rhs, '[') && !first_line_ends_with(next_line_rhs, '[')
}

fn rewrite_expr_addrof(
    context: &RewriteContext<'_>,
    borrow_kind: ast::BorrowKind,
    mutability: ast::Mutability,
    expr: &ast::Expr,
    shape: Shape,
) -> Option<String> {
    let operator_str = match (mutability, borrow_kind) {
        (ast::Mutability::Not, ast::BorrowKind::Ref) => "&",
        (ast::Mutability::Not, ast::BorrowKind::Raw) => "&raw const ",
        (ast::Mutability::Mut, ast::BorrowKind::Ref) => "&mut ",
        (ast::Mutability::Mut, ast::BorrowKind::Raw) => "&raw mut ",
    };
    rewrite_unary_prefix(context, operator_str, expr, shape)
}

pub(crate) fn is_method_call(expr: &ast::Expr) -> bool {
    match expr.kind {
        ast::ExprKind::MethodCall(..) => true,
        ast::ExprKind::AddrOf(_, _, ref expr)
        | ast::ExprKind::Box(ref expr)
        | ast::ExprKind::Cast(ref expr, _)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Unary(_, ref expr) => is_method_call(expr),
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use super::last_line_offsetted;

    #[test]
    fn test_last_line_offsetted() {
        let lines = "one\n    two";
        assert_eq!(last_line_offsetted(2, lines), true);
        assert_eq!(last_line_offsetted(4, lines), false);
        assert_eq!(last_line_offsetted(6, lines), false);

        let lines = "one    two";
        assert_eq!(last_line_offsetted(2, lines), false);
        assert_eq!(last_line_offsetted(0, lines), false);

        let lines = "\ntwo";
        assert_eq!(last_line_offsetted(2, lines), false);
        assert_eq!(last_line_offsetted(0, lines), false);

        let lines = "one\n    two      three";
        assert_eq!(last_line_offsetted(2, lines), true);
        let lines = "one\n two      three";
        assert_eq!(last_line_offsetted(2, lines), false);
    }
}
