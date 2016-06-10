// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering;
use std::borrow::Borrow;
use std::mem::swap;
use std::ops::Deref;
use std::iter::ExactSizeIterator;
use std::fmt::Write;

use {Indent, Spanned};
use codemap::SpanUtils;
use rewrite::{Rewrite, RewriteContext};
use lists::{write_list, itemize_list, ListFormatting, SeparatorTactic, ListTactic,
            DefinitiveListTactic, definitive_tactic, ListItem, format_item_list};
use string::{StringFormat, rewrite_string};
use utils::{extra_offset, last_line_width, wrap_str, binary_search, first_line_width,
            semicolon_for_stmt, trimmed_last_line_width, left_most_sub_expr};
use visitor::FmtVisitor;
use config::{Config, StructLitStyle, MultilineStyle, ElseIfBraceStyle, ControlBraceStyle};
use comment::{FindUncommented, rewrite_comment, contains_comment, recover_comment_removed};
use types::rewrite_path;
use items::{span_lo_for_arg, span_hi_for_arg};
use chains::rewrite_chain;
use macros::rewrite_macro;

use syntax::{ast, ptr};
use syntax::codemap::{CodeMap, Span, BytePos, mk_sp};
use syntax::parse::classify;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        format_expr(self, ExprType::SubExpression, context, width, offset)
    }
}

#[derive(PartialEq)]
enum ExprType {
    Statement,
    SubExpression,
}

fn format_expr(expr: &ast::Expr,
               expr_type: ExprType,
               context: &RewriteContext,
               width: usize,
               offset: Indent)
               -> Option<String> {
    let result = match expr.node {
        ast::ExprKind::Vec(ref expr_vec) => {
            rewrite_array(expr_vec.iter().map(|e| &**e),
                          mk_sp(context.codemap.span_after(expr.span, "["), expr.span.hi),
                          context,
                          width,
                          offset)
        }
        ast::ExprKind::Lit(ref l) => {
            match l.node {
                ast::LitKind::Str(_, ast::StrStyle::Cooked) => {
                    rewrite_string_lit(context, l.span, width, offset)
                }
                _ => {
                    wrap_str(context.snippet(expr.span),
                             context.config.max_width,
                             width,
                             offset)
                }
            }
        }
        ast::ExprKind::Call(ref callee, ref args) => {
            let inner_span = mk_sp(callee.span.hi, expr.span.hi);
            rewrite_call(context, &**callee, args, inner_span, width, offset)
        }
        ast::ExprKind::Paren(ref subexpr) => rewrite_paren(context, subexpr, width, offset),
        ast::ExprKind::Binary(ref op, ref lhs, ref rhs) => {
            rewrite_binary_op(context, op, lhs, rhs, width, offset)
        }
        ast::ExprKind::Unary(ref op, ref subexpr) => {
            rewrite_unary_op(context, op, subexpr, width, offset)
        }
        ast::ExprKind::Struct(ref path, ref fields, ref base) => {
            rewrite_struct_lit(context,
                               path,
                               fields,
                               base.as_ref().map(|e| &**e),
                               expr.span,
                               width,
                               offset)
        }
        ast::ExprKind::Tup(ref items) => {
            rewrite_tuple(context,
                          items.iter().map(|x| &**x),
                          expr.span,
                          width,
                          offset)
        }
        ast::ExprKind::While(ref cond, ref block, label) => {
            Loop::new_while(None, cond, block, label).rewrite(context, width, offset)
        }
        ast::ExprKind::WhileLet(ref pat, ref cond, ref block, label) => {
            Loop::new_while(Some(pat), cond, block, label).rewrite(context, width, offset)
        }
        ast::ExprKind::ForLoop(ref pat, ref cond, ref block, label) => {
            Loop::new_for(pat, cond, block, label).rewrite(context, width, offset)
        }
        ast::ExprKind::Loop(ref block, label) => {
            Loop::new_loop(block, label).rewrite(context, width, offset)
        }
        ast::ExprKind::Block(ref block) => block.rewrite(context, width, offset),
        ast::ExprKind::If(ref cond, ref if_block, ref else_block) => {
            rewrite_if_else(context,
                            cond,
                            expr_type,
                            if_block,
                            else_block.as_ref().map(|e| &**e),
                            expr.span,
                            None,
                            width,
                            offset,
                            true)
        }
        ast::ExprKind::IfLet(ref pat, ref cond, ref if_block, ref else_block) => {
            rewrite_if_else(context,
                            cond,
                            expr_type,
                            if_block,
                            else_block.as_ref().map(|e| &**e),
                            expr.span,
                            Some(pat),
                            width,
                            offset,
                            true)
        }
        ast::ExprKind::Match(ref cond, ref arms) => {
            rewrite_match(context, cond, arms, width, offset, expr.span)
        }
        ast::ExprKind::Path(ref qself, ref path) => {
            rewrite_path(context, true, qself.as_ref(), path, width, offset)
        }
        ast::ExprKind::Assign(ref lhs, ref rhs) => {
            rewrite_assignment(context, lhs, rhs, None, width, offset)
        }
        ast::ExprKind::AssignOp(ref op, ref lhs, ref rhs) => {
            rewrite_assignment(context, lhs, rhs, Some(op), width, offset)
        }
        ast::ExprKind::Again(ref opt_ident) => {
            let id_str = match *opt_ident {
                Some(ident) => format!(" {}", ident.node),
                None => String::new(),
            };
            wrap_str(format!("continue{}", id_str),
                     context.config.max_width,
                     width,
                     offset)
        }
        ast::ExprKind::Break(ref opt_ident) => {
            let id_str = match *opt_ident {
                Some(ident) => format!(" {}", ident.node),
                None => String::new(),
            };
            wrap_str(format!("break{}", id_str),
                     context.config.max_width,
                     width,
                     offset)
        }
        ast::ExprKind::Closure(capture, ref fn_decl, ref body, _) => {
            rewrite_closure(capture, fn_decl, body, expr.span, context, width, offset)
        }
        ast::ExprKind::Try(..) |
        ast::ExprKind::Field(..) |
        ast::ExprKind::TupField(..) |
        ast::ExprKind::MethodCall(..) => rewrite_chain(expr, context, width, offset),
        ast::ExprKind::Mac(ref mac) => {
            // Failure to rewrite a marco should not imply failure to
            // rewrite the expression.
            rewrite_macro(mac, None, context, width, offset).or_else(|| {
                wrap_str(context.snippet(expr.span),
                         context.config.max_width,
                         width,
                         offset)
            })
        }
        ast::ExprKind::Ret(None) => {
            wrap_str("return".to_owned(), context.config.max_width, width, offset)
        }
        ast::ExprKind::Ret(Some(ref expr)) => {
            rewrite_unary_prefix(context, "return ", &**expr, width, offset)
        }
        ast::ExprKind::Box(ref expr) => {
            rewrite_unary_prefix(context, "box ", &**expr, width, offset)
        }
        ast::ExprKind::AddrOf(mutability, ref expr) => {
            rewrite_expr_addrof(context, mutability, expr, width, offset)
        }
        ast::ExprKind::Cast(ref expr, ref ty) => {
            rewrite_pair(&**expr, &**ty, "", " as ", "", context, width, offset)
        }
        ast::ExprKind::Type(ref expr, ref ty) => {
            rewrite_pair(&**expr, &**ty, "", ": ", "", context, width, offset)
        }
        ast::ExprKind::Index(ref expr, ref index) => {
            rewrite_pair(&**expr, &**index, "", "[", "]", context, width, offset)
        }
        ast::ExprKind::Repeat(ref expr, ref repeats) => {
            rewrite_pair(&**expr, &**repeats, "[", "; ", "]", context, width, offset)
        }
        ast::ExprKind::Range(ref lhs, ref rhs, limits) => {
            let delim = match limits {
                ast::RangeLimits::HalfOpen => "..",
                ast::RangeLimits::Closed => "...",
            };

            match (lhs.as_ref().map(|x| &**x), rhs.as_ref().map(|x| &**x)) {
                (Some(ref lhs), Some(ref rhs)) => {
                    rewrite_pair(&**lhs, &**rhs, "", delim, "", context, width, offset)
                }
                (None, Some(ref rhs)) => {
                    rewrite_unary_prefix(context, delim, &**rhs, width, offset)
                }
                (Some(ref lhs), None) => {
                    rewrite_unary_suffix(context, delim, &**lhs, width, offset)
                }
                (None, None) => wrap_str(delim.into(), context.config.max_width, width, offset),
            }
        }
        // We do not format these expressions yet, but they should still
        // satisfy our width restrictions.
        ast::ExprKind::InPlace(..) |
        ast::ExprKind::InlineAsm(..) => {
            wrap_str(context.snippet(expr.span),
                     context.config.max_width,
                     width,
                     offset)
        }
    };
    result.and_then(|res| recover_comment_removed(res, expr.span, context, width, offset))
}

pub fn rewrite_pair<LHS, RHS>(lhs: &LHS,
                              rhs: &RHS,
                              prefix: &str,
                              infix: &str,
                              suffix: &str,
                              context: &RewriteContext,
                              width: usize,
                              offset: Indent)
                              -> Option<String>
    where LHS: Rewrite,
          RHS: Rewrite
{
    let max_width = try_opt!(width.checked_sub(prefix.len() + infix.len() + suffix.len()));

    binary_search(1, max_width, |lhs_budget| {
        let lhs_offset = offset + prefix.len();
        let lhs_str = match lhs.rewrite(context, lhs_budget, lhs_offset) {
            Some(result) => result,
            None => return Err(Ordering::Greater),
        };

        let last_line_width = last_line_width(&lhs_str);
        let rhs_budget = match max_width.checked_sub(last_line_width) {
            Some(b) => b,
            None => return Err(Ordering::Less),
        };
        let rhs_indent = offset + last_line_width + prefix.len() + infix.len();

        let rhs_str = match rhs.rewrite(context, rhs_budget, rhs_indent) {
            Some(result) => result,
            None => return Err(Ordering::Less),
        };

        Ok(format!("{}{}{}{}{}", prefix, lhs_str, infix, rhs_str, suffix))
    })
}

pub fn rewrite_array<'a, I>(expr_iter: I,
                            span: Span,
                            context: &RewriteContext,
                            width: usize,
                            offset: Indent)
                            -> Option<String>
    where I: Iterator<Item = &'a ast::Expr>
{
    // 2 for brackets;
    let offset = offset + 1;
    let inner_context = &RewriteContext { block_indent: offset, ..*context };
    let max_item_width = try_opt!(width.checked_sub(2));
    let items = itemize_list(context.codemap,
                             expr_iter,
                             "]",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             // 1 = [
                             |item| item.rewrite(&inner_context, max_item_width, offset),
                             span.lo,
                             span.hi)
        .collect::<Vec<_>>();

    let has_long_item = try_opt!(items.iter()
        .map(|li| li.item.as_ref().map(|s| s.len() > 10))
        .fold(Some(false), |acc, x| acc.and_then(|y| x.map(|x| x || y))));

    let tactic = if has_long_item || items.iter().any(ListItem::is_multiline) {
        definitive_tactic(&items, ListTactic::HorizontalVertical, max_item_width)
    } else {
        DefinitiveListTactic::Mixed
    };

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: offset,
        width: max_item_width,
        ends_with_newline: false,
        config: context.config,
    };
    let list_str = try_opt!(write_list(&items, &fmt));

    Some(format!("[{}]", list_str))
}

// This functions is pretty messy because of the rules around closures and blocks:
//   * the body of a closure is represented by an ast::Block, but that does not
//     imply there are `{}` (unless the block is empty) (see rust issue #27872),
//   * if there is a return type, then there must be braces,
//   * given a closure with braces, whether that is parsed to give an inner block
//     or not depends on if there is a return type and if there are statements
//     in that block,
//   * if the first expression in the body ends with a block (i.e., is a
//     statement without needing a semi-colon), then adding or removing braces
//     can change whether it is treated as an expression or statement.
fn rewrite_closure(capture: ast::CaptureBy,
                   fn_decl: &ast::FnDecl,
                   body: &ast::Block,
                   span: Span,
                   context: &RewriteContext,
                   width: usize,
                   offset: Indent)
                   -> Option<String> {
    let mover = if capture == ast::CaptureBy::Value {
        "move "
    } else {
        ""
    };
    let offset = offset + mover.len();

    // 4 = "|| {".len(), which is overconservative when the closure consists of
    // a single expression.
    let budget = try_opt!(width.checked_sub(4 + mover.len()));
    // 1 = |
    let argument_offset = offset + 1;
    let ret_str = try_opt!(fn_decl.output.rewrite(context, budget, argument_offset));
    let force_block = !ret_str.is_empty();

    // 1 = space between arguments and return type.
    let horizontal_budget = budget.checked_sub(ret_str.len() + 1).unwrap_or(0);

    let arg_items = itemize_list(context.codemap,
                                 fn_decl.inputs.iter(),
                                 "|",
                                 |arg| span_lo_for_arg(arg),
                                 |arg| span_hi_for_arg(arg),
                                 |arg| arg.rewrite(context, budget, argument_offset),
                                 context.codemap.span_after(span, "|"),
                                 body.span.lo);
    let item_vec = arg_items.collect::<Vec<_>>();
    let tactic = definitive_tactic(&item_vec, ListTactic::HorizontalVertical, horizontal_budget);
    let budget = match tactic {
        DefinitiveListTactic::Horizontal => horizontal_budget,
        _ => budget,
    };

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: argument_offset,
        width: budget,
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

    if body.expr.is_none() && body.stmts.is_empty() {
        return Some(format!("{} {{}}", prefix));
    }

    // 1 = space between `|...|` and body.
    let extra_offset = extra_offset(&prefix, offset) + 1;
    let budget = try_opt!(width.checked_sub(extra_offset));

    // This is where we figure out whether to use braces or not.
    let mut had_braces = true;
    let mut inner_block = body;

    // If there is an inner block and we can ignore it, do so.
    if body.stmts.is_empty() {
        if let ast::ExprKind::Block(ref inner) = inner_block.expr.as_ref().unwrap().node {
            inner_block = inner;
        } else if !force_block {
            had_braces = false;
        }
    }

    let try_single_line = is_simple_block(inner_block, context.codemap) &&
                          inner_block.rules == ast::BlockCheckMode::Default;

    if try_single_line && !force_block {
        let must_preserve_braces =
            !classify::expr_requires_semi_to_be_stmt(left_most_sub_expr(inner_block.expr
                .as_ref()
                .unwrap()));
        if !(must_preserve_braces && had_braces) &&
           (must_preserve_braces || !prefix.contains('\n')) {
            // If we got here, then we can try to format without braces.

            let inner_expr = inner_block.expr.as_ref().unwrap();
            let mut rewrite = inner_expr.rewrite(context, budget, offset + extra_offset);

            if must_preserve_braces {
                // If we are here, then failure to rewrite is unacceptable.
                if rewrite.is_none() {
                    return None;
                }
            } else {
                // Checks if rewrite succeeded and fits on a single line.
                rewrite = and_one_line(rewrite);
            }

            if let Some(rewrite) = rewrite {
                return Some(format!("{} {}", prefix, rewrite));
            }
        }
    }

    // If we fell through the above block, then we need braces, but we might
    // still prefer a one-liner (we might also have fallen through because of
    // lack of space).
    if try_single_line && !prefix.contains('\n') {
        let inner_expr = inner_block.expr.as_ref().unwrap();
        // 4 = braces and spaces.
        let mut rewrite = inner_expr.rewrite(context,
                                             try_opt!(budget.checked_sub(4)),
                                             offset + extra_offset);

        // Checks if rewrite succeeded and fits on a single line.
        rewrite = and_one_line(rewrite);

        if let Some(rewrite) = rewrite {
            return Some(format!("{} {{ {} }}", prefix, rewrite));
        }
    }

    // We couldn't format the closure body as a single line expression; fall
    // back to block formatting.
    let body_rewrite = try_opt!(inner_block.rewrite(&context, budget, Indent::empty()));

    let block_threshold = context.config.closure_block_indent_threshold;
    if block_threshold < 0 || body_rewrite.matches('\n').count() <= block_threshold as usize {
        return Some(format!("{} {}", prefix, body_rewrite));
    }

    // The body of the closure is big enough to be block indented, that means we
    // must re-format.
    let mut context = context.clone();
    context.block_indent.alignment = 0;
    let body_rewrite = try_opt!(inner_block.rewrite(&context, budget, Indent::empty()));
    Some(format!("{} {}", prefix, body_rewrite))
}

fn and_one_line(x: Option<String>) -> Option<String> {
    x.and_then(|x| if x.contains('\n') { None } else { Some(x) })
}

fn nop_block_collapse(block_str: Option<String>, budget: usize) -> Option<String> {
    block_str.map(|block_str| {
        if block_str.starts_with("{") && budget >= 2 &&
           (block_str[1..].find(|c: char| !c.is_whitespace()).unwrap() == block_str.len() - 2) {
            "{}".to_owned()
        } else {
            block_str.to_owned()
        }
    })
}

impl Rewrite for ast::Block {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        let user_str = context.snippet(self.span);
        if user_str == "{}" && width >= 2 {
            return Some(user_str);
        }

        let mut visitor = FmtVisitor::from_codemap(context.parse_session, context.config);
        visitor.block_indent = context.block_indent;

        let prefix = match self.rules {
            ast::BlockCheckMode::Unsafe(..) => {
                let snippet = context.snippet(self.span);
                let open_pos = try_opt!(snippet.find_uncommented("{"));
                visitor.last_pos = self.span.lo + BytePos(open_pos as u32);

                // Extract comment between unsafe and block start.
                let trimmed = &snippet[6..open_pos].trim();

                let prefix = if !trimmed.is_empty() {
                    // 9 = "unsafe  {".len(), 7 = "unsafe ".len()
                    let budget = try_opt!(width.checked_sub(9));
                    format!("unsafe {} ",
                            try_opt!(rewrite_comment(trimmed,
                                                     true,
                                                     budget,
                                                     offset + 7,
                                                     context.config)))
                } else {
                    "unsafe ".to_owned()
                };

                if is_simple_block(self, context.codemap) && prefix.len() < width {
                    let body = self.expr
                        .as_ref()
                        .unwrap()
                        .rewrite(context, width - prefix.len(), offset);
                    if let Some(ref expr_str) = body {
                        let result = format!("{}{{ {} }}", prefix, expr_str);
                        if result.len() <= width && !result.contains('\n') {
                            return Some(result);
                        }
                    }
                }

                prefix
            }
            ast::BlockCheckMode::Default => {
                visitor.last_pos = self.span.lo;

                String::new()
            }
        };

        visitor.visit_block(self);

        Some(format!("{}{}", prefix, visitor.buffer))
    }
}

impl Rewrite for ast::Stmt {
    fn rewrite(&self, context: &RewriteContext, _width: usize, offset: Indent) -> Option<String> {
        let result = match self.node {
            ast::StmtKind::Decl(ref decl, _) => {
                if let ast::DeclKind::Local(ref local) = decl.node {
                    local.rewrite(context, context.config.max_width, offset)
                } else {
                    None
                }
            }
            ast::StmtKind::Expr(ref ex, _) |
            ast::StmtKind::Semi(ref ex, _) => {
                let suffix = if semicolon_for_stmt(self) { ";" } else { "" };

                format_expr(ex,
                            ExprType::Statement,
                            context,
                            context.config.max_width - offset.width() - suffix.len(),
                            offset)
                    .map(|s| s + suffix)
            }
            ast::StmtKind::Mac(..) => None,
        };
        result.and_then(|res| recover_comment_removed(res, self.span, context, _width, offset))
    }
}

// Abstraction over for, while and loop expressions
struct Loop<'a> {
    cond: Option<&'a ast::Expr>,
    block: &'a ast::Block,
    label: Option<ast::SpannedIdent>,
    pat: Option<&'a ast::Pat>,
    keyword: &'a str,
    matcher: &'a str,
    connector: &'a str,
}

impl<'a> Loop<'a> {
    fn new_loop(block: &'a ast::Block, label: Option<ast::SpannedIdent>) -> Loop<'a> {
        Loop {
            cond: None,
            block: block,
            label: label,
            pat: None,
            keyword: "loop",
            matcher: "",
            connector: "",
        }
    }

    fn new_while(pat: Option<&'a ast::Pat>,
                 cond: &'a ast::Expr,
                 block: &'a ast::Block,
                 label: Option<ast::SpannedIdent>)
                 -> Loop<'a> {
        Loop {
            cond: Some(cond),
            block: block,
            label: label,
            pat: pat,
            keyword: "while ",
            matcher: match pat {
                Some(..) => "let ",
                None => "",
            },
            connector: " =",
        }
    }

    fn new_for(pat: &'a ast::Pat,
               cond: &'a ast::Expr,
               block: &'a ast::Block,
               label: Option<ast::SpannedIdent>)
               -> Loop<'a> {
        Loop {
            cond: Some(cond),
            block: block,
            label: label,
            pat: Some(pat),
            keyword: "for ",
            matcher: "",
            connector: " in",
        }
    }
}

impl<'a> Rewrite for Loop<'a> {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        let label_string = rewrite_label(self.label);
        // 2 = " {".len()
        let inner_width = try_opt!(width.checked_sub(self.keyword.len() + 2 + label_string.len()));
        let inner_offset = offset + self.keyword.len() + label_string.len();

        let pat_expr_string = match self.cond {
            Some(cond) => {
                try_opt!(rewrite_pat_expr(context,
                                          self.pat,
                                          cond,
                                          self.matcher,
                                          self.connector,
                                          inner_width,
                                          inner_offset))
            }
            None => String::new(),
        };

        let alt_block_sep = String::from("\n") + &context.block_indent.to_string(context.config);
        let block_sep = match context.config.control_brace_style {
            ControlBraceStyle::AlwaysNextLine => alt_block_sep.as_str(),
            ControlBraceStyle::AlwaysSameLine => " ",
        };
        // FIXME: this drops any comment between "loop" and the block.
        self.block
            .rewrite(context, width, offset)
            .map(|result| {
                format!("{}{}{}{}{}",
                        label_string,
                        self.keyword,
                        pat_expr_string,
                        block_sep,
                        result)
            })
    }
}

fn rewrite_label(label: Option<ast::SpannedIdent>) -> String {
    match label {
        Some(ident) => format!("{}: ", ident.node),
        None => "".to_owned(),
    }
}

fn extract_comment(span: Span,
                   context: &RewriteContext,
                   offset: Indent,
                   width: usize)
                   -> Option<String> {
    let comment_str = context.snippet(span);
    if contains_comment(&comment_str) {
        let comment =
            try_opt!(rewrite_comment(comment_str.trim(), false, width, offset, context.config));
        Some(format!("\n{indent}{}\n{indent}",
                     comment,
                     indent = offset.to_string(context.config)))
    } else {
        None
    }
}

// Rewrites if-else blocks. If let Some(_) = pat, the expression is
// treated as an if-let-else expression.
fn rewrite_if_else(context: &RewriteContext,
                   cond: &ast::Expr,
                   expr_type: ExprType,
                   if_block: &ast::Block,
                   else_block_opt: Option<&ast::Expr>,
                   span: Span,
                   pat: Option<&ast::Pat>,
                   width: usize,
                   offset: Indent,
                   allow_single_line: bool)
                   -> Option<String> {
    let (budget, indent) = if !allow_single_line {
        // We are part of an if-elseif-else chain. Our constraints are tightened.
        // 7 = "} else" .len()
        (try_opt!(width.checked_sub(7)), offset + 7)
    } else {
        (width, offset)
    };

    // 3 = "if ", 2 = " {"
    let pat_penalty = match context.config.else_if_brace_style {
        ElseIfBraceStyle::AlwaysNextLine => 3,
        _ => 3 + 2,
    };
    let pat_expr_string = try_opt!(rewrite_pat_expr(context,
                                                    pat,
                                                    cond,
                                                    "let ",
                                                    " =",
                                                    try_opt!(budget.checked_sub(pat_penalty)),
                                                    indent + 3));

    // Try to format if-else on single line.
    if expr_type == ExprType::SubExpression && allow_single_line &&
       context.config.single_line_if_else_max_width > 0 {
        let trial = single_line_if_else(context, &pat_expr_string, if_block, else_block_opt, width);

        if trial.is_some() &&
           trial.as_ref().unwrap().len() <= context.config.single_line_if_else_max_width {
            return trial;
        }
    }

    let if_block_string = try_opt!(if_block.rewrite(context, width, offset));

    let between_if_cond = mk_sp(context.codemap.span_after(span, "if"),
                                pat.map_or(cond.span.lo,
                                           |_| context.codemap.span_before(span, "let")));

    let between_if_cond_comment = extract_comment(between_if_cond, &context, offset, width);

    let after_cond_comment = extract_comment(mk_sp(cond.span.hi, if_block.span.lo),
                                             context,
                                             offset,
                                             width);

    let alt_block_sep = String::from("\n") + &context.block_indent.to_string(context.config);
    let after_sep = match context.config.else_if_brace_style {
        ElseIfBraceStyle::AlwaysNextLine => alt_block_sep.as_str(),
        _ => " ",
    };
    let mut result = format!("if{}{}{}{}",
                             between_if_cond_comment.as_ref().map_or(" ", |str| &**str),
                             pat_expr_string,
                             after_cond_comment.as_ref().map_or(after_sep, |str| &**str),
                             if_block_string);

    if let Some(else_block) = else_block_opt {
        let mut last_in_chain = false;
        let rewrite = match else_block.node {
            // If the else expression is another if-else expression, prevent it
            // from being formatted on a single line.
            // Note how we're passing the original width and offset, as the
            // cost of "else" should not cascade.
            ast::ExprKind::IfLet(ref pat, ref cond, ref if_block, ref next_else_block) => {
                rewrite_if_else(context,
                                cond,
                                expr_type,
                                if_block,
                                next_else_block.as_ref().map(|e| &**e),
                                mk_sp(else_block.span.lo, span.hi),
                                Some(pat),
                                width,
                                offset,
                                false)
            }
            ast::ExprKind::If(ref cond, ref if_block, ref next_else_block) => {
                rewrite_if_else(context,
                                cond,
                                expr_type,
                                if_block,
                                next_else_block.as_ref().map(|e| &**e),
                                mk_sp(else_block.span.lo, span.hi),
                                None,
                                width,
                                offset,
                                false)
            }
            _ => {
                last_in_chain = true;
                else_block.rewrite(context, width, offset)
            }
        };

        let between_if_else_block =
            mk_sp(if_block.span.hi,
                  context.codemap.span_before(mk_sp(if_block.span.hi, else_block.span.lo), "else"));
        let between_if_else_block_comment =
            extract_comment(between_if_else_block, &context, offset, width);

        let after_else = mk_sp(context.codemap
                                   .span_after(mk_sp(if_block.span.hi, else_block.span.lo),
                                               "else"),
                               else_block.span.lo);
        let after_else_comment = extract_comment(after_else, &context, offset, width);

        let between_sep = match context.config.else_if_brace_style {
            ElseIfBraceStyle::AlwaysNextLine |
            ElseIfBraceStyle::ClosingNextLine => alt_block_sep.as_str(),
            ElseIfBraceStyle::AlwaysSameLine => " ",
        };
        let after_sep = match context.config.else_if_brace_style {
            ElseIfBraceStyle::AlwaysNextLine if last_in_chain => alt_block_sep.as_str(),
            _ => " ",
        };
        try_opt!(write!(&mut result,
                        "{}else{}",
                        between_if_else_block_comment.as_ref()
                            .map_or(between_sep, |str| &**str),
                        after_else_comment.as_ref().map_or(after_sep, |str| &**str))
            .ok());
        result.push_str(&&try_opt!(rewrite));
    }

    Some(result)
}

fn single_line_if_else(context: &RewriteContext,
                       pat_expr_str: &str,
                       if_node: &ast::Block,
                       else_block_opt: Option<&ast::Expr>,
                       width: usize)
                       -> Option<String> {
    let else_block = try_opt!(else_block_opt);
    let fixed_cost = "if  {  } else {  }".len();

    if let ast::ExprKind::Block(ref else_node) = else_block.node {
        if !is_simple_block(if_node, context.codemap) ||
           !is_simple_block(else_node, context.codemap) || pat_expr_str.contains('\n') {
            return None;
        }

        let new_width = try_opt!(width.checked_sub(pat_expr_str.len() + fixed_cost));
        let if_expr = if_node.expr.as_ref().unwrap();
        let if_str = try_opt!(if_expr.rewrite(context, new_width, Indent::empty()));

        let new_width = try_opt!(new_width.checked_sub(if_str.len()));
        let else_expr = else_node.expr.as_ref().unwrap();
        let else_str = try_opt!(else_expr.rewrite(context, new_width, Indent::empty()));

        // FIXME: this check shouldn't be necessary. Rewrites should either fail
        // or wrap to a newline when the object does not fit the width.
        let fits_line = fixed_cost + pat_expr_str.len() + if_str.len() + else_str.len() <= width;

        if fits_line && !if_str.contains('\n') && !else_str.contains('\n') {
            return Some(format!("if {} {{ {} }} else {{ {} }}",
                                pat_expr_str,
                                if_str,
                                else_str));
        }
    }

    None
}

fn block_contains_comment(block: &ast::Block, codemap: &CodeMap) -> bool {
    let snippet = codemap.span_to_snippet(block.span).unwrap();
    contains_comment(&snippet)
}

// Checks that a block contains no statements, an expression and no comments.
// FIXME: incorrectly returns false when comment is contained completely within
// the expression.
pub fn is_simple_block(block: &ast::Block, codemap: &CodeMap) -> bool {
    block.stmts.is_empty() && block.expr.is_some() && !block_contains_comment(block, codemap)
}

/// Checks whether a block contains at most one statement or expression, and no comments.
pub fn is_simple_block_stmt(block: &ast::Block, codemap: &CodeMap) -> bool {
    (block.stmts.is_empty() || (block.stmts.len() == 1 && block.expr.is_none())) &&
    !block_contains_comment(block, codemap)
}

/// Checks whether a block contains no statements, expressions, or comments.
pub fn is_empty_block(block: &ast::Block, codemap: &CodeMap) -> bool {
    block.stmts.is_empty() && block.expr.is_none() && !block_contains_comment(block, codemap)
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
fn rewrite_match_arm_comment(context: &RewriteContext,
                             missed_str: &str,
                             width: usize,
                             arm_indent: Indent,
                             arm_indent_str: &str)
                             -> Option<String> {
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

    let first = missed_str.find(|c: char| !c.is_whitespace()).unwrap_or(missed_str.len());
    if missed_str[..first].chars().filter(|c| c == &'\n').count() >= 2 {
        // Excessive vertical whitespace before comment should be preserved
        // TODO handle vertical whitespace better
        result.push('\n');
    }
    let missed_str = missed_str[first..].trim();
    if !missed_str.is_empty() {
        let comment =
            try_opt!(rewrite_comment(&missed_str, false, width, arm_indent, context.config));
        result.push('\n');
        result.push_str(arm_indent_str);
        result.push_str(&comment);
    }

    Some(result)
}

fn rewrite_match(context: &RewriteContext,
                 cond: &ast::Expr,
                 arms: &[ast::Arm],
                 width: usize,
                 offset: Indent,
                 span: Span)
                 -> Option<String> {
    if arms.is_empty() {
        return None;
    }

    // `match `cond` {`
    let cond_budget = try_opt!(width.checked_sub(8));
    let cond_str = try_opt!(cond.rewrite(context, cond_budget, offset + 6));
    let alt_block_sep = String::from("\n") + &context.block_indent.to_string(context.config);
    let block_sep = match context.config.control_brace_style {
        ControlBraceStyle::AlwaysSameLine => " ",
        ControlBraceStyle::AlwaysNextLine => alt_block_sep.as_str(),
    };
    let mut result = format!("match {}{}{{", cond_str, block_sep);

    let nested_context = context.nested_context();
    let arm_indent = nested_context.block_indent;
    let arm_indent_str = arm_indent.to_string(context.config);

    let open_brace_pos = context.codemap
        .span_after(mk_sp(cond.span.hi, arm_start_pos(&arms[0])), "{");

    for (i, arm) in arms.iter().enumerate() {
        // Make sure we get the stuff between arms.
        let missed_str = if i == 0 {
            context.snippet(mk_sp(open_brace_pos, arm_start_pos(arm)))
        } else {
            context.snippet(mk_sp(arm_end_pos(&arms[i - 1]), arm_start_pos(arm)))
        };
        let comment = try_opt!(rewrite_match_arm_comment(context,
                                                         &missed_str,
                                                         width,
                                                         arm_indent,
                                                         &arm_indent_str));
        result.push_str(&comment);
        result.push('\n');
        result.push_str(&arm_indent_str);

        let arm_str = arm.rewrite(&nested_context,
                                  context.config.max_width - arm_indent.width(),
                                  arm_indent);
        if let Some(ref arm_str) = arm_str {
            result.push_str(arm_str);
        } else {
            // We couldn't format the arm, just reproduce the source.
            let snippet = context.snippet(mk_sp(arm_start_pos(arm), arm_end_pos(arm)));
            result.push_str(&snippet);
            result.push_str(arm_comma(&context.config, &arm, &arm.body));
        }
    }
    // BytePos(1) = closing match brace.
    let last_span = mk_sp(arm_end_pos(&arms[arms.len() - 1]), span.hi - BytePos(1));
    let last_comment = context.snippet(last_span);
    let comment = try_opt!(rewrite_match_arm_comment(context,
                                                     &last_comment,
                                                     width,
                                                     arm_indent,
                                                     &arm_indent_str));
    result.push_str(&comment);
    result.push('\n');
    result.push_str(&context.block_indent.to_string(context.config));
    result.push('}');
    Some(result)
}

fn arm_start_pos(arm: &ast::Arm) -> BytePos {
    let &ast::Arm { ref attrs, ref pats, .. } = arm;
    if !attrs.is_empty() {
        return attrs[0].span.lo;
    }

    pats[0].span.lo
}

fn arm_end_pos(arm: &ast::Arm) -> BytePos {
    arm.body.span.hi
}

fn arm_comma(config: &Config, arm: &ast::Arm, body: &ast::Expr) -> &'static str {
    if !config.match_wildcard_trailing_comma {
        if arm.pats.len() == 1 && arm.pats[0].node == ast::PatKind::Wild && arm.guard.is_none() {
            return "";
        }
    }

    if config.match_block_trailing_comma {
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
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        let &ast::Arm { ref attrs, ref pats, ref guard, ref body } = self;

        // FIXME this is all a bit grotty, would be nice to abstract out the
        // treatment of attributes.
        let attr_str = if !attrs.is_empty() {
            // We only use this visitor for the attributes, should we use it for
            // more?
            let mut attr_visitor = FmtVisitor::from_codemap(context.parse_session, context.config);
            attr_visitor.block_indent = context.block_indent;
            attr_visitor.last_pos = attrs[0].span.lo;
            if attr_visitor.visit_attrs(attrs) {
                // Attributes included a skip instruction.
                return None;
            }
            attr_visitor.format_missing(pats[0].span.lo);
            attr_visitor.buffer.to_string()
        } else {
            String::new()
        };

        // Patterns
        // 5 = ` => {`
        let pat_budget = try_opt!(width.checked_sub(5));
        let pat_strs = try_opt!(pats.iter()
            .map(|p| p.rewrite(context, pat_budget, offset))
            .collect::<Option<Vec<_>>>());

        let all_simple = pat_strs.iter().all(|p| pat_is_simple(&p));
        let items: Vec<_> = pat_strs.into_iter().map(ListItem::from_str).collect();
        let fmt = ListFormatting {
            tactic: if all_simple {
                DefinitiveListTactic::Mixed
            } else {
                DefinitiveListTactic::Vertical
            },
            separator: " |",
            trailing_separator: SeparatorTactic::Never,
            indent: offset,
            width: pat_budget,
            ends_with_newline: false,
            config: context.config,
        };
        let pats_str = try_opt!(write_list(items, &fmt));

        let budget = if pats_str.contains('\n') {
            context.config.max_width - offset.width()
        } else {
            width
        };

        let guard_str = try_opt!(rewrite_guard(context,
                                               guard,
                                               budget,
                                               offset,
                                               trimmed_last_line_width(&pats_str)));

        let pats_str = format!("{}{}", pats_str, guard_str);
        // Where the next text can start.
        let mut line_start = last_line_width(&pats_str);
        if !pats_str.contains('\n') {
            line_start += offset.width();
        }

        let body = match **body {
            ast::Expr { node: ast::ExprKind::Block(ref block), .. }
                if !is_unsafe_block(block) && is_simple_block(block, context.codemap) &&
                   context.config.wrap_match_arms => block.expr.as_ref().map(|e| &**e).unwrap(),
            ref x => x,
        };

        let comma = arm_comma(&context.config, self, body);
        let alt_block_sep = String::from("\n") + &context.block_indent.to_string(context.config);

        // Let's try and get the arm body on the same line as the condition.
        // 4 = ` => `.len()
        if context.config.max_width > line_start + comma.len() + 4 {
            let budget = context.config.max_width - line_start - comma.len() - 4;
            let offset = Indent::new(offset.block_indent, line_start + 4 - offset.block_indent);
            let rewrite = nop_block_collapse(body.rewrite(context, budget, offset), budget);
            let is_block = if let ast::ExprKind::Block(..) = body.node {
                true
            } else {
                false
            };

            let block_sep = match context.config.control_brace_style {
                ControlBraceStyle::AlwaysNextLine if is_block => alt_block_sep.as_str(),
                _ => " ",
            };
            match rewrite {
                Some(ref body_str) if !body_str.contains('\n') || !context.config.wrap_match_arms ||
                                      is_block => {
                    return Some(format!("{}{} =>{}{}{}",
                                        attr_str.trim_left(),
                                        pats_str,
                                        block_sep,
                                        body_str,
                                        comma));
                }
                _ => {}
            }
        }

        // FIXME: we're doing a second rewrite of the expr; This may not be
        // necessary.
        let body_budget = try_opt!(width.checked_sub(context.config.tab_spaces));
        let indent = context.block_indent.block_indent(context.config);
        let inner_context = &RewriteContext { block_indent: indent, ..*context };
        let next_line_body =
            try_opt!(nop_block_collapse(body.rewrite(inner_context, body_budget, indent),
                                        body_budget));
        let indent_str = offset.block_indent(context.config).to_string(context.config);
        let (body_prefix, body_suffix) = if context.config.wrap_match_arms {
            if context.config.match_block_trailing_comma {
                (" {", "},")
            } else {
                (" {", "}")
            }
        } else {
            ("", "")
        };

        let block_sep = match context.config.control_brace_style {
            ControlBraceStyle::AlwaysNextLine => alt_block_sep,
            ControlBraceStyle::AlwaysSameLine => String::from(body_prefix) + "\n",
        };
        Some(format!("{}{} =>{}{}{}\n{}{}",
                     attr_str.trim_left(),
                     pats_str,
                     block_sep,
                     indent_str,
                     next_line_body,
                     offset.to_string(context.config),
                     body_suffix))
    }
}

// A pattern is simple if it is very short or it is short-ish and just a path.
// E.g. `Foo::Bar` is simple, but `Foo(..)` is not.
fn pat_is_simple(pat_str: &str) -> bool {
    pat_str.len() <= 16 ||
    (pat_str.len() <= 24 && pat_str.chars().all(|c| c.is_alphabetic() || c == ':'))
}

// The `if ...` guard on a match arm.
fn rewrite_guard(context: &RewriteContext,
                 guard: &Option<ptr::P<ast::Expr>>,
                 width: usize,
                 offset: Indent,
                 // The amount of space used up on this line for the pattern in
                 // the arm (excludes offset).
                 pattern_width: usize)
                 -> Option<String> {
    if let Some(ref guard) = *guard {
        // First try to fit the guard string on the same line as the pattern.
        // 4 = ` if `, 5 = ` => {`
        let overhead = pattern_width + 4 + 5;
        if overhead < width {
            let cond_str = guard.rewrite(context, width - overhead, offset + pattern_width + 4);
            if let Some(cond_str) = cond_str {
                return Some(format!(" if {}", cond_str));
            }
        }

        // Not enough space to put the guard after the pattern, try a newline.
        let overhead = offset.block_indent(context.config).width() + 4 + 5;
        if overhead < width {
            let cond_str = guard.rewrite(context,
                                         width - overhead,
                                         // 3 == `if `
                                         offset.block_indent(context.config) + 3);
            if let Some(cond_str) = cond_str {
                return Some(format!("\n{}if {}",
                                    offset.block_indent(context.config).to_string(context.config),
                                    cond_str));
            }
        }

        None
    } else {
        Some(String::new())
    }
}

fn rewrite_pat_expr(context: &RewriteContext,
                    pat: Option<&ast::Pat>,
                    expr: &ast::Expr,
                    matcher: &str,
                    // Connecting piece between pattern and expression,
                    // *without* trailing space.
                    connector: &str,
                    width: usize,
                    offset: Indent)
                    -> Option<String> {
    let pat_offset = offset + matcher.len();
    let mut result = match pat {
        Some(pat) => {
            let pat_budget = try_opt!(width.checked_sub(connector.len() + matcher.len()));
            let pat_string = try_opt!(pat.rewrite(context, pat_budget, pat_offset));
            format!("{}{}{}", matcher, pat_string, connector)
        }
        None => String::new(),
    };

    // Consider only the last line of the pat string.
    let extra_offset = extra_offset(&result, offset);

    // The expression may (partionally) fit on the current line.
    if width > extra_offset + 1 {
        let spacer = if pat.is_some() { " " } else { "" };

        let expr_rewrite = expr.rewrite(context,
                                        try_opt!(width.checked_sub(extra_offset + spacer.len())),
                                        offset + extra_offset + spacer.len());

        if let Some(expr_string) = expr_rewrite {
            result.push_str(spacer);
            result.push_str(&expr_string);
            return Some(result);
        }
    }

    // The expression won't fit on the current line, jump to next.
    result.push('\n');
    result.push_str(&pat_offset.to_string(context.config));

    let expr_rewrite =
        expr.rewrite(context,
                     try_opt!(context.config.max_width.checked_sub(pat_offset.width())),
                     pat_offset);
    result.push_str(&&try_opt!(expr_rewrite));

    Some(result)
}

fn rewrite_string_lit(context: &RewriteContext,
                      span: Span,
                      width: usize,
                      offset: Indent)
                      -> Option<String> {
    let string_lit = context.snippet(span);

    if !context.config.format_strings && !context.config.force_format_strings {
        return Some(string_lit);
    }

    if !context.config.force_format_strings &&
       !string_requires_rewrite(context, span, &string_lit, width, offset) {
        return Some(string_lit);
    }

    let fmt = StringFormat {
        opener: "\"",
        closer: "\"",
        line_start: " ",
        line_end: "\\",
        width: width,
        offset: offset,
        trim_end: false,
        config: context.config,
    };

    // Remove the quote characters.
    let str_lit = &string_lit[1..string_lit.len() - 1];

    rewrite_string(str_lit, &fmt)
}

fn string_requires_rewrite(context: &RewriteContext,
                           span: Span,
                           string: &str,
                           width: usize,
                           offset: Indent)
                           -> bool {
    if context.codemap.lookup_char_pos(span.lo).col.0 != offset.width() {
        return true;
    }

    for (i, line) in string.lines().enumerate() {
        if i == 0 {
            if line.len() > width {
                return true;
            }
        } else {
            if line.len() > width + offset.width() {
                return true;
            }
        }
    }

    false
}

pub fn rewrite_call<R>(context: &RewriteContext,
                       callee: &R,
                       args: &[ptr::P<ast::Expr>],
                       span: Span,
                       width: usize,
                       offset: Indent)
                       -> Option<String>
    where R: Rewrite
{
    let closure = |callee_max_width| {
        rewrite_call_inner(context, callee, callee_max_width, args, span, width, offset)
    };

    // 2 is for parens
    let max_width = try_opt!(width.checked_sub(2));
    binary_search(1, max_width, closure)
}

fn rewrite_call_inner<R>(context: &RewriteContext,
                         callee: &R,
                         max_callee_width: usize,
                         args: &[ptr::P<ast::Expr>],
                         span: Span,
                         width: usize,
                         offset: Indent)
                         -> Result<String, Ordering>
    where R: Rewrite
{
    let callee = callee.borrow();
    // FIXME using byte lens instead of char lens (and probably all over the
    // place too)
    let callee_str = match callee.rewrite(context, max_callee_width, offset) {
        Some(string) => {
            if !string.contains('\n') && string.len() > max_callee_width {
                panic!("{:?} {}", string, max_callee_width);
            } else {
                string
            }
        }
        None => return Err(Ordering::Greater),
    };

    let span_lo = context.codemap.span_after(span, "(");
    let span = mk_sp(span_lo, span.hi);

    let extra_offset = extra_offset(&callee_str, offset);
    // 2 is for parens.
    let remaining_width = match width.checked_sub(extra_offset + 2) {
        Some(str) => str,
        None => return Err(Ordering::Greater),
    };
    let offset = offset + extra_offset + 1;
    let arg_count = args.len();
    let block_indent = if arg_count == 1 {
        context.block_indent
    } else {
        offset
    };
    let inner_context = &RewriteContext { block_indent: block_indent, ..*context };

    let items = itemize_list(context.codemap,
                             args.iter(),
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             |item| item.rewrite(&inner_context, remaining_width, offset),
                             span.lo,
                             span.hi);
    let mut item_vec: Vec<_> = items.collect();

    // Try letting the last argument overflow to the next line with block
    // indentation. If its first line fits on one line with the other arguments,
    // we format the function arguments horizontally.
    let overflow_last = match args.last().map(|x| &x.node) {
        Some(&ast::ExprKind::Closure(..)) |
        Some(&ast::ExprKind::Block(..)) if arg_count > 1 => true,
        _ => false,
    } && context.config.chains_overflow_last;

    let mut orig_last = None;
    let mut placeholder = None;

    // Replace the last item with its first line to see if it fits with
    // first arguments.
    if overflow_last {
        let inner_context = &RewriteContext { block_indent: context.block_indent, ..*context };
        let rewrite = args.last().unwrap().rewrite(&inner_context, remaining_width, offset);

        if let Some(rewrite) = rewrite {
            let rewrite_first_line = Some(rewrite[..first_line_width(&rewrite)].to_owned());
            placeholder = Some(rewrite);

            swap(&mut item_vec[arg_count - 1].item, &mut orig_last);
            item_vec[arg_count - 1].item = rewrite_first_line;
        }
    }

    let tactic = definitive_tactic(&item_vec,
                                   ListTactic::LimitedHorizontalVertical(context.config
                                       .fn_call_width),
                                   remaining_width);

    // Replace the stub with the full overflowing last argument if the rewrite
    // succeeded and its first line fits with the other arguments.
    match (overflow_last, tactic, placeholder) {
        (true, DefinitiveListTactic::Horizontal, placeholder @ Some(..)) => {
            item_vec[arg_count - 1].item = placeholder;
        }
        (true, _, _) => {
            item_vec[arg_count - 1].item = orig_last;
        }
        (false, _, _) => {}
    }

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: offset,
        width: width,
        ends_with_newline: false,
        config: context.config,
    };

    let list_str = match write_list(&item_vec, &fmt) {
        Some(str) => str,
        None => return Err(Ordering::Less),
    };

    Ok(format!("{}({})", callee_str, list_str))
}

fn rewrite_paren(context: &RewriteContext,
                 subexpr: &ast::Expr,
                 width: usize,
                 offset: Indent)
                 -> Option<String> {
    debug!("rewrite_paren, width: {}, offset: {:?}", width, offset);
    // 1 is for opening paren, 2 is for opening+closing, we want to keep the closing
    // paren on the same line as the subexpr.
    let subexpr_str = subexpr.rewrite(context, try_opt!(width.checked_sub(2)), offset + 1);
    debug!("rewrite_paren, subexpr_str: `{:?}`", subexpr_str);
    subexpr_str.map(|s| format!("({})", s))
}

fn rewrite_struct_lit<'a>(context: &RewriteContext,
                          path: &ast::Path,
                          fields: &'a [ast::Field],
                          base: Option<&'a ast::Expr>,
                          span: Span,
                          width: usize,
                          offset: Indent)
                          -> Option<String> {
    debug!("rewrite_struct_lit: width {}, offset {:?}", width, offset);

    enum StructLitField<'a> {
        Regular(&'a ast::Field),
        Base(&'a ast::Expr),
    }

    // 2 = " {".len()
    let path_budget = try_opt!(width.checked_sub(2));
    let path_str = try_opt!(rewrite_path(context, true, None, path, path_budget, offset));

    // Foo { a: Foo } - indent is +3, width is -5.
    let h_budget = width.checked_sub(path_str.len() + 5).unwrap_or(0);
    // The 1 taken from the v_budget is for the comma.
    let (indent, v_budget) = match context.config.struct_lit_style {
        StructLitStyle::Visual => (offset + path_str.len() + 3, h_budget),
        StructLitStyle::Block => {
            // If we are all on one line, then we'll ignore the indent, and we
            // have a smaller budget.
            let indent = context.block_indent.block_indent(context.config);
            let v_budget = context.config.max_width.checked_sub(indent.width()).unwrap_or(0);
            (indent, v_budget)
        }
    };

    let field_iter = fields.into_iter()
        .map(StructLitField::Regular)
        .chain(base.into_iter().map(StructLitField::Base));

    let inner_context = &RewriteContext { block_indent: indent, ..*context };

    let items = itemize_list(context.codemap,
                             field_iter,
                             "}",
                             |item| {
        match *item {
            StructLitField::Regular(ref field) => field.span.lo,
            StructLitField::Base(ref expr) => {
                let last_field_hi = fields.last().map_or(span.lo, |field| field.span.hi);
                let snippet = context.snippet(mk_sp(last_field_hi, expr.span.lo));
                let pos = snippet.find_uncommented("..").unwrap();
                last_field_hi + BytePos(pos as u32)
            }
        }
    },
                             |item| {
                                 match *item {
                                     StructLitField::Regular(ref field) => field.span.hi,
                                     StructLitField::Base(ref expr) => expr.span.hi,
                                 }
                             },
                             |item| {
        match *item {
            StructLitField::Regular(ref field) => {
                rewrite_field(inner_context,
                              &field,
                              v_budget.checked_sub(1).unwrap_or(0),
                              indent)
            }
            StructLitField::Base(ref expr) => {
                // 2 = ..
                expr.rewrite(inner_context, try_opt!(v_budget.checked_sub(2)), indent + 2)
                    .map(|s| format!("..{}", s))
            }
        }
    },
                             context.codemap.span_after(span, "{"),
                             span.hi);
    let item_vec = items.collect::<Vec<_>>();

    let tactic = {
        let mut prelim_tactic = match (context.config.struct_lit_style, fields.len()) {
            (StructLitStyle::Visual, 1) => ListTactic::HorizontalVertical,
            _ => context.config.struct_lit_multiline_style.to_list_tactic(),
        };

        if prelim_tactic == ListTactic::HorizontalVertical && fields.len() > 1 {
            prelim_tactic = ListTactic::LimitedHorizontalVertical(context.config.struct_lit_width);
        }

        definitive_tactic(&item_vec, prelim_tactic, h_budget)
    };

    let budget = match tactic {
        DefinitiveListTactic::Horizontal => h_budget,
        _ => v_budget,
    };

    let ends_with_newline = context.config.struct_lit_style != StructLitStyle::Visual &&
                            tactic == DefinitiveListTactic::Vertical;

    let fmt = ListFormatting {
        tactic: tactic,
        separator: ",",
        trailing_separator: if base.is_some() {
            SeparatorTactic::Never
        } else {
            context.config.struct_lit_trailing_comma
        },
        indent: indent,
        width: budget,
        ends_with_newline: ends_with_newline,
        config: context.config,
    };
    let fields_str = try_opt!(write_list(&item_vec, &fmt));

    if fields_str.is_empty() {
        return Some(format!("{} {{}}", path_str));
    }

    let format_on_newline = || {
        let inner_indent = context.block_indent
            .block_indent(context.config)
            .to_string(context.config);
        let outer_indent = context.block_indent.to_string(context.config);
        Some(format!("{} {{\n{}{}\n{}}}",
                     path_str,
                     inner_indent,
                     fields_str,
                     outer_indent))
    };

    match (context.config.struct_lit_style, context.config.struct_lit_multiline_style) {
        (StructLitStyle::Block, _) if fields_str.contains('\n') || fields_str.len() > h_budget => {
            format_on_newline()
        }
        (StructLitStyle::Block, MultilineStyle::ForceMulti) => format_on_newline(),
        _ => Some(format!("{} {{ {} }}", path_str, fields_str)),
    }

    // FIXME if context.config.struct_lit_style == Visual, but we run out
    // of space, we should fall back to BlockIndent.
}

fn rewrite_field(context: &RewriteContext,
                 field: &ast::Field,
                 width: usize,
                 offset: Indent)
                 -> Option<String> {
    let name = &field.ident.node.to_string();
    let overhead = name.len() + 2;
    let expr = field.expr.rewrite(context,
                                  try_opt!(width.checked_sub(overhead)),
                                  offset + overhead);

    match expr {
        Some(e) => Some(format!("{}: {}", name, e)),
        None => {
            let expr_offset = offset.block_indent(&context.config);
            let expr = field.expr.rewrite(context,
                                          try_opt!(context.config
                                              .max_width
                                              .checked_sub(expr_offset.width())),
                                          expr_offset);
            expr.map(|s| format!("{}:\n{}{}", name, expr_offset.to_string(&context.config), s))
        }
    }
}

pub fn rewrite_tuple<'a, I>(context: &RewriteContext,
                            mut items: I,
                            span: Span,
                            width: usize,
                            offset: Indent)
                            -> Option<String>
    where I: ExactSizeIterator,
          <I as Iterator>::Item: Deref,
          <I::Item as Deref>::Target: Rewrite + Spanned + 'a
{
    let indent = offset + 1;
    // In case of length 1, need a trailing comma
    if items.len() == 1 {
        // 3 = "(" + ",)"
        let budget = try_opt!(width.checked_sub(3));
        return items.next().unwrap().rewrite(context, budget, indent).map(|s| format!("({},)", s));
    }

    let list_lo = context.codemap.span_after(span, "(");
    let items = itemize_list(context.codemap,
                             items,
                             ")",
                             |item| item.span().lo,
                             |item| item.span().hi,
                             |item| {
                                 let inner_width = try_opt!(context.config
                                     .max_width
                                     .checked_sub(indent.width() + 1));
                                 item.rewrite(context, inner_width, indent)
                             },
                             list_lo,
                             span.hi - BytePos(1));
    let budget = try_opt!(width.checked_sub(2));
    let list_str = try_opt!(format_item_list(items, budget, indent, context.config));

    Some(format!("({})", list_str))
}

fn rewrite_binary_op(context: &RewriteContext,
                     op: &ast::BinOp,
                     lhs: &ast::Expr,
                     rhs: &ast::Expr,
                     width: usize,
                     offset: Indent)
                     -> Option<String> {
    // FIXME: format comments between operands and operator

    let operator_str = context.snippet(op.span);

    // Get "full width" rhs and see if it fits on the current line. This
    // usually works fairly well since it tends to place operands of
    // operations with high precendence close together.
    let rhs_result = try_opt!(rhs.rewrite(context, width, offset));

    // Second condition is needed in case of line break not caused by a
    // shortage of space, but by end-of-line comments, for example.
    // Note that this is non-conservative, but its just to see if it's even
    // worth trying to put everything on one line.
    if rhs_result.len() + 2 + operator_str.len() < width && !rhs_result.contains('\n') {
        // 1 = space between lhs expr and operator
        if let Some(mut result) = lhs.rewrite(context, width - 1 - operator_str.len(), offset) {
            result.push(' ');
            result.push_str(&operator_str);
            result.push(' ');

            let remaining_width = width.checked_sub(last_line_width(&result)).unwrap_or(0);

            if rhs_result.len() <= remaining_width {
                result.push_str(&rhs_result);
                return Some(result);
            }

            if let Some(rhs_result) = rhs.rewrite(context, remaining_width, offset + result.len()) {
                if rhs_result.len() <= remaining_width {
                    result.push_str(&rhs_result);
                    return Some(result);
                }
            }
        }
    }

    // We have to use multiple lines.

    // Re-evaluate the lhs because we have more space now:
    let budget = try_opt!(context.config
        .max_width
        .checked_sub(offset.width() + 1 + operator_str.len()));
    Some(format!("{} {}\n{}{}",
                 try_opt!(lhs.rewrite(context, budget, offset)),
                 operator_str,
                 offset.to_string(context.config),
                 rhs_result))
}

pub fn rewrite_unary_prefix<R: Rewrite>(context: &RewriteContext,
                                        prefix: &str,
                                        rewrite: &R,
                                        width: usize,
                                        offset: Indent)
                                        -> Option<String> {
    rewrite.rewrite(context,
                 try_opt!(width.checked_sub(prefix.len())),
                 offset + prefix.len())
        .map(|r| format!("{}{}", prefix, r))
}

// FIXME: this is probably not correct for multi-line Rewrites. we should
// subtract suffix.len() from the last line budget, not the first!
pub fn rewrite_unary_suffix<R: Rewrite>(context: &RewriteContext,
                                        suffix: &str,
                                        rewrite: &R,
                                        width: usize,
                                        offset: Indent)
                                        -> Option<String> {
    rewrite.rewrite(context, try_opt!(width.checked_sub(suffix.len())), offset)
        .map(|mut r| {
            r.push_str(suffix);
            r
        })
}

fn rewrite_unary_op(context: &RewriteContext,
                    op: &ast::UnOp,
                    expr: &ast::Expr,
                    width: usize,
                    offset: Indent)
                    -> Option<String> {
    // For some reason, an UnOp is not spanned like BinOp!
    let operator_str = match *op {
        ast::UnOp::Deref => "*",
        ast::UnOp::Not => "!",
        ast::UnOp::Neg => "-",
    };
    rewrite_unary_prefix(context, operator_str, expr, width, offset)
}

fn rewrite_assignment(context: &RewriteContext,
                      lhs: &ast::Expr,
                      rhs: &ast::Expr,
                      op: Option<&ast::BinOp>,
                      width: usize,
                      offset: Indent)
                      -> Option<String> {
    let operator_str = match op {
        Some(op) => context.snippet(op.span),
        None => "=".to_owned(),
    };

    // 1 = space between lhs and operator.
    let max_width = try_opt!(width.checked_sub(operator_str.len() + 1));
    let lhs_str = format!("{} {}",
                          try_opt!(lhs.rewrite(context, max_width, offset)),
                          operator_str);

    rewrite_assign_rhs(&context, lhs_str, rhs, width, offset)
}

// The left hand side must contain everything up to, and including, the
// assignment operator.
pub fn rewrite_assign_rhs<S: Into<String>>(context: &RewriteContext,
                                           lhs: S,
                                           ex: &ast::Expr,
                                           width: usize,
                                           offset: Indent)
                                           -> Option<String> {
    let mut result = lhs.into();
    let last_line_width = last_line_width(&result) -
                          if result.contains('\n') {
        offset.width()
    } else {
        0
    };
    // 1 = space between operator and rhs.
    let max_width = try_opt!(width.checked_sub(last_line_width + 1));
    let rhs = ex.rewrite(&context, max_width, offset + last_line_width + 1);

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
            let new_offset = offset.block_indent(context.config);
            let max_width = try_opt!((width + offset.width()).checked_sub(new_offset.width()));
            let inner_context = context.nested_context();
            let new_rhs = ex.rewrite(&inner_context, max_width, new_offset);

            // FIXME: DRY!
            match (rhs, new_rhs) {
                (Some(ref orig_rhs), Some(ref replacement_rhs))
                    if count_line_breaks(orig_rhs) >
                       count_line_breaks(replacement_rhs) + 1 => {
                    result.push_str(&format!("\n{}", new_offset.to_string(context.config)));
                    result.push_str(replacement_rhs);
                }
                (None, Some(ref final_rhs)) => {
                    result.push_str(&format!("\n{}", new_offset.to_string(context.config)));
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

fn rewrite_expr_addrof(context: &RewriteContext,
                       mutability: ast::Mutability,
                       expr: &ast::Expr,
                       width: usize,
                       offset: Indent)
                       -> Option<String> {
    let operator_str = match mutability {
        ast::Mutability::Immutable => "&",
        ast::Mutability::Mutable => "&mut ",
    };
    rewrite_unary_prefix(context, operator_str, expr, width, offset)
}
