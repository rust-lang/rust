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

use {Indent, Spanned};
use rewrite::{Rewrite, RewriteContext};
use lists::{write_list, itemize_list, ListFormatting, SeparatorTactic, ListTactic,
            DefinitiveListTactic, definitive_tactic, ListItem, format_fn_args};
use string::{StringFormat, rewrite_string};
use utils::{span_after, extra_offset, last_line_width, wrap_str, binary_search, first_line_width,
            semicolon_for_stmt};
use visitor::FmtVisitor;
use config::{StructLitStyle, MultilineStyle};
use comment::{FindUncommented, rewrite_comment, contains_comment};
use types::rewrite_path;
use items::{span_lo_for_arg, span_hi_for_arg};
use chains::rewrite_chain;
use macros::rewrite_macro;

use syntax::{ast, ptr};
use syntax::codemap::{CodeMap, Span, BytePos, mk_sp};
use syntax::visit::Visitor;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match self.node {
            ast::Expr_::ExprVec(ref expr_vec) => {
                rewrite_array(expr_vec.iter().map(|e| &**e),
                              mk_sp(span_after(self.span, "[", context.codemap), self.span.hi),
                              context,
                              width,
                              offset)
            }
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(_, ast::StrStyle::CookedStr) => {
                        rewrite_string_lit(context, l.span, width, offset)
                    }
                    _ => {
                        wrap_str(context.snippet(self.span),
                                 context.config.max_width,
                                 width,
                                 offset)
                    }
                }
            }
            ast::Expr_::ExprCall(ref callee, ref args) => {
                let inner_span = mk_sp(callee.span.hi, self.span.hi);
                rewrite_call(context, &**callee, args, inner_span, width, offset)
            }
            ast::Expr_::ExprParen(ref subexpr) => {
                rewrite_paren(context, subexpr, width, offset)
            }
            ast::Expr_::ExprBinary(ref op, ref lhs, ref rhs) => {
                rewrite_binary_op(context, op, lhs, rhs, width, offset)
            }
            ast::Expr_::ExprUnary(ref op, ref subexpr) => {
                rewrite_unary_op(context, op, subexpr, width, offset)
            }
            ast::Expr_::ExprStruct(ref path, ref fields, ref base) => {
                rewrite_struct_lit(context,
                                   path,
                                   fields,
                                   base.as_ref().map(|e| &**e),
                                   self.span,
                                   width,
                                   offset)
            }
            ast::Expr_::ExprTup(ref items) => {
                rewrite_tuple(context, items, self.span, width, offset)
            }
            ast::Expr_::ExprWhile(ref cond, ref block, label) => {
                Loop::new_while(None, cond, block, label).rewrite(context, width, offset)
            }
            ast::Expr_::ExprWhileLet(ref pat, ref cond, ref block, label) => {
                Loop::new_while(Some(pat), cond, block, label).rewrite(context, width, offset)
            }
            ast::Expr_::ExprForLoop(ref pat, ref cond, ref block, label) => {
                Loop::new_for(pat, cond, block, label).rewrite(context, width, offset)
            }
            ast::Expr_::ExprLoop(ref block, label) => {
                Loop::new_loop(block, label).rewrite(context, width, offset)
            }
            ast::Expr_::ExprBlock(ref block) => {
                block.rewrite(context, width, offset)
            }
            ast::Expr_::ExprIf(ref cond, ref if_block, ref else_block) => {
                rewrite_if_else(context,
                                cond,
                                if_block,
                                else_block.as_ref().map(|e| &**e),
                                None,
                                width,
                                offset,
                                true)
            }
            ast::Expr_::ExprIfLet(ref pat, ref cond, ref if_block, ref else_block) => {
                rewrite_if_else(context,
                                cond,
                                if_block,
                                else_block.as_ref().map(|e| &**e),
                                Some(pat),
                                width,
                                offset,
                                true)
            }
            ast::Expr_::ExprMatch(ref cond, ref arms) => {
                rewrite_match(context, cond, arms, width, offset, self.span)
            }
            ast::Expr_::ExprPath(ref qself, ref path) => {
                rewrite_path(context, true, qself.as_ref(), path, width, offset)
            }
            ast::Expr_::ExprAssign(ref lhs, ref rhs) => {
                rewrite_assignment(context, lhs, rhs, None, width, offset)
            }
            ast::Expr_::ExprAssignOp(ref op, ref lhs, ref rhs) => {
                rewrite_assignment(context, lhs, rhs, Some(op), width, offset)
            }
            ast::Expr_::ExprAgain(ref opt_ident) => {
                let id_str = match *opt_ident {
                    Some(ident) => format!(" {}", ident.node),
                    None => String::new(),
                };
                Some(format!("continue{}", id_str))
            }
            ast::Expr_::ExprBreak(ref opt_ident) => {
                let id_str = match *opt_ident {
                    Some(ident) => format!(" {}", ident.node),
                    None => String::new(),
                };
                Some(format!("break{}", id_str))
            }
            ast::Expr_::ExprClosure(capture, ref fn_decl, ref body) => {
                rewrite_closure(capture, fn_decl, body, self.span, context, width, offset)
            }
            ast::Expr_::ExprField(..) |
            ast::Expr_::ExprTupField(..) |
            ast::Expr_::ExprMethodCall(..) => {
                rewrite_chain(self, context, width, offset)
            }
            ast::Expr_::ExprMac(ref mac) => {
                // Failure to rewrite a marco should not imply failure to
                // rewrite the expression.
                rewrite_macro(mac, context, width, offset).or_else(|| {
                    wrap_str(context.snippet(self.span),
                             context.config.max_width,
                             width,
                             offset)
                })
            }
            ast::Expr_::ExprRet(None) => {
                wrap_str("return".to_owned(), context.config.max_width, width, offset)
            }
            ast::Expr_::ExprRet(Some(ref expr)) => {
                rewrite_unary_prefix(context, "return ", &**expr, width, offset)
            }
            ast::Expr_::ExprBox(ref expr) => {
                rewrite_unary_prefix(context, "box ", &**expr, width, offset)
            }
            ast::Expr_::ExprAddrOf(mutability, ref expr) => {
                rewrite_expr_addrof(context, mutability, expr, width, offset)
            }
            ast::Expr_::ExprCast(ref expr, ref ty) => {
                rewrite_pair(&**expr, &**ty, "", " as ", "", context, width, offset)
            }
            ast::Expr_::ExprIndex(ref expr, ref index) => {
                rewrite_pair(&**expr, &**index, "", "[", "]", context, width, offset)
            }
            ast::Expr_::ExprRepeat(ref expr, ref repeats) => {
                rewrite_pair(&**expr, &**repeats, "[", "; ", "]", context, width, offset)
            }
            ast::Expr_::ExprRange(Some(ref lhs), Some(ref rhs)) => {
                rewrite_pair(&**lhs, &**rhs, "", "..", "", context, width, offset)
            }
            ast::Expr_::ExprRange(None, Some(ref rhs)) => {
                rewrite_unary_prefix(context, "..", &**rhs, width, offset)
            }
            ast::Expr_::ExprRange(Some(ref lhs), None) => {
                Some(format!("{}..",
                             try_opt!(lhs.rewrite(context,
                                                  try_opt!(width.checked_sub(2)),
                                                  offset))))
            }
            ast::Expr_::ExprRange(None, None) => {
                if width >= 2 {
                    Some("..".into())
                } else {
                    None
                }
            }
            // We do not format these expressions yet, but they should still
            // satisfy our width restrictions.
            ast::Expr_::ExprInPlace(..) |
            ast::Expr_::ExprInlineAsm(..) => {
                wrap_str(context.snippet(self.span),
                         context.config.max_width,
                         width,
                         offset)
            }
        }
    }
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
                                      .fold(Some(false),
                                            |acc, x| acc.and_then(|y| x.map(|x| x || y))));

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

// This functions is pretty messy because of the wrapping and unwrapping of
// expressions into and from blocks. See rust issue #27872.
fn rewrite_closure(capture: ast::CaptureClause,
                   fn_decl: &ast::FnDecl,
                   body: &ast::Block,
                   span: Span,
                   context: &RewriteContext,
                   width: usize,
                   offset: Indent)
                   -> Option<String> {
    let mover = if capture == ast::CaptureClause::CaptureByValue {
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
    // 1 = space between arguments and return type.
    let horizontal_budget = budget.checked_sub(ret_str.len() + 1).unwrap_or(0);

    let arg_items = itemize_list(context.codemap,
                                 fn_decl.inputs.iter(),
                                 "|",
                                 |arg| span_lo_for_arg(arg),
                                 |arg| span_hi_for_arg(arg),
                                 |arg| arg.rewrite(context, budget, argument_offset),
                                 span_after(span, "|", context.codemap),
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

    // Try to format closure body as a single line expression without braces.
    if is_simple_block(body, context.codemap) && !prefix.contains('\n') {
        let (spacer, closer) = if ret_str.is_empty() {
            (" ", "")
        } else {
            (" { ", " }")
        };
        let expr = body.expr.as_ref().unwrap();
        // All closure bodies are blocks in the eyes of the AST, but we may not
        // want to unwrap them when they only contain a single expression.
        let inner_expr = match expr.node {
            ast::Expr_::ExprBlock(ref inner) if inner.stmts.is_empty() && inner.expr.is_some() &&
                                                inner.rules ==
                                                ast::BlockCheckMode::DefaultBlock => {
                inner.expr.as_ref().unwrap()
            }
            _ => expr,
        };
        let extra_offset = extra_offset(&prefix, offset) + spacer.len();
        let budget = try_opt!(width.checked_sub(extra_offset + closer.len()));
        let rewrite = inner_expr.rewrite(context, budget, offset + extra_offset);

        // Checks if rewrite succeeded and fits on a single line.
        let accept_rewrite = rewrite.as_ref().map(|result| !result.contains('\n')).unwrap_or(false);

        if accept_rewrite {
            return Some(format!("{}{}{}{}", prefix, spacer, rewrite.unwrap(), closer));
        }
    }

    // We couldn't format the closure body as a single line expression; fall
    // back to block formatting.
    let body_rewrite = body.expr
                           .as_ref()
                           .and_then(|body_expr| {
                               if let ast::Expr_::ExprBlock(ref inner) = body_expr.node {
                                   Some(inner.rewrite(&context, 2, Indent::empty()))
                               } else {
                                   None
                               }
                           })
                           .unwrap_or_else(|| body.rewrite(&context, 2, Indent::empty()));

    Some(format!("{} {}", prefix, try_opt!(body_rewrite)))
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

        let mut visitor = FmtVisitor::from_codemap(context.parse_session, context.config, None);
        visitor.block_indent = context.block_indent;

        let prefix = match self.rules {
            ast::BlockCheckMode::UnsafeBlock(..) => {
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
            ast::BlockCheckMode::DefaultBlock => {
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
        match self.node {
            ast::Stmt_::StmtDecl(ref decl, _) => {
                if let ast::Decl_::DeclLocal(ref local) = decl.node {
                    local.rewrite(context, context.config.max_width, offset)
                } else {
                    None
                }
            }
            ast::Stmt_::StmtExpr(ref ex, _) | ast::Stmt_::StmtSemi(ref ex, _) => {
                let suffix = if semicolon_for_stmt(self) {
                    ";"
                } else {
                    ""
                };

                ex.rewrite(context,
                           context.config.max_width - offset.width() - suffix.len(),
                           offset)
                  .map(|s| s + suffix)
            }
            ast::Stmt_::StmtMac(..) => None,
        }
    }
}

// Abstraction over for, while and loop expressions
struct Loop<'a> {
    cond: Option<&'a ast::Expr>,
    block: &'a ast::Block,
    label: Option<ast::Ident>,
    pat: Option<&'a ast::Pat>,
    keyword: &'a str,
    matcher: &'a str,
    connector: &'a str,
}

impl<'a> Loop<'a> {
    fn new_loop(block: &'a ast::Block, label: Option<ast::Ident>) -> Loop<'a> {
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
                 label: Option<ast::Ident>)
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
               label: Option<ast::Ident>)
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

        // FIXME: this drops any comment between "loop" and the block.
        self.block
            .rewrite(context, width, offset)
            .map(|result| {
                format!("{}{}{} {}",
                        label_string,
                        self.keyword,
                        pat_expr_string,
                        result)
            })
    }
}

fn rewrite_label(label: Option<ast::Ident>) -> String {
    match label {
        Some(ident) => format!("{}: ", ident),
        None => "".to_owned(),
    }
}

// Rewrites if-else blocks. If let Some(_) = pat, the expression is
// treated as an if-let-else expression.
fn rewrite_if_else(context: &RewriteContext,
                   cond: &ast::Expr,
                   if_block: &ast::Block,
                   else_block_opt: Option<&ast::Expr>,
                   pat: Option<&ast::Pat>,
                   width: usize,
                   offset: Indent,
                   allow_single_line: bool)
                   -> Option<String> {
    // 3 = "if ", 2 = " {"
    let pat_expr_string = try_opt!(rewrite_pat_expr(context,
                                                    pat,
                                                    cond,
                                                    "let ",
                                                    " =",
                                                    try_opt!(width.checked_sub(3 + 2)),
                                                    offset + 3));

    // Try to format if-else on single line.
    if allow_single_line && context.config.single_line_if_else {
        let trial = single_line_if_else(context, &pat_expr_string, if_block, else_block_opt, width);

        if trial.is_some() {
            return trial;
        }
    }

    let if_block_string = try_opt!(if_block.rewrite(context, width, offset));
    let mut result = format!("if {} {}", pat_expr_string, if_block_string);

    if let Some(else_block) = else_block_opt {
        let rewrite = match else_block.node {
            // If the else expression is another if-else expression, prevent it
            // from being formatted on a single line.
            ast::Expr_::ExprIfLet(ref pat, ref cond, ref if_block, ref else_block) => {
                rewrite_if_else(context,
                                cond,
                                if_block,
                                else_block.as_ref().map(|e| &**e),
                                Some(pat),
                                width,
                                offset,
                                false)
            }
            ast::Expr_::ExprIf(ref cond, ref if_block, ref else_block) => {
                rewrite_if_else(context,
                                cond,
                                if_block,
                                else_block.as_ref().map(|e| &**e),
                                None,
                                width,
                                offset,
                                false)
            }
            _ => else_block.rewrite(context, width, offset),
        };

        result.push_str(" else ");
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

    if let ast::ExprBlock(ref else_node) = else_block.node {
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
        let comment = try_opt!(rewrite_comment(&missed_str,
                                               false,
                                               width,
                                               arm_indent,
                                               context.config));
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
    let mut result = format!("match {} {{", cond_str);

    let nested_context = context.nested_context();
    let arm_indent = nested_context.block_indent;
    let arm_indent_str = arm_indent.to_string(context.config);

    let open_brace_pos = span_after(mk_sp(cond.span.hi, arm_start_pos(&arms[0])),
                                    "{",
                                    context.codemap);

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
            result.push_str(arm_comma(&arm.body));
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

fn arm_comma(body: &ast::Expr) -> &'static str {
    if let ast::ExprBlock(ref block) = body.node {
        if let ast::DefaultBlock = block.rules {
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
        let indent_str = offset.to_string(context.config);

        // FIXME this is all a bit grotty, would be nice to abstract out the
        // treatment of attributes.
        let attr_str = if !attrs.is_empty() {
            // We only use this visitor for the attributes, should we use it for
            // more?
            let mut attr_visitor = FmtVisitor::from_codemap(context.parse_session,
                                                            context.config,
                                                            None);
            attr_visitor.block_indent = context.block_indent;
            attr_visitor.last_pos = attrs[0].span.lo;
            if attr_visitor.visit_attrs(attrs) {
                // Attributes included a skip instruction.
                let snippet = context.snippet(mk_sp(attrs[0].span.lo, body.span.hi));
                return Some(snippet);
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

        let mut total_width = pat_strs.iter().fold(0, |a, p| a + p.len());
        // Add ` | `.len().
        total_width += (pat_strs.len() - 1) * 3;

        let mut vertical = total_width > pat_budget || pat_strs.iter().any(|p| p.contains('\n'));
        if !vertical && context.config.take_source_hints {
            // If the patterns were previously stacked, keep them stacked.
            let pat_span = mk_sp(pats[0].span.lo, pats[pats.len() - 1].span.hi);
            let pat_str = context.snippet(pat_span);
            vertical = pat_str.contains('\n');
        }

        let pats_width = if vertical {
            pat_strs[pat_strs.len() - 1].len()
        } else {
            total_width
        };

        let mut pats_str = String::new();
        for p in pat_strs {
            if !pats_str.is_empty() {
                if vertical {
                    pats_str.push_str(" |\n");
                    pats_str.push_str(&indent_str);
                } else {
                    pats_str.push_str(" | ");
                }
            }
            pats_str.push_str(&p);
        }

        let guard_str = try_opt!(rewrite_guard(context, guard, width, offset, pats_width));

        let pats_str = format!("{}{}", pats_str, guard_str);
        // Where the next text can start.
        let mut line_start = last_line_width(&pats_str);
        if !pats_str.contains('\n') {
            line_start += offset.width();
        }

        let comma = arm_comma(body);

        // let body = match *body {
        //     ast::ExprBlock(ref b) if is_simple_block(b, context.codemap) => b.expr,
        //     ref x => x,
        // };

        // Let's try and get the arm body on the same line as the condition.
        // 4 = ` => `.len()
        let same_line_body = if context.config.max_width > line_start + comma.len() + 4 {
            let budget = context.config.max_width - line_start - comma.len() - 4;
            let offset = Indent::new(offset.block_indent, line_start + 4 - offset.block_indent);
            let rewrite = nop_block_collapse(body.rewrite(context, budget, offset), budget);

            match rewrite {
                Some(ref body_str) if body_str.len() <= budget || comma.is_empty() => {
                    return Some(format!("{}{} => {}{}",
                                        attr_str.trim_left(),
                                        pats_str,
                                        body_str,
                                        comma));
                }
                _ => rewrite,
            }
        } else {
            None
        };

        if let ast::ExprBlock(_) = body.node {
            // We're trying to fit a block in, but it still failed, give up.
            return None;
        }

        let mut result = format!("{}{} =>", attr_str.trim_left(), pats_str);

        match same_line_body {
            // FIXME: also take this branch is expr is block
            Some(ref body) if !body.contains('\n') => {
                result.push(' ');
                result.push_str(&body);
            }
            _ => {
                let body_budget = try_opt!(width.checked_sub(context.config.tab_spaces));
                let indent = context.block_indent.block_indent(context.config);
                let inner_context = &RewriteContext { block_indent: indent, ..*context };
                let next_line_body = try_opt!(nop_block_collapse(body.rewrite(inner_context,
                                                                              body_budget,
                                                                              indent),
                                                                 body_budget));

                result.push_str(" {\n");
                let indent_str = offset.block_indent(context.config).to_string(context.config);
                result.push_str(&indent_str);
                result.push_str(&next_line_body);
                result.push('\n');
                result.push_str(&offset.to_string(context.config));
                result.push('}');
            }
        };

        Some(result)
    }
}

// Takes two possible rewrites for the match arm body and chooses the "nicest".
fn match_arm_heuristic<'a>(former: Option<&'a str>, latter: Option<&'a str>) -> Option<&'a str> {
    match (former, latter) {
        (f @ Some(..), None) => f,
        (Some(f), Some(l)) if f.chars().filter(|&c| c == '\n').count() <=
                              l.chars().filter(|&c| c == '\n').count() => {
            Some(f)
        }
        (_, l) => l,
    }
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
    if let &Some(ref guard) = guard {
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
        let overhead = context.config.tab_spaces + 4 + 5;
        if overhead < width {
            let cond_str = guard.rewrite(context,
                                         width - overhead,
                                         offset.block_indent(context.config));
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
        let spacer = if pat.is_some() {
            " "
        } else {
            ""
        };

        let expr_rewrite = expr.rewrite(context,
                                        width - extra_offset - spacer.len(),
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

    let expr_rewrite = expr.rewrite(context,
                                    context.config.max_width - pat_offset.width(),
                                    pat_offset);
    result.push_str(&&try_opt!(expr_rewrite));

    Some(result)
}

fn rewrite_string_lit(context: &RewriteContext,
                      span: Span,
                      width: usize,
                      offset: Indent)
                      -> Option<String> {
    if !context.config.format_strings {
        return Some(context.snippet(span));
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

    let string_lit = context.snippet(span);
    let str_lit = &string_lit[1..string_lit.len() - 1]; // Remove the quote characters.

    rewrite_string(str_lit, &fmt)
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

    let span_lo = span_after(span, "(", context.codemap);
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
        Some(&ast::Expr_::ExprClosure(..)) |
        Some(&ast::Expr_::ExprBlock(..)) if arg_count > 1 => true,
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
        (true,
         DefinitiveListTactic::Horizontal,
         placeholder @ Some(..)) => {
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
    assert!(!fields.is_empty() || base.is_some());

    enum StructLitField<'a> {
        Regular(&'a ast::Field),
        Base(&'a ast::Expr),
    }

    // 2 = " {".len()
    let path_budget = try_opt!(width.checked_sub(2));
    let path_str = try_opt!(rewrite_path(context, true, None, path, path_budget, offset));

    // Foo { a: Foo } - indent is +3, width is -5.
    let h_budget = width.checked_sub(path_str.len() + 5).unwrap_or(0);
    let (indent, v_budget) = match context.config.struct_lit_style {
        StructLitStyle::Visual => {
            (offset + path_str.len() + 3, h_budget)
        }
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
                                         let last_field_hi = fields.last().map_or(span.lo,
                                                                                  |field| {
                                                                                      field.span.hi
                                                                                  });
                                         let snippet = context.snippet(mk_sp(last_field_hi,
                                                                             expr.span.lo));
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
                                         rewrite_field(inner_context, &field, v_budget, indent)
                                     }
                                     StructLitField::Base(ref expr) => {
                                         // 2 = ..
                                         expr.rewrite(inner_context,
                                                      try_opt!(v_budget.checked_sub(2)),
                                                      indent + 2)
                                             .map(|s| format!("..{}", s))
                                     }
                                 }
                             },
                             span_after(span, "{", context.codemap),
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
        ends_with_newline: match tactic {
            DefinitiveListTactic::Horizontal => false,
            DefinitiveListTactic::Vertical => true,
            DefinitiveListTactic::Mixed => unreachable!(),
        },
        config: context.config,
    };
    let fields_str = try_opt!(write_list(&item_vec, &fmt));

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

    match (context.config.struct_lit_style,
           context.config.struct_lit_multiline_style) {
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
    expr.map(|s| format!("{}: {}", name, s))
}

pub fn rewrite_tuple<'a, R>(context: &RewriteContext,
                            items: &'a [ptr::P<R>],
                            span: Span,
                            width: usize,
                            offset: Indent)
                            -> Option<String>
    where R: Rewrite + Spanned + 'a
{
    debug!("rewrite_tuple_lit: width: {}, offset: {:?}", width, offset);
    let indent = offset + 1;
    // In case of length 1, need a trailing comma
    if items.len() == 1 {
        // 3 = "(" + ",)"
        let budget = try_opt!(width.checked_sub(3));
        return items[0].rewrite(context, budget, indent).map(|s| format!("({},)", s));
    }

    let items = itemize_list(context.codemap,
                             items.iter(),
                             ")",
                             |item| item.span().lo,
                             |item| item.span().hi,
                             |item| {
                                 let inner_width = try_opt!(context.config
                                                                   .max_width
                                                                   .checked_sub(indent.width() +
                                                                                1));
                                 item.rewrite(context, inner_width, indent)
                             },
                             span.lo + BytePos(1), // Remove parens
                             span.hi - BytePos(1));
    let budget = try_opt!(width.checked_sub(2));
    let list_str = try_opt!(format_fn_args(items, budget, indent, context.config));

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

fn rewrite_unary_op(context: &RewriteContext,
                    op: &ast::UnOp,
                    expr: &ast::Expr,
                    width: usize,
                    offset: Indent)
                    -> Option<String> {
    // For some reason, an UnOp is not spanned like BinOp!
    let operator_str = match *op {
        ast::UnOp::UnDeref => "*",
        ast::UnOp::UnNot => "!",
        ast::UnOp::UnNeg => "-",
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

    match rhs {
        Some(new_str) => {
            result.push(' ');
            result.push_str(&new_str)
        }
        None => {
            // Expression did not fit on the same line as the identifier. Retry
            // on the next line.
            let new_offset = offset.block_indent(context.config);
            result.push_str(&format!("\n{}", new_offset.to_string(context.config)));

            // FIXME: we probably should related max_width to width instead of
            // config.max_width where is the 1 coming from anyway?
            let max_width = try_opt!(context.config.max_width.checked_sub(new_offset.width() + 1));
            let inner_context = context.nested_context();
            let rhs = ex.rewrite(&inner_context, max_width, new_offset);

            result.push_str(&&try_opt!(rhs));
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
        ast::Mutability::MutImmutable => "&",
        ast::Mutability::MutMutable => "&mut ",
    };
    rewrite_unary_prefix(context, operator_str, expr, width, offset)
}
