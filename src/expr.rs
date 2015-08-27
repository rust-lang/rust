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
use lists::{write_list, itemize_list, ListFormatting, SeparatorTactic, ListTactic};
use string::{StringFormat, rewrite_string};
use StructLitStyle;
use utils::{span_after, make_indent, extra_offset, first_line_width, last_line_width};
use visitor::FmtVisitor;
use config::BlockIndentStyle;
use comment::{FindUncommented, rewrite_comment, contains_comment};
use types::rewrite_path;
use items::{span_lo_for_arg, span_hi_for_arg, rewrite_fn_input};

use syntax::{ast, ptr};
use syntax::codemap::{CodeMap, Pos, Span, BytePos, mk_sp};
use syntax::visit::Visitor;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        match self.node {
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(ref is, ast::StrStyle::CookedStr) => {
                        rewrite_string_lit(context, &is, l.span, width, offset)
                    }
                    _ => context.codemap.span_to_snippet(self.span).ok(),
                }
            }
            ast::Expr_::ExprCall(ref callee, ref args) => {
                rewrite_call(context, callee, args, self.span, width, offset)
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
                rewrite_tuple_lit(context, items, self.span, width, offset)
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
            // We reformat it ourselves because rustc gives us a bad span
            // for ranges, see rust#27162
            ast::Expr_::ExprRange(ref left, ref right) => {
                rewrite_range(context,
                              left.as_ref().map(|e| &**e),
                              right.as_ref().map(|e| &**e),
                              width,
                              offset)
            }
            ast::Expr_::ExprMatch(ref cond, ref arms, _) => {
                rewrite_match(context, cond, arms, width, offset)
            }
            ast::Expr_::ExprPath(ref qself, ref path) => {
                rewrite_path(context, qself.as_ref(), path, width, offset)
            }
            ast::Expr_::ExprAssign(ref lhs, ref rhs) => {
                rewrite_assignment(context, lhs, rhs, None, width, offset)
            }
            ast::Expr_::ExprAssignOp(ref op, ref lhs, ref rhs) => {
                rewrite_assignment(context, lhs, rhs, Some(op), width, offset)
            }
            // FIXME #184 Note that this formatting is broken due to a bad span
            // from the parser.
            // `continue`
            ast::Expr_::ExprAgain(ref opt_ident) => {
                let id_str = match *opt_ident {
                    Some(ident) => format!(" {}", ident),
                    None => String::new(),
                };
                Some(format!("continue{}", id_str))
            }
            ast::Expr_::ExprBreak(ref opt_ident) => {
                let id_str = match *opt_ident {
                    Some(ident) => format!(" {}", ident),
                    None => String::new(),
                };
                Some(format!("break{}", id_str))
            }
            ast::Expr_::ExprClosure(capture, ref fn_decl, ref body) => {
                rewrite_closure(capture, fn_decl, body, self.span, context, width, offset)
            }
            _ => {
                // We do not format these expressions yet, but they should still
                // satisfy our width restrictions.
                let snippet = context.codemap.span_to_snippet(self.span).unwrap();

                {
                    let mut lines = snippet.lines();

                    // The caller of this function has already placed `offset`
                    // characters on the first line.
                    let first_line_max_len = try_opt!(context.config.max_width.checked_sub(offset));
                    if lines.next().unwrap().len() > first_line_max_len {
                        return None;
                    }

                    // The other lines must fit within the maximum width.
                    if lines.find(|line| line.len() > context.config.max_width).is_some() {
                        return None;
                    }

                    // `width` is the maximum length of the last line, excluding
                    // indentation.
                    // A special check for the last line, since the caller may
                    // place trailing characters on this line.
                    if snippet.lines().rev().next().unwrap().len() > offset + width {
                        return None;
                    }
                }

                Some(snippet)
            }
        }
    }
}

// This functions is pretty messy because of the wrapping and unwrapping of
// expressions into and from blocks. See rust issue #27872.
fn rewrite_closure(capture: ast::CaptureClause,
                   fn_decl: &ast::FnDecl,
                   body: &ast::Block,
                   span: Span,
                   context: &RewriteContext,
                   width: usize,
                   offset: usize)
                   -> Option<String> {
    let mover = if capture == ast::CaptureClause::CaptureByValue {
        "move "
    } else {
        ""
    };
    let offset = offset + mover.len();

    // 4 = "|| {".len(), which is overconservative when the closure consists of
    // a single expression.
    let argument_budget = try_opt!(width.checked_sub(4 + mover.len()));
    // 1 = |
    let argument_offset = offset + 1;

    let arg_items = itemize_list(context.codemap,
                                 fn_decl.inputs.iter(),
                                 "|",
                                 |arg| span_lo_for_arg(arg),
                                 |arg| span_hi_for_arg(arg),
                                 |arg| rewrite_fn_input(arg),
                                 span_after(span, "|", context.codemap),
                                 body.span.lo);

    let fmt = ListFormatting::for_fn(argument_budget, argument_offset);
    let prefix = format!("{}|{}|", mover, write_list(&arg_items.collect::<Vec<_>>(), &fmt));
    let block_indent = closure_block_indent(context, offset);

    // Try to format closure body as a single line expression without braces.
    if body.stmts.is_empty() {
        let expr = body.expr.as_ref().unwrap();
        // All closure bodies are blocks in the eyes of the AST, but we may not
        // want to unwrap them when they only contain a single expression.
        let inner_expr = match expr.node {
            ast::Expr_::ExprBlock(ref inner) if inner.stmts.is_empty() && inner.expr.is_some() => {
                inner.expr.as_ref().unwrap()
            }
            _ => expr,
        };

        // 1 = the separating space between arguments and the body.
        let extra_offset = extra_offset(&prefix, offset) + 1;
        let rewrite = inner_expr.rewrite(context, width - extra_offset, offset + extra_offset);

        // Checks if rewrite succeeded and fits on a single line.
        let accept_rewrite = rewrite.as_ref().map(|result| !result.contains('\n')).unwrap_or(false);

        if accept_rewrite {
            return Some(format!("{} {}", prefix, rewrite.unwrap()));
        }
    }

    // We couldn't format the closure body as a single line expression; fall
    // back to block formatting.
    let inner_context = &RewriteContext { block_indent: block_indent, ..*context };
    let body_rewrite = if let ast::Expr_::ExprBlock(ref inner) = body.expr.as_ref().unwrap().node {
        inner.rewrite(inner_context, 0, 0)
    } else {
        body.rewrite(inner_context, 0, 0)
    };

    Some(format!("{} {}", prefix, try_opt!(body_rewrite)))
}

impl Rewrite for ast::Block {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        let user_str = context.codemap.span_to_snippet(self.span).unwrap();
        if user_str == "{}" && width >= 2 {
            return Some(user_str);
        }

        let mut visitor = FmtVisitor::from_codemap(context.codemap, context.config);
        visitor.block_indent = context.block_indent;

        let prefix = match self.rules {
            ast::BlockCheckMode::PushUnsafeBlock(..) |
            ast::BlockCheckMode::UnsafeBlock(..) => {
                let snippet = try_opt!(context.codemap.span_to_snippet(self.span).ok());
                let open_pos = try_opt!(snippet.find_uncommented("{"));
                visitor.last_pos = self.span.lo + BytePos(open_pos as u32);

                // Extract comment between unsafe and block start.
                let trimmed = &snippet[6..open_pos].trim();

                if !trimmed.is_empty() {
                    // 9 = "unsafe  {".len(), 7 = "unsafe ".len()
                    format!("unsafe {} ", rewrite_comment(trimmed, true, width - 9, offset + 7))
                } else {
                    "unsafe ".to_owned()
                }
            }
            ast::BlockCheckMode::PopUnsafeBlock(..) |
            ast::BlockCheckMode::DefaultBlock => {
                visitor.last_pos = self.span.lo;

                String::new()
            }
        };

        visitor.visit_block(self);

        // Push text between last block item and end of block
        let snippet = visitor.snippet(mk_sp(visitor.last_pos, self.span.hi));
        visitor.buffer.push_str(&snippet);

        Some(format!("{}{}", prefix, visitor.buffer))
    }
}

// FIXME(#18): implement pattern formatting
impl Rewrite for ast::Pat {
    fn rewrite(&self, context: &RewriteContext, _: usize, _: usize) -> Option<String> {
        context.codemap.span_to_snippet(self.span).ok()
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
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        let label_string = rewrite_label(self.label);
        // 2 = " {".len()
        let inner_width = width - self.keyword.len() - 2 - label_string.len();
        let inner_offset = offset + self.keyword.len() + label_string.len();

        let pat_expr_string = match self.cond {
            Some(cond) => try_opt!(rewrite_pat_expr(context,
                                                    self.pat,
                                                    cond,
                                                    self.matcher,
                                                    self.connector,
                                                    inner_width,
                                                    inner_offset)),
            None => String::new(),
        };

        // FIXME: this drops any comment between "loop" and the block.
        self.block.rewrite(context, width, offset).map(|result| {
            format!("{}{}{} {}", label_string, self.keyword, pat_expr_string, result)
        })
    }
}

fn rewrite_label(label: Option<ast::Ident>) -> String {
    match label {
        Some(ident) => format!("{}: ", ident),
        None => "".to_owned(),
    }
}

// FIXME: this doesn't play well with line breaks
fn rewrite_range(context: &RewriteContext,
                 left: Option<&ast::Expr>,
                 right: Option<&ast::Expr>,
                 width: usize,
                 offset: usize)
                 -> Option<String> {
    let left_string = match left {
        Some(expr) => try_opt!(expr.rewrite(context, width - 2, offset)),
        None => String::new(),
    };

    let right_string = match right {
        Some(expr) => {
            // 2 = ..
            let max_width = (width - 2).checked_sub(left_string.len()).unwrap_or(0);
            try_opt!(expr.rewrite(context, max_width, offset + 2 + left_string.len()))
        }
        None => String::new(),
    };

    Some(format!("{}..{}", left_string, right_string))
}

// Rewrites if-else blocks. If let Some(_) = pat, the expression is
// treated as an if-let-else expression.
fn rewrite_if_else(context: &RewriteContext,
                   cond: &ast::Expr,
                   if_block: &ast::Block,
                   else_block_opt: Option<&ast::Expr>,
                   pat: Option<&ast::Pat>,
                   width: usize,
                   offset: usize,
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
        let if_str = try_opt!(if_expr.rewrite(context, new_width, 0));

        let new_width = try_opt!(new_width.checked_sub(if_str.len()));
        let else_expr = else_node.expr.as_ref().unwrap();
        let else_str = try_opt!(else_expr.rewrite(context, new_width, 0));

        // FIXME: this check shouldn't be necessary. Rewrites should either fail
        // or wrap to a newline when the object does not fit the width.
        let fits_line = fixed_cost + pat_expr_str.len() + if_str.len() + else_str.len() <= width;

        if fits_line && !if_str.contains('\n') && !else_str.contains('\n') {
            return Some(format!("if {} {{ {} }} else {{ {} }}", pat_expr_str, if_str, else_str));
        }
    }

    None
}

// Checks that a block contains no statements, an expression and no comments.
fn is_simple_block(block: &ast::Block, codemap: &CodeMap) -> bool {
    if !block.stmts.is_empty() || block.expr.is_none() {
        return false;
    }

    let snippet = codemap.span_to_snippet(block.span).unwrap();

    !contains_comment(&snippet)
}

fn rewrite_match(context: &RewriteContext,
                 cond: &ast::Expr,
                 arms: &[ast::Arm],
                 width: usize,
                 offset: usize)
                 -> Option<String> {
    if arms.is_empty() {
        return None;
    }

    // `match `cond` {`
    let cond_str = try_opt!(cond.rewrite(context, width - 8, offset + 6));
    let mut result = format!("match {} {{", cond_str);

    let block_indent = context.block_indent;
    let nested_context = context.nested_context();
    let arm_indent_str = make_indent(nested_context.block_indent);

    let open_brace_pos = span_after(mk_sp(cond.span.hi, arm_start_pos(&arms[0])),
                                    "{",
                                    context.codemap);

    for (i, arm) in arms.iter().enumerate() {
        // Make sure we get the stuff between arms.
        let missed_str = if i == 0 {
            context.codemap.span_to_snippet(mk_sp(open_brace_pos + BytePos(1),
                                                  arm_start_pos(arm))).unwrap()
        } else {
            context.codemap.span_to_snippet(mk_sp(arm_end_pos(&arms[i-1]),
                                                  arm_start_pos(arm))).unwrap()
        };
        let missed_str = match missed_str.find_uncommented(",") {
            Some(n) => &missed_str[n+1..],
            None => &missed_str[..],
        };
        // first = first non-whitespace byte index.
        let first = missed_str.find(|c: char| !c.is_whitespace()).unwrap_or(missed_str.len());
        if missed_str[..first].chars().filter(|c| c == &'\n').count() >= 2 {
            // There were multiple line breaks which got trimmed to nothing
            // that means there should be some vertical white space. Lets
            // replace that with just one blank line.
            result.push('\n');
        }
        let missed_str = missed_str.trim();
        if !missed_str.is_empty() {
            result.push('\n');
            result.push_str(&arm_indent_str);
            result.push_str(missed_str);
        }
        result.push('\n');
        result.push_str(&arm_indent_str);

        let arm_str = arm.rewrite(&nested_context,
                                  context.config.max_width -
                                      nested_context.block_indent,
                                  nested_context.block_indent);
        if let Some(ref arm_str) = arm_str {
            result.push_str(arm_str);
        } else {
            // We couldn't format the arm, just reproduce the source.
            let snippet = context.codemap.span_to_snippet(mk_sp(arm_start_pos(arm),
                                                                arm_end_pos(arm))).unwrap();
            result.push_str(&snippet);
        }
    }

    // We'll miss any comments etc. between the last arm and the end of the
    // match expression, but meh.

    result.push('\n');
    result.push_str(&make_indent(block_indent));
    result.push('}');
    Some(result)
}

fn arm_start_pos(arm: &ast::Arm) -> BytePos {
    let &ast::Arm { ref attrs, ref pats, .. } = arm;
    if !attrs.is_empty() {
        return attrs[0].span.lo
    }

    pats[0].span.lo
}

fn arm_end_pos(arm: &ast::Arm) -> BytePos {
    arm.body.span.hi
}

// Match arms.
impl Rewrite for ast::Arm {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        let &ast::Arm { ref attrs, ref pats, ref guard, ref body } = self;
        let indent_str = make_indent(offset);

        // FIXME this is all a bit grotty, would be nice to abstract out the
        // treatment of attributes.
        let attr_str = if !attrs.is_empty() {
            // We only use this visitor for the attributes, should we use it for
            // more?
            let mut attr_visitor = FmtVisitor::from_codemap(context.codemap, context.config);
            attr_visitor.block_indent = context.block_indent;
            attr_visitor.last_pos = attrs[0].span.lo;
            if attr_visitor.visit_attrs(attrs) {
                // Attributes included a skip instruction.
                let snippet = context.codemap.span_to_snippet(mk_sp(attrs[0].span.lo,
                                                                    body.span.hi)).unwrap();
                return Some(snippet);
            }
            attr_visitor.format_missing(pats[0].span.lo);
            attr_visitor.buffer.to_string()
        } else {
            String::new()
        };

        // Patterns
        let pat_strs = try_opt!(pats.iter().map(|p| p.rewrite(context,
                                                     // 5 = ` => {`
                                                     width - 5,
                                                     offset + context.config.tab_spaces))
                                           .collect::<Option<Vec<_>>>());

        let mut total_width = pat_strs.iter().fold(0, |a, p| a + p.len());
        // Add ` | `.len().
        total_width += (pat_strs.len() - 1) * 3;

        let mut vertical = total_width > width - 5 || pat_strs.iter().any(|p| p.contains('\n'));
        if !vertical {
            // If the patterns were previously stacked, keep them stacked.
            // FIXME should be an option.
            let pat_span = mk_sp(pats[0].span.lo, pats[pats.len() - 1].span.hi);
            let pat_str = context.codemap.span_to_snippet(pat_span).unwrap();
            vertical = pat_str.find('\n').is_some();
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
        if pats_str.find('\n').is_none() {
            line_start += offset;
        }

        let comma = if let ast::ExprBlock(_) = body.node {
            ""
        } else {
            ","
        };
        let nested_indent = context.block_indent + context.config.tab_spaces;

        // Let's try and get the arm body on the same line as the condition.
        // 4 = ` => `.len()
        if context.config.max_width > line_start + comma.len() + 4 {
            let budget = context.config.max_width - line_start - comma.len() - 4;
            if let Some(ref body_str) = body.rewrite(context,
                                                     budget,
                                                     nested_indent) {
                if first_line_width(body_str) <= budget {
                    return Some(format!("{}{} => {}{}",
                                        attr_str.trim_left(),
                                        pats_str,
                                        body_str,
                                        comma));
                }
            }
        }

        // We have to push the body to the next line.
        if comma.is_empty() {
            // We're trying to fit a block in, but it still failed, give up.
            return None;
        }

        let body_str = try_opt!(body.rewrite(context,
                                             width - context.config.tab_spaces,
                                             nested_indent));
        Some(format!("{}{} =>\n{}{},",
                     attr_str.trim_left(),
                     pats_str,
                     make_indent(offset + context.config.tab_spaces),
                     body_str))
    }
}

// The `if ...` guard on a match arm.
fn rewrite_guard(context: &RewriteContext,
                 guard: &Option<ptr::P<ast::Expr>>,
                 width: usize,
                 offset: usize,
                 // The amount of space used up on this line for the pattern in
                 // the arm (excludes offset).
                 pattern_width: usize)
                 -> Option<String> {
    if let &Some(ref guard) = guard {
        // First try to fit the guard string on the same line as the pattern.
        // 4 = ` if `, 5 = ` => {`
        let overhead = pattern_width + 4 + 5;
        if overhead < width {
            let cond_str = guard.rewrite(context,
                                         width - overhead,
                                         offset + pattern_width + 4);
            if let Some(cond_str) = cond_str {
                return Some(format!(" if {}", cond_str));
            }
        }

        // Not enough space to put the guard after the pattern, try a newline.
        let overhead = context.config.tab_spaces + 4 + 5;
        if overhead < width {
            let cond_str = guard.rewrite(context,
                                         width - overhead,
                                         offset + context.config.tab_spaces);
            if let Some(cond_str) = cond_str {
                return Some(format!("\n{}if {}",
                                    make_indent(offset + context.config.tab_spaces),
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
                    offset: usize)
                    -> Option<String> {
    let pat_offset = offset + matcher.len();
    let mut result = match pat {
        Some(pat) => {
            let pat_string = try_opt!(pat.rewrite(context,
                                                  width - connector.len() - matcher.len(),
                                                  pat_offset));
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
    result.push_str(&make_indent(pat_offset));

    let expr_rewrite = expr.rewrite(context, context.config.max_width - pat_offset, pat_offset);
    result.push_str(&&try_opt!(expr_rewrite));

    Some(result)
}

fn rewrite_string_lit(context: &RewriteContext,
                      s: &str,
                      span: Span,
                      width: usize,
                      offset: usize)
                      -> Option<String> {
    // Check if there is anything to fix: we always try to fixup multi-line
    // strings, or if the string is too long for the line.
    let l_loc = context.codemap.lookup_char_pos(span.lo);
    let r_loc = context.codemap.lookup_char_pos(span.hi);
    if l_loc.line == r_loc.line && r_loc.col.to_usize() <= context.config.max_width {
        return context.codemap.span_to_snippet(span).ok();
    }
    let fmt = StringFormat {
        opener: "\"",
        closer: "\"",
        line_start: " ",
        line_end: "\\",
        width: width,
        offset: offset,
        trim_end: false,
    };

    Some(rewrite_string(&s.escape_default(), &fmt))
}

fn rewrite_call(context: &RewriteContext,
                callee: &ast::Expr,
                args: &[ptr::P<ast::Expr>],
                span: Span,
                width: usize,
                offset: usize)
                -> Option<String> {
    debug!("rewrite_call, width: {}, offset: {}", width, offset);

    // FIXME using byte lens instead of char lens (and probably all over the place too)
    // 2 is for parens
    let max_callee_width = try_opt!(width.checked_sub(2));
    let callee_str = try_opt!(callee.rewrite(context, max_callee_width, offset));
    debug!("rewrite_call, callee_str: `{}`", callee_str);

    if args.is_empty() {
        return Some(format!("{}()", callee_str));
    }

    let extra_offset = extra_offset(&callee_str, offset);
    // 2 is for parens.
    let remaining_width = try_opt!(width.checked_sub(extra_offset + 2));
    let offset = offset + extra_offset + 1;
    let block_indent = expr_block_indent(context, offset);
    let inner_context = &RewriteContext { block_indent: block_indent, ..*context };

    let items = itemize_list(context.codemap,
                             args.iter(),
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             // Take old span when rewrite fails.
                             |item| {
                                 item.rewrite(inner_context, remaining_width, offset)
                                     .unwrap_or(context.codemap.span_to_snippet(item.span).unwrap())
                             },
                             callee.span.hi + BytePos(1),
                             span.hi);

    let fmt = ListFormatting::for_fn(remaining_width, offset);

    Some(format!("{}({})", callee_str, write_list(&items.collect::<Vec<_>>(), &fmt)))
}

macro_rules! block_indent_helper {
    ($name:ident, $option:ident) => (
        fn $name(context: &RewriteContext, offset: usize) -> usize {
            match context.config.$option {
                BlockIndentStyle::Inherit => context.block_indent,
                BlockIndentStyle::Tabbed => context.block_indent + context.config.tab_spaces,
                BlockIndentStyle::Visual => offset,
            }
        }
    );
}

block_indent_helper!(expr_block_indent, expr_indent_style);
block_indent_helper!(closure_block_indent, closure_indent_style);

fn rewrite_paren(context: &RewriteContext,
                 subexpr: &ast::Expr,
                 width: usize,
                 offset: usize)
                 -> Option<String> {
    debug!("rewrite_paren, width: {}, offset: {}", width, offset);
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
                          offset: usize)
                          -> Option<String> {
    debug!("rewrite_struct_lit: width {}, offset {}", width, offset);
    assert!(!fields.is_empty() || base.is_some());

    enum StructLitField<'a> {
        Regular(&'a ast::Field),
        Base(&'a ast::Expr),
    }

    // 2 = " {".len()
    let path_str = try_opt!(path.rewrite(context, width - 2, offset));

    // Foo { a: Foo } - indent is +3, width is -5.
    let h_budget = try_opt!(width.checked_sub(path_str.len() + 5));
    let (indent, v_budget) = match context.config.struct_lit_style {
        StructLitStyle::VisualIndent => {
            (offset + path_str.len() + 3, h_budget)
        }
        StructLitStyle::BlockIndent => {
            // If we are all on one line, then we'll ignore the indent, and we
            // have a smaller budget.
            let indent = context.block_indent + context.config.tab_spaces;
            let v_budget = context.config.max_width.checked_sub(indent).unwrap_or(0);
            (indent, v_budget)
        }
    };

    let field_iter = fields.into_iter().map(StructLitField::Regular)
                           .chain(base.into_iter().map(StructLitField::Base));

    let inner_context = &RewriteContext { block_indent: indent, ..*context };

    let items = itemize_list(context.codemap,
                             field_iter,
                             "}",
                             |item| {
                                 match *item {
                                     StructLitField::Regular(ref field) => field.span.lo,
                                     // 2 = ..
                                     StructLitField::Base(ref expr) => expr.span.lo - BytePos(2),
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
                                         rewrite_field(inner_context, &field, h_budget, indent)
                                            .unwrap_or(context.codemap.span_to_snippet(field.span)
                                                                      .unwrap())
                                     }
                                     StructLitField::Base(ref expr) => {
                                         // 2 = ..
                                         expr.rewrite(inner_context, h_budget - 2, indent + 2)
                                             .map(|s| format!("..{}", s))
                                             .unwrap_or(context.codemap.span_to_snippet(expr.span)
                                                                       .unwrap())
                                     }
                                 }
                             },
                             span_after(span, "{", context.codemap),
                             span.hi);

    let fmt = ListFormatting {
        tactic: ListTactic::HorizontalVertical,
        separator: ",",
        trailing_separator: if base.is_some() {
            SeparatorTactic::Never
        } else {
            context.config.struct_lit_trailing_comma
        },
        indent: indent,
        h_width: h_budget,
        v_width: v_budget,
        ends_with_newline: false,
    };
    let fields_str = write_list(&items.collect::<Vec<_>>(), &fmt);

    match context.config.struct_lit_style {
        StructLitStyle::BlockIndent if fields_str.contains('\n') => {
            let inner_indent = make_indent(context.block_indent + context.config.tab_spaces);
            let outer_indent = make_indent(context.block_indent);
            Some(format!("{} {{\n{}{}\n{}}}", path_str, inner_indent, fields_str, outer_indent))
        }
        _ => Some(format!("{} {{ {} }}", path_str, fields_str)),
    }

    // FIXME if context.config.struct_lit_style == VisualIndent, but we run out
    // of space, we should fall back to BlockIndent.
}

fn rewrite_field(context: &RewriteContext,
                 field: &ast::Field,
                 width: usize,
                 offset: usize)
                 -> Option<String> {
    let name = &field.ident.node.to_string();
    let overhead = name.len() + 2;
    let expr = field.expr.rewrite(context, width - overhead, offset + overhead);
    expr.map(|s| format!("{}: {}", name, s))
}

fn rewrite_tuple_lit(context: &RewriteContext,
                     items: &[ptr::P<ast::Expr>],
                     span: Span,
                     width: usize,
                     offset: usize)
                     -> Option<String> {
    debug!("rewrite_tuple_lit: width: {}, offset: {}", width, offset);
    let indent = offset + 1;
    // In case of length 1, need a trailing comma
    if items.len() == 1 {
        // 3 = "(" + ",)"
        return items[0].rewrite(context, width - 3, indent).map(|s| format!("({},)", s));
    }

    let items = itemize_list(context.codemap,
                             items.iter(),
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             |item| {
                                 let inner_width = context.config.max_width - indent - 1;
                                 item.rewrite(context, inner_width, indent)
                                     .unwrap_or(context.codemap.span_to_snippet(item.span).unwrap())
                             },
                             span.lo + BytePos(1), // Remove parens
                             span.hi - BytePos(1));

    let fmt = ListFormatting::for_fn(width - 2, indent);

    Some(format!("({})", write_list(&items.collect::<Vec<_>>(), &fmt)))
}

fn rewrite_binary_op(context: &RewriteContext,
                     op: &ast::BinOp,
                     lhs: &ast::Expr,
                     rhs: &ast::Expr,
                     width: usize,
                     offset: usize)
                     -> Option<String> {
    // FIXME: format comments between operands and operator

    let operator_str = context.codemap.span_to_snippet(op.span).unwrap();

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
        if let Some(mut result) = lhs.rewrite(context,
                                              width - 1 - operator_str.len(),
                                              offset) {

            result.push(' ');
            result.push_str(&operator_str);
            result.push(' ');

            let remaining_width = width.checked_sub(last_line_width(&result)).unwrap_or(0);

            if rhs_result.len() <= remaining_width {
                result.push_str(&rhs_result);
                return Some(result);
            }

            if let Some(rhs_result) = rhs.rewrite(context,
                                                  remaining_width,
                                                  offset + result.len()) {
                if rhs_result.len() <= remaining_width {
                    result.push_str(&rhs_result);
                    return Some(result);
                }
            }
        }
    }

    // We have to use multiple lines.

    // Re-evaluate the lhs because we have more space now:
    let budget = try_opt!(context.config.max_width.checked_sub(offset + 1 + operator_str.len()));
    Some(format!("{} {}\n{}{}",
                 try_opt!(lhs.rewrite(context, budget, offset)),
                 operator_str,
                 make_indent(offset),
                 rhs_result))
}

fn rewrite_unary_op(context: &RewriteContext,
                    op: &ast::UnOp,
                    expr: &ast::Expr,
                    width: usize,
                    offset: usize)
                    -> Option<String> {
    // For some reason, an UnOp is not spanned like BinOp!
    let operator_str = match *op {
        ast::UnOp::UnUniq => "box ",
        ast::UnOp::UnDeref => "*",
        ast::UnOp::UnNot => "!",
        ast::UnOp::UnNeg => "-",
    };

    let subexpr =
        try_opt!(expr.rewrite(context, try_opt!(width.checked_sub(operator_str.len())), offset));

    Some(format!("{}{}", operator_str, subexpr))
}

fn rewrite_assignment(context: &RewriteContext,
                      lhs: &ast::Expr,
                      rhs: &ast::Expr,
                      op: Option<&ast::BinOp>,
                      width: usize,
                      offset: usize)
                      -> Option<String> {
    let operator_str = match op {
        Some(op) => context.codemap.span_to_snippet(op.span).unwrap(),
        None => "=".to_owned(),
    };

    // 1 = space between lhs and operator.
    let max_width = try_opt!(width.checked_sub(operator_str.len() + 1));
    let lhs_str = format!("{} {}", try_opt!(lhs.rewrite(context, max_width, offset)), operator_str);

    rewrite_assign_rhs(&context, lhs_str, rhs, width, offset)
}

// The left hand side must contain everything up to, and including, the
// assignment operator.
pub fn rewrite_assign_rhs<S: Into<String>>(context: &RewriteContext,
                                           lhs: S,
                                           ex: &ast::Expr,
                                           width: usize,
                                           offset: usize)
                                           -> Option<String> {
    let mut result = lhs.into();

    // 1 = space between operator and rhs.
    let max_width = try_opt!(width.checked_sub(result.len() + 1));
    let rhs = ex.rewrite(&context, max_width, offset + result.len() + 1);

    match rhs {
        Some(new_str) => {
            result.push(' ');
            result.push_str(&new_str)
        }
        None => {
            // Expression did not fit on the same line as the identifier. Retry
            // on the next line.
            let new_offset = offset + context.config.tab_spaces;
            result.push_str(&format!("\n{}", make_indent(new_offset)));

            let max_width = try_opt!(context.config.max_width.checked_sub(new_offset + 1));
            let rhs = try_opt!(ex.rewrite(&context, max_width, new_offset));

            result.push_str(&rhs);
        }
    }

    Some(result)
}
