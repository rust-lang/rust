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
use utils::{span_after, make_indent};
use visitor::FmtVisitor;

use syntax::{ast, ptr};
use syntax::codemap::{Pos, Span, BytePos, mk_sp};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::visit::Visitor;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        match self.node {
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(ref is, _) => {
                        rewrite_string_lit(context, &is, l.span, width, offset)
                    }
                    _ => context.codemap.span_to_snippet(self.span).ok()
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
            ast::Expr_::ExprLoop(ref block, _) => {
                // FIXME: this drops any comment between "loop" and the block.
                // TODO: format label
                block.rewrite(context, width, offset).map(|result| {
                    format!("loop {}", result)
                })
            }
            _ => context.codemap.span_to_snippet(self.span).ok()
        }
    }
}

impl Rewrite for ast::Block {
    fn rewrite(&self, context: &RewriteContext, _: usize, _: usize) -> Option<String> {
        let mut visitor = FmtVisitor::from_codemap(context.codemap, context.config);
        visitor.last_pos = self.span.lo;
        visitor.block_indent = context.block_indent;

        visitor.visit_block(self);

        // Push text between last block item and end of block
        let snippet = visitor.snippet(mk_sp(visitor.last_pos, self.span.hi));
        visitor.changes.push_str_span(self.span, &snippet);

        // Stringify visitor
        let file_name = context.codemap.span_to_filename(self.span);
        let string_buffer = visitor.changes.get(&file_name);

        Some(string_buffer.to_string())
    }
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

    // TODO using byte lens instead of char lens (and probably all over the place too)
    let callee_str = try_opt!(callee.rewrite(context, width, offset));
    debug!("rewrite_call, callee_str: `{}`", callee_str);

    if args.len() == 0 {
        return Some(format!("{}()", callee_str));
    }

    // 2 is for parens.
    let remaining_width = try_opt!(width.checked_sub(callee_str.len() + 2));
    let offset = callee_str.len() + 1 + offset;

    let items = itemize_list(context.codemap,
                             Vec::new(),
                             args.iter(),
                             ",",
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             // Take old span when rewrite fails.
                             |item| item.rewrite(context, remaining_width, offset)
                                        .unwrap_or(context.codemap.span_to_snippet(item.span)
                                                                  .unwrap()),
                             callee.span.hi + BytePos(1),
                             span.hi);

    let fmt = ListFormatting {
        tactic: ListTactic::HorizontalVertical,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: offset,
        h_width: remaining_width,
        v_width: remaining_width,
        ends_with_newline: true,
    };

    Some(format!("{}({})", callee_str, write_list(&items, &fmt)))
}

fn rewrite_paren(context: &RewriteContext,
                 subexpr: &ast::Expr,
                 width: usize,
                 offset: usize)
                 -> Option<String> {
    debug!("rewrite_paren, width: {}, offset: {}", width, offset);
    // 1 is for opening paren, 2 is for opening+closing, we want to keep the closing
    // paren on the same line as the subexpr
    let subexpr_str = subexpr.rewrite(context, width-2, offset+1);
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
    assert!(fields.len() > 0 || base.is_some());

    enum StructLitField<'a> {
        Regular(&'a ast::Field),
        Base(&'a ast::Expr),
    }

    let path_str = pprust::path_to_string(path);
    let (indent, h_budget, v_budget) = match context.config.struct_lit_style {
        StructLitStyle::VisualIndent => {
            // Foo { a: Foo } - indent is +3, width is -5.
            let budget = width - (path_str.len() + 5);
            (offset + path_str.len() + 3, budget, budget)
        }
        StructLitStyle::BlockIndent => {
            // If we are all on one line, then we'll ignore the indent, and we
            // have a smaller budget.
            let indent = context.block_indent + context.config.tab_spaces;
            (indent, width - (path_str.len() + 5), width - indent)
        }
    };

    let field_iter = fields.into_iter().map(StructLitField::Regular)
                           .chain(base.into_iter().map(StructLitField::Base));

    let items = itemize_list(context.codemap,
                             Vec::new(),
                             field_iter,
                             ",",
                             "}",
                             |item| {
                                 match *item {
                                     StructLitField::Regular(ref field) => field.span.lo,
                                     // 2 = ..
                                     StructLitField::Base(ref expr) => expr.span.lo - BytePos(2)
                                 }
                             },
                             |item| {
                                 match *item {
                                     StructLitField::Regular(ref field) => field.span.hi,
                                     StructLitField::Base(ref expr) => expr.span.hi
                                 }
                             },
                             |item| {
                                 match *item {
                                     StructLitField::Regular(ref field) => {
                                         rewrite_field(context, &field, h_budget, indent)
                                            .unwrap_or(context.codemap.span_to_snippet(field.span)
                                                                      .unwrap())
                                     },
                                     StructLitField::Base(ref expr) => {
                                         // 2 = ..
                                         expr.rewrite(context, h_budget - 2, indent + 2)
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
        ends_with_newline: true,
    };
    let fields_str = write_list(&items, &fmt);

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
    let name = &token::get_ident(field.ident.node);
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
                             Vec::new(),
                             items.into_iter(),
                             ",",
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             |item| item.rewrite(context,
                                                 context.config.max_width - indent - 1,
                                                 indent)
                                        .unwrap_or(context.codemap.span_to_snippet(item.span)
                                                                  .unwrap()),
                             span.lo + BytePos(1), // Remove parens
                             span.hi - BytePos(1));

    let fmt = ListFormatting {
        tactic: ListTactic::HorizontalVertical,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: indent,
        h_width: width - 2,
        v_width: width - 2,
        ends_with_newline: true,
    };

    Some(format!("({})", write_list(&items, &fmt)))
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

    // 1 = space between lhs expr and operator
    let mut result =
        try_opt!(lhs.rewrite(context,
                             context.config.max_width - offset - 1 - operator_str.len(),
                             offset));

    result.push(' ');
    result.push_str(&operator_str);

    // 1 = space between operator and rhs
    let used_width = result.len() + operator_str.len() + 1;
    let remaining_width = match result.rfind('\n') {
        Some(idx) => (offset + width + idx).checked_sub(used_width).unwrap_or(0),
        None => width.checked_sub(used_width).unwrap_or(0)
    };

    // Get "full width" rhs and see if it fits on the current line. This
    // usually works fairly well since it tends to place operands of
    // operations with high precendence close together.
    let rhs_result = try_opt!(rhs.rewrite(context, width, offset));

    // Second condition is needed in case of line break not caused by a
    // shortage of space, but by end-of-line comments, for example.
    if rhs_result.len() > remaining_width || rhs_result.contains('\n') {
        result.push('\n');
        result.push_str(&make_indent(offset));
    } else {
        result.push(' ');
    };

    result.push_str(&rhs_result);
    Some(result)
}

fn rewrite_unary_op(context: &RewriteContext,
                    op: &ast::UnOp,
                    expr: &ast::Expr,
                    width: usize,
                    offset: usize)
                    -> Option<String> {
    // For some reason, an UnOp is not spanned like BinOp!
    let operator_str = match *op {
        ast::UnOp::UnUniq => "&",
        ast::UnOp::UnDeref => "*",
        ast::UnOp::UnNot => "!",
        ast::UnOp::UnNeg => "-"
    };

    Some(format!("{}{}", operator_str, try_opt!(expr.rewrite(context, width - 1, offset))))
}
