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
use lists::{write_list, itemize_list, ListItem, ListFormatting, SeparatorTactic, ListTactic};
use string::{StringFormat, rewrite_string};

use syntax::{ast, ptr};
use syntax::codemap::{Pos, Span, BytePos};
use syntax::parse::token;
use syntax::print::pprust;

impl Rewrite for ast::Expr {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: usize) -> Option<String> {
        match self.node {
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(ref is, _) => {
                        let result = rewrite_string_lit(context, &is, l.span, width, offset);
                        debug!("string lit: `{:?}`", result);
                        return result;
                    }
                    _ => {}
                }
            }
            ast::Expr_::ExprCall(ref callee, ref args) => {
                return rewrite_call(context, callee, args, self.span, width, offset);
            }
            ast::Expr_::ExprParen(ref subexpr) => {
                return rewrite_paren(context, subexpr, width, offset);
            }
            ast::Expr_::ExprStruct(ref path, ref fields, ref base) => {
                return rewrite_struct_lit(context, path,
                                               fields,
                                               base.as_ref().map(|e| &**e),
                                               width,
                                               offset);
            }
            ast::Expr_::ExprTup(ref items) => {
                return rewrite_tuple_lit(context, items, self.span, width, offset);
            }
            _ => {}
        }

        context.codemap.span_to_snippet(self.span).ok()
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
        trim_end: false
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
    let remaining_width = width - callee_str.len() - 2;
    let offset = callee_str.len() + 1 + offset;

    let items = itemize_list(context.codemap,
                             Vec::new(),
                             args.iter(),
                             ",",
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             |item| item.rewrite(context, remaining_width, offset)
                                        .unwrap(), // FIXME: don't unwrap, take span literal
                             callee.span.hi + BytePos(1),
                             span.hi);

    let fmt = ListFormatting {
        tactic: ListTactic::HorizontalVertical,
        separator: ",",
        trailing_separator: SeparatorTactic::Never,
        indent: offset,
        h_width: remaining_width,
        v_width: remaining_width,
        is_expression: true,
    };

    Some(format!("{}({})", callee_str, write_list(&items, &fmt)))
}

fn rewrite_paren(context: &RewriteContext, subexpr: &ast::Expr, width: usize, offset: usize) -> Option<String> {
    debug!("rewrite_paren, width: {}, offset: {}", width, offset);
    // 1 is for opening paren, 2 is for opening+closing, we want to keep the closing
    // paren on the same line as the subexpr
    let subexpr_str = subexpr.rewrite(context, width-2, offset+1);
    debug!("rewrite_paren, subexpr_str: `{:?}`", subexpr_str);
    subexpr_str.map(|s| format!("({})", s))
}

fn rewrite_struct_lit(context: &RewriteContext,
                      path: &ast::Path,
                      fields: &[ast::Field],
                      base: Option<&ast::Expr>,
                      width: usize,
                      offset: usize)
        -> Option<String>
{
    debug!("rewrite_struct_lit: width {}, offset {}", width, offset);
    assert!(fields.len() > 0 || base.is_some());

    let path_str = pprust::path_to_string(path);
    // Foo { a: Foo } - indent is +3, width is -5.
    let indent = offset + path_str.len() + 3;
    let budget = width - (path_str.len() + 5);

    let field_strs: Vec<_> =
        try_opt!(fields.iter()
                       .map(|field| rewrite_field(context, field, budget, indent))
                       .chain(base.iter()
                                  .map(|expr| expr.rewrite(context,
                                                           // 2 = ".."
                                                           budget - 2,
                                                           indent + 2)
                                                  .map(|s| format!("..{}", s))))
                       .collect());

    // FIXME comments
    let field_strs: Vec<_> = field_strs.into_iter().map(ListItem::from_str).collect();
    let fmt = ListFormatting {
        tactic: ListTactic::HorizontalVertical,
        separator: ",",
        trailing_separator: if base.is_some() {
            SeparatorTactic::Never
        } else {
            context.config.struct_lit_trailing_comma
        },
        indent: indent,
        h_width: budget,
        v_width: budget,
        is_expression: true,
    };
    let fields_str = write_list(&field_strs, &fmt);
    Some(format!("{} {{ {} }}", path_str, fields_str))

    // FIXME if the usual multi-line layout is too wide, we should fall back to
    // Foo {
    //     a: ...,
    // }
}

fn rewrite_field(context: &RewriteContext, field: &ast::Field, width: usize, offset: usize) -> Option<String> {
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
    let indent = offset + 1;

    let items = itemize_list(context.codemap,
                             Vec::new(),
                             items.into_iter(),
                             ",",
                             ")",
                             |item| item.span.lo,
                             |item| item.span.hi,
                             |item| item.rewrite(context,
                                                 context.config.max_width - indent - 2,
                                                 indent)
                                        .unwrap(), // FIXME: don't unwrap, take span literal
                             span.lo + BytePos(1), // Remove parens
                             span.hi - BytePos(1));

    // In case of length 1, need a trailing comma
    let trailing_separator_tactic = if items.len() == 1 {
        SeparatorTactic::Always
    } else {
        SeparatorTactic::Never
    };

    let fmt = ListFormatting {
        tactic: ListTactic::HorizontalVertical,
        separator: ",",
        trailing_separator: trailing_separator_tactic,
        indent: indent,
        h_width: width - 2,
        v_width: width - 2,
        is_expression: true,
    };

    Some(format!("({})", write_list(&items, &fmt)))
}
