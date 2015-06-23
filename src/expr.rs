// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use utils::*;
use lists::{write_list, ListFormatting, SeparatorTactic, ListTactic};
use rewrite::{Rewrite, RewriteContext};

use syntax::{ast, ptr};
use syntax::codemap::{Pos, Span};
use syntax::parse::token;
use syntax::print::pprust;

use MIN_STRING;

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
                return rewrite_call(context, callee, args, width, offset);
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
                return rewrite_tuple_lit(context, items, width, offset);
            }
            _ => {}
        }

        context.codemap.span_to_snippet(self.span).ok()
    }
}

fn rewrite_string_lit(context: &RewriteContext, s: &str, span: Span, width: usize, offset: usize) -> Option<String> {
    // FIXME I bet this stomps unicode escapes in the source string

    // Check if there is anything to fix: we always try to fixup multi-line
    // strings, or if the string is too long for the line.
    let l_loc = context.codemap.lookup_char_pos(span.lo);
    let r_loc = context.codemap.lookup_char_pos(span.hi);
    if l_loc.line == r_loc.line && r_loc.col.to_usize() <= context.config.max_width {
        return context.codemap.span_to_snippet(span).ok();
    }

    // TODO if lo.col > IDEAL - 10, start a new line (need cur indent for that)

    let s = s.escape_default();

    let offset = offset + 1;
    let indent = make_indent(offset);
    let indent = &indent;

    let mut cur_start = 0;
    let mut result = String::with_capacity(round_up_to_power_of_two(s.len()));
    result.push('"');
    loop {
        let max_chars = if cur_start == 0 {
            // First line.
            width - 2 // 2 = " + \
        } else {
            context.config.max_width - offset - 1 // 1 = either \ or ;
        };

        let mut cur_end = cur_start + max_chars;

        if cur_end >= s.len() {
            result.push_str(&s[cur_start..]);
            break;
        }

        // Make sure we're on a char boundary.
        cur_end = next_char(&s, cur_end);

        // Push cur_end left until we reach whitespace
        while !s.char_at(cur_end-1).is_whitespace() {
            cur_end = prev_char(&s, cur_end);

            if cur_end - cur_start < MIN_STRING {
                // We can't break at whitespace, fall back to splitting
                // anywhere that doesn't break an escape sequence
                cur_end = next_char(&s, cur_start + max_chars);
                while s.char_at(prev_char(&s, cur_end)) == '\\' {
                    cur_end = prev_char(&s, cur_end);
                }
                break;
            }
        }
        // Make sure there is no whitespace to the right of the break.
        while cur_end < s.len() && s.char_at(cur_end).is_whitespace() {
            cur_end = next_char(&s, cur_end+1);
        }
        result.push_str(&s[cur_start..cur_end]);
        result.push_str("\\\n");
        result.push_str(indent);

        cur_start = cur_end;
    }
    result.push('"');

    Some(result)
}

fn rewrite_call(context: &RewriteContext,
                callee: &ast::Expr,
                args: &[ptr::P<ast::Expr>],
                width: usize,
                offset: usize)
        -> Option<String>
{
    debug!("rewrite_call, width: {}, offset: {}", width, offset);

    // TODO using byte lens instead of char lens (and probably all over the place too)
    let callee_str = try_opt!(callee.rewrite(context, width, offset));
    debug!("rewrite_call, callee_str: `{:?}`", callee_str);
    // 2 is for parens.
    let remaining_width = width - callee_str.len() - 2;
    let offset = callee_str.len() + 1 + offset;
    let arg_count = args.len();

    let args_str = if arg_count > 0 {
        let args_rewritten: Vec<_> =
            try_opt!(args.iter()
                         .map(|arg| arg.rewrite(context, remaining_width, offset)
                                       .map(|arg_str| (arg_str, String::new())))
                         .collect());
        let fmt = ListFormatting {
            tactic: ListTactic::HorizontalVertical,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: offset,
            h_width: remaining_width,
            v_width: remaining_width,
        };
        write_list(&args_rewritten, &fmt)
    } else {
        String::new()
    };

    Some(format!("{}({})", callee_str, args_str))
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
    let field_strs: Vec<_> = field_strs.into_iter().map(|s| (s, String::new())).collect();
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
                     width: usize,
                     offset: usize)
    -> Option<String> {
        // opening paren
        let indent = offset + 1;
        // In case of length 1, need a trailing comma
        if items.len() == 1 {
            return items[0].rewrite(context, width - 3, indent).map(|s| format!("({},)", s));
        }
        // Only last line has width-1 as budget, other may take max_width
        let item_strs: Vec<_> =
            try_opt!(items.iter()
                          .enumerate()
                          .map(|(i, item)| {
                              let rem_width = if i == items.len() - 1 {
                                  width - 2
                              } else {
                                  context.config.max_width - indent - 2
                              };
                              item.rewrite(context, rem_width, indent)
                          })
                          .collect());
        let tactics = if item_strs.iter().any(|s| s.contains('\n')) {
            ListTactic::Vertical
        } else {
            ListTactic::HorizontalVertical
        };
        // FIXME handle comments
        let item_strs: Vec<_> = item_strs.into_iter().map(|s| (s, String::new())).collect();
        let fmt = ListFormatting {
            tactic: tactics,
            separator: ",",
            trailing_separator: SeparatorTactic::Never,
            indent: indent,
            h_width: width - 2,
            v_width: width - 2,
        };
        let item_str = write_list(&item_strs, &fmt);
        Some(format!("({})", item_str))
    }
