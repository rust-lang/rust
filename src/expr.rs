// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use visitor::FmtVisitor;
use utils::*;
use lists::{write_list, ListFormatting, SeparatorTactic, ListTactic};

use syntax::{ast, ptr};
use syntax::codemap::{Span, Pos};

use {MAX_WIDTH, MIN_STRING};

impl<'a> FmtVisitor<'a> {
    // TODO NEEDS TESTS
    fn rewrite_string_lit(&mut self, s: &str, span: Span, width: usize, offset: usize) -> String {
        // FIXME I bet this stomps unicode escapes in the source string

        // Check if there is anything to fix: we always try to fixup multi-line
        // strings, or if the string is too long for the line.
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let r_loc = self.codemap.lookup_char_pos(span.hi);
        if l_loc.line == r_loc.line && r_loc.col.to_usize() <= MAX_WIDTH {
            return self.snippet(span);
        }

        // TODO if lo.col > IDEAL - 10, start a new line (need cur indent for that)

        let s = s.escape_default();

        let offset = offset + 1;
        let indent = make_indent(offset);
        let indent = &indent;

        let max_chars = width - 1;

        let mut cur_start = 0;
        let mut result = String::new();
        result.push('"');
        loop {
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
                    while s.char_at(cur_end) == '\\' {
                        cur_end = prev_char(&s, cur_end);
                    }
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

        result
    }

    fn rewrite_call(&mut self,
                    callee: &ast::Expr,
                    args: &[ptr::P<ast::Expr>],
                    width: usize,
                    offset: usize)
        -> String
    {
        debug!("rewrite_call, width: {}, offset: {}", width, offset);

        // TODO using byte lens instead of char lens (and probably all over the place too)
        let callee_str = self.rewrite_expr(callee, width, offset);
        debug!("rewrite_call, callee_str: `{}`", callee_str);
        // 2 is for parens.
        let remaining_width = width - callee_str.len() - 2;
        let offset = callee_str.len() + 1 + offset;
        let arg_count = args.len();

        let args_str = if arg_count > 0 {
            let args: Vec<_> = args.iter().map(|e| (self.rewrite_expr(e,
                                                                      remaining_width,
                                                                      offset), String::new())).collect();
            // TODO move this into write_list
            let tactics = if args.iter().any(|&(ref s, _)| s.contains('\n')) {
                ListTactic::Vertical
            } else {
                ListTactic::HorizontalVertical
            };
            let fmt = ListFormatting {
                tactic: tactics,
                separator: ",",
                trailing_separator: SeparatorTactic::Never,
                indent: offset,
                h_width: remaining_width,
                v_width: remaining_width,
            };
            write_list(&args, &fmt)
        } else {
            String::new()
        };

        format!("{}({})", callee_str, args_str)
    }

    pub fn rewrite_expr(&mut self, expr: &ast::Expr, width: usize, offset: usize) -> String {
        match expr.node {
            ast::Expr_::ExprLit(ref l) => {
                match l.node {
                    ast::Lit_::LitStr(ref is, _) => {
                        let result = self.rewrite_string_lit(&is, l.span, width, offset);
                        debug!("string lit: `{}`", result);
                        return result;
                    }
                    _ => {}
                }
            }
            ast::Expr_::ExprCall(ref callee, ref args) => {
                return self.rewrite_call(callee, args, width, offset);
            }
            _ => {}
        }

        let result = self.snippet(expr.span);
        result
    }
}
