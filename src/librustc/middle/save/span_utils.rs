// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session::Session;

use middle::save::generated_code;

use std::cell::Cell;

use syntax::ast;
use syntax::codemap::*;
use syntax::parse::lexer;
use syntax::parse::lexer::{Reader,StringReader};
use syntax::parse::token;
use syntax::parse::token::{is_keyword,keywords,is_ident,Token};

pub struct SpanUtils<'a> {
    pub sess: &'a Session,
    pub err_count: Cell<int>,
}

impl<'a> SpanUtils<'a> {
    // Standard string for extents/location.
    pub fn extent_str(&self, span: Span) -> String {
        let lo_loc = self.sess.codemap().lookup_char_pos(span.lo);
        let hi_loc = self.sess.codemap().lookup_char_pos(span.hi);
        let lo_pos = self.sess.codemap().lookup_byte_offset(span.lo).pos;
        let hi_pos = self.sess.codemap().lookup_byte_offset(span.hi).pos;

        format!("file_name,{},file_line,{},file_col,{},extent_start,{},\
                 file_line_end,{},file_col_end,{},extent_end,{}",
                lo_loc.file.name, lo_loc.line, lo_loc.col.to_uint(), lo_pos.to_uint(),
                hi_loc.line, hi_loc.col.to_uint(), hi_pos.to_uint())
    }

    // sub_span starts at span.lo, so we need to adjust the positions etc.
    // If sub_span is None, we don't need to adjust.
    pub fn make_sub_span(&self, span: Span, sub_span: Option<Span>) -> Option<Span> {
        let loc = self.sess.codemap().lookup_char_pos(span.lo);
        assert!(!generated_code(span),
                "generated code; we should not be processing this `{}` in {}, line {}",
                 self.snippet(span), loc.file.name, loc.line);

        match sub_span {
            None => None,
            Some(sub) => {
                let FileMapAndBytePos {fm, pos} =
                    self.sess.codemap().lookup_byte_offset(span.lo);
                let base = pos + fm.start_pos;
                Some(Span {
                    lo: base + self.sess.codemap().lookup_byte_offset(sub.lo).pos,
                    hi: base + self.sess.codemap().lookup_byte_offset(sub.hi).pos,
                    expn_id: NO_EXPANSION,
                })
            }
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.sess.codemap().span_to_snippet(span) {
            Some(s) => s,
            None => String::new(),
        }
    }

    pub fn retokenise_span(&self, span: Span) -> StringReader<'a> {
        // sadness - we don't have spans for sub-expressions nor access to the tokens
        // so in order to get extents for the function name itself (which dxr expects)
        // we need to re-tokenise the fn definition

        // Note: this is a bit awful - it adds the contents of span to the end of
        // the codemap as a new filemap. This is mostly OK, but means we should
        // not iterate over the codemap. Also, any spans over the new filemap
        // are incompatible with spans over other filemaps.
        let filemap = self.sess.codemap().new_filemap(String::from_str("<anon-dxr>"),
                                                      self.snippet(span));
        let s = self.sess;
        lexer::StringReader::new(s.diagnostic(), filemap)
    }

    // Re-parses a path and returns the span for the last identifier in the path
    pub fn span_for_last_ident(&self, span: Span) -> Option<Span> {
        let mut result = None;

        let mut toks = self.retokenise_span(span);
        let mut bracket_count = 0u;
        loop {
            let ts = toks.next_token();
            if ts.tok == token::EOF {
                return self.make_sub_span(span, result)
            }
            if bracket_count == 0 &&
               (is_ident(&ts.tok) || is_keyword(keywords::Self, &ts.tok)) {
                result = Some(ts.sp);
            }

            bracket_count += match ts.tok {
                token::LT => 1,
                token::GT => -1,
                token::BINOP(token::SHR) => -2,
                _ => 0
            }
        }
    }

    // Return the span for the first identifier in the path.
    pub fn span_for_first_ident(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut bracket_count = 0u;
        loop {
            let ts = toks.next_token();
            if ts.tok == token::EOF {
                return None;
            }
            if bracket_count == 0 &&
               (is_ident(&ts.tok) || is_keyword(keywords::Self, &ts.tok)) {
                return self.make_sub_span(span, Some(ts.sp));
            }

            bracket_count += match ts.tok {
                token::LT => 1,
                token::GT => -1,
                token::BINOP(token::SHR) => -2,
                _ => 0
            }
        }
    }

    // Return the span for the last ident before a `(` or `<` or '::<' and outside any
    // any brackets, or the last span.
    pub fn sub_span_for_meth_name(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.next_token();
        let mut result = None;
        let mut bracket_count = 0u;
        let mut last_span = None;
        while prev.tok != token::EOF {
            last_span = None;
            let mut next = toks.next_token();

            if (next.tok == token::LPAREN ||
                next.tok == token::LT) &&
               bracket_count == 0 &&
               is_ident(&prev.tok) {
                result = Some(prev.sp);
            }

            if bracket_count == 0 &&
                next.tok == token::MOD_SEP {
                let old = prev;
                prev = next;
                next = toks.next_token();
                if next.tok == token::LT &&
                   is_ident(&old.tok) {
                    result = Some(old.sp);
                }
            }

            bracket_count += match prev.tok {
                token::LPAREN | token::LT => 1,
                token::RPAREN | token::GT => -1,
                token::BINOP(token::SHR) => -2,
                _ => 0
            };

            if is_ident(&prev.tok) && bracket_count == 0 {
                last_span = Some(prev.sp);
            }
            prev = next;
        }
        if result.is_none() && last_span.is_some() {
            return self.make_sub_span(span, last_span);
        }
        return self.make_sub_span(span, result);
    }

    // Return the span for the last ident before a `<` and outside any
    // brackets, or the last span.
    pub fn sub_span_for_type_name(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.next_token();
        let mut result = None;
        let mut bracket_count = 0u;
        loop {
            let next = toks.next_token();

            if (next.tok == token::LT ||
                next.tok == token::COLON) &&
               bracket_count == 0 &&
               is_ident(&prev.tok) {
                result = Some(prev.sp);
            }

            bracket_count += match prev.tok {
                token::LT => 1,
                token::GT => -1,
                token::BINOP(token::SHR) => -2,
                _ => 0
            };

            if next.tok == token::EOF {
                break;
            }
            prev = next;
        }
        if bracket_count != 0 {
            let loc = self.sess.codemap().lookup_char_pos(span.lo);
            self.sess.span_bug(span,
                format!("Mis-counted brackets when breaking path? Parsing '{}' in {}, line {}",
                        self.snippet(span), loc.file.name, loc.line).as_slice());
        }
        if result.is_none() && is_ident(&prev.tok) && bracket_count == 0 {
            return self.make_sub_span(span, Some(prev.sp));
        }
        self.make_sub_span(span, result)
    }

    // Reparse span and return an owned vector of sub spans of the first limit
    // identifier tokens in the given nesting level.
    // example with Foo<Bar<T,V>, Bar<T,V>>
    // Nesting = 0: all idents outside of brackets: ~[Foo]
    // Nesting = 1: idents within one level of brackets: ~[Bar, Bar]
    pub fn spans_with_brackets(&self, span: Span, nesting: int, limit: int) -> Vec<Span> {
        let mut result: Vec<Span> = vec!();

        let mut toks = self.retokenise_span(span);
        // We keep track of how many brackets we're nested in
        let mut bracket_count = 0i;
        loop {
            let ts = toks.next_token();
            if ts.tok == token::EOF {
                if bracket_count != 0 {
                    let loc = self.sess.codemap().lookup_char_pos(span.lo);
                    self.sess.span_bug(span, format!(
                        "Mis-counted brackets when breaking path? Parsing '{}' in {}, line {}",
                         self.snippet(span), loc.file.name, loc.line).as_slice());
                }
                return result
            }
            if (result.len() as int) == limit {
                return result;
            }
            bracket_count += match ts.tok {
                token::LT => 1,
                token::GT => -1,
                token::BINOP(token::SHL) => 2,
                token::BINOP(token::SHR) => -2,
                _ => 0
            };
            if is_ident(&ts.tok) &&
               bracket_count == nesting {
                result.push(self.make_sub_span(span, Some(ts.sp)).unwrap());
            }
        }
    }

    pub fn sub_span_before_token(&self, span: Span, tok: Token) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.next_token();
        loop {
            if prev.tok == token::EOF {
                return None;
            }
            let next = toks.next_token();
            if next.tok == tok {
                return self.make_sub_span(span, Some(prev.sp));
            }
            prev = next;
        }
    }

    // Return an owned vector of the subspans of the tokens that come before tok2
    // which is before tok1. If there is no instance of tok2 before tok1, then that
    // place in the result is None.
    // Everything returned must be inside a set of (non-angle) brackets, but no
    // more deeply nested than that.
    pub fn sub_spans_before_tokens(&self,
                               span: Span,
                               tok1: Token,
                               tok2: Token) -> Vec<Option<Span>> {
        let mut sub_spans : Vec<Option<Span>> = vec!();
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.next_token();
        let mut next = toks.next_token();
        let mut stored_val = false;
        let mut found_val = false;
        let mut bracket_count = 0u;
        while next.tok != token::EOF {
            if bracket_count == 1 {
                if next.tok == tok2 {
                    sub_spans.push(self.make_sub_span(span, Some(prev.sp)));
                    stored_val = true;
                    found_val = false;
                }
                if next.tok == tok1 {
                    if !stored_val {
                        sub_spans.push(None);
                    } else {
                        stored_val = false;
                    }
                    found_val = false;
                }
                if !stored_val &&
                   is_ident(&next.tok) {
                    found_val = true;
                }
            }

            bracket_count += match next.tok {
                token::LPAREN | token::LBRACE => 1,
                token::RPAREN | token::RBRACE => -1,
                _ => 0
            };

            prev = next;
            next = toks.next_token();
        }
        if found_val {
            sub_spans.push(None);
        }
        return sub_spans;
    }

    pub fn sub_span_after_keyword(&self,
                              span: Span,
                              keyword: keywords::Keyword) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        loop {
            let ts = toks.next_token();
            if ts.tok == token::EOF {
                return None;
            }
            if is_keyword(keyword, &ts.tok) {
                let ts = toks.next_token();
                if ts.tok == token::EOF {
                    return None
                } else {
                    return self.make_sub_span(span, Some(ts.sp));
                }
            }
        }
    }

    // Returns a list of the spans of idents in a patch.
    // E.g., For foo::bar<x,t>::baz, we return [foo, bar, baz] (well, their spans)
    pub fn spans_for_path_segments(&self, path: &ast::Path) -> Vec<Span> {
        if generated_code(path.span) {
            return vec!();
        }

        self.spans_with_brackets(path.span, 0, -1)
    }

    // Return an owned vector of the subspans of the param identifier
    // tokens found in span.
    pub fn spans_for_ty_params(&self, span: Span, number: int) -> Vec<Span> {
        if generated_code(span) {
            return vec!();
        }
        // Type params are nested within one level of brackets:
        // i.e. we want ~[A, B] from Foo<A, B<T,U>>
        self.spans_with_brackets(span, 1, number)
    }

    pub fn report_span_err(&self, kind: &str, span: Span) {
        let loc = self.sess.codemap().lookup_char_pos(span.lo);
        info!("({}) Could not find sub_span in `{}` in {}, line {}",
              kind, self.snippet(span), loc.file.name, loc.line);
        self.err_count.set(self.err_count.get()+1);
        if self.err_count.get() > 1000 {
            self.sess.bug("span errors reached 1000, giving up");
        }
    }
}
