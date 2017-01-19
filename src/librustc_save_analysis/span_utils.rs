// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::session::Session;

use generated_code;

use std::cell::Cell;
use std::env;
use std::path::Path;

use syntax::ast;
use syntax::parse::filemap_to_tts;
use syntax::parse::lexer::{self, StringReader};
use syntax::parse::token::{self, Token};
use syntax::symbol::keywords;
use syntax::tokenstream::TokenTree;
use syntax_pos::*;

#[derive(Clone)]
pub struct SpanUtils<'a> {
    pub sess: &'a Session,
    // FIXME given that we clone SpanUtils all over the place, this err_count is
    // probably useless and any logic relying on it is bogus.
    pub err_count: Cell<isize>,
}

impl<'a> SpanUtils<'a> {
    pub fn new(sess: &'a Session) -> SpanUtils<'a> {
        SpanUtils {
            sess: sess,
            err_count: Cell::new(0),
        }
    }

    pub fn make_path_string(file_name: &str) -> String {
        let path = Path::new(file_name);
        if path.is_absolute() {
            path.clone().display().to_string()
        } else {
            env::current_dir().unwrap().join(&path).display().to_string()
        }
    }

    // sub_span starts at span.lo, so we need to adjust the positions etc.
    // If sub_span is None, we don't need to adjust.
    pub fn make_sub_span(&self, span: Span, sub_span: Option<Span>) -> Option<Span> {
        match sub_span {
            None => None,
            Some(sub) => {
                let FileMapAndBytePos {fm, pos} = self.sess.codemap().lookup_byte_offset(span.lo);
                let base = pos + fm.start_pos;
                Some(Span {
                    lo: base + self.sess.codemap().lookup_byte_offset(sub.lo).pos,
                    hi: base + self.sess.codemap().lookup_byte_offset(sub.hi).pos,
                    expn_id: span.expn_id,
                })
            }
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.sess.codemap().span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => String::new(),
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
        let filemap = self.sess
                          .codemap()
                          .new_filemap(String::from("<anon-dxr>"), None, self.snippet(span));
        lexer::StringReader::new(&self.sess.parse_sess, filemap)
    }

    fn span_to_tts(&self, span: Span) -> Vec<TokenTree> {
        let filename = String::from("<anon-dxr>");
        let filemap = self.sess.codemap().new_filemap(filename, None, self.snippet(span));
        filemap_to_tts(&self.sess.parse_sess, filemap)
    }

    // Re-parses a path and returns the span for the last identifier in the path
    pub fn span_for_last_ident(&self, span: Span) -> Option<Span> {
        let mut result = None;

        let mut toks = self.retokenise_span(span);
        let mut bracket_count = 0;
        loop {
            let ts = toks.real_token();
            if ts.tok == token::Eof {
                return self.make_sub_span(span, result)
            }
            if bracket_count == 0 && (ts.tok.is_ident() || ts.tok.is_keyword(keywords::SelfValue)) {
                result = Some(ts.sp);
            }

            bracket_count += match ts.tok {
                token::Lt => 1,
                token::Gt => -1,
                token::BinOp(token::Shr) => -2,
                _ => 0,
            }
        }
    }

    // Return the span for the first identifier in the path.
    pub fn span_for_first_ident(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut bracket_count = 0;
        loop {
            let ts = toks.real_token();
            if ts.tok == token::Eof {
                return None;
            }
            if bracket_count == 0 && (ts.tok.is_ident() || ts.tok.is_keyword(keywords::SelfValue)) {
                return self.make_sub_span(span, Some(ts.sp));
            }

            bracket_count += match ts.tok {
                token::Lt => 1,
                token::Gt => -1,
                token::BinOp(token::Shr) => -2,
                _ => 0,
            }
        }
    }

    // Return the span for the last ident before a `(` or `<` or '::<' and outside any
    // any brackets, or the last span.
    pub fn sub_span_for_meth_name(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.real_token();
        let mut result = None;
        let mut bracket_count = 0;
        let mut prev_span = None;
        while prev.tok != token::Eof {
            prev_span = None;
            let mut next = toks.real_token();

            if (next.tok == token::OpenDelim(token::Paren) || next.tok == token::Lt) &&
               bracket_count == 0 && prev.tok.is_ident() {
                result = Some(prev.sp);
            }

            if bracket_count == 0 && next.tok == token::ModSep {
                let old = prev;
                prev = next;
                next = toks.real_token();
                if next.tok == token::Lt && old.tok.is_ident() {
                    result = Some(old.sp);
                }
            }

            bracket_count += match prev.tok {
                token::OpenDelim(token::Paren) | token::Lt => 1,
                token::CloseDelim(token::Paren) | token::Gt => -1,
                token::BinOp(token::Shr) => -2,
                _ => 0,
            };

            if prev.tok.is_ident() && bracket_count == 0 {
                prev_span = Some(prev.sp);
            }
            prev = next;
        }
        if result.is_none() && prev_span.is_some() {
            return self.make_sub_span(span, prev_span);
        }
        return self.make_sub_span(span, result);
    }

    // Return the span for the last ident before a `<` and outside any
    // angle brackets, or the last span.
    pub fn sub_span_for_type_name(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.real_token();
        let mut result = None;

        // We keep track of the following two counts - the depth of nesting of
        // angle brackets, and the depth of nesting of square brackets. For the
        // angle bracket count, we only count tokens which occur outside of any
        // square brackets (i.e. bracket_count == 0). The intutition here is
        // that we want to count angle brackets in the type, but not any which
        // could be in expression context (because these could mean 'less than',
        // etc.).
        let mut angle_count = 0;
        let mut bracket_count = 0;
        loop {
            let next = toks.real_token();

            if (next.tok == token::Lt || next.tok == token::Colon) &&
               angle_count == 0 &&
               bracket_count == 0 &&
               prev.tok.is_ident() {
                result = Some(prev.sp);
            }

            if bracket_count == 0 {
                angle_count += match prev.tok {
                    token::Lt => 1,
                    token::Gt => -1,
                    token::BinOp(token::Shl) => 2,
                    token::BinOp(token::Shr) => -2,
                    _ => 0,
                };
            }

            bracket_count += match prev.tok {
                token::OpenDelim(token::Bracket) => 1,
                token::CloseDelim(token::Bracket) => -1,
                _ => 0,
            };

            if next.tok == token::Eof {
                break;
            }
            prev = next;
        }
        if angle_count != 0 || bracket_count != 0 {
            let loc = self.sess.codemap().lookup_char_pos(span.lo);
            span_bug!(span,
                      "Mis-counted brackets when breaking path? Parsing '{}' \
                       in {}, line {}",
                      self.snippet(span),
                      loc.file.name,
                      loc.line);
        }
        if result.is_none() && prev.tok.is_ident() && angle_count == 0 {
            return self.make_sub_span(span, Some(prev.sp));
        }
        self.make_sub_span(span, result)
    }

    // Reparse span and return an owned vector of sub spans of the first limit
    // identifier tokens in the given nesting level.
    // example with Foo<Bar<T,V>, Bar<T,V>>
    // Nesting = 0: all idents outside of angle brackets: [Foo]
    // Nesting = 1: idents within one level of angle brackets: [Bar, Bar]
    pub fn spans_with_brackets(&self, span: Span, nesting: isize, limit: isize) -> Vec<Span> {
        let mut result: Vec<Span> = vec![];

        let mut toks = self.retokenise_span(span);
        // We keep track of how many brackets we're nested in
        let mut angle_count: isize = 0;
        let mut bracket_count: isize = 0;
        let mut found_ufcs_sep = false;
        loop {
            let ts = toks.real_token();
            if ts.tok == token::Eof {
                if angle_count != 0 || bracket_count != 0 {
                    if generated_code(span) {
                        return vec![];
                    }
                    let loc = self.sess.codemap().lookup_char_pos(span.lo);
                    span_bug!(span,
                              "Mis-counted brackets when breaking path? \
                               Parsing '{}' in {}, line {}",
                              self.snippet(span),
                              loc.file.name,
                              loc.line);
                }
                return result
            }
            if (result.len() as isize) == limit {
                return result;
            }
            bracket_count += match ts.tok {
                token::OpenDelim(token::Bracket) => 1,
                token::CloseDelim(token::Bracket) => -1,
                _ => 0,
            };
            if bracket_count > 0 {
                continue;
            }
            angle_count += match ts.tok {
                token::Lt => 1,
                token::Gt => -1,
                token::BinOp(token::Shl) => 2,
                token::BinOp(token::Shr) => -2,
                _ => 0,
            };

            // Ignore the `>::` in `<Type as Trait>::AssocTy`.

            // The root cause of this hack is that the AST representation of
            // qpaths is horrible. It treats <A as B>::C as a path with two
            // segments, B and C and notes that there is also a self type A at
            // position 0. Because we don't have spans for individual idents,
            // only the whole path, we have to iterate over the tokens in the
            // path, trying to pull out the non-nested idents (e.g., avoiding 'a
            // in `<A as B<'a>>::C`). So we end up with a span for `B>::C` from
            // the start of the first ident to the end of the path.
            if !found_ufcs_sep && angle_count == -1 {
                found_ufcs_sep = true;
                angle_count += 1;
            }
            if ts.tok.is_ident() && angle_count == nesting {
                result.push(self.make_sub_span(span, Some(ts.sp)).unwrap());
            }
        }
    }

    /// `span` must be the span for an item such as a function or struct. This
    /// function returns the program text from the start of the span until the
    /// end of the 'signature' part, that is up to, but not including an opening
    /// brace or semicolon.
    pub fn signature_string_for_span(&self, span: Span) -> String {
        let mut toks = self.span_to_tts(span).into_iter();
        let mut prev = toks.next().unwrap();
        let first_span = prev.get_span();
        let mut angle_count = 0;
        for tok in toks {
            if let TokenTree::Token(_, ref tok) = prev {
                angle_count += match *tok {
                    token::Eof => { break; }
                    token::Lt => 1,
                    token::Gt => -1,
                    token::BinOp(token::Shl) => 2,
                    token::BinOp(token::Shr) => -2,
                    _ => 0,
                };
            }
            if angle_count > 0 {
                prev = tok;
                continue;
            }
            if let TokenTree::Token(_, token::Semi) = tok {
                return self.snippet(mk_sp(first_span.lo, prev.get_span().hi));
            } else if let TokenTree::Delimited(_, ref d) = tok {
                if d.delim == token::Brace {
                    return self.snippet(mk_sp(first_span.lo, prev.get_span().hi));
                }
            }
            prev = tok;
        }
        self.snippet(span)
    }

    pub fn sub_span_before_token(&self, span: Span, tok: Token) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.real_token();
        loop {
            if prev.tok == token::Eof {
                return None;
            }
            let next = toks.real_token();
            if next.tok == tok {
                return self.make_sub_span(span, Some(prev.sp));
            }
            prev = next;
        }
    }

    pub fn sub_span_of_token(&self, span: Span, tok: Token) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        loop {
            let next = toks.real_token();
            if next.tok == token::Eof {
                return None;
            }
            if next.tok == tok {
                return self.make_sub_span(span, Some(next.sp));
            }
        }
    }

    pub fn sub_span_after_keyword(&self, span: Span, keyword: keywords::Keyword) -> Option<Span> {
        self.sub_span_after(span, |t| t.is_keyword(keyword))
    }

    pub fn sub_span_after_token(&self, span: Span, tok: Token) -> Option<Span> {
        self.sub_span_after(span, |t| t == tok)
    }

    fn sub_span_after<F: Fn(Token) -> bool>(&self, span: Span, f: F) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        loop {
            let ts = toks.real_token();
            if ts.tok == token::Eof {
                return None;
            }
            if f(ts.tok) {
                let ts = toks.real_token();
                if ts.tok == token::Eof {
                    return None
                } else {
                    return self.make_sub_span(span, Some(ts.sp));
                }
            }
        }
    }


    // Returns a list of the spans of idents in a path.
    // E.g., For foo::bar<x,t>::baz, we return [foo, bar, baz] (well, their spans)
    pub fn spans_for_path_segments(&self, path: &ast::Path) -> Vec<Span> {
        self.spans_with_brackets(path.span, 0, -1)
    }

    // Return an owned vector of the subspans of the param identifier
    // tokens found in span.
    pub fn spans_for_ty_params(&self, span: Span, number: isize) -> Vec<Span> {
        // Type params are nested within one level of brackets:
        // i.e. we want Vec<A, B> from Foo<A, B<T,U>>
        self.spans_with_brackets(span, 1, number)
    }

    pub fn report_span_err(&self, kind: &str, span: Span) {
        let loc = self.sess.codemap().lookup_char_pos(span.lo);
        info!("({}) Could not find sub_span in `{}` in {}, line {}",
              kind,
              self.snippet(span),
              loc.file.name,
              loc.line);
        self.err_count.set(self.err_count.get() + 1);
        if self.err_count.get() > 1000 {
            bug!("span errors reached 1000, giving up");
        }
    }

    // Return the name for a macro definition (identifier after first `!`)
    pub fn span_for_macro_def_name(&self, span: Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        loop {
            let ts = toks.real_token();
            if ts.tok == token::Eof {
                return None;
            }
            if ts.tok == token::Not {
                let ts = toks.real_token();
                if ts.tok.is_ident() {
                    return self.make_sub_span(span, Some(ts.sp));
                } else {
                    return None;
                }
            }
        }
    }

    // Return the name for a macro use (identifier before first `!`).
    pub fn span_for_macro_use_name(&self, span:Span) -> Option<Span> {
        let mut toks = self.retokenise_span(span);
        let mut prev = toks.real_token();
        loop {
            if prev.tok == token::Eof {
                return None;
            }
            let ts = toks.real_token();
            if ts.tok == token::Not {
                if prev.tok.is_ident() {
                    return self.make_sub_span(span, Some(prev.sp));
                } else {
                    return None;
                }
            }
            prev = ts;
        }
    }

    /// Return true if the span is generated code, and
    /// it is not a subspan of the root callsite.
    ///
    /// Used to filter out spans of minimal value,
    /// such as references to macro internal variables.
    pub fn filter_generated(&self, sub_span: Option<Span>, parent: Span) -> bool {
        if !generated_code(parent) {
            if sub_span.is_none() {
                // Edge case - this occurs on generated code with incorrect expansion info.
                return true;
            }
            return false;
        }
        // If sub_span is none, filter out generated code.
        if sub_span.is_none() {
            return true;
        }

        //If the span comes from a fake filemap, filter it.
        if !self.sess.codemap().lookup_char_pos(parent.lo).file.is_real_file() {
            return true;
        }

        // Otherwise, a generated span is deemed invalid if it is not a sub-span of the root
        // callsite. This filters out macro internal variables and most malformed spans.
        let span = self.sess.codemap().source_callsite(parent);
        !(span.contains(parent))
    }
}

macro_rules! filter {
    ($util: expr, $span: ident, $parent: expr, None) => {
        if $util.filter_generated($span, $parent) {
            return None;
        }
    };
    ($util: expr, $span: ident, $parent: expr) => {
        if $util.filter_generated($span, $parent) {
            return;
        }
    };
}
