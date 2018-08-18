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

use syntax::parse::lexer::{self, StringReader};
use syntax::parse::token::{self, Token};
use syntax::symbol::keywords;
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
            sess,
            err_count: Cell::new(0),
        }
    }

    pub fn make_path_string(&self, path: &FileName) -> String {
        match *path {
            FileName::Real(ref path) if !path.is_absolute() =>
                self.sess.working_dir.0
                    .join(&path)
                    .display()
                    .to_string(),
            _ => path.to_string(),
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.sess.codemap().span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => String::new(),
        }
    }

    pub fn retokenise_span(&self, span: Span) -> StringReader<'a> {
        lexer::StringReader::retokenize(&self.sess.parse_sess, span)
    }

    // Re-parses a path and returns the span for the last identifier in the path
    pub fn span_for_last_ident(&self, span: Span) -> Option<Span> {
        let mut result = None;

        let mut toks = self.retokenise_span(span);
        let mut bracket_count = 0;
        loop {
            let ts = toks.real_token();
            if ts.tok == token::Eof {
                return result;
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
                return Some(ts.sp);
            }

            bracket_count += match ts.tok {
                token::Lt => 1,
                token::Gt => -1,
                token::BinOp(token::Shr) => -2,
                _ => 0,
            }
        }
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
        // square brackets (i.e. bracket_count == 0). The intuition here is
        // that we want to count angle brackets in the type, but not any which
        // could be in expression context (because these could mean 'less than',
        // etc.).
        let mut angle_count = 0;
        let mut bracket_count = 0;
        loop {
            let next = toks.real_token();

            if (next.tok == token::Lt || next.tok == token::Colon) && angle_count == 0
                && bracket_count == 0 && prev.tok.is_ident()
            {
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
        #[cfg(debug_assertions)] {
            if angle_count != 0 || bracket_count != 0 {
                let loc = self.sess.codemap().lookup_char_pos(span.lo());
                span_bug!(
                    span,
                    "Mis-counted brackets when breaking path? Parsing '{}' \
                     in {}, line {}",
                    self.snippet(span),
                    loc.file.name,
                    loc.line
                );
            }
        }
        if result.is_none() && prev.tok.is_ident() {
            return Some(prev.sp);
        }
        result
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
                return Some(prev.sp);
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
                return Some(next.sp);
            }
        }
    }

    pub fn sub_span_after_keyword(&self, span: Span, keyword: keywords::Keyword) -> Option<Span> {
        self.sub_span_after(span, |t| t.is_keyword(keyword))
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
                    return None;
                } else {
                    return Some(ts.sp);
                }
            }
        }
    }

    // // Return the name for a macro definition (identifier after first `!`)
    // pub fn span_for_macro_def_name(&self, span: Span) -> Option<Span> {
    //     let mut toks = self.retokenise_span(span);
    //     loop {
    //         let ts = toks.real_token();
    //         if ts.tok == token::Eof {
    //             return None;
    //         }
    //         if ts.tok == token::Not {
    //             let ts = toks.real_token();
    //             if ts.tok.is_ident() {
    //                 return Some(ts.sp);
    //             } else {
    //                 return None;
    //             }
    //         }
    //     }
    // }

    // // Return the name for a macro use (identifier before first `!`).
    // pub fn span_for_macro_use_name(&self, span:Span) -> Option<Span> {
    //     let mut toks = self.retokenise_span(span);
    //     let mut prev = toks.real_token();
    //     loop {
    //         if prev.tok == token::Eof {
    //             return None;
    //         }
    //         let ts = toks.real_token();
    //         if ts.tok == token::Not {
    //             if prev.tok.is_ident() {
    //                 return Some(prev.sp);
    //             } else {
    //                 return None;
    //             }
    //         }
    //         prev = ts;
    //     }
    // }

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
        let sub_span = match sub_span {
            Some(ss) => ss,
            None => return true,
        };

        //If the span comes from a fake source_file, filter it.
        if !self.sess
            .codemap()
            .lookup_char_pos(parent.lo())
            .file
            .is_real_file()
        {
            return true;
        }

        // Otherwise, a generated span is deemed invalid if it is not a sub-span of the root
        // callsite. This filters out macro internal variables and most malformed spans.
        !parent.source_callsite().contains(sub_span)
    }
}

macro_rules! filter {
    ($util: expr, $span: expr, $parent: expr, None) => {
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
