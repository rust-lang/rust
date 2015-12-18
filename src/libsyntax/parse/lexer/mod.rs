// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::{BytePos, CharPos, CodeMap, Pos, Span};
use codemap;
use diagnostic::FatalError;
use diagnostic::SpanHandler;
use ext::tt::transcribe::tt_next_token;
use parse::token::str_to_ident;
use parse::token;
use str::char_at;

use std::borrow::Cow;
use std::char;
use std::mem::replace;
use std::rc::Rc;

pub use ext::tt::transcribe::{TtReader, new_tt_reader, new_tt_reader_with_doc_flag};

pub mod comments;
mod unicode_chars;

pub trait Reader {
    fn is_eof(&self) -> bool;
    fn next_token(&mut self) -> TokenAndSpan;
    /// Report a fatal error with the current span.
    fn fatal(&self, &str) -> FatalError;
    /// Report a non-fatal error with the current span.
    fn err(&self, &str);
    fn peek(&self) -> TokenAndSpan;
    /// Get a token the parser cares about.
    fn real_token(&mut self) -> TokenAndSpan {
        let mut t = self.next_token();
        loop {
            match t.tok {
                token::Whitespace | token::Comment | token::Shebang(_) => {
                    t = self.next_token();
                }
                _ => break,
            }
        }
        t
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TokenAndSpan {
    pub tok: token::Token,
    pub sp: Span,
}

pub struct StringReader<'a> {
    pub span_diagnostic: &'a SpanHandler,
    /// The absolute offset within the codemap of the next character to read
    pub pos: BytePos,
    /// The absolute offset within the codemap of the last character read(curr)
    pub last_pos: BytePos,
    /// The column of the next character to read
    pub col: CharPos,
    /// The last character to be read
    pub curr: Option<char>,
    pub filemap: Rc<codemap::FileMap>,
    // cached:
    pub peek_tok: token::Token,
    pub peek_span: Span,

    // cache a direct reference to the source text, so that we don't have to
    // retrieve it via `self.filemap.src.as_ref().unwrap()` all the time.
    source_text: Rc<String>,
}

impl<'a> Reader for StringReader<'a> {
    fn is_eof(&self) -> bool {
        self.curr.is_none()
    }
    /// Return the next token. EFFECT: advances the string_reader.
    fn next_token(&mut self) -> TokenAndSpan {
        let ret_val = TokenAndSpan {
            tok: replace(&mut self.peek_tok, token::Underscore),
            sp: self.peek_span,
        };
        self.advance_token();
        ret_val
    }
    fn fatal(&self, m: &str) -> FatalError {
        self.fatal_span(self.peek_span, m)
    }
    fn err(&self, m: &str) {
        self.err_span(self.peek_span, m)
    }
    fn peek(&self) -> TokenAndSpan {
        // FIXME(pcwalton): Bad copy!
        TokenAndSpan {
            tok: self.peek_tok.clone(),
            sp: self.peek_span,
        }
    }
}

impl<'a> Reader for TtReader<'a> {
    fn is_eof(&self) -> bool {
        self.cur_tok == token::Eof
    }
    fn next_token(&mut self) -> TokenAndSpan {
        let r = tt_next_token(self);
        debug!("TtReader: r={:?}", r);
        r
    }
    fn fatal(&self, m: &str) -> FatalError {
        self.sp_diag.span_fatal(self.cur_span, m)
    }
    fn err(&self, m: &str) {
        self.sp_diag.span_err(self.cur_span, m);
    }
    fn peek(&self) -> TokenAndSpan {
        TokenAndSpan {
            tok: self.cur_tok.clone(),
            sp: self.cur_span,
        }
    }
}

impl<'a> StringReader<'a> {
    /// For comments.rs, which hackily pokes into pos and curr
    pub fn new_raw<'b>(span_diagnostic: &'b SpanHandler,
                       filemap: Rc<codemap::FileMap>)
                       -> StringReader<'b> {
        if filemap.src.is_none() {
            span_diagnostic.handler
                           .bug(&format!("Cannot lex filemap \
                                          without source: {}",
                                         filemap.name)[..]);
        }

        let source_text = (*filemap.src.as_ref().unwrap()).clone();

        let mut sr = StringReader {
            span_diagnostic: span_diagnostic,
            pos: filemap.start_pos,
            last_pos: filemap.start_pos,
            col: CharPos(0),
            curr: Some('\n'),
            filemap: filemap,
            // dummy values; not read
            peek_tok: token::Eof,
            peek_span: codemap::DUMMY_SP,
            source_text: source_text,
        };
        sr.bump();
        sr
    }

    pub fn new<'b>(span_diagnostic: &'b SpanHandler,
                   filemap: Rc<codemap::FileMap>)
                   -> StringReader<'b> {
        let mut sr = StringReader::new_raw(span_diagnostic, filemap);
        sr.advance_token();
        sr
    }

    pub fn curr_is(&self, c: char) -> bool {
        self.curr == Some(c)
    }

    /// Report a fatal lexical error with a given span.
    pub fn fatal_span(&self, sp: Span, m: &str) -> FatalError {
        self.span_diagnostic.span_fatal(sp, m)
    }

    /// Report a lexical error with a given span.
    pub fn err_span(&self, sp: Span, m: &str) {
        self.span_diagnostic.span_err(sp, m)
    }

    /// Suggest some help with a given span.
    pub fn help_span(&self, sp: Span, m: &str) {
        self.span_diagnostic.span_help(sp, m)
    }

    /// Report a fatal error spanning [`from_pos`, `to_pos`).
    fn fatal_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) -> FatalError {
        self.fatal_span(codemap::mk_sp(from_pos, to_pos), m)
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`).
    fn err_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) {
        self.err_span(codemap::mk_sp(from_pos, to_pos), m)
    }

    /// Suggest some help spanning [`from_pos`, `to_pos`).
    fn help_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) {
        self.help_span(codemap::mk_sp(from_pos, to_pos), m)
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`), appending an
    /// escaped character to the error message
    fn fatal_span_char(&self, from_pos: BytePos, to_pos: BytePos, m: &str, c: char) -> FatalError {
        let mut m = m.to_string();
        m.push_str(": ");
        for c in c.escape_default() {
            m.push(c);
        }
        self.fatal_span_(from_pos, to_pos, &m[..])
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`), appending an
    /// escaped character to the error message
    fn err_span_char(&self, from_pos: BytePos, to_pos: BytePos, m: &str, c: char) {
        let mut m = m.to_string();
        m.push_str(": ");
        for c in c.escape_default() {
            m.push(c);
        }
        self.err_span_(from_pos, to_pos, &m[..]);
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`), appending the
    /// offending string to the error message
    fn fatal_span_verbose(&self, from_pos: BytePos, to_pos: BytePos, mut m: String) -> FatalError {
        m.push_str(": ");
        let from = self.byte_offset(from_pos).to_usize();
        let to = self.byte_offset(to_pos).to_usize();
        m.push_str(&self.source_text[from..to]);
        self.fatal_span_(from_pos, to_pos, &m[..])
    }

    /// Advance peek_tok and peek_span to refer to the next token, and
    /// possibly update the interner.
    fn advance_token(&mut self) {
        match self.scan_whitespace_or_comment() {
            Some(comment) => {
                self.peek_span = comment.sp;
                self.peek_tok = comment.tok;
            }
            None => {
                if self.is_eof() {
                    self.peek_tok = token::Eof;
                    self.peek_span = codemap::mk_sp(self.filemap.end_pos, self.filemap.end_pos);
                } else {
                    let start_bytepos = self.last_pos;
                    self.peek_tok = self.next_token_inner();
                    self.peek_span = codemap::mk_sp(start_bytepos, self.last_pos);
                };
            }
        }
    }

    fn byte_offset(&self, pos: BytePos) -> BytePos {
        (pos - self.filemap.start_pos)
    }

    /// Calls `f` with a string slice of the source text spanning from `start`
    /// up to but excluding `self.last_pos`, meaning the slice does not include
    /// the character `self.curr`.
    pub fn with_str_from<T, F>(&self, start: BytePos, f: F) -> T
        where F: FnOnce(&str) -> T
    {
        self.with_str_from_to(start, self.last_pos, f)
    }

    /// Create a Name from a given offset to the current offset, each
    /// adjusted 1 towards each other (assumes that on either side there is a
    /// single-byte delimiter).
    pub fn name_from(&self, start: BytePos) -> ast::Name {
        debug!("taking an ident from {:?} to {:?}", start, self.last_pos);
        self.with_str_from(start, token::intern)
    }

    /// As name_from, with an explicit endpoint.
    pub fn name_from_to(&self, start: BytePos, end: BytePos) -> ast::Name {
        debug!("taking an ident from {:?} to {:?}", start, end);
        self.with_str_from_to(start, end, token::intern)
    }

    /// Calls `f` with a string slice of the source text spanning from `start`
    /// up to but excluding `end`.
    fn with_str_from_to<T, F>(&self, start: BytePos, end: BytePos, f: F) -> T
        where F: FnOnce(&str) -> T
    {
        f(&self.source_text[self.byte_offset(start).to_usize()..self.byte_offset(end).to_usize()])
    }

    /// Converts CRLF to LF in the given string, raising an error on bare CR.
    fn translate_crlf<'b>(&self, start: BytePos, s: &'b str, errmsg: &'b str) -> Cow<'b, str> {
        let mut i = 0;
        while i < s.len() {
            let ch = char_at(s, i);
            let next = i + ch.len_utf8();
            if ch == '\r' {
                if next < s.len() && char_at(s, next) == '\n' {
                    return translate_crlf_(self, start, s, errmsg, i).into();
                }
                let pos = start + BytePos(i as u32);
                let end_pos = start + BytePos(next as u32);
                self.err_span_(pos, end_pos, errmsg);
            }
            i = next;
        }
        return s.into();

        fn translate_crlf_(rdr: &StringReader,
                           start: BytePos,
                           s: &str,
                           errmsg: &str,
                           mut i: usize)
                           -> String {
            let mut buf = String::with_capacity(s.len());
            let mut j = 0;
            while i < s.len() {
                let ch = char_at(s, i);
                let next = i + ch.len_utf8();
                if ch == '\r' {
                    if j < i {
                        buf.push_str(&s[j..i]);
                    }
                    j = next;
                    if next >= s.len() || char_at(s, next) != '\n' {
                        let pos = start + BytePos(i as u32);
                        let end_pos = start + BytePos(next as u32);
                        rdr.err_span_(pos, end_pos, errmsg);
                    }
                }
                i = next;
            }
            if j < s.len() {
                buf.push_str(&s[j..]);
            }
            buf
        }
    }


    /// Advance the StringReader by one character. If a newline is
    /// discovered, add it to the FileMap's list of line start offsets.
    pub fn bump(&mut self) {
        self.last_pos = self.pos;
        let current_byte_offset = self.byte_offset(self.pos).to_usize();
        if current_byte_offset < self.source_text.len() {
            assert!(self.curr.is_some());
            let last_char = self.curr.unwrap();
            let ch = char_at(&self.source_text, current_byte_offset);
            let next = current_byte_offset + ch.len_utf8();
            let byte_offset_diff = next - current_byte_offset;
            self.pos = self.pos + Pos::from_usize(byte_offset_diff);
            self.curr = Some(ch);
            self.col = self.col + CharPos(1);
            if last_char == '\n' {
                self.filemap.next_line(self.last_pos);
                self.col = CharPos(0);
            }

            if byte_offset_diff > 1 {
                self.filemap.record_multibyte_char(self.last_pos, byte_offset_diff);
            }
        } else {
            self.curr = None;
        }
    }

    pub fn nextch(&self) -> Option<char> {
        let offset = self.byte_offset(self.pos).to_usize();
        if offset < self.source_text.len() {
            Some(char_at(&self.source_text, offset))
        } else {
            None
        }
    }

    pub fn nextch_is(&self, c: char) -> bool {
        self.nextch() == Some(c)
    }

    pub fn nextnextch(&self) -> Option<char> {
        let offset = self.byte_offset(self.pos).to_usize();
        let s = &self.source_text[..];
        if offset >= s.len() {
            return None;
        }
        let next = offset + char_at(s, offset).len_utf8();
        if next < s.len() {
            Some(char_at(s, next))
        } else {
            None
        }
    }

    pub fn nextnextch_is(&self, c: char) -> bool {
        self.nextnextch() == Some(c)
    }

    /// Eats <XID_start><XID_continue>*, if possible.
    fn scan_optional_raw_name(&mut self) -> Option<ast::Name> {
        if !ident_start(self.curr) {
            return None;
        }
        let start = self.last_pos;
        while ident_continue(self.curr) {
            self.bump();
        }

        self.with_str_from(start, |string| {
            if string == "_" {
                None
            } else {
                Some(token::intern(string))
            }
        })
    }

    /// PRECONDITION: self.curr is not whitespace
    /// Eats any kind of comment.
    fn scan_comment(&mut self) -> Option<TokenAndSpan> {
        match self.curr {
            Some(c) => {
                if c.is_whitespace() {
                    self.span_diagnostic.span_err(codemap::mk_sp(self.last_pos, self.last_pos),
                                                  "called consume_any_line_comment, but there \
                                                   was whitespace");
                }
            }
            None => {}
        }

        if self.curr_is('/') {
            match self.nextch() {
                Some('/') => {
                    self.bump();
                    self.bump();

                    // line comments starting with "///" or "//!" are doc-comments
                    let doc_comment = self.curr_is('/') || self.curr_is('!');
                    let start_bpos = if doc_comment {
                        self.pos - BytePos(3)
                    } else {
                        self.last_pos - BytePos(2)
                    };

                    while !self.is_eof() {
                        match self.curr.unwrap() {
                            '\n' => break,
                            '\r' => {
                                if self.nextch_is('\n') {
                                    // CRLF
                                    break;
                                } else if doc_comment {
                                    self.err_span_(self.last_pos,
                                                   self.pos,
                                                   "bare CR not allowed in doc-comment");
                                }
                            }
                            _ => (),
                        }
                        self.bump();
                    }

                    return if doc_comment {
                        self.with_str_from(start_bpos, |string| {
                            // comments with only more "/"s are not doc comments
                            let tok = if is_doc_comment(string) {
                                token::DocComment(token::intern(string))
                            } else {
                                token::Comment
                            };

                            Some(TokenAndSpan {
                                tok: tok,
                                sp: codemap::mk_sp(start_bpos, self.last_pos),
                            })
                        })
                    } else {
                        Some(TokenAndSpan {
                            tok: token::Comment,
                            sp: codemap::mk_sp(start_bpos, self.last_pos),
                        })
                    };
                }
                Some('*') => {
                    self.bump();
                    self.bump();
                    self.scan_block_comment()
                }
                _ => None,
            }
        } else if self.curr_is('#') {
            if self.nextch_is('!') {

                // Parse an inner attribute.
                if self.nextnextch_is('[') {
                    return None;
                }

                // I guess this is the only way to figure out if
                // we're at the beginning of the file...
                let cmap = CodeMap::new();
                cmap.files.borrow_mut().push(self.filemap.clone());
                let loc = cmap.lookup_char_pos_adj(self.last_pos);
                debug!("Skipping a shebang");
                if loc.line == 1 && loc.col == CharPos(0) {
                    // FIXME: Add shebang "token", return it
                    let start = self.last_pos;
                    while !self.curr_is('\n') && !self.is_eof() {
                        self.bump();
                    }
                    return Some(TokenAndSpan {
                        tok: token::Shebang(self.name_from(start)),
                        sp: codemap::mk_sp(start, self.last_pos),
                    });
                }
            }
            None
        } else {
            None
        }
    }

    /// If there is whitespace, shebang, or a comment, scan it. Otherwise,
    /// return None.
    fn scan_whitespace_or_comment(&mut self) -> Option<TokenAndSpan> {
        match self.curr.unwrap_or('\0') {
            // # to handle shebang at start of file -- this is the entry point
            // for skipping over all "junk"
            '/' | '#' => {
                let c = self.scan_comment();
                debug!("scanning a comment {:?}", c);
                c
            }
            c if is_whitespace(Some(c)) => {
                let start_bpos = self.last_pos;
                while is_whitespace(self.curr) {
                    self.bump();
                }
                let c = Some(TokenAndSpan {
                    tok: token::Whitespace,
                    sp: codemap::mk_sp(start_bpos, self.last_pos),
                });
                debug!("scanning whitespace: {:?}", c);
                c
            }
            _ => None,
        }
    }

    /// Might return a sugared-doc-attr
    fn scan_block_comment(&mut self) -> Option<TokenAndSpan> {
        // block comments starting with "/**" or "/*!" are doc-comments
        let is_doc_comment = self.curr_is('*') || self.curr_is('!');
        let start_bpos = self.last_pos - BytePos(2);

        let mut level: isize = 1;
        let mut has_cr = false;
        while level > 0 {
            if self.is_eof() {
                let msg = if is_doc_comment {
                    "unterminated block doc-comment"
                } else {
                    "unterminated block comment"
                };
                let last_bpos = self.last_pos;
                panic!(self.fatal_span_(start_bpos, last_bpos, msg));
            }
            let n = self.curr.unwrap();
            match n {
                '/' if self.nextch_is('*') => {
                    level += 1;
                    self.bump();
                }
                '*' if self.nextch_is('/') => {
                    level -= 1;
                    self.bump();
                }
                '\r' => {
                    has_cr = true;
                }
                _ => (),
            }
            self.bump();
        }

        self.with_str_from(start_bpos, |string| {
            // but comments with only "*"s between two "/"s are not
            let tok = if is_block_doc_comment(string) {
                let string = if has_cr {
                    self.translate_crlf(start_bpos,
                                        string,
                                        "bare CR not allowed in block doc-comment")
                } else {
                    string.into()
                };
                token::DocComment(token::intern(&string[..]))
            } else {
                token::Comment
            };

            Some(TokenAndSpan {
                tok: tok,
                sp: codemap::mk_sp(start_bpos, self.last_pos),
            })
        })
    }

    /// Scan through any digits (base `scan_radix`) or underscores,
    /// and return how many digits there were.
    ///
    /// `real_radix` represents the true radix of the number we're
    /// interested in, and errors will be emitted for any digits
    /// between `real_radix` and `scan_radix`.
    fn scan_digits(&mut self, real_radix: u32, scan_radix: u32) -> usize {
        assert!(real_radix <= scan_radix);
        let mut len = 0;
        loop {
            let c = self.curr;
            if c == Some('_') {
                debug!("skipping a _");
                self.bump();
                continue;
            }
            match c.and_then(|cc| cc.to_digit(scan_radix)) {
                Some(_) => {
                    debug!("{:?} in scan_digits", c);
                    // check that the hypothetical digit is actually
                    // in range for the true radix
                    if c.unwrap().to_digit(real_radix).is_none() {
                        self.err_span_(self.last_pos,
                                       self.pos,
                                       &format!("invalid digit for a base {} literal", real_radix));
                    }
                    len += 1;
                    self.bump();
                }
                _ => return len,
            }
        }
    }

    /// Lex a LIT_INTEGER or a LIT_FLOAT
    fn scan_number(&mut self, c: char) -> token::Lit {
        let num_digits;
        let mut base = 10;
        let start_bpos = self.last_pos;

        self.bump();

        if c == '0' {
            match self.curr.unwrap_or('\0') {
                'b' => {
                    self.bump();
                    base = 2;
                    num_digits = self.scan_digits(2, 10);
                }
                'o' => {
                    self.bump();
                    base = 8;
                    num_digits = self.scan_digits(8, 10);
                }
                'x' => {
                    self.bump();
                    base = 16;
                    num_digits = self.scan_digits(16, 16);
                }
                '0'...'9' | '_' | '.' => {
                    num_digits = self.scan_digits(10, 10) + 1;
                }
                _ => {
                    // just a 0
                    return token::Integer(self.name_from(start_bpos));
                }
            }
        } else if c.is_digit(10) {
            num_digits = self.scan_digits(10, 10) + 1;
        } else {
            num_digits = 0;
        }

        if num_digits == 0 {
            self.err_span_(start_bpos,
                           self.last_pos,
                           "no valid digits found for number");
            return token::Integer(token::intern("0"));
        }

        // might be a float, but don't be greedy if this is actually an
        // integer literal followed by field/method access or a range pattern
        // (`0..2` and `12.foo()`)
        if self.curr_is('.') && !self.nextch_is('.') &&
           !self.nextch()
                .unwrap_or('\0')
                .is_xid_start() {
            // might have stuff after the ., and if it does, it needs to start
            // with a number
            self.bump();
            if self.curr.unwrap_or('\0').is_digit(10) {
                self.scan_digits(10, 10);
                self.scan_float_exponent();
            }
            let last_pos = self.last_pos;
            self.check_float_base(start_bpos, last_pos, base);
            return token::Float(self.name_from(start_bpos));
        } else {
            // it might be a float if it has an exponent
            if self.curr_is('e') || self.curr_is('E') {
                self.scan_float_exponent();
                let last_pos = self.last_pos;
                self.check_float_base(start_bpos, last_pos, base);
                return token::Float(self.name_from(start_bpos));
            }
            // but we certainly have an integer!
            return token::Integer(self.name_from(start_bpos));
        }
    }

    /// Scan over `n_digits` hex digits, stopping at `delim`, reporting an
    /// error if too many or too few digits are encountered.
    fn scan_hex_digits(&mut self, n_digits: usize, delim: char, below_0x7f_only: bool) -> bool {
        debug!("scanning {} digits until {:?}", n_digits, delim);
        let start_bpos = self.last_pos;
        let mut accum_int = 0;

        let mut valid = true;
        for _ in 0..n_digits {
            if self.is_eof() {
                let last_bpos = self.last_pos;
                panic!(self.fatal_span_(start_bpos,
                                        last_bpos,
                                        "unterminated numeric character escape"));
            }
            if self.curr_is(delim) {
                let last_bpos = self.last_pos;
                self.err_span_(start_bpos,
                               last_bpos,
                               "numeric character escape is too short");
                valid = false;
                break;
            }
            let c = self.curr.unwrap_or('\x00');
            accum_int *= 16;
            accum_int += c.to_digit(16).unwrap_or_else(|| {
                self.err_span_char(self.last_pos,
                                   self.pos,
                                   "invalid character in numeric character escape",
                                   c);

                valid = false;
                0
            });
            self.bump();
        }

        if below_0x7f_only && accum_int >= 0x80 {
            self.err_span_(start_bpos,
                           self.last_pos,
                           "this form of character escape may only be used with characters in \
                            the range [\\x00-\\x7f]");
            valid = false;
        }

        match char::from_u32(accum_int) {
            Some(_) => valid,
            None => {
                let last_bpos = self.last_pos;
                self.err_span_(start_bpos, last_bpos, "invalid numeric character escape");
                false
            }
        }
    }

    /// Scan for a single (possibly escaped) byte or char
    /// in a byte, (non-raw) byte string, char, or (non-raw) string literal.
    /// `start` is the position of `first_source_char`, which is already consumed.
    ///
    /// Returns true if there was a valid char/byte, false otherwise.
    fn scan_char_or_byte(&mut self,
                         start: BytePos,
                         first_source_char: char,
                         ascii_only: bool,
                         delim: char)
                         -> bool {
        match first_source_char {
            '\\' => {
                // '\X' for some X must be a character constant:
                let escaped = self.curr;
                let escaped_pos = self.last_pos;
                self.bump();
                match escaped {
                    None => {}  // EOF here is an error that will be checked later.
                    Some(e) => {
                        return match e {
                            'n' | 'r' | 't' | '\\' | '\'' | '"' | '0' => true,
                            'x' => self.scan_byte_escape(delim, !ascii_only),
                            'u' => {
                                let valid = if self.curr_is('{') {
                                    self.scan_unicode_escape(delim) && !ascii_only
                                } else {
                                    self.err_span_(start,
                                                   self.last_pos,
                                                   "incorrect unicode escape sequence");
                                    self.help_span_(start,
                                                    self.last_pos,
                                                    "format of unicode escape sequences is \
                                                     `\\u{…}`");
                                    false
                                };
                                if ascii_only {
                                    self.err_span_(start,
                                                   self.last_pos,
                                                   "unicode escape sequences cannot be used as a \
                                                    byte or in a byte string");
                                }
                                valid

                            }
                            '\n' if delim == '"' => {
                                self.consume_whitespace();
                                true
                            }
                            '\r' if delim == '"' && self.curr_is('\n') => {
                                self.consume_whitespace();
                                true
                            }
                            c => {
                                let last_pos = self.last_pos;
                                self.err_span_char(escaped_pos,
                                                   last_pos,
                                                   if ascii_only {
                                                       "unknown byte escape"
                                                   } else {
                                                       "unknown character escape"
                                                   },
                                                   c);
                                if e == '\r' {
                                    self.help_span_(escaped_pos,
                                                    last_pos,
                                                    "this is an isolated carriage return; \
                                                     consider checking your editor and version \
                                                     control settings")
                                }
                                if (e == '{' || e == '}') && !ascii_only {
                                    self.help_span_(escaped_pos,
                                                    last_pos,
                                                    "if used in a formatting string, curly braces \
                                                     are escaped with `{{` and `}}`")
                                }
                                false
                            }
                        }
                    }
                }
            }
            '\t' | '\n' | '\r' | '\'' if delim == '\'' => {
                let last_pos = self.last_pos;
                self.err_span_char(start,
                                   last_pos,
                                   if ascii_only {
                                       "byte constant must be escaped"
                                   } else {
                                       "character constant must be escaped"
                                   },
                                   first_source_char);
                return false;
            }
            '\r' => {
                if self.curr_is('\n') {
                    self.bump();
                    return true;
                } else {
                    self.err_span_(start,
                                   self.last_pos,
                                   "bare CR not allowed in string, use \\r instead");
                    return false;
                }
            }
            _ => {
                if ascii_only && first_source_char > '\x7F' {
                    let last_pos = self.last_pos;
                    self.err_span_char(start,
                                       last_pos,
                                       "byte constant must be ASCII. Use a \\xHH escape for a \
                                        non-ASCII byte",
                                       first_source_char);
                    return false;
                }
            }
        }
        true
    }

    /// Scan over a \u{...} escape
    ///
    /// At this point, we have already seen the \ and the u, the { is the current character. We
    /// will read at least one digit, and up to 6, and pass over the }.
    fn scan_unicode_escape(&mut self, delim: char) -> bool {
        self.bump(); // past the {
        let start_bpos = self.last_pos;
        let mut count = 0;
        let mut accum_int = 0;
        let mut valid = true;

        while !self.curr_is('}') && count <= 6 {
            let c = match self.curr {
                Some(c) => c,
                None => {
                    panic!(self.fatal_span_(start_bpos,
                                            self.last_pos,
                                            "unterminated unicode escape (found EOF)"));
                }
            };
            accum_int *= 16;
            accum_int += c.to_digit(16).unwrap_or_else(|| {
                if c == delim {
                    panic!(self.fatal_span_(self.last_pos,
                                            self.pos,
                                            "unterminated unicode escape (needed a `}`)"));
                } else {
                    self.err_span_char(self.last_pos,
                                       self.pos,
                                       "invalid character in unicode escape",
                                       c);
                }
                valid = false;
                0
            });
            self.bump();
            count += 1;
        }

        if count > 6 {
            self.err_span_(start_bpos,
                           self.last_pos,
                           "overlong unicode escape (can have at most 6 hex digits)");
            valid = false;
        }

        if valid && (char::from_u32(accum_int).is_none() || count == 0) {
            self.err_span_(start_bpos,
                           self.last_pos,
                           "invalid unicode character escape");
            valid = false;
        }

        self.bump(); // past the ending }
        valid
    }

    /// Scan over a float exponent.
    fn scan_float_exponent(&mut self) {
        if self.curr_is('e') || self.curr_is('E') {
            self.bump();
            if self.curr_is('-') || self.curr_is('+') {
                self.bump();
            }
            if self.scan_digits(10, 10) == 0 {
                self.err_span_(self.last_pos,
                               self.pos,
                               "expected at least one digit in exponent")
            }
        }
    }

    /// Check that a base is valid for a floating literal, emitting a nice
    /// error if it isn't.
    fn check_float_base(&mut self, start_bpos: BytePos, last_bpos: BytePos, base: usize) {
        match base {
            16 => {
                self.err_span_(start_bpos,
                               last_bpos,
                               "hexadecimal float literal is not supported")
            }
            8 => {
                self.err_span_(start_bpos,
                               last_bpos,
                               "octal float literal is not supported")
            }
            2 => {
                self.err_span_(start_bpos,
                               last_bpos,
                               "binary float literal is not supported")
            }
            _ => (),
        }
    }

    fn binop(&mut self, op: token::BinOpToken) -> token::Token {
        self.bump();
        if self.curr_is('=') {
            self.bump();
            return token::BinOpEq(op);
        } else {
            return token::BinOp(op);
        }
    }

    /// Return the next token from the string, advances the input past that
    /// token, and updates the interner
    fn next_token_inner(&mut self) -> token::Token {
        let c = self.curr;
        if ident_start(c) &&
           match (c.unwrap(), self.nextch(), self.nextnextch()) {
            // Note: r as in r" or r#" is part of a raw string literal,
            // b as in b' is part of a byte literal.
            // They are not identifiers, and are handled further down.
            ('r', Some('"'), _) |
            ('r', Some('#'), _) |
            ('b', Some('"'), _) |
            ('b', Some('\''), _) |
            ('b', Some('r'), Some('"')) |
            ('b', Some('r'), Some('#')) => false,
            _ => true,
        } {
            let start = self.last_pos;
            while ident_continue(self.curr) {
                self.bump();
            }

            return self.with_str_from(start, |string| {
                if string == "_" {
                    token::Underscore
                } else {
                    // FIXME: perform NFKC normalization here. (Issue #2253)
                    if self.curr_is(':') && self.nextch_is(':') {
                        token::Ident(str_to_ident(string), token::ModName)
                    } else {
                        token::Ident(str_to_ident(string), token::Plain)
                    }
                }
            });
        }

        if is_dec_digit(c) {
            let num = self.scan_number(c.unwrap());
            let suffix = self.scan_optional_raw_name();
            debug!("next_token_inner: scanned number {:?}, {:?}", num, suffix);
            return token::Literal(num, suffix);
        }

        match c.expect("next_token_inner called at EOF") {
            // One-byte tokens.
            ';' => {
                self.bump();
                return token::Semi;
            }
            ',' => {
                self.bump();
                return token::Comma;
            }
            '.' => {
                self.bump();
                return if self.curr_is('.') {
                    self.bump();
                    if self.curr_is('.') {
                        self.bump();
                        token::DotDotDot
                    } else {
                        token::DotDot
                    }
                } else {
                    token::Dot
                };
            }
            '(' => {
                self.bump();
                return token::OpenDelim(token::Paren);
            }
            ')' => {
                self.bump();
                return token::CloseDelim(token::Paren);
            }
            '{' => {
                self.bump();
                return token::OpenDelim(token::Brace);
            }
            '}' => {
                self.bump();
                return token::CloseDelim(token::Brace);
            }
            '[' => {
                self.bump();
                return token::OpenDelim(token::Bracket);
            }
            ']' => {
                self.bump();
                return token::CloseDelim(token::Bracket);
            }
            '@' => {
                self.bump();
                return token::At;
            }
            '#' => {
                self.bump();
                return token::Pound;
            }
            '~' => {
                self.bump();
                return token::Tilde;
            }
            '?' => {
                self.bump();
                return token::Question;
            }
            ':' => {
                self.bump();
                if self.curr_is(':') {
                    self.bump();
                    return token::ModSep;
                } else {
                    return token::Colon;
                }
            }

            '$' => {
                self.bump();
                return token::Dollar;
            }

            // Multi-byte tokens.
            '=' => {
                self.bump();
                if self.curr_is('=') {
                    self.bump();
                    return token::EqEq;
                } else if self.curr_is('>') {
                    self.bump();
                    return token::FatArrow;
                } else {
                    return token::Eq;
                }
            }
            '!' => {
                self.bump();
                if self.curr_is('=') {
                    self.bump();
                    return token::Ne;
                } else {
                    return token::Not;
                }
            }
            '<' => {
                self.bump();
                match self.curr.unwrap_or('\x00') {
                    '=' => {
                        self.bump();
                        return token::Le;
                    }
                    '<' => {
                        return self.binop(token::Shl);
                    }
                    '-' => {
                        self.bump();
                        match self.curr.unwrap_or('\x00') {
                            _ => {
                                return token::LArrow;
                            }
                        }
                    }
                    _ => {
                        return token::Lt;
                    }
                }
            }
            '>' => {
                self.bump();
                match self.curr.unwrap_or('\x00') {
                    '=' => {
                        self.bump();
                        return token::Ge;
                    }
                    '>' => {
                        return self.binop(token::Shr);
                    }
                    _ => {
                        return token::Gt;
                    }
                }
            }
            '\'' => {
                // Either a character constant 'a' OR a lifetime name 'abc
                self.bump();
                let start = self.last_pos;

                // the eof will be picked up by the final `'` check below
                let c2 = self.curr.unwrap_or('\x00');
                self.bump();

                // If the character is an ident start not followed by another single
                // quote, then this is a lifetime name:
                if ident_start(Some(c2)) && !self.curr_is('\'') {
                    while ident_continue(self.curr) {
                        self.bump();
                    }

                    // Include the leading `'` in the real identifier, for macro
                    // expansion purposes. See #12512 for the gory details of why
                    // this is necessary.
                    let ident = self.with_str_from(start, |lifetime_name| {
                        str_to_ident(&format!("'{}", lifetime_name))
                    });

                    // Conjure up a "keyword checking ident" to make sure that
                    // the lifetime name is not a keyword.
                    let keyword_checking_ident = self.with_str_from(start, |lifetime_name| {
                        str_to_ident(lifetime_name)
                    });
                    let keyword_checking_token = &token::Ident(keyword_checking_ident,
                                                               token::Plain);
                    let last_bpos = self.last_pos;
                    if keyword_checking_token.is_keyword(token::keywords::SelfValue) {
                        self.err_span_(start,
                                       last_bpos,
                                       "invalid lifetime name: 'self is no longer a special \
                                        lifetime");
                    } else if keyword_checking_token.is_any_keyword() &&
                       !keyword_checking_token.is_keyword(token::keywords::Static) {
                        self.err_span_(start, last_bpos, "invalid lifetime name");
                    }
                    return token::Lifetime(ident);
                }

                // Otherwise it is a character constant:
                let valid = self.scan_char_or_byte(start,
                                                   c2,
                                                   // ascii_only =
                                                   false,
                                                   '\'');
                if !self.curr_is('\'') {
                    let last_bpos = self.last_pos;
                    panic!(self.fatal_span_verbose(// Byte offsetting here is okay because the
                                                   // character before position `start` is an
                                                   // ascii single quote.
                                                   start - BytePos(1),
                                                   last_bpos,

                                                   String::from("character literal may only \
                                                                 contain one codepoint")));
                }
                let id = if valid {
                    self.name_from(start)
                } else {
                    token::intern("0")
                };
                self.bump(); // advance curr past token
                let suffix = self.scan_optional_raw_name();
                return token::Literal(token::Char(id), suffix);
            }
            'b' => {
                self.bump();
                let lit = match self.curr {
                    Some('\'') => self.scan_byte(),
                    Some('"') => self.scan_byte_string(),
                    Some('r') => self.scan_raw_byte_string(),
                    _ => unreachable!(),  // Should have been a token::Ident above.
                };
                let suffix = self.scan_optional_raw_name();
                return token::Literal(lit, suffix);
            }
            '"' => {
                let start_bpos = self.last_pos;
                let mut valid = true;
                self.bump();
                while !self.curr_is('"') {
                    if self.is_eof() {
                        let last_bpos = self.last_pos;
                        panic!(self.fatal_span_(start_bpos,
                                                last_bpos,
                                                "unterminated double quote string"));
                    }

                    let ch_start = self.last_pos;
                    let ch = self.curr.unwrap();
                    self.bump();
                    valid &= self.scan_char_or_byte(ch_start,
                                                    ch,
                                                    // ascii_only =
                                                    false,
                                                    '"');
                }
                // adjust for the ASCII " at the start of the literal
                let id = if valid {
                    self.name_from(start_bpos + BytePos(1))
                } else {
                    token::intern("??")
                };
                self.bump();
                let suffix = self.scan_optional_raw_name();
                return token::Literal(token::Str_(id), suffix);
            }
            'r' => {
                let start_bpos = self.last_pos;
                self.bump();
                let mut hash_count = 0;
                while self.curr_is('#') {
                    self.bump();
                    hash_count += 1;
                }

                if self.is_eof() {
                    let last_bpos = self.last_pos;
                    panic!(self.fatal_span_(start_bpos, last_bpos, "unterminated raw string"));
                } else if !self.curr_is('"') {
                    let last_bpos = self.last_pos;
                    let curr_char = self.curr.unwrap();
                    panic!(self.fatal_span_char(start_bpos,
                                                last_bpos,
                                                "found invalid character; only `#` is allowed \
                                                 in raw string delimitation",
                                                curr_char));
                }
                self.bump();
                let content_start_bpos = self.last_pos;
                let mut content_end_bpos;
                let mut valid = true;
                'outer: loop {
                    if self.is_eof() {
                        let last_bpos = self.last_pos;
                        panic!(self.fatal_span_(start_bpos, last_bpos, "unterminated raw string"));
                    }
                    // if self.curr_is('"') {
                    // content_end_bpos = self.last_pos;
                    // for _ in 0..hash_count {
                    // self.bump();
                    // if !self.curr_is('#') {
                    // continue 'outer;
                    let c = self.curr.unwrap();
                    match c {
                        '"' => {
                            content_end_bpos = self.last_pos;
                            for _ in 0..hash_count {
                                self.bump();
                                if !self.curr_is('#') {
                                    continue 'outer;
                                }
                            }
                            break;
                        }
                        '\r' => {
                            if !self.nextch_is('\n') {
                                let last_bpos = self.last_pos;
                                self.err_span_(start_bpos,
                                               last_bpos,
                                               "bare CR not allowed in raw string, use \\r \
                                                instead");
                                valid = false;
                            }
                        }
                        _ => (),
                    }
                    self.bump();
                }
                self.bump();
                let id = if valid {
                    self.name_from_to(content_start_bpos, content_end_bpos)
                } else {
                    token::intern("??")
                };
                let suffix = self.scan_optional_raw_name();
                return token::Literal(token::StrRaw(id, hash_count), suffix);
            }
            '-' => {
                if self.nextch_is('>') {
                    self.bump();
                    self.bump();
                    return token::RArrow;
                } else {
                    return self.binop(token::Minus);
                }
            }
            '&' => {
                if self.nextch_is('&') {
                    self.bump();
                    self.bump();
                    return token::AndAnd;
                } else {
                    return self.binop(token::And);
                }
            }
            '|' => {
                match self.nextch() {
                    Some('|') => {
                        self.bump();
                        self.bump();
                        return token::OrOr;
                    }
                    _ => {
                        return self.binop(token::Or);
                    }
                }
            }
            '+' => {
                return self.binop(token::Plus);
            }
            '*' => {
                return self.binop(token::Star);
            }
            '/' => {
                return self.binop(token::Slash);
            }
            '^' => {
                return self.binop(token::Caret);
            }
            '%' => {
                return self.binop(token::Percent);
            }
            c => {
                let last_bpos = self.last_pos;
                let bpos = self.pos;
                unicode_chars::check_for_substitution(&self, c);
                panic!(self.fatal_span_char(last_bpos, bpos, "unknown start of token", c))
            }
        }
    }

    fn consume_whitespace(&mut self) {
        while is_whitespace(self.curr) && !self.is_eof() {
            self.bump();
        }
    }

    fn read_to_eol(&mut self) -> String {
        let mut val = String::new();
        while !self.curr_is('\n') && !self.is_eof() {
            val.push(self.curr.unwrap());
            self.bump();
        }
        if self.curr_is('\n') {
            self.bump();
        }
        return val;
    }

    fn read_one_line_comment(&mut self) -> String {
        let val = self.read_to_eol();
        assert!((val.as_bytes()[0] == b'/' && val.as_bytes()[1] == b'/') ||
                (val.as_bytes()[0] == b'#' && val.as_bytes()[1] == b'!'));
        return val;
    }

    fn consume_non_eol_whitespace(&mut self) {
        while is_whitespace(self.curr) && !self.curr_is('\n') && !self.is_eof() {
            self.bump();
        }
    }

    fn peeking_at_comment(&self) -> bool {
        (self.curr_is('/') && self.nextch_is('/')) || (self.curr_is('/') && self.nextch_is('*')) ||
        (self.curr_is('#') && self.nextch_is('!') && !self.nextnextch_is('['))
    }

    fn scan_byte(&mut self) -> token::Lit {
        self.bump();
        let start = self.last_pos;

        // the eof will be picked up by the final `'` check below
        let c2 = self.curr.unwrap_or('\x00');
        self.bump();

        let valid = self.scan_char_or_byte(start,
                                           c2,
                                           // ascii_only =
                                           true,
                                           '\'');
        if !self.curr_is('\'') {
            // Byte offsetting here is okay because the
            // character before position `start` are an
            // ascii single quote and ascii 'b'.
            let last_pos = self.last_pos;
            panic!(self.fatal_span_verbose(start - BytePos(2),
                                           last_pos,
                                           "unterminated byte constant".to_string()));
        }

        let id = if valid {
            self.name_from(start)
        } else {
            token::intern("?")
        };
        self.bump(); // advance curr past token
        return token::Byte(id);
    }

    fn scan_byte_escape(&mut self, delim: char, below_0x7f_only: bool) -> bool {
        self.scan_hex_digits(2, delim, below_0x7f_only)
    }

    fn scan_byte_string(&mut self) -> token::Lit {
        self.bump();
        let start = self.last_pos;
        let mut valid = true;

        while !self.curr_is('"') {
            if self.is_eof() {
                let last_pos = self.last_pos;
                panic!(self.fatal_span_(start, last_pos, "unterminated double quote byte string"));
            }

            let ch_start = self.last_pos;
            let ch = self.curr.unwrap();
            self.bump();
            valid &= self.scan_char_or_byte(ch_start,
                                            ch,
                                            // ascii_only =
                                            true,
                                            '"');
        }
        let id = if valid {
            self.name_from(start)
        } else {
            token::intern("??")
        };
        self.bump();
        return token::ByteStr(id);
    }

    fn scan_raw_byte_string(&mut self) -> token::Lit {
        let start_bpos = self.last_pos;
        self.bump();
        let mut hash_count = 0;
        while self.curr_is('#') {
            self.bump();
            hash_count += 1;
        }

        if self.is_eof() {
            let last_pos = self.last_pos;
            panic!(self.fatal_span_(start_bpos, last_pos, "unterminated raw string"));
        } else if !self.curr_is('"') {
            let last_pos = self.last_pos;
            let ch = self.curr.unwrap();
            panic!(self.fatal_span_char(start_bpos,
                                        last_pos,
                                        "found invalid character; only `#` is allowed in raw \
                                         string delimitation",
                                        ch));
        }
        self.bump();
        let content_start_bpos = self.last_pos;
        let mut content_end_bpos;
        'outer: loop {
            match self.curr {
                None => {
                    let last_pos = self.last_pos;
                    panic!(self.fatal_span_(start_bpos, last_pos, "unterminated raw string"))
                }
                Some('"') => {
                    content_end_bpos = self.last_pos;
                    for _ in 0..hash_count {
                        self.bump();
                        if !self.curr_is('#') {
                            continue 'outer;
                        }
                    }
                    break;
                }
                Some(c) => {
                    if c > '\x7F' {
                        let last_pos = self.last_pos;
                        self.err_span_char(last_pos, last_pos, "raw byte string must be ASCII", c);
                    }
                }
            }
            self.bump();
        }
        self.bump();
        return token::ByteStrRaw(self.name_from_to(content_start_bpos, content_end_bpos),
                                 hash_count);
    }
}

pub fn is_whitespace(c: Option<char>) -> bool {
    match c.unwrap_or('\x00') { // None can be null for now... it's not whitespace
        ' ' | '\n' | '\t' | '\r' => true,
        _ => false,
    }
}

fn in_range(c: Option<char>, lo: char, hi: char) -> bool {
    match c {
        Some(c) => lo <= c && c <= hi,
        _ => false,
    }
}

fn is_dec_digit(c: Option<char>) -> bool {
    return in_range(c, '0', '9');
}

pub fn is_doc_comment(s: &str) -> bool {
    let res = (s.starts_with("///") && *s.as_bytes().get(3).unwrap_or(&b' ') != b'/') ||
              s.starts_with("//!");
    debug!("is {:?} a doc comment? {}", s, res);
    res
}

pub fn is_block_doc_comment(s: &str) -> bool {
    // Prevent `/**/` from being parsed as a doc comment
    let res = ((s.starts_with("/**") && *s.as_bytes().get(3).unwrap_or(&b' ') != b'*') ||
               s.starts_with("/*!")) && s.len() >= 5;
    debug!("is {:?} a doc comment? {}", s, res);
    res
}

fn ident_start(c: Option<char>) -> bool {
    let c = match c {
        Some(c) => c,
        None => return false,
    };

    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || (c > '\x7f' && c.is_xid_start())
}

fn ident_continue(c: Option<char>) -> bool {
    let c = match c {
        Some(c) => c,
        None => return false,
    };

    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
    (c > '\x7f' && c.is_xid_continue())
}

#[cfg(test)]
mod tests {
    use super::*;

    use codemap::{BytePos, CodeMap, Span, NO_EXPANSION};
    use diagnostic;
    use parse::token;
    use parse::token::str_to_ident;
    use std::io;

    fn mk_sh() -> diagnostic::SpanHandler {
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        let emitter = diagnostic::EmitterWriter::new(Box::new(io::sink()), None);
        let handler = diagnostic::Handler::with_emitter(true, Box::new(emitter));
        diagnostic::SpanHandler::new(handler, CodeMap::new())
    }

    // open a string reader for the given string
    fn setup<'a>(span_handler: &'a diagnostic::SpanHandler, teststr: String) -> StringReader<'a> {
        let fm = span_handler.cm.new_filemap("zebra.rs".to_string(), teststr);
        StringReader::new(span_handler, fm)
    }

    #[test]
    fn t1() {
        let span_handler = mk_sh();
        let mut string_reader = setup(&span_handler,
                                      "/* my source file */ fn main() { println!(\"zebra\"); }\n"
                                          .to_string());
        let id = str_to_ident("fn");
        assert_eq!(string_reader.next_token().tok, token::Comment);
        assert_eq!(string_reader.next_token().tok, token::Whitespace);
        let tok1 = string_reader.next_token();
        let tok2 = TokenAndSpan {
            tok: token::Ident(id, token::Plain),
            sp: Span {
                lo: BytePos(21),
                hi: BytePos(23),
                expn_id: NO_EXPANSION,
            },
        };
        assert_eq!(tok1, tok2);
        assert_eq!(string_reader.next_token().tok, token::Whitespace);
        // the 'main' id is already read:
        assert_eq!(string_reader.last_pos.clone(), BytePos(28));
        // read another token:
        let tok3 = string_reader.next_token();
        let tok4 = TokenAndSpan {
            tok: token::Ident(str_to_ident("main"), token::Plain),
            sp: Span {
                lo: BytePos(24),
                hi: BytePos(28),
                expn_id: NO_EXPANSION,
            },
        };
        assert_eq!(tok3, tok4);
        // the lparen is already read:
        assert_eq!(string_reader.last_pos.clone(), BytePos(29))
    }

    // check that the given reader produces the desired stream
    // of tokens (stop checking after exhausting the expected vec)
    fn check_tokenization(mut string_reader: StringReader, expected: Vec<token::Token>) {
        for expected_tok in &expected {
            assert_eq!(&string_reader.next_token().tok, expected_tok);
        }
    }

    // make the identifier by looking up the string in the interner
    fn mk_ident(id: &str, style: token::IdentStyle) -> token::Token {
        token::Ident(str_to_ident(id), style)
    }

    #[test]
    fn doublecolonparsing() {
        check_tokenization(setup(&mk_sh(), "a b".to_string()),
                           vec![mk_ident("a", token::Plain),
                                token::Whitespace,
                                mk_ident("b", token::Plain)]);
    }

    #[test]
    fn dcparsing_2() {
        check_tokenization(setup(&mk_sh(), "a::b".to_string()),
                           vec![mk_ident("a", token::ModName),
                                token::ModSep,
                                mk_ident("b", token::Plain)]);
    }

    #[test]
    fn dcparsing_3() {
        check_tokenization(setup(&mk_sh(), "a ::b".to_string()),
                           vec![mk_ident("a", token::Plain),
                                token::Whitespace,
                                token::ModSep,
                                mk_ident("b", token::Plain)]);
    }

    #[test]
    fn dcparsing_4() {
        check_tokenization(setup(&mk_sh(), "a:: b".to_string()),
                           vec![mk_ident("a", token::ModName),
                                token::ModSep,
                                token::Whitespace,
                                mk_ident("b", token::Plain)]);
    }

    #[test]
    fn character_a() {
        assert_eq!(setup(&mk_sh(), "'a'".to_string()).next_token().tok,
                   token::Literal(token::Char(token::intern("a")), None));
    }

    #[test]
    fn character_space() {
        assert_eq!(setup(&mk_sh(), "' '".to_string()).next_token().tok,
                   token::Literal(token::Char(token::intern(" ")), None));
    }

    #[test]
    fn character_escaped() {
        assert_eq!(setup(&mk_sh(), "'\\n'".to_string()).next_token().tok,
                   token::Literal(token::Char(token::intern("\\n")), None));
    }

    #[test]
    fn lifetime_name() {
        assert_eq!(setup(&mk_sh(), "'abc".to_string()).next_token().tok,
                   token::Lifetime(token::str_to_ident("'abc")));
    }

    #[test]
    fn raw_string() {
        assert_eq!(setup(&mk_sh(), "r###\"\"#a\\b\x00c\"\"###".to_string())
                       .next_token()
                       .tok,
                   token::Literal(token::StrRaw(token::intern("\"#a\\b\x00c\""), 3), None));
    }

    #[test]
    fn literal_suffixes() {
        macro_rules! test {
            ($input: expr, $tok_type: ident, $tok_contents: expr) => {{
                assert_eq!(setup(&mk_sh(), format!("{}suffix", $input)).next_token().tok,
                           token::Literal(token::$tok_type(token::intern($tok_contents)),
                                          Some(token::intern("suffix"))));
                // with a whitespace separator:
                assert_eq!(setup(&mk_sh(), format!("{} suffix", $input)).next_token().tok,
                           token::Literal(token::$tok_type(token::intern($tok_contents)),
                                          None));
            }}
        }

        test!("'a'", Char, "a");
        test!("b'a'", Byte, "a");
        test!("\"a\"", Str_, "a");
        test!("b\"a\"", ByteStr, "a");
        test!("1234", Integer, "1234");
        test!("0b101", Integer, "0b101");
        test!("0xABC", Integer, "0xABC");
        test!("1.0", Float, "1.0");
        test!("1.0e10", Float, "1.0e10");

        assert_eq!(setup(&mk_sh(), "2us".to_string()).next_token().tok,
                   token::Literal(token::Integer(token::intern("2")),
                                  Some(token::intern("us"))));
        assert_eq!(setup(&mk_sh(), "r###\"raw\"###suffix".to_string()).next_token().tok,
                   token::Literal(token::StrRaw(token::intern("raw"), 3),
                                  Some(token::intern("suffix"))));
        assert_eq!(setup(&mk_sh(), "br###\"raw\"###suffix".to_string()).next_token().tok,
                   token::Literal(token::ByteStrRaw(token::intern("raw"), 3),
                                  Some(token::intern("suffix"))));
    }

    #[test]
    fn line_doc_comments() {
        assert!(is_doc_comment("///"));
        assert!(is_doc_comment("/// blah"));
        assert!(!is_doc_comment("////"));
    }

    #[test]
    fn nested_block_comments() {
        let sh = mk_sh();
        let mut lexer = setup(&sh, "/* /* */ */'a'".to_string());
        match lexer.next_token().tok {
            token::Comment => {}
            _ => panic!("expected a comment!"),
        }
        assert_eq!(lexer.next_token().tok,
                   token::Literal(token::Char(token::intern("a")), None));
    }

    #[test]
    fn crlf_comments() {
        let sh = mk_sh();
        let mut lexer = setup(&sh, "// test\r\n/// test\r\n".to_string());
        let comment = lexer.next_token();
        assert_eq!(comment.tok, token::Comment);
        assert_eq!(comment.sp, ::codemap::mk_sp(BytePos(0), BytePos(7)));
        assert_eq!(lexer.next_token().tok, token::Whitespace);
        assert_eq!(lexer.next_token().tok,
                   token::DocComment(token::intern("/// test")));
    }
}
