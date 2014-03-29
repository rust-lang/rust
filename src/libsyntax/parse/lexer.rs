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
use diagnostic::SpanHandler;
use ext::tt::transcribe::tt_next_token;
use parse::token;
use parse::token::{str_to_ident};

use std::char;
use std::mem::replace;
use std::num::from_str_radix;
use std::rc::Rc;
use std::str;

pub use ext::tt::transcribe::{TtReader, new_tt_reader};

pub trait Reader {
    fn is_eof(&self) -> bool;
    fn next_token(&mut self) -> TokenAndSpan;
    fn fatal(&self, ~str) -> !;
    fn span_diag<'a>(&'a self) -> &'a SpanHandler;
    fn peek(&self) -> TokenAndSpan;
}

#[deriving(Clone, Eq, Show)]
pub struct TokenAndSpan {
    tok: token::Token,
    sp: Span,
}

pub struct StringReader<'a> {
    span_diagnostic: &'a SpanHandler,
    // The absolute offset within the codemap of the next character to read
    pos: BytePos,
    // The absolute offset within the codemap of the last character read(curr)
    last_pos: BytePos,
    // The column of the next character to read
    col: CharPos,
    // The last character to be read
    curr: Option<char>,
    filemap: Rc<codemap::FileMap>,
    /* cached: */
    peek_tok: token::Token,
    peek_span: Span,
}

impl<'a> StringReader<'a> {
    pub fn curr_is(&self, c: char) -> bool {
        self.curr == Some(c)
    }
}

pub fn new_string_reader<'a>(span_diagnostic: &'a SpanHandler,
                             filemap: Rc<codemap::FileMap>)
                             -> StringReader<'a> {
    let mut r = new_low_level_string_reader(span_diagnostic, filemap);
    string_advance_token(&mut r); /* fill in peek_* */
    r
}

/* For comments.rs, which hackily pokes into 'pos' and 'curr' */
pub fn new_low_level_string_reader<'a>(span_diagnostic: &'a SpanHandler,
                                       filemap: Rc<codemap::FileMap>)
                                       -> StringReader<'a> {
    // Force the initial reader bump to start on a fresh line
    let initial_char = '\n';
    let mut r = StringReader {
        span_diagnostic: span_diagnostic,
        pos: filemap.start_pos,
        last_pos: filemap.start_pos,
        col: CharPos(0),
        curr: Some(initial_char),
        filemap: filemap,
        /* dummy values; not read */
        peek_tok: token::EOF,
        peek_span: codemap::DUMMY_SP,
    };
    bump(&mut r);
    r
}

impl<'a> Reader for StringReader<'a> {
    fn is_eof(&self) -> bool { is_eof(self) }
    // return the next token. EFFECT: advances the string_reader.
    fn next_token(&mut self) -> TokenAndSpan {
        let ret_val = TokenAndSpan {
            tok: replace(&mut self.peek_tok, token::UNDERSCORE),
            sp: self.peek_span,
        };
        string_advance_token(self);
        ret_val
    }
    fn fatal(&self, m: ~str) -> ! {
        self.span_diagnostic.span_fatal(self.peek_span, m)
    }
    fn span_diag<'a>(&'a self) -> &'a SpanHandler { self.span_diagnostic }
    fn peek(&self) -> TokenAndSpan {
        // FIXME(pcwalton): Bad copy!
        TokenAndSpan {
            tok: self.peek_tok.clone(),
            sp: self.peek_span.clone(),
        }
    }
}

impl<'a> Reader for TtReader<'a> {
    fn is_eof(&self) -> bool {
        self.cur_tok == token::EOF
    }
    fn next_token(&mut self) -> TokenAndSpan {
        let r = tt_next_token(self);
        debug!("TtReader: r={:?}", r);
        r
    }
    fn fatal(&self, m: ~str) -> ! {
        self.sp_diag.span_fatal(self.cur_span, m);
    }
    fn span_diag<'a>(&'a self) -> &'a SpanHandler { self.sp_diag }
    fn peek(&self) -> TokenAndSpan {
        TokenAndSpan {
            tok: self.cur_tok.clone(),
            sp: self.cur_span.clone(),
        }
    }
}

// report a lexical error spanning [`from_pos`, `to_pos`)
fn fatal_span(rdr: &mut StringReader,
              from_pos: BytePos,
              to_pos: BytePos,
              m: ~str)
           -> ! {
    rdr.peek_span = codemap::mk_sp(from_pos, to_pos);
    rdr.fatal(m);
}

// report a lexical error spanning [`from_pos`, `to_pos`), appending an
// escaped character to the error message
fn fatal_span_char(rdr: &mut StringReader,
                   from_pos: BytePos,
                   to_pos: BytePos,
                   m: ~str,
                   c: char)
                -> ! {
    let mut m = m;
    m.push_str(": ");
    char::escape_default(c, |c| m.push_char(c));
    fatal_span(rdr, from_pos, to_pos, m);
}

// report a lexical error spanning [`from_pos`, `to_pos`), appending the
// offending string to the error message
fn fatal_span_verbose(rdr: &mut StringReader,
                      from_pos: BytePos,
                      to_pos: BytePos,
                      m: ~str)
                   -> ! {
    let mut m = m;
    m.push_str(": ");
    let from = byte_offset(rdr, from_pos).to_uint();
    let to = byte_offset(rdr, to_pos).to_uint();
    m.push_str(rdr.filemap.src.slice(from, to));
    fatal_span(rdr, from_pos, to_pos, m);
}

// EFFECT: advance peek_tok and peek_span to refer to the next token.
// EFFECT: update the interner, maybe.
fn string_advance_token(r: &mut StringReader) {
    match consume_whitespace_and_comments(r) {
        Some(comment) => {
            r.peek_span = comment.sp;
            r.peek_tok = comment.tok;
        },
        None => {
            if is_eof(r) {
                r.peek_tok = token::EOF;
            } else {
                let start_bytepos = r.last_pos;
                r.peek_tok = next_token_inner(r);
                r.peek_span = codemap::mk_sp(start_bytepos,
                                             r.last_pos);
            };
        }
    }
}

fn byte_offset(rdr: &StringReader, pos: BytePos) -> BytePos {
    (pos - rdr.filemap.start_pos)
}

/// Calls `f` with a string slice of the source text spanning from `start`
/// up to but excluding `rdr.last_pos`, meaning the slice does not include
/// the character `rdr.curr`.
pub fn with_str_from<T>(
                     rdr: &StringReader,
                     start: BytePos,
                     f: |s: &str| -> T)
                     -> T {
    with_str_from_to(rdr, start, rdr.last_pos, f)
}

/// Calls `f` with astring slice of the source text spanning from `start`
/// up to but excluding `end`.
fn with_str_from_to<T>(
                    rdr: &StringReader,
                    start: BytePos,
                    end: BytePos,
                    f: |s: &str| -> T)
                    -> T {
    f(rdr.filemap.src.slice(
            byte_offset(rdr, start).to_uint(),
            byte_offset(rdr, end).to_uint()))
}

// EFFECT: advance the StringReader by one character. If a newline is
// discovered, add it to the FileMap's list of line start offsets.
pub fn bump(rdr: &mut StringReader) {
    rdr.last_pos = rdr.pos;
    let current_byte_offset = byte_offset(rdr, rdr.pos).to_uint();
    if current_byte_offset < rdr.filemap.src.len() {
        assert!(rdr.curr.is_some());
        let last_char = rdr.curr.unwrap();
        let next = rdr.filemap.src.char_range_at(current_byte_offset);
        let byte_offset_diff = next.next - current_byte_offset;
        rdr.pos = rdr.pos + Pos::from_uint(byte_offset_diff);
        rdr.curr = Some(next.ch);
        rdr.col = rdr.col + CharPos(1u);
        if last_char == '\n' {
            rdr.filemap.next_line(rdr.last_pos);
            rdr.col = CharPos(0u);
        }

        if byte_offset_diff > 1 {
            rdr.filemap.record_multibyte_char(rdr.last_pos, byte_offset_diff);
        }
    } else {
        rdr.curr = None;
    }
}

pub fn is_eof(rdr: &StringReader) -> bool {
    rdr.curr.is_none()
}

pub fn nextch(rdr: &StringReader) -> Option<char> {
    let offset = byte_offset(rdr, rdr.pos).to_uint();
    if offset < rdr.filemap.src.len() {
        Some(rdr.filemap.src.char_at(offset))
    } else {
        None
    }
}
pub fn nextch_is(rdr: &StringReader, c: char) -> bool {
    nextch(rdr) == Some(c)
}

pub fn nextnextch(rdr: &StringReader) -> Option<char> {
    let offset = byte_offset(rdr, rdr.pos).to_uint();
    let s = rdr.filemap.deref().src.as_slice();
    if offset >= s.len() { return None }
    let str::CharRange { next, .. } = s.char_range_at(offset);
    if next < s.len() {
        Some(s.char_at(next))
    } else {
        None
    }
}
pub fn nextnextch_is(rdr: &StringReader, c: char) -> bool {
    nextnextch(rdr) == Some(c)
}

fn hex_digit_val(c: Option<char>) -> int {
    let d = c.unwrap_or('\x00');

    if in_range(c, '0', '9') { return (d as int) - ('0' as int); }
    if in_range(c, 'a', 'f') { return (d as int) - ('a' as int) + 10; }
    if in_range(c, 'A', 'F') { return (d as int) - ('A' as int) + 10; }
    fail!();
}

pub fn is_whitespace(c: Option<char>) -> bool {
    match c.unwrap_or('\x00') { // None can be null for now... it's not whitespace
        ' ' | '\n' | '\t' | '\r' => true,
        _ => false
    }
}

fn in_range(c: Option<char>, lo: char, hi: char) -> bool {
    match c {
        Some(c) => lo <= c && c <= hi,
        _ => false
    }
}

fn is_dec_digit(c: Option<char>) -> bool { return in_range(c, '0', '9'); }

fn is_hex_digit(c: Option<char>) -> bool {
    return in_range(c, '0', '9') || in_range(c, 'a', 'f') ||
            in_range(c, 'A', 'F');
}

// EFFECT: eats whitespace and comments.
// returns a Some(sugared-doc-attr) if one exists, None otherwise.
fn consume_whitespace_and_comments(rdr: &mut StringReader)
                                -> Option<TokenAndSpan> {
    while is_whitespace(rdr.curr) { bump(rdr); }
    return consume_any_line_comment(rdr);
}

pub fn is_line_non_doc_comment(s: &str) -> bool {
    s.starts_with("////")
}

// PRECONDITION: rdr.curr is not whitespace
// EFFECT: eats any kind of comment.
// returns a Some(sugared-doc-attr) if one exists, None otherwise
fn consume_any_line_comment(rdr: &mut StringReader)
                         -> Option<TokenAndSpan> {
    if rdr.curr_is('/') {
        match nextch(rdr) {
            Some('/') => {
                bump(rdr);
                bump(rdr);
                // line comments starting with "///" or "//!" are doc-comments
                if rdr.curr_is('/') || rdr.curr_is('!') {
                    let start_bpos = rdr.pos - BytePos(3);
                    while !rdr.curr_is('\n') && !is_eof(rdr) {
                        bump(rdr);
                    }
                    let ret = with_str_from(rdr, start_bpos, |string| {
                        // but comments with only more "/"s are not
                        if !is_line_non_doc_comment(string) {
                            Some(TokenAndSpan{
                                tok: token::DOC_COMMENT(str_to_ident(string)),
                                sp: codemap::mk_sp(start_bpos, rdr.pos)
                            })
                        } else {
                            None
                        }
                    });

                    if ret.is_some() {
                        return ret;
                    }
                } else {
                    while !rdr.curr_is('\n') && !is_eof(rdr) { bump(rdr); }
                }
                // Restart whitespace munch.
                consume_whitespace_and_comments(rdr)
            }
            Some('*') => { bump(rdr); bump(rdr); consume_block_comment(rdr) }
            _ => None
        }
    } else if rdr.curr_is('#') {
        if nextch_is(rdr, '!') {

            // Parse an inner attribute.
            if nextnextch_is(rdr, '[') {
                return None;
            }

            // I guess this is the only way to figure out if
            // we're at the beginning of the file...
            let cmap = CodeMap::new();
            cmap.files.borrow_mut().push(rdr.filemap.clone());
            let loc = cmap.lookup_char_pos_adj(rdr.last_pos);
            if loc.line == 1u && loc.col == CharPos(0u) {
                while !rdr.curr_is('\n') && !is_eof(rdr) { bump(rdr); }
                return consume_whitespace_and_comments(rdr);
            }
        }
        None
    } else {
        None
    }
}

pub fn is_block_non_doc_comment(s: &str) -> bool {
    s.starts_with("/***")
}

// might return a sugared-doc-attr
fn consume_block_comment(rdr: &mut StringReader) -> Option<TokenAndSpan> {
    // block comments starting with "/**" or "/*!" are doc-comments
    let is_doc_comment = rdr.curr_is('*') || rdr.curr_is('!');
    let start_bpos = rdr.pos - BytePos(if is_doc_comment {3} else {2});

    let mut level: int = 1;
    while level > 0 {
        if is_eof(rdr) {
            let msg = if is_doc_comment {
                ~"unterminated block doc-comment"
            } else {
                ~"unterminated block comment"
            };
            fatal_span(rdr, start_bpos, rdr.last_pos, msg);
        } else if rdr.curr_is('/') && nextch_is(rdr, '*') {
            level += 1;
            bump(rdr);
            bump(rdr);
        } else if rdr.curr_is('*') && nextch_is(rdr, '/') {
            level -= 1;
            bump(rdr);
            bump(rdr);
        } else {
            bump(rdr);
        }
    }

    let res = if is_doc_comment {
        with_str_from(rdr, start_bpos, |string| {
            // but comments with only "*"s between two "/"s are not
            if !is_block_non_doc_comment(string) {
                Some(TokenAndSpan{
                        tok: token::DOC_COMMENT(str_to_ident(string)),
                        sp: codemap::mk_sp(start_bpos, rdr.pos)
                    })
            } else {
                None
            }
        })
    } else {
        None
    };

    // restart whitespace munch.
    if res.is_some() { res } else { consume_whitespace_and_comments(rdr) }
}

fn scan_exponent(rdr: &mut StringReader, start_bpos: BytePos) -> Option<~str> {
    // \x00 hits the `return None` case immediately, so this is fine.
    let mut c = rdr.curr.unwrap_or('\x00');
    let mut rslt = ~"";
    if c == 'e' || c == 'E' {
        rslt.push_char(c);
        bump(rdr);
        c = rdr.curr.unwrap_or('\x00');
        if c == '-' || c == '+' {
            rslt.push_char(c);
            bump(rdr);
        }
        let exponent = scan_digits(rdr, 10u);
        if exponent.len() > 0u {
            return Some(rslt + exponent);
        } else {
            fatal_span(rdr, start_bpos, rdr.last_pos,
                       ~"scan_exponent: bad fp literal");
        }
    } else { return None::<~str>; }
}

fn scan_digits(rdr: &mut StringReader, radix: uint) -> ~str {
    let mut rslt = ~"";
    loop {
        let c = rdr.curr;
        if c == Some('_') { bump(rdr); continue; }
        match c.and_then(|cc| char::to_digit(cc, radix)) {
          Some(_) => {
            rslt.push_char(c.unwrap());
            bump(rdr);
          }
          _ => return rslt
        }
    };
}

fn check_float_base(rdr: &mut StringReader, start_bpos: BytePos, last_bpos: BytePos,
                    base: uint) {
    match base {
      16u => fatal_span(rdr, start_bpos, last_bpos,
                      ~"hexadecimal float literal is not supported"),
      8u => fatal_span(rdr, start_bpos, last_bpos,
                     ~"octal float literal is not supported"),
      2u => fatal_span(rdr, start_bpos, last_bpos,
                     ~"binary float literal is not supported"),
      _ => ()
    }
}

fn scan_number(c: char, rdr: &mut StringReader) -> token::Token {
    let mut num_str;
    let mut base = 10u;
    let mut c = c;
    let mut n = nextch(rdr).unwrap_or('\x00');
    let start_bpos = rdr.last_pos;
    if c == '0' && n == 'x' {
        bump(rdr);
        bump(rdr);
        base = 16u;
    } else if c == '0' && n == 'o' {
        bump(rdr);
        bump(rdr);
        base = 8u;
    } else if c == '0' && n == 'b' {
        bump(rdr);
        bump(rdr);
        base = 2u;
    }
    num_str = scan_digits(rdr, base);
    c = rdr.curr.unwrap_or('\x00');
    nextch(rdr);
    if c == 'u' || c == 'i' {
        enum Result { Signed(ast::IntTy), Unsigned(ast::UintTy) }
        let signed = c == 'i';
        let mut tp = {
            if signed { Signed(ast::TyI) }
            else { Unsigned(ast::TyU) }
        };
        bump(rdr);
        c = rdr.curr.unwrap_or('\x00');
        if c == '8' {
            bump(rdr);
            tp = if signed { Signed(ast::TyI8) }
                      else { Unsigned(ast::TyU8) };
        }
        n = nextch(rdr).unwrap_or('\x00');
        if c == '1' && n == '6' {
            bump(rdr);
            bump(rdr);
            tp = if signed { Signed(ast::TyI16) }
                      else { Unsigned(ast::TyU16) };
        } else if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            tp = if signed { Signed(ast::TyI32) }
                      else { Unsigned(ast::TyU32) };
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
            tp = if signed { Signed(ast::TyI64) }
                      else { Unsigned(ast::TyU64) };
        }
        if num_str.len() == 0u {
            fatal_span(rdr, start_bpos, rdr.last_pos,
                       ~"no valid digits found for number");
        }
        let parsed = match from_str_radix::<u64>(num_str, base as uint) {
            Some(p) => p,
            None => fatal_span(rdr, start_bpos, rdr.last_pos,
                               ~"int literal is too large")
        };

        match tp {
          Signed(t) => return token::LIT_INT(parsed as i64, t),
          Unsigned(t) => return token::LIT_UINT(parsed, t)
        }
    }
    let mut is_float = false;
    if rdr.curr_is('.') && !(ident_start(nextch(rdr)) || nextch_is(rdr, '.')) {
        is_float = true;
        bump(rdr);
        let dec_part = scan_digits(rdr, 10u);
        num_str.push_char('.');
        num_str.push_str(dec_part);
    }
    match scan_exponent(rdr, start_bpos) {
      Some(ref s) => {
        is_float = true;
        num_str.push_str(*s);
      }
      None => ()
    }

    if rdr.curr_is('f') {
        bump(rdr);
        c = rdr.curr.unwrap_or('\x00');
        n = nextch(rdr).unwrap_or('\x00');
        if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            check_float_base(rdr, start_bpos, rdr.last_pos, base);
            return token::LIT_FLOAT(str_to_ident(num_str), ast::TyF32);
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
            check_float_base(rdr, start_bpos, rdr.last_pos, base);
            return token::LIT_FLOAT(str_to_ident(num_str), ast::TyF64);
            /* FIXME (#2252): if this is out of range for either a
            32-bit or 64-bit float, it won't be noticed till the
            back-end.  */
        } else {
            fatal_span(rdr, start_bpos, rdr.last_pos,
                       ~"expected `f32` or `f64` suffix");
        }
    }
    if is_float {
        check_float_base(rdr, start_bpos, rdr.last_pos, base);
        return token::LIT_FLOAT_UNSUFFIXED(str_to_ident(num_str));
    } else {
        if num_str.len() == 0u {
            fatal_span(rdr, start_bpos, rdr.last_pos,
                       ~"no valid digits found for number");
        }
        let parsed = match from_str_radix::<u64>(num_str, base as uint) {
            Some(p) => p,
            None => fatal_span(rdr, start_bpos, rdr.last_pos,
                               ~"int literal is too large")
        };

        debug!("lexing {} as an unsuffixed integer literal", num_str);
        return token::LIT_INT_UNSUFFIXED(parsed as i64);
    }
}

fn scan_numeric_escape(rdr: &mut StringReader, n_hex_digits: uint) -> char {
    let mut accum_int = 0;
    let mut i = n_hex_digits;
    let start_bpos = rdr.last_pos;
    while i != 0u && !is_eof(rdr) {
        let n = rdr.curr;
        if !is_hex_digit(n) {
            fatal_span_char(rdr, rdr.last_pos, rdr.pos,
                            ~"illegal character in numeric character escape",
                            n.unwrap());
        }
        bump(rdr);
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        i -= 1u;
    }
    if i != 0 && is_eof(rdr) {
        fatal_span(rdr, start_bpos, rdr.last_pos,
                   ~"unterminated numeric character escape");
    }

    match char::from_u32(accum_int as u32) {
        Some(x) => x,
        None => fatal_span(rdr, start_bpos, rdr.last_pos,
                           ~"illegal numeric character escape")
    }
}

fn ident_start(c: Option<char>) -> bool {
    let c = match c { Some(c) => c, None => return false };

    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && char::is_XID_start(c))
}

fn ident_continue(c: Option<char>) -> bool {
    let c = match c { Some(c) => c, None => return false };

    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9')
        || c == '_'
        || (c > '\x7f' && char::is_XID_continue(c))
}

// return the next token from the string
// EFFECT: advances the input past that token
// EFFECT: updates the interner
fn next_token_inner(rdr: &mut StringReader) -> token::Token {
    let c = rdr.curr;
    if ident_start(c) && !nextch_is(rdr, '"') && !nextch_is(rdr, '#') {
        // Note: r as in r" or r#" is part of a raw string literal,
        // not an identifier, and is handled further down.

        let start = rdr.last_pos;
        while ident_continue(rdr.curr) {
            bump(rdr);
        }

        return with_str_from(rdr, start, |string| {
            if string == "_" {
                token::UNDERSCORE
            } else {
                let is_mod_name = rdr.curr_is(':') && nextch_is(rdr, ':');

                // FIXME: perform NFKC normalization here. (Issue #2253)
                token::IDENT(str_to_ident(string), is_mod_name)
            }
        })
    }
    if is_dec_digit(c) {
        return scan_number(c.unwrap(), rdr);
    }
    fn binop(rdr: &mut StringReader, op: token::BinOp) -> token::Token {
        bump(rdr);
        if rdr.curr_is('=') {
            bump(rdr);
            return token::BINOPEQ(op);
        } else { return token::BINOP(op); }
    }
    match c.expect("next_token_inner called at EOF") {





      // One-byte tokens.
      ';' => { bump(rdr); return token::SEMI; }
      ',' => { bump(rdr); return token::COMMA; }
      '.' => {
          bump(rdr);
          return if rdr.curr_is('.') {
              bump(rdr);
              if rdr.curr_is('.') {
                  bump(rdr);
                  token::DOTDOTDOT
              } else {
                  token::DOTDOT
              }
          } else {
              token::DOT
          };
      }
      '(' => { bump(rdr); return token::LPAREN; }
      ')' => { bump(rdr); return token::RPAREN; }
      '{' => { bump(rdr); return token::LBRACE; }
      '}' => { bump(rdr); return token::RBRACE; }
      '[' => { bump(rdr); return token::LBRACKET; }
      ']' => { bump(rdr); return token::RBRACKET; }
      '@' => { bump(rdr); return token::AT; }
      '#' => { bump(rdr); return token::POUND; }
      '~' => { bump(rdr); return token::TILDE; }
      ':' => {
        bump(rdr);
        if rdr.curr_is(':') {
            bump(rdr);
            return token::MOD_SEP;
        } else { return token::COLON; }
      }

      '$' => { bump(rdr); return token::DOLLAR; }





      // Multi-byte tokens.
      '=' => {
        bump(rdr);
        if rdr.curr_is('=') {
            bump(rdr);
            return token::EQEQ;
        } else if rdr.curr_is('>') {
            bump(rdr);
            return token::FAT_ARROW;
        } else {
            return token::EQ;
        }
      }
      '!' => {
        bump(rdr);
        if rdr.curr_is('=') {
            bump(rdr);
            return token::NE;
        } else { return token::NOT; }
      }
      '<' => {
        bump(rdr);
        match rdr.curr.unwrap_or('\x00') {
          '=' => { bump(rdr); return token::LE; }
          '<' => { return binop(rdr, token::SHL); }
          '-' => {
            bump(rdr);
            match rdr.curr.unwrap_or('\x00') {
              '>' => { bump(rdr); return token::DARROW; }
              _ => { return token::LARROW; }
            }
          }
          _ => { return token::LT; }
        }
      }
      '>' => {
        bump(rdr);
        match rdr.curr.unwrap_or('\x00') {
          '=' => { bump(rdr); return token::GE; }
          '>' => { return binop(rdr, token::SHR); }
          _ => { return token::GT; }
        }
      }
      '\'' => {
        // Either a character constant 'a' OR a lifetime name 'abc
        bump(rdr);
        let start = rdr.last_pos;

        // the eof will be picked up by the final `'` check below
        let mut c2 = rdr.curr.unwrap_or('\x00');
        bump(rdr);

        // If the character is an ident start not followed by another single
        // quote, then this is a lifetime name:
        if ident_start(Some(c2)) && !rdr.curr_is('\'') {
            while ident_continue(rdr.curr) {
                bump(rdr);
            }
            let ident = with_str_from(rdr, start, |lifetime_name| {
                str_to_ident(lifetime_name)
            });
            let tok = &token::IDENT(ident, false);

            if token::is_keyword(token::keywords::Self, tok) {
                fatal_span(rdr, start, rdr.last_pos,
                           ~"invalid lifetime name: 'self is no longer a special lifetime");
            } else if token::is_any_keyword(tok) &&
                !token::is_keyword(token::keywords::Static, tok) {
                fatal_span(rdr, start, rdr.last_pos,
                           ~"invalid lifetime name");
            } else {
                return token::LIFETIME(ident);
            }
        }

        // Otherwise it is a character constant:
        match c2 {
            '\\' => {
                // '\X' for some X must be a character constant:
                let escaped = rdr.curr;
                let escaped_pos = rdr.last_pos;
                bump(rdr);
                match escaped {
                    None => {}
                    Some(e) => {
                        c2 = match e {
                            'n' => '\n',
                            'r' => '\r',
                            't' => '\t',
                            '\\' => '\\',
                            '\'' => '\'',
                            '"' => '"',
                            '0' => '\x00',
                            'x' => scan_numeric_escape(rdr, 2u),
                            'u' => scan_numeric_escape(rdr, 4u),
                            'U' => scan_numeric_escape(rdr, 8u),
                            c2 => {
                                fatal_span_char(rdr, escaped_pos, rdr.last_pos,
                                                ~"unknown character escape", c2)
                            }
                        }
                    }
                }
            }
            '\t' | '\n' | '\r' | '\'' => {
                fatal_span_char(rdr, start, rdr.last_pos,
                                ~"character constant must be escaped", c2);
            }
            _ => {}
        }
        if !rdr.curr_is('\'') {
            fatal_span_verbose(rdr,
                               // Byte offsetting here is okay because the
                               // character before position `start` is an
                               // ascii single quote.
                               start - BytePos(1),
                               rdr.last_pos,
                               ~"unterminated character constant");
        }
        bump(rdr); // advance curr past token
        return token::LIT_CHAR(c2 as u32);
      }
      '"' => {
        let mut accum_str = ~"";
        let start_bpos = rdr.last_pos;
        bump(rdr);
        while !rdr.curr_is('"') {
            if is_eof(rdr) {
                fatal_span(rdr, start_bpos, rdr.last_pos,
                           ~"unterminated double quote string");
            }

            let ch = rdr.curr.unwrap();
            bump(rdr);
            match ch {
              '\\' => {
                if is_eof(rdr) {
                    fatal_span(rdr, start_bpos, rdr.last_pos,
                           ~"unterminated double quote string");
                }

                let escaped = rdr.curr.unwrap();
                let escaped_pos = rdr.last_pos;
                bump(rdr);
                match escaped {
                  'n' => accum_str.push_char('\n'),
                  'r' => accum_str.push_char('\r'),
                  't' => accum_str.push_char('\t'),
                  '\\' => accum_str.push_char('\\'),
                  '\'' => accum_str.push_char('\''),
                  '"' => accum_str.push_char('"'),
                  '\n' => consume_whitespace(rdr),
                  '0' => accum_str.push_char('\x00'),
                  'x' => {
                    accum_str.push_char(scan_numeric_escape(rdr, 2u));
                  }
                  'u' => {
                    accum_str.push_char(scan_numeric_escape(rdr, 4u));
                  }
                  'U' => {
                    accum_str.push_char(scan_numeric_escape(rdr, 8u));
                  }
                  c2 => {
                    fatal_span_char(rdr, escaped_pos, rdr.last_pos,
                                    ~"unknown string escape", c2);
                  }
                }
              }
              _ => accum_str.push_char(ch)
            }
        }
        bump(rdr);
        return token::LIT_STR(str_to_ident(accum_str));
      }
      'r' => {
        let start_bpos = rdr.last_pos;
        bump(rdr);
        let mut hash_count = 0u;
        while rdr.curr_is('#') {
            bump(rdr);
            hash_count += 1;
        }

        if is_eof(rdr) {
            fatal_span(rdr, start_bpos, rdr.last_pos,
                       ~"unterminated raw string");
        } else if !rdr.curr_is('"') {
            fatal_span_char(rdr, start_bpos, rdr.last_pos,
                            ~"only `#` is allowed in raw string delimitation; \
                              found illegal character",
                            rdr.curr.unwrap());
        }
        bump(rdr);
        let content_start_bpos = rdr.last_pos;
        let mut content_end_bpos;
        'outer: loop {
            if is_eof(rdr) {
                fatal_span(rdr, start_bpos, rdr.last_pos,
                           ~"unterminated raw string");
            }
            if rdr.curr_is('"') {
                content_end_bpos = rdr.last_pos;
                for _ in range(0, hash_count) {
                    bump(rdr);
                    if !rdr.curr_is('#') {
                        continue 'outer;
                    }
                }
                break;
            }
            bump(rdr);
        }
        bump(rdr);
        let str_content = with_str_from_to(rdr,
                                           content_start_bpos,
                                           content_end_bpos,
                                           str_to_ident);
        return token::LIT_STR_RAW(str_content, hash_count);
      }
      '-' => {
        if nextch_is(rdr, '>') {
            bump(rdr);
            bump(rdr);
            return token::RARROW;
        } else { return binop(rdr, token::MINUS); }
      }
      '&' => {
        if nextch_is(rdr, '&') {
            bump(rdr);
            bump(rdr);
            return token::ANDAND;
        } else { return binop(rdr, token::AND); }
      }
      '|' => {
        match nextch(rdr) {
          Some('|') => { bump(rdr); bump(rdr); return token::OROR; }
          _ => { return binop(rdr, token::OR); }
        }
      }
      '+' => { return binop(rdr, token::PLUS); }
      '*' => { return binop(rdr, token::STAR); }
      '/' => { return binop(rdr, token::SLASH); }
      '^' => { return binop(rdr, token::CARET); }
      '%' => { return binop(rdr, token::PERCENT); }
      c => {
          fatal_span_char(rdr, rdr.last_pos, rdr.pos,
                          ~"unknown start of token", c);
      }
    }
}

fn consume_whitespace(rdr: &mut StringReader) {
    while is_whitespace(rdr.curr) && !is_eof(rdr) { bump(rdr); }
}

#[cfg(test)]
mod test {
    use super::*;

    use codemap::{BytePos, CodeMap, Span};
    use diagnostic;
    use parse::token;
    use parse::token::{str_to_ident};
    use std::io::util;

    fn mk_sh() -> diagnostic::SpanHandler {
        let emitter = diagnostic::EmitterWriter::new(~util::NullWriter);
        let handler = diagnostic::mk_handler(~emitter);
        diagnostic::mk_span_handler(handler, CodeMap::new())
    }

    // open a string reader for the given string
    fn setup<'a>(span_handler: &'a diagnostic::SpanHandler,
                 teststr: ~str) -> StringReader<'a> {
        let fm = span_handler.cm.new_filemap(~"zebra.rs", teststr);
        new_string_reader(span_handler, fm)
    }

    #[test] fn t1 () {
        let span_handler = mk_sh();
        let mut string_reader = setup(&span_handler,
            ~"/* my source file */ \
              fn main() { println!(\"zebra\"); }\n");
        let id = str_to_ident("fn");
        let tok1 = string_reader.next_token();
        let tok2 = TokenAndSpan{
            tok:token::IDENT(id, false),
            sp:Span {lo:BytePos(21),hi:BytePos(23),expn_info: None}};
        assert_eq!(tok1,tok2);
        // the 'main' id is already read:
        assert_eq!(string_reader.last_pos.clone(), BytePos(28));
        // read another token:
        let tok3 = string_reader.next_token();
        let tok4 = TokenAndSpan{
            tok:token::IDENT(str_to_ident("main"), false),
            sp:Span {lo:BytePos(24),hi:BytePos(28),expn_info: None}};
        assert_eq!(tok3,tok4);
        // the lparen is already read:
        assert_eq!(string_reader.last_pos.clone(), BytePos(29))
    }

    // check that the given reader produces the desired stream
    // of tokens (stop checking after exhausting the expected vec)
    fn check_tokenization (mut string_reader: StringReader, expected: Vec<token::Token> ) {
        for expected_tok in expected.iter() {
            assert_eq!(&string_reader.next_token().tok, expected_tok);
        }
    }

    // make the identifier by looking up the string in the interner
    fn mk_ident (id: &str, is_mod_name: bool) -> token::Token {
        token::IDENT (str_to_ident(id),is_mod_name)
    }

    #[test] fn doublecolonparsing () {
        check_tokenization(setup(&mk_sh(), ~"a b"),
                           vec!(mk_ident("a",false),
                             mk_ident("b",false)));
    }

    #[test] fn dcparsing_2 () {
        check_tokenization(setup(&mk_sh(), ~"a::b"),
                           vec!(mk_ident("a",true),
                             token::MOD_SEP,
                             mk_ident("b",false)));
    }

    #[test] fn dcparsing_3 () {
        check_tokenization(setup(&mk_sh(), ~"a ::b"),
                           vec!(mk_ident("a",false),
                             token::MOD_SEP,
                             mk_ident("b",false)));
    }

    #[test] fn dcparsing_4 () {
        check_tokenization(setup(&mk_sh(), ~"a:: b"),
                           vec!(mk_ident("a",true),
                             token::MOD_SEP,
                             mk_ident("b",false)));
    }

    #[test] fn character_a() {
        assert_eq!(setup(&mk_sh(), ~"'a'").next_token().tok,
                   token::LIT_CHAR('a' as u32));
    }

    #[test] fn character_space() {
        assert_eq!(setup(&mk_sh(), ~"' '").next_token().tok,
                   token::LIT_CHAR(' ' as u32));
    }

    #[test] fn character_escaped() {
        assert_eq!(setup(&mk_sh(), ~"'\\n'").next_token().tok,
                   token::LIT_CHAR('\n' as u32));
    }

    #[test] fn lifetime_name() {
        assert_eq!(setup(&mk_sh(), ~"'abc").next_token().tok,
                   token::LIFETIME(token::str_to_ident("abc")));
    }

    #[test] fn raw_string() {
        assert_eq!(setup(&mk_sh(), ~"r###\"\"#a\\b\x00c\"\"###").next_token().tok,
                   token::LIT_STR_RAW(token::str_to_ident("\"#a\\b\x00c\""), 3));
    }

    #[test] fn line_doc_comments() {
        assert!(!is_line_non_doc_comment("///"));
        assert!(!is_line_non_doc_comment("/// blah"));
        assert!(is_line_non_doc_comment("////"));
    }

    #[test] fn nested_block_comments() {
        assert_eq!(setup(&mk_sh(), ~"/* /* */ */'a'").next_token().tok,
                   token::LIT_CHAR('a' as u32));
    }

}
