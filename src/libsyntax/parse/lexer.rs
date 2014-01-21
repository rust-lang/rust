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
use ext::tt::transcribe::{tt_next_token};
use ext::tt::transcribe::{dup_tt_reader};
use parse::token;
use parse::token::{str_to_ident};

use std::cast::transmute;
use std::cell::{Cell, RefCell};
use std::char;
use std::num::from_str_radix;
use std::util;

pub use ext::tt::transcribe::{TtReader, new_tt_reader};

pub trait Reader {
    fn is_eof(@self) -> bool;
    fn next_token(@self) -> TokenAndSpan;
    fn fatal(@self, ~str) -> !;
    fn span_diag(@self) -> @SpanHandler;
    fn peek(@self) -> TokenAndSpan;
    fn dup(@self) -> @Reader;
}

#[deriving(Clone, Eq)]
pub struct TokenAndSpan {
    tok: token::Token,
    sp: Span,
}

pub struct StringReader {
    span_diagnostic: @SpanHandler,
    src: @str,
    // The absolute offset within the codemap of the next character to read
    pos: Cell<BytePos>,
    // The absolute offset within the codemap of the last character read(curr)
    last_pos: Cell<BytePos>,
    // The column of the next character to read
    col: Cell<CharPos>,
    // The last character to be read
    curr: Cell<char>,
    filemap: @codemap::FileMap,
    /* cached: */
    peek_tok: RefCell<token::Token>,
    peek_span: RefCell<Span>,
}

pub fn new_string_reader(span_diagnostic: @SpanHandler,
                         filemap: @codemap::FileMap)
                      -> @StringReader {
    let r = new_low_level_string_reader(span_diagnostic, filemap);
    string_advance_token(r); /* fill in peek_* */
    return r;
}

/* For comments.rs, which hackily pokes into 'pos' and 'curr' */
pub fn new_low_level_string_reader(span_diagnostic: @SpanHandler,
                                   filemap: @codemap::FileMap)
                                -> @StringReader {
    // Force the initial reader bump to start on a fresh line
    let initial_char = '\n';
    let r = @StringReader {
        span_diagnostic: span_diagnostic,
        src: filemap.src,
        pos: Cell::new(filemap.start_pos),
        last_pos: Cell::new(filemap.start_pos),
        col: Cell::new(CharPos(0)),
        curr: Cell::new(initial_char),
        filemap: filemap,
        /* dummy values; not read */
        peek_tok: RefCell::new(token::EOF),
        peek_span: RefCell::new(codemap::DUMMY_SP),
    };
    bump(r);
    return r;
}

// duplicating the string reader is probably a bad idea, in
// that using them will cause interleaved pushes of line
// offsets to the underlying filemap...
fn dup_string_reader(r: @StringReader) -> @StringReader {
    @StringReader {
        span_diagnostic: r.span_diagnostic,
        src: r.src,
        pos: Cell::new(r.pos.get()),
        last_pos: Cell::new(r.last_pos.get()),
        col: Cell::new(r.col.get()),
        curr: Cell::new(r.curr.get()),
        filemap: r.filemap,
        peek_tok: r.peek_tok.clone(),
        peek_span: r.peek_span.clone(),
    }
}

impl Reader for StringReader {
    fn is_eof(@self) -> bool { is_eof(self) }
    // return the next token. EFFECT: advances the string_reader.
    fn next_token(@self) -> TokenAndSpan {
        let ret_val = {
            let mut peek_tok = self.peek_tok.borrow_mut();
            TokenAndSpan {
                tok: util::replace(peek_tok.get(), token::UNDERSCORE),
                sp: self.peek_span.get(),
            }
        };
        string_advance_token(self);
        ret_val
    }
    fn fatal(@self, m: ~str) -> ! {
        self.span_diagnostic.span_fatal(self.peek_span.get(), m)
    }
    fn span_diag(@self) -> @SpanHandler { self.span_diagnostic }
    fn peek(@self) -> TokenAndSpan {
        // XXX(pcwalton): Bad copy!
        TokenAndSpan {
            tok: self.peek_tok.get(),
            sp: self.peek_span.get(),
        }
    }
    fn dup(@self) -> @Reader { dup_string_reader(self) as @Reader }
}

impl Reader for TtReader {
    fn is_eof(@self) -> bool {
        let cur_tok = self.cur_tok.borrow();
        *cur_tok.get() == token::EOF
    }
    fn next_token(@self) -> TokenAndSpan {
        let r = tt_next_token(self);
        debug!("TtReader: r={:?}", r);
        return r;
    }
    fn fatal(@self, m: ~str) -> ! {
        self.sp_diag.span_fatal(self.cur_span.get(), m);
    }
    fn span_diag(@self) -> @SpanHandler { self.sp_diag }
    fn peek(@self) -> TokenAndSpan {
        TokenAndSpan {
            tok: self.cur_tok.get(),
            sp: self.cur_span.get(),
        }
    }
    fn dup(@self) -> @Reader { dup_tt_reader(self) as @Reader }
}

// report a lexical error spanning [`from_pos`, `to_pos`)
fn fatal_span(rdr: @StringReader,
              from_pos: BytePos,
              to_pos: BytePos,
              m: ~str)
           -> ! {
    rdr.peek_span.set(codemap::mk_sp(from_pos, to_pos));
    rdr.fatal(m);
}

// report a lexical error spanning [`from_pos`, `to_pos`), appending an
// escaped character to the error message
fn fatal_span_char(rdr: @StringReader,
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
fn fatal_span_verbose(rdr: @StringReader,
                      from_pos: BytePos,
                      to_pos: BytePos,
                      m: ~str)
                   -> ! {
    let mut m = m;
    m.push_str(": ");
    let s = rdr.src.slice(
                  byte_offset(rdr, from_pos).to_uint(),
                  byte_offset(rdr, to_pos).to_uint());
    m.push_str(s);
    fatal_span(rdr, from_pos, to_pos, m);
}

// EFFECT: advance peek_tok and peek_span to refer to the next token.
// EFFECT: update the interner, maybe.
fn string_advance_token(r: @StringReader) {
    match consume_whitespace_and_comments(r) {
        Some(comment) => {
            r.peek_span.set(comment.sp);
            r.peek_tok.set(comment.tok);
        },
        None => {
            if is_eof(r) {
                r.peek_tok.set(token::EOF);
            } else {
                let start_bytepos = r.last_pos.get();
                r.peek_tok.set(next_token_inner(r));
                r.peek_span.set(codemap::mk_sp(start_bytepos,
                                               r.last_pos.get()));
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
                     rdr: @StringReader,
                     start: BytePos,
                     f: |s: &str| -> T)
                     -> T {
    with_str_from_to(rdr, start, rdr.last_pos.get(), f)
}

/// Calls `f` with astring slice of the source text spanning from `start`
/// up to but excluding `end`.
fn with_str_from_to<T>(
                    rdr: @StringReader,
                    start: BytePos,
                    end: BytePos,
                    f: |s: &str| -> T)
                    -> T {
    f(rdr.src.slice(
            byte_offset(rdr, start).to_uint(),
            byte_offset(rdr, end).to_uint()))
}

// EFFECT: advance the StringReader by one character. If a newline is
// discovered, add it to the FileMap's list of line start offsets.
pub fn bump(rdr: &StringReader) {
    rdr.last_pos.set(rdr.pos.get());
    let current_byte_offset = byte_offset(rdr, rdr.pos.get()).to_uint();
    if current_byte_offset < (rdr.src).len() {
        assert!(rdr.curr.get() != unsafe {
            transmute(-1u32)
        }); // FIXME: #8971: unsound
        let last_char = rdr.curr.get();
        let next = rdr.src.char_range_at(current_byte_offset);
        let byte_offset_diff = next.next - current_byte_offset;
        rdr.pos.set(rdr.pos.get() + Pos::from_uint(byte_offset_diff));
        rdr.curr.set(next.ch);
        rdr.col.set(rdr.col.get() + CharPos(1u));
        if last_char == '\n' {
            rdr.filemap.next_line(rdr.last_pos.get());
            rdr.col.set(CharPos(0u));
        }

        if byte_offset_diff > 1 {
            rdr.filemap.record_multibyte_char(
                Pos::from_uint(current_byte_offset), byte_offset_diff);
        }
    } else {
        rdr.curr.set(unsafe { transmute(-1u32) }); // FIXME: #8971: unsound
    }
}
pub fn is_eof(rdr: @StringReader) -> bool {
    rdr.curr.get() == unsafe { transmute(-1u32) } // FIXME: #8971: unsound
}
pub fn nextch(rdr: @StringReader) -> char {
    let offset = byte_offset(rdr, rdr.pos.get()).to_uint();
    if offset < (rdr.src).len() {
        return rdr.src.char_at(offset);
    } else { return unsafe { transmute(-1u32) }; } // FIXME: #8971: unsound
}

fn hex_digit_val(c: char) -> int {
    if in_range(c, '0', '9') { return (c as int) - ('0' as int); }
    if in_range(c, 'a', 'f') { return (c as int) - ('a' as int) + 10; }
    if in_range(c, 'A', 'F') { return (c as int) - ('A' as int) + 10; }
    fail!();
}

pub fn is_whitespace(c: char) -> bool {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

fn in_range(c: char, lo: char, hi: char) -> bool {
    return lo <= c && c <= hi
}

fn is_dec_digit(c: char) -> bool { return in_range(c, '0', '9'); }

fn is_hex_digit(c: char) -> bool {
    return in_range(c, '0', '9') || in_range(c, 'a', 'f') ||
            in_range(c, 'A', 'F');
}

// EFFECT: eats whitespace and comments.
// returns a Some(sugared-doc-attr) if one exists, None otherwise.
fn consume_whitespace_and_comments(rdr: @StringReader)
                                -> Option<TokenAndSpan> {
    while is_whitespace(rdr.curr.get()) { bump(rdr); }
    return consume_any_line_comment(rdr);
}

pub fn is_line_non_doc_comment(s: &str) -> bool {
    s.starts_with("////")
}

// PRECONDITION: rdr.curr is not whitespace
// EFFECT: eats any kind of comment.
// returns a Some(sugared-doc-attr) if one exists, None otherwise
fn consume_any_line_comment(rdr: @StringReader)
                         -> Option<TokenAndSpan> {
    if rdr.curr.get() == '/' {
        match nextch(rdr) {
          '/' => {
            bump(rdr);
            bump(rdr);
            // line comments starting with "///" or "//!" are doc-comments
            if rdr.curr.get() == '/' || rdr.curr.get() == '!' {
                let start_bpos = rdr.pos.get() - BytePos(3);
                while rdr.curr.get() != '\n' && !is_eof(rdr) {
                    bump(rdr);
                }
                let ret = with_str_from(rdr, start_bpos, |string| {
                    // but comments with only more "/"s are not
                    if !is_line_non_doc_comment(string) {
                        Some(TokenAndSpan{
                            tok: token::DOC_COMMENT(str_to_ident(string)),
                            sp: codemap::mk_sp(start_bpos, rdr.pos.get())
                        })
                    } else {
                        None
                    }
                });

                if ret.is_some() {
                    return ret;
                }
            } else {
                while rdr.curr.get() != '\n' && !is_eof(rdr) { bump(rdr); }
            }
            // Restart whitespace munch.
            return consume_whitespace_and_comments(rdr);
          }
          '*' => { bump(rdr); bump(rdr); return consume_block_comment(rdr); }
          _ => ()
        }
    } else if rdr.curr.get() == '#' {
        if nextch(rdr) == '!' {
            // I guess this is the only way to figure out if
            // we're at the beginning of the file...
            let cmap = @CodeMap::new();
            {
                let mut files = cmap.files.borrow_mut();
                files.get().push(rdr.filemap);
            }
            let loc = cmap.lookup_char_pos_adj(rdr.last_pos.get());
            if loc.line == 1u && loc.col == CharPos(0u) {
                while rdr.curr.get() != '\n' && !is_eof(rdr) { bump(rdr); }
                return consume_whitespace_and_comments(rdr);
            }
        }
    }
    return None;
}

pub fn is_block_non_doc_comment(s: &str) -> bool {
    s.starts_with("/***")
}

// might return a sugared-doc-attr
fn consume_block_comment(rdr: @StringReader) -> Option<TokenAndSpan> {
    // block comments starting with "/**" or "/*!" are doc-comments
    let is_doc_comment = rdr.curr.get() == '*' || rdr.curr.get() == '!';
    let start_bpos = rdr.pos.get() - BytePos(if is_doc_comment {3} else {2});

    let mut level: int = 1;
    while level > 0 {
        if is_eof(rdr) {
            let msg = if is_doc_comment {
                ~"unterminated block doc-comment"
            } else {
                ~"unterminated block comment"
            };
            fatal_span(rdr, start_bpos, rdr.last_pos.get(), msg);
        } else if rdr.curr.get() == '/' && nextch(rdr) == '*' {
            level += 1;
            bump(rdr);
            bump(rdr);
        } else if rdr.curr.get() == '*' && nextch(rdr) == '/' {
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
                        sp: codemap::mk_sp(start_bpos, rdr.pos.get())
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

fn scan_exponent(rdr: @StringReader, start_bpos: BytePos) -> Option<~str> {
    let mut c = rdr.curr.get();
    let mut rslt = ~"";
    if c == 'e' || c == 'E' {
        rslt.push_char(c);
        bump(rdr);
        c = rdr.curr.get();
        if c == '-' || c == '+' {
            rslt.push_char(c);
            bump(rdr);
        }
        let exponent = scan_digits(rdr, 10u);
        if exponent.len() > 0u {
            return Some(rslt + exponent);
        } else {
            fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                       ~"scan_exponent: bad fp literal");
        }
    } else { return None::<~str>; }
}

fn scan_digits(rdr: @StringReader, radix: uint) -> ~str {
    let mut rslt = ~"";
    loop {
        let c = rdr.curr.get();
        if c == '_' { bump(rdr); continue; }
        match char::to_digit(c, radix) {
          Some(_) => {
            rslt.push_char(c);
            bump(rdr);
          }
          _ => return rslt
        }
    };
}

fn check_float_base(rdr: @StringReader, start_bpos: BytePos, last_bpos: BytePos,
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

fn scan_number(c: char, rdr: @StringReader) -> token::Token {
    let mut num_str;
    let mut base = 10u;
    let mut c = c;
    let mut n = nextch(rdr);
    let start_bpos = rdr.last_pos.get();
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
    c = rdr.curr.get();
    nextch(rdr);
    if c == 'u' || c == 'i' {
        enum Result { Signed(ast::IntTy), Unsigned(ast::UintTy) }
        let signed = c == 'i';
        let mut tp = {
            if signed { Signed(ast::TyI) }
            else { Unsigned(ast::TyU) }
        };
        bump(rdr);
        c = rdr.curr.get();
        if c == '8' {
            bump(rdr);
            tp = if signed { Signed(ast::TyI8) }
                      else { Unsigned(ast::TyU8) };
        }
        n = nextch(rdr);
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
            fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                       ~"no valid digits found for number");
        }
        let parsed = match from_str_radix::<u64>(num_str, base as uint) {
            Some(p) => p,
            None => fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                               ~"int literal is too large")
        };

        match tp {
          Signed(t) => return token::LIT_INT(parsed as i64, t),
          Unsigned(t) => return token::LIT_UINT(parsed, t)
        }
    }
    let mut is_float = false;
    if rdr.curr.get() == '.' && !(ident_start(nextch(rdr)) || nextch(rdr) ==
                                  '.') {
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

    if rdr.curr.get() == 'f' {
        bump(rdr);
        c = rdr.curr.get();
        n = nextch(rdr);
        if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            check_float_base(rdr, start_bpos, rdr.last_pos.get(), base);
            return token::LIT_FLOAT(str_to_ident(num_str), ast::TyF32);
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
            check_float_base(rdr, start_bpos, rdr.last_pos.get(), base);
            return token::LIT_FLOAT(str_to_ident(num_str), ast::TyF64);
            /* FIXME (#2252): if this is out of range for either a
            32-bit or 64-bit float, it won't be noticed till the
            back-end.  */
        } else {
            fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                       ~"expected `f32` or `f64` suffix");
        }
    }
    if is_float {
        check_float_base(rdr, start_bpos, rdr.last_pos.get(), base);
        return token::LIT_FLOAT_UNSUFFIXED(str_to_ident(num_str));
    } else {
        if num_str.len() == 0u {
            fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                       ~"no valid digits found for number");
        }
        let parsed = match from_str_radix::<u64>(num_str, base as uint) {
            Some(p) => p,
            None => fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                               ~"int literal is too large")
        };

        debug!("lexing {} as an unsuffixed integer literal", num_str);
        return token::LIT_INT_UNSUFFIXED(parsed as i64);
    }
}

fn scan_numeric_escape(rdr: @StringReader, n_hex_digits: uint) -> char {
    let mut accum_int = 0;
    let mut i = n_hex_digits;
    let start_bpos = rdr.last_pos.get();
    while i != 0u {
        let n = rdr.curr.get();
        if !is_hex_digit(n) {
            fatal_span_char(rdr, rdr.last_pos.get(), rdr.pos.get(),
                            ~"illegal character in numeric character escape",
                            n);
        }
        bump(rdr);
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        i -= 1u;
    }
    match char::from_u32(accum_int as u32) {
        Some(x) => x,
        None => fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                           ~"illegal numeric character escape")
    }
}

fn ident_start(c: char) -> bool {
    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && char::is_XID_start(c))
}

fn ident_continue(c: char) -> bool {
    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9')
        || c == '_'
        || (c > '\x7f' && char::is_XID_continue(c))
}

// return the next token from the string
// EFFECT: advances the input past that token
// EFFECT: updates the interner
fn next_token_inner(rdr: @StringReader) -> token::Token {
    let c = rdr.curr.get();
    if ident_start(c) && nextch(rdr) != '"' && nextch(rdr) != '#' {
        // Note: r as in r" or r#" is part of a raw string literal,
        // not an identifier, and is handled further down.

        let start = rdr.last_pos.get();
        while ident_continue(rdr.curr.get()) {
            bump(rdr);
        }

        return with_str_from(rdr, start, |string| {
            if string == "_" {
                token::UNDERSCORE
            } else {
                let is_mod_name = rdr.curr.get() == ':' && nextch(rdr) == ':';

                // FIXME: perform NFKC normalization here. (Issue #2253)
                token::IDENT(str_to_ident(string), is_mod_name)
            }
        })
    }
    if is_dec_digit(c) {
        return scan_number(c, rdr);
    }
    fn binop(rdr: @StringReader, op: token::BinOp) -> token::Token {
        bump(rdr);
        if rdr.curr.get() == '=' {
            bump(rdr);
            return token::BINOPEQ(op);
        } else { return token::BINOP(op); }
    }
    match c {





      // One-byte tokens.
      ';' => { bump(rdr); return token::SEMI; }
      ',' => { bump(rdr); return token::COMMA; }
      '.' => {
          bump(rdr);
          return if rdr.curr.get() == '.' {
              bump(rdr);
              if rdr.curr.get() == '.' {
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
        if rdr.curr.get() == ':' {
            bump(rdr);
            return token::MOD_SEP;
        } else { return token::COLON; }
      }

      '$' => { bump(rdr); return token::DOLLAR; }





      // Multi-byte tokens.
      '=' => {
        bump(rdr);
        if rdr.curr.get() == '=' {
            bump(rdr);
            return token::EQEQ;
        } else if rdr.curr.get() == '>' {
            bump(rdr);
            return token::FAT_ARROW;
        } else {
            return token::EQ;
        }
      }
      '!' => {
        bump(rdr);
        if rdr.curr.get() == '=' {
            bump(rdr);
            return token::NE;
        } else { return token::NOT; }
      }
      '<' => {
        bump(rdr);
        match rdr.curr.get() {
          '=' => { bump(rdr); return token::LE; }
          '<' => { return binop(rdr, token::SHL); }
          '-' => {
            bump(rdr);
            match rdr.curr.get() {
              '>' => { bump(rdr); return token::DARROW; }
              _ => { return token::LARROW; }
            }
          }
          _ => { return token::LT; }
        }
      }
      '>' => {
        bump(rdr);
        match rdr.curr.get() {
          '=' => { bump(rdr); return token::GE; }
          '>' => { return binop(rdr, token::SHR); }
          _ => { return token::GT; }
        }
      }
      '\'' => {
        // Either a character constant 'a' OR a lifetime name 'abc
        bump(rdr);
        let start = rdr.last_pos.get();
        let mut c2 = rdr.curr.get();
        bump(rdr);

        // If the character is an ident start not followed by another single
        // quote, then this is a lifetime name:
        if ident_start(c2) && rdr.curr.get() != '\'' {
            while ident_continue(rdr.curr.get()) {
                bump(rdr);
            }
            return with_str_from(rdr, start, |lifetime_name| {
                let ident = str_to_ident(lifetime_name);
                let tok = &token::IDENT(ident, false);

                if token::is_keyword(token::keywords::Self, tok) {
                    fatal_span(rdr, start, rdr.last_pos.get(),
                               ~"invalid lifetime name: 'self is no longer a special lifetime");
                } else if token::is_any_keyword(tok) &&
                    !token::is_keyword(token::keywords::Static, tok) {
                    fatal_span(rdr, start, rdr.last_pos.get(),
                               ~"invalid lifetime name");
                } else {
                    token::LIFETIME(ident)
                }
            })
        }

        // Otherwise it is a character constant:
        match c2 {
            '\\' => {
                // '\X' for some X must be a character constant:
                let escaped = rdr.curr.get();
                let escaped_pos = rdr.last_pos.get();
                bump(rdr);
                match escaped {
                    'n' => { c2 = '\n'; }
                    'r' => { c2 = '\r'; }
                    't' => { c2 = '\t'; }
                    '\\' => { c2 = '\\'; }
                    '\'' => { c2 = '\''; }
                    '"' => { c2 = '"'; }
                    '0' => { c2 = '\x00'; }
                    'x' => { c2 = scan_numeric_escape(rdr, 2u); }
                    'u' => { c2 = scan_numeric_escape(rdr, 4u); }
                    'U' => { c2 = scan_numeric_escape(rdr, 8u); }
                    c2 => {
                        fatal_span_char(rdr, escaped_pos, rdr.last_pos.get(),
                                        ~"unknown character escape", c2);
                    }
                }
            }
            '\t' | '\n' | '\r' | '\'' => {
                fatal_span_char(rdr, start, rdr.last_pos.get(),
                                ~"character constant must be escaped", c2);
            }
            _ => {}
        }
        if rdr.curr.get() != '\'' {
            fatal_span_verbose(rdr,
                               // Byte offsetting here is okay because the
                               // character before position `start` is an
                               // ascii single quote.
                               start - BytePos(1),
                               rdr.last_pos.get(),
                               ~"unterminated character constant");
        }
        bump(rdr); // advance curr past token
        return token::LIT_CHAR(c2 as u32);
      }
      '"' => {
        let mut accum_str = ~"";
        let start_bpos = rdr.last_pos.get();
        bump(rdr);
        while rdr.curr.get() != '"' {
            if is_eof(rdr) {
                fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                           ~"unterminated double quote string");
            }

            let ch = rdr.curr.get();
            bump(rdr);
            match ch {
              '\\' => {
                let escaped = rdr.curr.get();
                let escaped_pos = rdr.last_pos.get();
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
                    fatal_span_char(rdr, escaped_pos, rdr.last_pos.get(),
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
        let start_bpos = rdr.last_pos.get();
        bump(rdr);
        let mut hash_count = 0u;
        while rdr.curr.get() == '#' {
            bump(rdr);
            hash_count += 1;
        }
        if rdr.curr.get() != '"' {
            fatal_span_char(rdr, start_bpos, rdr.last_pos.get(),
                            ~"only `#` is allowed in raw string delimitation; \
                              found illegal character",
                            rdr.curr.get());
        }
        bump(rdr);
        let content_start_bpos = rdr.last_pos.get();
        let mut content_end_bpos;
        'outer: loop {
            if is_eof(rdr) {
                fatal_span(rdr, start_bpos, rdr.last_pos.get(),
                           ~"unterminated raw string");
            }
            if rdr.curr.get() == '"' {
                content_end_bpos = rdr.last_pos.get();
                for _ in range(0, hash_count) {
                    bump(rdr);
                    if rdr.curr.get() != '#' {
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
        if nextch(rdr) == '>' {
            bump(rdr);
            bump(rdr);
            return token::RARROW;
        } else { return binop(rdr, token::MINUS); }
      }
      '&' => {
        if nextch(rdr) == '&' {
            bump(rdr);
            bump(rdr);
            return token::ANDAND;
        } else { return binop(rdr, token::AND); }
      }
      '|' => {
        match nextch(rdr) {
          '|' => { bump(rdr); bump(rdr); return token::OROR; }
          _ => { return binop(rdr, token::OR); }
        }
      }
      '+' => { return binop(rdr, token::PLUS); }
      '*' => { return binop(rdr, token::STAR); }
      '/' => { return binop(rdr, token::SLASH); }
      '^' => { return binop(rdr, token::CARET); }
      '%' => { return binop(rdr, token::PERCENT); }
      c => {
          fatal_span_char(rdr, rdr.last_pos.get(), rdr.pos.get(),
                          ~"unknown start of token", c);
      }
    }
}

fn consume_whitespace(rdr: @StringReader) {
    while is_whitespace(rdr.curr.get()) && !is_eof(rdr) { bump(rdr); }
}

#[cfg(test)]
mod test {
    use super::*;

    use codemap::{BytePos, CodeMap, Span};
    use diagnostic;
    use parse::token;
    use parse::token::{str_to_ident};

    // represents a testing reader (incl. both reader and interner)
    struct Env {
        string_reader: @StringReader
    }

    // open a string reader for the given string
    fn setup(teststr: @str) -> Env {
        let cm = CodeMap::new();
        let fm = cm.new_filemap(@"zebra.rs", teststr);
        let span_handler =
            diagnostic::mk_span_handler(diagnostic::mk_handler(None),@cm);
        Env {
            string_reader: new_string_reader(span_handler,fm)
        }
    }

    #[test] fn t1 () {
        let Env {string_reader} =
            setup(@"/* my source file */ \
                    fn main() { println!(\"zebra\"); }\n");
        let id = str_to_ident("fn");
        let tok1 = string_reader.next_token();
        let tok2 = TokenAndSpan{
            tok:token::IDENT(id, false),
            sp:Span {lo:BytePos(21),hi:BytePos(23),expn_info: None}};
        assert_eq!(tok1,tok2);
        // the 'main' id is already read:
        assert_eq!(string_reader.last_pos.get().clone(), BytePos(28));
        // read another token:
        let tok3 = string_reader.next_token();
        let tok4 = TokenAndSpan{
            tok:token::IDENT(str_to_ident("main"), false),
            sp:Span {lo:BytePos(24),hi:BytePos(28),expn_info: None}};
        assert_eq!(tok3,tok4);
        // the lparen is already read:
        assert_eq!(string_reader.last_pos.get().clone(), BytePos(29))
    }

    // check that the given reader produces the desired stream
    // of tokens (stop checking after exhausting the expected vec)
    fn check_tokenization (env: Env, expected: ~[token::Token]) {
        for expected_tok in expected.iter() {
            let TokenAndSpan {tok:actual_tok, sp: _} =
                env.string_reader.next_token();
            assert_eq!(&actual_tok,expected_tok);
        }
    }

    // make the identifier by looking up the string in the interner
    fn mk_ident (id: &str, is_mod_name: bool) -> token::Token {
        token::IDENT (str_to_ident(id),is_mod_name)
    }

    #[test] fn doublecolonparsing () {
        let env = setup (@"a b");
        check_tokenization (env,
                           ~[mk_ident("a",false),
                             mk_ident("b",false)]);
    }

    #[test] fn dcparsing_2 () {
        let env = setup (@"a::b");
        check_tokenization (env,
                           ~[mk_ident("a",true),
                             token::MOD_SEP,
                             mk_ident("b",false)]);
    }

    #[test] fn dcparsing_3 () {
        let env = setup (@"a ::b");
        check_tokenization (env,
                           ~[mk_ident("a",false),
                             token::MOD_SEP,
                             mk_ident("b",false)]);
    }

    #[test] fn dcparsing_4 () {
        let env = setup (@"a:: b");
        check_tokenization (env,
                           ~[mk_ident("a",true),
                             token::MOD_SEP,
                             mk_ident("b",false)]);
    }

    #[test] fn character_a() {
        let env = setup(@"'a'");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        assert_eq!(tok,token::LIT_CHAR('a' as u32));
    }

    #[test] fn character_space() {
        let env = setup(@"' '");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        assert_eq!(tok, token::LIT_CHAR(' ' as u32));
    }

    #[test] fn character_escaped() {
        let env = setup(@"'\\n'");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        assert_eq!(tok, token::LIT_CHAR('\n' as u32));
    }

    #[test] fn lifetime_name() {
        let env = setup(@"'abc");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        let id = token::str_to_ident("abc");
        assert_eq!(tok, token::LIFETIME(id));
    }

    #[test] fn raw_string() {
        let env = setup(@"r###\"\"#a\\b\x00c\"\"###");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        let id = token::str_to_ident("\"#a\\b\x00c\"");
        assert_eq!(tok, token::LIT_STR_RAW(id, 3));
    }

    #[test] fn line_doc_comments() {
        assert!(!is_line_non_doc_comment("///"));
        assert!(!is_line_non_doc_comment("/// blah"));
        assert!(is_line_non_doc_comment("////"));
    }

    #[test] fn nested_block_comments() {
        let env = setup(@"/* /* */ */'a'");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        assert_eq!(tok,token::LIT_CHAR('a' as u32));
    }

}
