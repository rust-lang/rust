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
use codemap::{BytePos, CharPos, CodeMap, Pos, span};
use codemap;
use diagnostic::span_handler;
use ext::tt::transcribe::{tt_next_token};
use ext::tt::transcribe::{dup_tt_reader};
use parse::token;
use parse::token::{str_to_ident};

use std::char;
use std::either;
use std::u64;
use std::util;

pub use ext::tt::transcribe::{TtReader, new_tt_reader};

pub trait reader {
    fn is_eof(@mut self) -> bool;
    fn next_token(@mut self) -> TokenAndSpan;
    fn fatal(@mut self, ~str) -> !;
    fn span_diag(@mut self) -> @mut span_handler;
    fn peek(@mut self) -> TokenAndSpan;
    fn dup(@mut self) -> @mut reader;
}

#[deriving(Clone, Eq)]
pub struct TokenAndSpan {
    tok: token::Token,
    sp: span,
}

pub struct StringReader {
    span_diagnostic: @mut span_handler,
    src: @str,
    // The absolute offset within the codemap of the next character to read
    pos: BytePos,
    // The absolute offset within the codemap of the last character read(curr)
    last_pos: BytePos,
    // The column of the next character to read
    col: CharPos,
    // The last character to be read
    curr: char,
    filemap: @codemap::FileMap,
    /* cached: */
    peek_tok: token::Token,
    peek_span: span
}

pub fn new_string_reader(span_diagnostic: @mut span_handler,
                         filemap: @codemap::FileMap)
                      -> @mut StringReader {
    let r = new_low_level_string_reader(span_diagnostic, filemap);
    string_advance_token(r); /* fill in peek_* */
    return r;
}

/* For comments.rs, which hackily pokes into 'pos' and 'curr' */
pub fn new_low_level_string_reader(span_diagnostic: @mut span_handler,
                                   filemap: @codemap::FileMap)
                                -> @mut StringReader {
    // Force the initial reader bump to start on a fresh line
    let initial_char = '\n';
    let r = @mut StringReader {
        span_diagnostic: span_diagnostic,
        src: filemap.src,
        pos: filemap.start_pos,
        last_pos: filemap.start_pos,
        col: CharPos(0),
        curr: initial_char,
        filemap: filemap,
        /* dummy values; not read */
        peek_tok: token::EOF,
        peek_span: codemap::dummy_sp()
    };
    bump(r);
    return r;
}

// duplicating the string reader is probably a bad idea, in
// that using them will cause interleaved pushes of line
// offsets to the underlying filemap...
fn dup_string_reader(r: @mut StringReader) -> @mut StringReader {
    @mut StringReader {
        span_diagnostic: r.span_diagnostic,
        src: r.src,
        pos: r.pos,
        last_pos: r.last_pos,
        col: r.col,
        curr: r.curr,
        filemap: r.filemap,
        peek_tok: r.peek_tok.clone(),
        peek_span: r.peek_span
    }
}

impl reader for StringReader {
    fn is_eof(@mut self) -> bool { is_eof(self) }
    // return the next token. EFFECT: advances the string_reader.
    fn next_token(@mut self) -> TokenAndSpan {
        let ret_val = TokenAndSpan {
            tok: util::replace(&mut self.peek_tok, token::UNDERSCORE),
            sp: self.peek_span,
        };
        string_advance_token(self);
        ret_val
    }
    fn fatal(@mut self, m: ~str) -> ! {
        self.span_diagnostic.span_fatal(self.peek_span, m)
    }
    fn span_diag(@mut self) -> @mut span_handler { self.span_diagnostic }
    fn peek(@mut self) -> TokenAndSpan {
        // XXX(pcwalton): Bad copy!
        TokenAndSpan {
            tok: self.peek_tok.clone(),
            sp: self.peek_span,
        }
    }
    fn dup(@mut self) -> @mut reader { dup_string_reader(self) as @mut reader }
}

impl reader for TtReader {
    fn is_eof(@mut self) -> bool { self.cur_tok == token::EOF }
    fn next_token(@mut self) -> TokenAndSpan {
        let r = tt_next_token(self);
        debug!("TtReader: r=%?", r);
        return r;
    }
    fn fatal(@mut self, m: ~str) -> ! {
        self.sp_diag.span_fatal(self.cur_span, m);
    }
    fn span_diag(@mut self) -> @mut span_handler { self.sp_diag }
    fn peek(@mut self) -> TokenAndSpan {
        TokenAndSpan {
            tok: self.cur_tok.clone(),
            sp: self.cur_span,
        }
    }
    fn dup(@mut self) -> @mut reader { dup_tt_reader(self) as @mut reader }
}

// EFFECT: advance peek_tok and peek_span to refer to the next token.
// EFFECT: update the interner, maybe.
fn string_advance_token(r: @mut StringReader) {
    match (consume_whitespace_and_comments(r)) {
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
                r.peek_span = codemap::mk_sp(start_bytepos, r.last_pos);
            };
        }
    }
}

fn byte_offset(rdr: &StringReader, pos: BytePos) -> BytePos {
    (pos - rdr.filemap.start_pos)
}

pub fn with_str_from<T>(rdr: @mut StringReader, start: BytePos, f: &fn(s: &str) -> T) -> T {
    f(rdr.src.slice(
            byte_offset(rdr, start).to_uint(),
            byte_offset(rdr, rdr.last_pos).to_uint()))
}

// EFFECT: advance the StringReader by one character. If a newline is
// discovered, add it to the FileMap's list of line start offsets.
pub fn bump(rdr: &mut StringReader) {
    rdr.last_pos = rdr.pos;
    let current_byte_offset = byte_offset(rdr, rdr.pos).to_uint();
    if current_byte_offset < (rdr.src).len() {
        assert!(rdr.curr != -1 as char);
        let last_char = rdr.curr;
        let next = rdr.src.char_range_at(current_byte_offset);
        let byte_offset_diff = next.next - current_byte_offset;
        rdr.pos = rdr.pos + BytePos(byte_offset_diff);
        rdr.curr = next.ch;
        rdr.col = rdr.col + CharPos(1u);
        if last_char == '\n' {
            rdr.filemap.next_line(rdr.last_pos);
            rdr.col = CharPos(0u);
        }

        if byte_offset_diff > 1 {
            rdr.filemap.record_multibyte_char(
                BytePos(current_byte_offset), byte_offset_diff);
        }
    } else {
        rdr.curr = -1 as char;
    }
}
pub fn is_eof(rdr: @mut StringReader) -> bool {
    rdr.curr == -1 as char
}
pub fn nextch(rdr: @mut StringReader) -> char {
    let offset = byte_offset(rdr, rdr.pos).to_uint();
    if offset < (rdr.src).len() {
        return rdr.src.char_at(offset);
    } else { return -1 as char; }
}

fn dec_digit_val(c: char) -> int { return (c as int) - ('0' as int); }

fn hex_digit_val(c: char) -> int {
    if in_range(c, '0', '9') { return (c as int) - ('0' as int); }
    if in_range(c, 'a', 'f') { return (c as int) - ('a' as int) + 10; }
    if in_range(c, 'A', 'F') { return (c as int) - ('A' as int) + 10; }
    fail!();
}

fn bin_digit_value(c: char) -> int { if c == '0' { return 0; } return 1; }

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

fn is_bin_digit(c: char) -> bool { return c == '0' || c == '1'; }

// EFFECT: eats whitespace and comments.
// returns a Some(sugared-doc-attr) if one exists, None otherwise.
fn consume_whitespace_and_comments(rdr: @mut StringReader)
                                -> Option<TokenAndSpan> {
    while is_whitespace(rdr.curr) { bump(rdr); }
    return consume_any_line_comment(rdr);
}

pub fn is_line_non_doc_comment(s: &str) -> bool {
    let s = s.trim_right();
    s.len() > 3 && s.iter().all(|ch| ch == '/')
}

// PRECONDITION: rdr.curr is not whitespace
// EFFECT: eats any kind of comment.
// returns a Some(sugared-doc-attr) if one exists, None otherwise
fn consume_any_line_comment(rdr: @mut StringReader)
                         -> Option<TokenAndSpan> {
    if rdr.curr == '/' {
        match nextch(rdr) {
          '/' => {
            bump(rdr);
            bump(rdr);
            // line comments starting with "///" or "//!" are doc-comments
            if rdr.curr == '/' || rdr.curr == '!' {
                let start_bpos = rdr.pos - BytePos(3u);
                while rdr.curr != '\n' && !is_eof(rdr) {
                    bump(rdr);
                }
                let ret = do with_str_from(rdr, start_bpos) |string| {
                    // but comments with only more "/"s are not
                    if !is_line_non_doc_comment(string) {
                        Some(TokenAndSpan{
                            tok: token::DOC_COMMENT(str_to_ident(string)),
                            sp: codemap::mk_sp(start_bpos, rdr.pos)
                        })
                    } else {
                        None
                    }
                };

                if ret.is_some() {
                    return ret;
                }
            } else {
                while rdr.curr != '\n' && !is_eof(rdr) { bump(rdr); }
            }
            // Restart whitespace munch.
            return consume_whitespace_and_comments(rdr);
          }
          '*' => { bump(rdr); bump(rdr); return consume_block_comment(rdr); }
          _ => ()
        }
    } else if rdr.curr == '#' {
        if nextch(rdr) == '!' {
            // I guess this is the only way to figure out if
            // we're at the beginning of the file...
            let cmap = @CodeMap::new();
            (*cmap).files.push(rdr.filemap);
            let loc = cmap.lookup_char_pos_adj(rdr.last_pos);
            if loc.line == 1u && loc.col == CharPos(0u) {
                while rdr.curr != '\n' && !is_eof(rdr) { bump(rdr); }
                return consume_whitespace_and_comments(rdr);
            }
        }
    }
    return None;
}

pub fn is_block_non_doc_comment(s: &str) -> bool {
    assert!(s.len() >= 1u);
    s.slice(1u, s.len() - 1u).iter().all(|ch| ch == '*')
}

// might return a sugared-doc-attr
fn consume_block_comment(rdr: @mut StringReader)
                      -> Option<TokenAndSpan> {
    // block comments starting with "/**" or "/*!" are doc-comments
    let res = if rdr.curr == '*' || rdr.curr == '!' {
        let start_bpos = rdr.pos - BytePos(3u);
        while !(rdr.curr == '*' && nextch(rdr) == '/') && !is_eof(rdr) {
            bump(rdr);
        }
        if is_eof(rdr) {
            rdr.fatal(~"unterminated block doc-comment");
        } else {
            bump(rdr);
            bump(rdr);
            do with_str_from(rdr, start_bpos) |string| {
                // but comments with only "*"s between two "/"s are not
                if !is_block_non_doc_comment(string) {
                    Some(TokenAndSpan{
                         tok: token::DOC_COMMENT(str_to_ident(string)),
                         sp: codemap::mk_sp(start_bpos, rdr.pos)
                         })
                } else {
                    None
                }
            }
        }
    } else {
        loop {
            if is_eof(rdr) { rdr.fatal(~"unterminated block comment"); }
            if rdr.curr == '*' && nextch(rdr) == '/' {
                bump(rdr);
                bump(rdr);
                break;
            } else {
                bump(rdr);
            }
        }
        None
    };
    // restart whitespace munch.

   if res.is_some() { res } else { consume_whitespace_and_comments(rdr) }
}

fn scan_exponent(rdr: @mut StringReader) -> Option<~str> {
    let mut c = rdr.curr;
    let mut rslt = ~"";
    if c == 'e' || c == 'E' {
        rslt.push_char(c);
        bump(rdr);
        c = rdr.curr;
        if c == '-' || c == '+' {
            rslt.push_char(c);
            bump(rdr);
        }
        let exponent = scan_digits(rdr, 10u);
        if exponent.len() > 0u {
            return Some(rslt + exponent);
        } else { rdr.fatal(~"scan_exponent: bad fp literal"); }
    } else { return None::<~str>; }
}

fn scan_digits(rdr: @mut StringReader, radix: uint) -> ~str {
    let mut rslt = ~"";
    loop {
        let c = rdr.curr;
        if c == '_' { bump(rdr); loop; }
        match char::to_digit(c, radix) {
          Some(_) => {
            rslt.push_char(c);
            bump(rdr);
          }
          _ => return rslt
        }
    };
}

fn scan_number(c: char, rdr: @mut StringReader) -> token::Token {
    let mut num_str;
    let mut base = 10u;
    let mut c = c;
    let mut n = nextch(rdr);
    if c == '0' && n == 'x' {
        bump(rdr);
        bump(rdr);
        base = 16u;
    } else if c == '0' && n == 'b' {
        bump(rdr);
        bump(rdr);
        base = 2u;
    }
    num_str = scan_digits(rdr, base);
    c = rdr.curr;
    nextch(rdr);
    if c == 'u' || c == 'i' {
        let signed = c == 'i';
        let mut tp = {
            if signed { either::Left(ast::ty_i) }
            else { either::Right(ast::ty_u) }
        };
        bump(rdr);
        c = rdr.curr;
        if c == '8' {
            bump(rdr);
            tp = if signed { either::Left(ast::ty_i8) }
                      else { either::Right(ast::ty_u8) };
        }
        n = nextch(rdr);
        if c == '1' && n == '6' {
            bump(rdr);
            bump(rdr);
            tp = if signed { either::Left(ast::ty_i16) }
                      else { either::Right(ast::ty_u16) };
        } else if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            tp = if signed { either::Left(ast::ty_i32) }
                      else { either::Right(ast::ty_u32) };
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
            tp = if signed { either::Left(ast::ty_i64) }
                      else { either::Right(ast::ty_u64) };
        }
        if num_str.len() == 0u {
            rdr.fatal(~"no valid digits found for number");
        }
        let parsed = match u64::from_str_radix(num_str, base as uint) {
            Some(p) => p,
            None => rdr.fatal(~"int literal is too large")
        };

        match tp {
          either::Left(t) => return token::LIT_INT(parsed as i64, t),
          either::Right(t) => return token::LIT_UINT(parsed, t)
        }
    }
    let mut is_float = false;
    if rdr.curr == '.' && !(ident_start(nextch(rdr)) || nextch(rdr) == '.') {
        is_float = true;
        bump(rdr);
        let dec_part = scan_digits(rdr, 10u);
        num_str.push_char('.');
        num_str.push_str(dec_part);
    }
    if is_float {
        match base {
          16u => rdr.fatal(~"hexadecimal float literal is not supported"),
          2u => rdr.fatal(~"binary float literal is not supported"),
          _ => ()
        }
    }
    match scan_exponent(rdr) {
      Some(ref s) => {
        is_float = true;
        num_str.push_str(*s);
      }
      None => ()
    }

    let mut is_machine_float = false;
    if rdr.curr == 'f' {
        bump(rdr);
        c = rdr.curr;
        n = nextch(rdr);
        if c == '3' && n == '2' {
            bump(rdr);
            bump(rdr);
            return token::LIT_FLOAT(str_to_ident(num_str),
                                 ast::ty_f32);
        } else if c == '6' && n == '4' {
            bump(rdr);
            bump(rdr);
            return token::LIT_FLOAT(str_to_ident(num_str),
                                 ast::ty_f64);
            /* FIXME (#2252): if this is out of range for either a
            32-bit or 64-bit float, it won't be noticed till the
            back-end.  */
        } else {
            is_float = true;
            is_machine_float = true;
        }
    }
    if is_float {
        if is_machine_float {
            return token::LIT_FLOAT(str_to_ident(num_str), ast::ty_f);
        }
        return token::LIT_FLOAT_UNSUFFIXED(str_to_ident(num_str));
    } else {
        if num_str.len() == 0u {
            rdr.fatal(~"no valid digits found for number");
        }
        let parsed = match u64::from_str_radix(num_str, base as uint) {
            Some(p) => p,
            None => rdr.fatal(~"int literal is too large")
        };

        debug!("lexing %s as an unsuffixed integer literal",
               num_str);
        return token::LIT_INT_UNSUFFIXED(parsed as i64);
    }
}

fn scan_numeric_escape(rdr: @mut StringReader, n_hex_digits: uint) -> char {
    let mut accum_int = 0;
    let mut i = n_hex_digits;
    while i != 0u {
        let n = rdr.curr;
        bump(rdr);
        if !is_hex_digit(n) {
            rdr.fatal(fmt!("illegal numeric character escape: %d", n as int));
        }
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        i -= 1u;
    }
    return accum_int as char;
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
fn next_token_inner(rdr: @mut StringReader) -> token::Token {
    let c = rdr.curr;
    if ident_start(c) {
        let start = rdr.last_pos;
        while ident_continue(rdr.curr) {
            bump(rdr);
        }

        return do with_str_from(rdr, start) |string| {
            if string == "_" {
                token::UNDERSCORE
            } else {
                let is_mod_name = rdr.curr == ':' && nextch(rdr) == ':';

                // FIXME: perform NFKC normalization here. (Issue #2253)
                token::IDENT(str_to_ident(string), is_mod_name)
            }
        }
    }
    if is_dec_digit(c) {
        return scan_number(c, rdr);
    }
    fn binop(rdr: @mut StringReader, op: token::binop) -> token::Token {
        bump(rdr);
        if rdr.curr == '=' {
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
        if rdr.curr == '.' && nextch(rdr) != '.' {
            bump(rdr);
            return token::DOTDOT;
        }
        return token::DOT;
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
        if rdr.curr == ':' {
            bump(rdr);
            return token::MOD_SEP;
        } else { return token::COLON; }
      }

      '$' => { bump(rdr); return token::DOLLAR; }





      // Multi-byte tokens.
      '=' => {
        bump(rdr);
        if rdr.curr == '=' {
            bump(rdr);
            return token::EQEQ;
        } else if rdr.curr == '>' {
            bump(rdr);
            return token::FAT_ARROW;
        } else {
            return token::EQ;
        }
      }
      '!' => {
        bump(rdr);
        if rdr.curr == '=' {
            bump(rdr);
            return token::NE;
        } else { return token::NOT; }
      }
      '<' => {
        bump(rdr);
        match rdr.curr {
          '=' => { bump(rdr); return token::LE; }
          '<' => { return binop(rdr, token::SHL); }
          '-' => {
            bump(rdr);
            match rdr.curr {
              '>' => { bump(rdr); return token::DARROW; }
              _ => { return token::LARROW; }
            }
          }
          _ => { return token::LT; }
        }
      }
      '>' => {
        bump(rdr);
        match rdr.curr {
          '=' => { bump(rdr); return token::GE; }
          '>' => { return binop(rdr, token::SHR); }
          _ => { return token::GT; }
        }
      }
      '\'' => {
        // Either a character constant 'a' OR a lifetime name 'abc
        bump(rdr);
        let start = rdr.last_pos;
        let mut c2 = rdr.curr;
        bump(rdr);

        // If the character is an ident start not followed by another single
        // quote, then this is a lifetime name:
        if ident_start(c2) && rdr.curr != '\'' {
            while ident_continue(rdr.curr) {
                bump(rdr);
            }
            return do with_str_from(rdr, start) |lifetime_name| {
                token::LIFETIME(str_to_ident(lifetime_name))
            }
        }

        // Otherwise it is a character constant:
        if c2 == '\\' {
            // '\X' for some X must be a character constant:
            let escaped = rdr.curr;
            bump(rdr);
            match escaped {
              'n' => { c2 = '\n'; }
              'r' => { c2 = '\r'; }
              't' => { c2 = '\t'; }
              '\\' => { c2 = '\\'; }
              '\'' => { c2 = '\''; }
              '"' => { c2 = '"'; }
              'x' => { c2 = scan_numeric_escape(rdr, 2u); }
              'u' => { c2 = scan_numeric_escape(rdr, 4u); }
              'U' => { c2 = scan_numeric_escape(rdr, 8u); }
              c2 => {
                rdr.fatal(fmt!("unknown character escape: %d", c2 as int));
              }
            }
        }
        if rdr.curr != '\'' {
            rdr.fatal(~"unterminated character constant");
        }
        bump(rdr); // advance curr past token
        return token::LIT_INT(c2 as i64, ast::ty_char);
      }
      '"' => {
        let mut accum_str = ~"";
        let n = rdr.last_pos;
        bump(rdr);
        while rdr.curr != '"' {
            if is_eof(rdr) {
                do with_str_from(rdr, n) |s| {
                    rdr.fatal(fmt!("unterminated double quote string: %s", s));
                }
            }

            let ch = rdr.curr;
            bump(rdr);
            match ch {
              '\\' => {
                let escaped = rdr.curr;
                bump(rdr);
                match escaped {
                  'n' => accum_str.push_char('\n'),
                  'r' => accum_str.push_char('\r'),
                  't' => accum_str.push_char('\t'),
                  '\\' => accum_str.push_char('\\'),
                  '\'' => accum_str.push_char('\''),
                  '"' => accum_str.push_char('"'),
                  '\n' => consume_whitespace(rdr),
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
                    rdr.fatal(fmt!("unknown string escape: %d", c2 as int));
                  }
                }
              }
              _ => accum_str.push_char(ch)
            }
        }
        bump(rdr);
        return token::LIT_STR(str_to_ident(accum_str));
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
          // So the error span points to the unrecognized character
          rdr.peek_span = codemap::mk_sp(rdr.last_pos, rdr.pos);
          let mut cs = ~"";
          char::escape_default(c, |c| cs.push_char(c));
          rdr.fatal(fmt!("unknown start of token: %s", cs));
      }
    }
}

fn consume_whitespace(rdr: @mut StringReader) {
    while is_whitespace(rdr.curr) && !is_eof(rdr) { bump(rdr); }
}

#[cfg(test)]
mod test {
    use super::*;

    use ast;
    use codemap::{BytePos, CodeMap, span};
    use diagnostic;
    use parse::token;
    use parse::token::{str_to_ident};

    // represents a testing reader (incl. both reader and interner)
    struct Env {
        string_reader: @mut StringReader
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
                    fn main() { io::println(~\"zebra\"); }\n");
        let id = str_to_ident("fn");
        let tok1 = string_reader.next_token();
        let tok2 = TokenAndSpan{
            tok:token::IDENT(id, false),
            sp:span {lo:BytePos(21),hi:BytePos(23),expn_info: None}};
        assert_eq!(tok1,tok2);
        // the 'main' id is already read:
        assert_eq!(string_reader.last_pos.clone(), BytePos(28));
        // read another token:
        let tok3 = string_reader.next_token();
        let tok4 = TokenAndSpan{
            tok:token::IDENT(str_to_ident("main"), false),
            sp:span {lo:BytePos(24),hi:BytePos(28),expn_info: None}};
        assert_eq!(tok3,tok4);
        // the lparen is already read:
        assert_eq!(string_reader.last_pos.clone(), BytePos(29))
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
        assert_eq!(tok,token::LIT_INT('a' as i64, ast::ty_char));
    }

    #[test] fn character_space() {
        let env = setup(@"' '");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        assert_eq!(tok, token::LIT_INT(' ' as i64, ast::ty_char));
    }

    #[test] fn character_escaped() {
        let env = setup(@"'\n'");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        assert_eq!(tok, token::LIT_INT('\n' as i64, ast::ty_char));
    }

    #[test] fn lifetime_name() {
        let env = setup(@"'abc");
        let TokenAndSpan {tok, sp: _} =
            env.string_reader.next_token();
        let id = token::str_to_ident("abc");
        assert_eq!(tok, token::LIFETIME(id));
    }

    #[test] fn line_doc_comments() {
        assert!(!is_line_non_doc_comment("///"));
        assert!(!is_line_non_doc_comment("/// blah"));
        assert!(is_line_non_doc_comment("////"));
    }
}
