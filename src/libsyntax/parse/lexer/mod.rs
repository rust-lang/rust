use crate::parse::ParseSess;
use crate::parse::token::{self, Token, TokenKind};
use crate::symbol::{sym, Symbol};
use crate::parse::unescape;
use crate::parse::unescape_error_reporting::{emit_unescape_error, push_escaped_char};

use errors::{FatalError, Diagnostic, DiagnosticBuilder};
use syntax_pos::{BytePos, Pos, Span, NO_EXPANSION};
use core::unicode::property::Pattern_White_Space;

use std::borrow::Cow;
use std::char;
use std::iter;
use rustc_data_structures::sync::Lrc;
use log::debug;

pub mod comments;
mod tokentrees;
mod unicode_chars;

#[derive(Clone, Debug)]
pub struct UnmatchedBrace {
    pub expected_delim: token::DelimToken,
    pub found_delim: token::DelimToken,
    pub found_span: Span,
    pub unclosed_span: Option<Span>,
    pub candidate_span: Option<Span>,
}

pub struct StringReader<'a> {
    crate sess: &'a ParseSess,
    /// The absolute offset within the source_map of the next character to read
    crate next_pos: BytePos,
    /// The absolute offset within the source_map of the current character
    crate pos: BytePos,
    /// The current character (which has been read from self.pos)
    crate ch: Option<char>,
    crate source_file: Lrc<syntax_pos::SourceFile>,
    /// Stop reading src at this index.
    crate end_src_index: usize,
    // cached:
    peek_token: Token,
    peek_span_src_raw: Span,
    fatal_errs: Vec<DiagnosticBuilder<'a>>,
    // cache a direct reference to the source text, so that we don't have to
    // retrieve it via `self.source_file.src.as_ref().unwrap()` all the time.
    src: Lrc<String>,
    override_span: Option<Span>,
}

impl<'a> StringReader<'a> {
    fn mk_sp(&self, lo: BytePos, hi: BytePos) -> Span {
        self.mk_sp_and_raw(lo, hi).0
    }

    fn mk_sp_and_raw(&self, lo: BytePos, hi: BytePos) -> (Span, Span) {
        let raw = Span::new(lo, hi, NO_EXPANSION);
        let real = self.override_span.unwrap_or(raw);

        (real, raw)
    }

    fn unwrap_or_abort(&mut self, res: Result<Token, ()>) -> Token {
        match res {
            Ok(tok) => tok,
            Err(_) => {
                self.emit_fatal_errors();
                FatalError.raise();
            }
        }
    }

    fn next_token(&mut self) -> Token where Self: Sized {
        let res = self.try_next_token();
        self.unwrap_or_abort(res)
    }

    /// Returns the next token. EFFECT: advances the string_reader.
    pub fn try_next_token(&mut self) -> Result<Token, ()> {
        assert!(self.fatal_errs.is_empty());
        let ret_val = self.peek_token.take();
        self.advance_token()?;
        Ok(ret_val)
    }

    fn try_real_token(&mut self) -> Result<Token, ()> {
        let mut t = self.try_next_token()?;
        loop {
            match t.kind {
                token::Whitespace | token::Comment | token::Shebang(_) => {
                    t = self.try_next_token()?;
                }
                _ => break,
            }
        }

        Ok(t)
    }

    pub fn real_token(&mut self) -> Token {
        let res = self.try_real_token();
        self.unwrap_or_abort(res)
    }

    #[inline]
    fn is_eof(&self) -> bool {
        self.ch.is_none()
    }

    fn fail_unterminated_raw_string(&self, pos: BytePos, hash_count: u16) -> ! {
        let mut err = self.struct_span_fatal(pos, pos, "unterminated raw string");
        err.span_label(self.mk_sp(pos, pos), "unterminated raw string");

        if hash_count > 0 {
            err.note(&format!("this raw string should be terminated with `\"{}`",
                              "#".repeat(hash_count as usize)));
        }

        err.emit();
        FatalError.raise();
    }

    fn fatal(&self, m: &str) -> FatalError {
        self.fatal_span(self.peek_token.span, m)
    }

    crate fn emit_fatal_errors(&mut self) {
        for err in &mut self.fatal_errs {
            err.emit();
        }

        self.fatal_errs.clear();
    }

    pub fn buffer_fatal_errors(&mut self) -> Vec<Diagnostic> {
        let mut buffer = Vec::new();

        for err in self.fatal_errs.drain(..) {
            err.buffer(&mut buffer);
        }

        buffer
    }

    pub fn peek(&self) -> &Token {
        &self.peek_token
    }

    /// For comments.rs, which hackily pokes into next_pos and ch
    fn new_raw(sess: &'a ParseSess,
               source_file: Lrc<syntax_pos::SourceFile>,
               override_span: Option<Span>) -> Self {
        let mut sr = StringReader::new_raw_internal(sess, source_file, override_span);
        sr.bump();

        sr
    }

    fn new_raw_internal(sess: &'a ParseSess, source_file: Lrc<syntax_pos::SourceFile>,
        override_span: Option<Span>) -> Self
    {
        if source_file.src.is_none() {
            sess.span_diagnostic.bug(&format!("Cannot lex source_file without source: {}",
                                              source_file.name));
        }

        let src = (*source_file.src.as_ref().unwrap()).clone();

        StringReader {
            sess,
            next_pos: source_file.start_pos,
            pos: source_file.start_pos,
            ch: Some('\n'),
            source_file,
            end_src_index: src.len(),
            peek_token: Token::dummy(),
            peek_span_src_raw: syntax_pos::DUMMY_SP,
            src,
            fatal_errs: Vec::new(),
            override_span,
        }
    }

    pub fn new_or_buffered_errs(sess: &'a ParseSess,
                                source_file: Lrc<syntax_pos::SourceFile>,
                                override_span: Option<Span>) -> Result<Self, Vec<Diagnostic>> {
        let mut sr = StringReader::new_raw(sess, source_file, override_span);
        if sr.advance_token().is_err() {
            Err(sr.buffer_fatal_errors())
        } else {
            Ok(sr)
        }
    }

    pub fn retokenize(sess: &'a ParseSess, mut span: Span) -> Self {
        let begin = sess.source_map().lookup_byte_offset(span.lo());
        let end = sess.source_map().lookup_byte_offset(span.hi());

        // Make the range zero-length if the span is invalid.
        if span.lo() > span.hi() || begin.sf.start_pos != end.sf.start_pos {
            span = span.shrink_to_lo();
        }

        let mut sr = StringReader::new_raw_internal(sess, begin.sf, None);

        // Seek the lexer to the right byte range.
        sr.next_pos = span.lo();
        sr.end_src_index = sr.src_index(span.hi());

        sr.bump();

        if sr.advance_token().is_err() {
            sr.emit_fatal_errors();
            FatalError.raise();
        }

        sr
    }

    #[inline]
    fn ch_is(&self, c: char) -> bool {
        self.ch == Some(c)
    }

    /// Report a fatal lexical error with a given span.
    fn fatal_span(&self, sp: Span, m: &str) -> FatalError {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }

    /// Report a lexical error with a given span.
    fn err_span(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.struct_span_err(sp, m).emit();
    }


    /// Report a fatal error spanning [`from_pos`, `to_pos`).
    fn fatal_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) -> FatalError {
        self.fatal_span(self.mk_sp(from_pos, to_pos), m)
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`).
    fn err_span_(&self, from_pos: BytePos, to_pos: BytePos, m: &str) {
        self.err_span(self.mk_sp(from_pos, to_pos), m)
    }

    /// Report a lexical error spanning [`from_pos`, `to_pos`), appending an
    /// escaped character to the error message
    fn fatal_span_char(&self, from_pos: BytePos, to_pos: BytePos, m: &str, c: char) -> FatalError {
        let mut m = m.to_string();
        m.push_str(": ");
        push_escaped_char(&mut m, c);

        self.fatal_span_(from_pos, to_pos, &m[..])
    }

    fn struct_span_fatal(&self, from_pos: BytePos, to_pos: BytePos, m: &str)
        -> DiagnosticBuilder<'a>
    {
        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), m)
    }

    fn struct_fatal_span_char(&self, from_pos: BytePos, to_pos: BytePos, m: &str, c: char)
        -> DiagnosticBuilder<'a>
    {
        let mut m = m.to_string();
        m.push_str(": ");
        push_escaped_char(&mut m, c);

        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), &m[..])
    }

    /// Advance peek_token to refer to the next token, and
    /// possibly update the interner.
    fn advance_token(&mut self) -> Result<(), ()> {
        match self.scan_whitespace_or_comment() {
            Some(comment) => {
                self.peek_span_src_raw = comment.span;
                self.peek_token = comment;
            }
            None => {
                let (kind, start_pos, end_pos) = if self.is_eof() {
                    (token::Eof, self.source_file.end_pos, self.source_file.end_pos)
                } else {
                    let start_pos = self.pos;
                    (self.next_token_inner()?, start_pos, self.pos)
                };
                let (real, raw) = self.mk_sp_and_raw(start_pos, end_pos);
                self.peek_token = Token::new(kind, real);
                self.peek_span_src_raw = raw;
            }
        }

        Ok(())
    }

    #[inline]
    fn src_index(&self, pos: BytePos) -> usize {
        (pos - self.source_file.start_pos).to_usize()
    }

    /// Slice of the source text from `start` up to but excluding `self.pos`,
    /// meaning the slice does not include the character `self.ch`.
    fn str_from(&self, start: BytePos) -> &str
    {
        self.str_from_to(start, self.pos)
    }

    /// Creates a Symbol from a given offset to the current offset.
    fn symbol_from(&self, start: BytePos) -> Symbol {
        debug!("taking an ident from {:?} to {:?}", start, self.pos);
        Symbol::intern(self.str_from(start))
    }

    /// As symbol_from, with an explicit endpoint.
    fn symbol_from_to(&self, start: BytePos, end: BytePos) -> Symbol {
        debug!("taking an ident from {:?} to {:?}", start, end);
        Symbol::intern(self.str_from_to(start, end))
    }

    /// Slice of the source text spanning from `start` up to but excluding `end`.
    fn str_from_to(&self, start: BytePos, end: BytePos) -> &str
    {
        &self.src[self.src_index(start)..self.src_index(end)]
    }

    /// Converts CRLF to LF in the given string, raising an error on bare CR.
    fn translate_crlf<'b>(&self, start: BytePos, s: &'b str, errmsg: &'b str) -> Cow<'b, str> {
        let mut chars = s.char_indices().peekable();
        while let Some((i, ch)) = chars.next() {
            if ch == '\r' {
                if let Some((lf_idx, '\n')) = chars.peek() {
                    return translate_crlf_(self, start, s, *lf_idx, chars, errmsg).into();
                }
                let pos = start + BytePos(i as u32);
                let end_pos = start + BytePos((i + ch.len_utf8()) as u32);
                self.err_span_(pos, end_pos, errmsg);
            }
        }
        return s.into();

        fn translate_crlf_(rdr: &StringReader<'_>,
                           start: BytePos,
                           s: &str,
                           mut j: usize,
                           mut chars: iter::Peekable<impl Iterator<Item = (usize, char)>>,
                           errmsg: &str)
                           -> String {
            let mut buf = String::with_capacity(s.len());
            // Skip first CR
            buf.push_str(&s[.. j - 1]);
            while let Some((i, ch)) = chars.next() {
                if ch == '\r' {
                    if j < i {
                        buf.push_str(&s[j..i]);
                    }
                    let next = i + ch.len_utf8();
                    j = next;
                    if chars.peek().map(|(_, ch)| *ch) != Some('\n') {
                        let pos = start + BytePos(i as u32);
                        let end_pos = start + BytePos(next as u32);
                        rdr.err_span_(pos, end_pos, errmsg);
                    }
                }
            }
            if j < s.len() {
                buf.push_str(&s[j..]);
            }
            buf
        }
    }

    /// Advance the StringReader by one character.
    crate fn bump(&mut self) {
        let next_src_index = self.src_index(self.next_pos);
        if next_src_index < self.end_src_index {
            let next_ch = char_at(&self.src, next_src_index);
            let next_ch_len = next_ch.len_utf8();

            self.ch = Some(next_ch);
            self.pos = self.next_pos;
            self.next_pos = self.next_pos + Pos::from_usize(next_ch_len);
        } else {
            self.ch = None;
            self.pos = self.next_pos;
        }
    }

    fn nextch(&self) -> Option<char> {
        let next_src_index = self.src_index(self.next_pos);
        if next_src_index < self.end_src_index {
            Some(char_at(&self.src, next_src_index))
        } else {
            None
        }
    }

    #[inline]
    fn nextch_is(&self, c: char) -> bool {
        self.nextch() == Some(c)
    }

    fn nextnextch(&self) -> Option<char> {
        let next_src_index = self.src_index(self.next_pos);
        if next_src_index < self.end_src_index {
            let next_next_src_index =
                next_src_index + char_at(&self.src, next_src_index).len_utf8();
            if next_next_src_index < self.end_src_index {
                return Some(char_at(&self.src, next_next_src_index));
            }
        }
        None
    }

    #[inline]
    fn nextnextch_is(&self, c: char) -> bool {
        self.nextnextch() == Some(c)
    }

    /// Eats <XID_start><XID_continue>*, if possible.
    fn scan_optional_raw_name(&mut self) -> Option<Symbol> {
        if !ident_start(self.ch) {
            return None;
        }

        let start = self.pos;
        self.bump();

        while ident_continue(self.ch) {
            self.bump();
        }

        match self.str_from(start) {
            "_" => {
                self.sess.span_diagnostic
                    .struct_span_warn(self.mk_sp(start, self.pos),
                                      "underscore literal suffix is not allowed")
                    .warn("this was previously accepted by the compiler but is \
                          being phased out; it will become a hard error in \
                          a future release!")
                    .note("for more information, see issue #42326 \
                          <https://github.com/rust-lang/rust/issues/42326>")
                    .emit();
                None
            }
            name => Some(Symbol::intern(name))
        }
    }

    /// PRECONDITION: self.ch is not whitespace
    /// Eats any kind of comment.
    fn scan_comment(&mut self) -> Option<Token> {
        if let Some(c) = self.ch {
            if c.is_whitespace() {
                let msg = "called consume_any_line_comment, but there was whitespace";
                self.sess.span_diagnostic.span_err(self.mk_sp(self.pos, self.pos), msg);
            }
        }

        if self.ch_is('/') {
            match self.nextch() {
                Some('/') => {
                    self.bump();
                    self.bump();

                    // line comments starting with "///" or "//!" are doc-comments
                    let doc_comment = (self.ch_is('/') && !self.nextch_is('/')) || self.ch_is('!');
                    let start_bpos = self.pos - BytePos(2);

                    while !self.is_eof() {
                        match self.ch.unwrap() {
                            '\n' => break,
                            '\r' => {
                                if self.nextch_is('\n') {
                                    // CRLF
                                    break;
                                } else if doc_comment {
                                    self.err_span_(self.pos,
                                                   self.next_pos,
                                                   "bare CR not allowed in doc-comment");
                                }
                            }
                            _ => (),
                        }
                        self.bump();
                    }

                    let kind = if doc_comment {
                        token::DocComment(self.symbol_from(start_bpos))
                    } else {
                        token::Comment
                    };
                    Some(Token::new(kind, self.mk_sp(start_bpos, self.pos)))
                }
                Some('*') => {
                    self.bump();
                    self.bump();
                    self.scan_block_comment()
                }
                _ => None,
            }
        } else if self.ch_is('#') {
            if self.nextch_is('!') {

                // Parse an inner attribute.
                if self.nextnextch_is('[') {
                    return None;
                }

                let is_beginning_of_file = self.pos == self.source_file.start_pos;
                if is_beginning_of_file {
                    debug!("Skipping a shebang");
                    let start = self.pos;
                    while !self.ch_is('\n') && !self.is_eof() {
                        self.bump();
                    }
                    return Some(Token::new(
                        token::Shebang(self.symbol_from(start)),
                        self.mk_sp(start, self.pos),
                    ));
                }
            }
            None
        } else {
            None
        }
    }

    /// If there is whitespace, shebang, or a comment, scan it. Otherwise,
    /// return `None`.
    fn scan_whitespace_or_comment(&mut self) -> Option<Token> {
        match self.ch.unwrap_or('\0') {
            // # to handle shebang at start of file -- this is the entry point
            // for skipping over all "junk"
            '/' | '#' => {
                let c = self.scan_comment();
                debug!("scanning a comment {:?}", c);
                c
            },
            c if is_pattern_whitespace(Some(c)) => {
                let start_bpos = self.pos;
                while is_pattern_whitespace(self.ch) {
                    self.bump();
                }
                let c = Some(Token::new(token::Whitespace, self.mk_sp(start_bpos, self.pos)));
                debug!("scanning whitespace: {:?}", c);
                c
            }
            _ => None,
        }
    }

    /// Might return a sugared-doc-attr
    fn scan_block_comment(&mut self) -> Option<Token> {
        // block comments starting with "/**" or "/*!" are doc-comments
        let is_doc_comment = self.ch_is('*') || self.ch_is('!');
        let start_bpos = self.pos - BytePos(2);

        let mut level: isize = 1;
        let mut has_cr = false;
        while level > 0 {
            if self.is_eof() {
                let msg = if is_doc_comment {
                    "unterminated block doc-comment"
                } else {
                    "unterminated block comment"
                };
                let last_bpos = self.pos;
                self.fatal_span_(start_bpos, last_bpos, msg).raise();
            }
            let n = self.ch.unwrap();
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

        let string = self.str_from(start_bpos);
        // but comments with only "*"s between two "/"s are not
        let kind = if is_block_doc_comment(string) {
            let string = if has_cr {
                self.translate_crlf(start_bpos,
                                    string,
                                    "bare CR not allowed in block doc-comment")
            } else {
                string.into()
            };
            token::DocComment(Symbol::intern(&string[..]))
        } else {
            token::Comment
        };

        Some(Token::new(kind, self.mk_sp(start_bpos, self.pos)))
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
            let c = self.ch;
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
                        self.err_span_(self.pos,
                                       self.next_pos,
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
    fn scan_number(&mut self, c: char) -> (token::LitKind, Symbol) {
        let mut base = 10;
        let start_bpos = self.pos;
        self.bump();

        let num_digits = if c == '0' {
            match self.ch.unwrap_or('\0') {
                'b' => {
                    self.bump();
                    base = 2;
                    self.scan_digits(2, 10)
                }
                'o' => {
                    self.bump();
                    base = 8;
                    self.scan_digits(8, 10)
                }
                'x' => {
                    self.bump();
                    base = 16;
                    self.scan_digits(16, 16)
                }
                '0'..='9' | '_' | '.' | 'e' | 'E' => {
                    self.scan_digits(10, 10) + 1
                }
                _ => {
                    // just a 0
                    return (token::Integer, sym::integer(0));
                }
            }
        } else if c.is_digit(10) {
            self.scan_digits(10, 10) + 1
        } else {
            0
        };

        if num_digits == 0 {
            self.err_span_(start_bpos, self.pos, "no valid digits found for number");

            return (token::Integer, Symbol::intern("0"));
        }

        // might be a float, but don't be greedy if this is actually an
        // integer literal followed by field/method access or a range pattern
        // (`0..2` and `12.foo()`)
        if self.ch_is('.') && !self.nextch_is('.') &&
           !ident_start(self.nextch()) {
            // might have stuff after the ., and if it does, it needs to start
            // with a number
            self.bump();
            if self.ch.unwrap_or('\0').is_digit(10) {
                self.scan_digits(10, 10);
                self.scan_float_exponent();
            }
            let pos = self.pos;
            self.check_float_base(start_bpos, pos, base);

            (token::Float, self.symbol_from(start_bpos))
        } else {
            // it might be a float if it has an exponent
            if self.ch_is('e') || self.ch_is('E') {
                self.scan_float_exponent();
                let pos = self.pos;
                self.check_float_base(start_bpos, pos, base);
                return (token::Float, self.symbol_from(start_bpos));
            }
            // but we certainly have an integer!
            (token::Integer, self.symbol_from(start_bpos))
        }
    }

    /// Scan over a float exponent.
    fn scan_float_exponent(&mut self) {
        if self.ch_is('e') || self.ch_is('E') {
            self.bump();

            if self.ch_is('-') || self.ch_is('+') {
                self.bump();
            }

            if self.scan_digits(10, 10) == 0 {
                let mut err = self.struct_span_fatal(
                    self.pos, self.next_pos,
                    "expected at least one digit in exponent"
                );
                if let Some(ch) = self.ch {
                    // check for e.g., Unicode minus '−' (Issue #49746)
                    if unicode_chars::check_for_substitution(self, ch, &mut err) {
                        self.bump();
                        self.scan_digits(10, 10);
                    }
                }
                err.emit();
            }
        }
    }

    /// Checks that a base is valid for a floating literal, emitting a nice
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

    fn binop(&mut self, op: token::BinOpToken) -> TokenKind {
        self.bump();
        if self.ch_is('=') {
            self.bump();
            token::BinOpEq(op)
        } else {
            token::BinOp(op)
        }
    }

    /// Returns the next token from the string, advances the input past that
    /// token, and updates the interner
    fn next_token_inner(&mut self) -> Result<TokenKind, ()> {
        let c = self.ch;

        if ident_start(c) {
            let (is_ident_start, is_raw_ident) =
                match (c.unwrap(), self.nextch(), self.nextnextch()) {
                    // r# followed by an identifier starter is a raw identifier.
                    // This is an exception to the r# case below.
                    ('r', Some('#'), x) if ident_start(x) => (true, true),
                    // r as in r" or r#" is part of a raw string literal.
                    // b as in b' is part of a byte literal.
                    // They are not identifiers, and are handled further down.
                    ('r', Some('"'), _) |
                    ('r', Some('#'), _) |
                    ('b', Some('"'), _) |
                    ('b', Some('\''), _) |
                    ('b', Some('r'), Some('"')) |
                    ('b', Some('r'), Some('#')) => (false, false),
                    _ => (true, false),
                };

            if is_ident_start {
                let raw_start = self.pos;
                if is_raw_ident {
                    // Consume the 'r#' characters.
                    self.bump();
                    self.bump();
                }

                let start = self.pos;
                self.bump();

                while ident_continue(self.ch) {
                    self.bump();
                }

                // FIXME: perform NFKC normalization here. (Issue #2253)
                let name = self.symbol_from(start);
                if is_raw_ident {
                    let span = self.mk_sp(raw_start, self.pos);
                    if !name.can_be_raw() {
                        self.err_span(span, &format!("`{}` cannot be a raw identifier", name));
                    }
                    self.sess.raw_identifier_spans.borrow_mut().push(span);
                }

                return Ok(token::Ident(name, is_raw_ident));
            }
        }

        if is_dec_digit(c) {
            let (kind, symbol) = self.scan_number(c.unwrap());
            let suffix = self.scan_optional_raw_name();
            debug!("next_token_inner: scanned number {:?}, {:?}, {:?}", kind, symbol, suffix);
            return Ok(TokenKind::lit(kind, symbol, suffix));
        }

        match c.expect("next_token_inner called at EOF") {
            // One-byte tokens.
            ';' => {
                self.bump();
                Ok(token::Semi)
            }
            ',' => {
                self.bump();
                Ok(token::Comma)
            }
            '.' => {
                self.bump();
                if self.ch_is('.') {
                    self.bump();
                    if self.ch_is('.') {
                        self.bump();
                        Ok(token::DotDotDot)
                    } else if self.ch_is('=') {
                        self.bump();
                        Ok(token::DotDotEq)
                    } else {
                        Ok(token::DotDot)
                    }
                } else {
                    Ok(token::Dot)
                }
            }
            '(' => {
                self.bump();
                Ok(token::OpenDelim(token::Paren))
            }
            ')' => {
                self.bump();
                Ok(token::CloseDelim(token::Paren))
            }
            '{' => {
                self.bump();
                Ok(token::OpenDelim(token::Brace))
            }
            '}' => {
                self.bump();
                Ok(token::CloseDelim(token::Brace))
            }
            '[' => {
                self.bump();
                Ok(token::OpenDelim(token::Bracket))
            }
            ']' => {
                self.bump();
                Ok(token::CloseDelim(token::Bracket))
            }
            '@' => {
                self.bump();
                Ok(token::At)
            }
            '#' => {
                self.bump();
                Ok(token::Pound)
            }
            '~' => {
                self.bump();
                Ok(token::Tilde)
            }
            '?' => {
                self.bump();
                Ok(token::Question)
            }
            ':' => {
                self.bump();
                if self.ch_is(':') {
                    self.bump();
                    Ok(token::ModSep)
                } else {
                    Ok(token::Colon)
                }
            }

            '$' => {
                self.bump();
                Ok(token::Dollar)
            }

            // Multi-byte tokens.
            '=' => {
                self.bump();
                if self.ch_is('=') {
                    self.bump();
                    Ok(token::EqEq)
                } else if self.ch_is('>') {
                    self.bump();
                    Ok(token::FatArrow)
                } else {
                    Ok(token::Eq)
                }
            }
            '!' => {
                self.bump();
                if self.ch_is('=') {
                    self.bump();
                    Ok(token::Ne)
                } else {
                    Ok(token::Not)
                }
            }
            '<' => {
                self.bump();
                match self.ch.unwrap_or('\x00') {
                    '=' => {
                        self.bump();
                        Ok(token::Le)
                    }
                    '<' => {
                        Ok(self.binop(token::Shl))
                    }
                    '-' => {
                        self.bump();
                        Ok(token::LArrow)
                    }
                    _ => {
                        Ok(token::Lt)
                    }
                }
            }
            '>' => {
                self.bump();
                match self.ch.unwrap_or('\x00') {
                    '=' => {
                        self.bump();
                        Ok(token::Ge)
                    }
                    '>' => {
                        Ok(self.binop(token::Shr))
                    }
                    _ => {
                        Ok(token::Gt)
                    }
                }
            }
            '\'' => {
                // Either a character constant 'a' OR a lifetime name 'abc
                let start_with_quote = self.pos;
                self.bump();
                let start = self.pos;

                // If the character is an ident start not followed by another single
                // quote, then this is a lifetime name:
                let starts_with_number = self.ch.unwrap_or('\x00').is_numeric();
                if (ident_start(self.ch) || starts_with_number) && !self.nextch_is('\'') {
                    self.bump();
                    while ident_continue(self.ch) {
                        self.bump();
                    }
                    // lifetimes shouldn't end with a single quote
                    // if we find one, then this is an invalid character literal
                    if self.ch_is('\'') {
                        let symbol = self.symbol_from(start);
                        self.bump();
                        self.validate_char_escape(start_with_quote);
                        return Ok(TokenKind::lit(token::Char, symbol, None));
                    }

                    if starts_with_number {
                        // this is a recovered lifetime written `'1`, error but accept it
                        self.err_span_(
                            start_with_quote,
                            self.pos,
                            "lifetimes cannot start with a number",
                        );
                    }

                    // Include the leading `'` in the real identifier, for macro
                    // expansion purposes. See #12512 for the gory details of why
                    // this is necessary.
                    return Ok(token::Lifetime(self.symbol_from(start_with_quote)));
                }
                let msg = "unterminated character literal";
                let symbol = self.scan_single_quoted_string(start_with_quote, msg);
                self.validate_char_escape(start_with_quote);
                let suffix = self.scan_optional_raw_name();
                Ok(TokenKind::lit(token::Char, symbol, suffix))
            }
            'b' => {
                self.bump();
                let (kind, symbol) = match self.ch {
                    Some('\'') => {
                        let start_with_quote = self.pos;
                        self.bump();
                        let msg = "unterminated byte constant";
                        let symbol = self.scan_single_quoted_string(start_with_quote, msg);
                        self.validate_byte_escape(start_with_quote);
                        (token::Byte, symbol)
                    },
                    Some('"') => {
                        let start_with_quote = self.pos;
                        let msg = "unterminated double quote byte string";
                        let symbol = self.scan_double_quoted_string(msg);
                        self.validate_byte_str_escape(start_with_quote);
                        (token::ByteStr, symbol)
                    },
                    Some('r') => {
                        let (start, end, hash_count) = self.scan_raw_string();
                        let symbol = self.symbol_from_to(start, end);
                        self.validate_raw_byte_str_escape(start, end);

                        (token::ByteStrRaw(hash_count), symbol)
                    }
                    _ => unreachable!(),  // Should have been a token::Ident above.
                };
                let suffix = self.scan_optional_raw_name();

                Ok(TokenKind::lit(kind, symbol, suffix))
            }
            '"' => {
                let start_with_quote = self.pos;
                let msg = "unterminated double quote string";
                let symbol = self.scan_double_quoted_string(msg);
                self.validate_str_escape(start_with_quote);
                let suffix = self.scan_optional_raw_name();
                Ok(TokenKind::lit(token::Str, symbol, suffix))
            }
            'r' => {
                let (start, end, hash_count) = self.scan_raw_string();
                let symbol = self.symbol_from_to(start, end);
                self.validate_raw_str_escape(start, end);
                let suffix = self.scan_optional_raw_name();

                Ok(TokenKind::lit(token::StrRaw(hash_count), symbol, suffix))
            }
            '-' => {
                if self.nextch_is('>') {
                    self.bump();
                    self.bump();
                    Ok(token::RArrow)
                } else {
                    Ok(self.binop(token::Minus))
                }
            }
            '&' => {
                if self.nextch_is('&') {
                    self.bump();
                    self.bump();
                    Ok(token::AndAnd)
                } else {
                    Ok(self.binop(token::And))
                }
            }
            '|' => {
                match self.nextch() {
                    Some('|') => {
                        self.bump();
                        self.bump();
                        Ok(token::OrOr)
                    }
                    _ => {
                        Ok(self.binop(token::Or))
                    }
                }
            }
            '+' => {
                Ok(self.binop(token::Plus))
            }
            '*' => {
                Ok(self.binop(token::Star))
            }
            '/' => {
                Ok(self.binop(token::Slash))
            }
            '^' => {
                Ok(self.binop(token::Caret))
            }
            '%' => {
                Ok(self.binop(token::Percent))
            }
            c => {
                let last_bpos = self.pos;
                let bpos = self.next_pos;
                let mut err = self.struct_fatal_span_char(last_bpos,
                                                          bpos,
                                                          "unknown start of token",
                                                          c);
                unicode_chars::check_for_substitution(self, c, &mut err);
                self.fatal_errs.push(err);

                Err(())
            }
        }
    }

    fn read_to_eol(&mut self) -> String {
        let mut val = String::new();
        while !self.ch_is('\n') && !self.is_eof() {
            val.push(self.ch.unwrap());
            self.bump();
        }

        if self.ch_is('\n') {
            self.bump();
        }

        val
    }

    fn read_one_line_comment(&mut self) -> String {
        let val = self.read_to_eol();
        assert!((val.as_bytes()[0] == b'/' && val.as_bytes()[1] == b'/') ||
                (val.as_bytes()[0] == b'#' && val.as_bytes()[1] == b'!'));
        val
    }

    fn consume_non_eol_whitespace(&mut self) {
        while is_pattern_whitespace(self.ch) && !self.ch_is('\n') && !self.is_eof() {
            self.bump();
        }
    }

    fn peeking_at_comment(&self) -> bool {
        (self.ch_is('/') && self.nextch_is('/')) || (self.ch_is('/') && self.nextch_is('*')) ||
        // consider shebangs comments, but not inner attributes
        (self.ch_is('#') && self.nextch_is('!') && !self.nextnextch_is('['))
    }

    fn scan_single_quoted_string(&mut self,
                                 start_with_quote: BytePos,
                                 unterminated_msg: &str) -> Symbol {
        // assumes that first `'` is consumed
        let start = self.pos;
        // lex `'''` as a single char, for recovery
        if self.ch_is('\'') && self.nextch_is('\'') {
            self.bump();
        } else {
            let mut first = true;
            loop {
                if self.ch_is('\'') {
                    break;
                }
                if self.ch_is('\\') && (self.nextch_is('\'') || self.nextch_is('\\')) {
                    self.bump();
                    self.bump();
                } else {
                    // Only attempt to infer single line string literals. If we encounter
                    // a slash, bail out in order to avoid nonsensical suggestion when
                    // involving comments.
                    if self.is_eof()
                        || (self.ch_is('/') && !first)
                        || (self.ch_is('\n') && !self.nextch_is('\'')) {

                        self.fatal_span_(start_with_quote, self.pos, unterminated_msg.into())
                            .raise()
                    }
                    self.bump();
                }
                first = false;
            }
        }

        let id = self.symbol_from(start);
        self.bump();
        id
    }

    fn scan_double_quoted_string(&mut self, unterminated_msg: &str) -> Symbol {
        debug_assert!(self.ch_is('\"'));
        let start_with_quote = self.pos;
        self.bump();
        let start = self.pos;
        while !self.ch_is('"') {
            if self.is_eof() {
                let pos = self.pos;
                self.fatal_span_(start_with_quote, pos, unterminated_msg).raise();
            }
            if self.ch_is('\\') && (self.nextch_is('\\') || self.nextch_is('"')) {
                self.bump();
            }
            self.bump();
        }
        let id = self.symbol_from(start);
        self.bump();
        id
    }

    /// Scans a raw (byte) string, returning byte position range for `"<literal>"`
    /// (including quotes) along with `#` character count in `(b)r##..."<literal>"##...`;
    fn scan_raw_string(&mut self) -> (BytePos, BytePos, u16) {
        let start_bpos = self.pos;
        self.bump();
        let mut hash_count: u16 = 0;
        while self.ch_is('#') {
            if hash_count == 65535 {
                let bpos = self.next_pos;
                self.fatal_span_(start_bpos,
                                 bpos,
                                 "too many `#` symbols: raw strings may be \
                                 delimited by up to 65535 `#` symbols").raise();
            }
            self.bump();
            hash_count += 1;
        }

        if self.is_eof() {
            self.fail_unterminated_raw_string(start_bpos, hash_count);
        } else if !self.ch_is('"') {
            let last_bpos = self.pos;
            let curr_char = self.ch.unwrap();
            self.fatal_span_char(start_bpos,
                                 last_bpos,
                                 "found invalid character; only `#` is allowed \
                                 in raw string delimitation",
                                 curr_char).raise();
        }
        self.bump();
        let content_start_bpos = self.pos;
        let mut content_end_bpos;
        'outer: loop {
            match self.ch {
                None => {
                    self.fail_unterminated_raw_string(start_bpos, hash_count);
                }
                Some('"') => {
                    content_end_bpos = self.pos;
                    for _ in 0..hash_count {
                        self.bump();
                        if !self.ch_is('#') {
                            continue 'outer;
                        }
                    }
                    break;
                }
                _ => (),
            }
            self.bump();
        }

        self.bump();

        (content_start_bpos, content_end_bpos, hash_count)
    }

    fn validate_char_escape(&self, start_with_quote: BytePos) {
        let lit = self.str_from_to(start_with_quote + BytePos(1), self.pos - BytePos(1));
        if let Err((off, err)) = unescape::unescape_char(lit) {
            emit_unescape_error(
                &self.sess.span_diagnostic,
                lit,
                self.mk_sp(start_with_quote, self.pos),
                unescape::Mode::Char,
                0..off,
                err,
            )
        }
    }

    fn validate_byte_escape(&self, start_with_quote: BytePos) {
        let lit = self.str_from_to(start_with_quote + BytePos(1), self.pos - BytePos(1));
        if let Err((off, err)) = unescape::unescape_byte(lit) {
            emit_unescape_error(
                &self.sess.span_diagnostic,
                lit,
                self.mk_sp(start_with_quote, self.pos),
                unescape::Mode::Byte,
                0..off,
                err,
            )
        }
    }

    fn validate_str_escape(&self, start_with_quote: BytePos) {
        let lit = self.str_from_to(start_with_quote + BytePos(1), self.pos - BytePos(1));
        unescape::unescape_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(start_with_quote, self.pos),
                    unescape::Mode::Str,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_raw_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_raw_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::Str,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_raw_byte_str_escape(&self, content_start: BytePos, content_end: BytePos) {
        let lit = self.str_from_to(content_start, content_end);
        unescape::unescape_raw_byte_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(content_start - BytePos(1), content_end + BytePos(1)),
                    unescape::Mode::ByteStr,
                    range,
                    err,
                )
            }
        })
    }

    fn validate_byte_str_escape(&self, start_with_quote: BytePos) {
        let lit = self.str_from_to(start_with_quote + BytePos(1), self.pos - BytePos(1));
        unescape::unescape_byte_str(lit, &mut |range, c| {
            if let Err(err) = c {
                emit_unescape_error(
                    &self.sess.span_diagnostic,
                    lit,
                    self.mk_sp(start_with_quote, self.pos),
                    unescape::Mode::ByteStr,
                    range,
                    err,
                )
            }
        })
    }
}

// This tests the character for the unicode property 'PATTERN_WHITE_SPACE' which
// is guaranteed to be forward compatible. http://unicode.org/reports/tr31/#R3
#[inline]
crate fn is_pattern_whitespace(c: Option<char>) -> bool {
    c.map_or(false, Pattern_White_Space)
}

#[inline]
fn in_range(c: Option<char>, lo: char, hi: char) -> bool {
    c.map_or(false, |c| lo <= c && c <= hi)
}

#[inline]
fn is_dec_digit(c: Option<char>) -> bool {
    in_range(c, '0', '9')
}

fn is_doc_comment(s: &str) -> bool {
    let res = (s.starts_with("///") && *s.as_bytes().get(3).unwrap_or(&b' ') != b'/') ||
              s.starts_with("//!");
    debug!("is {:?} a doc comment? {}", s, res);
    res
}

fn is_block_doc_comment(s: &str) -> bool {
    // Prevent `/**/` from being parsed as a doc comment
    let res = ((s.starts_with("/**") && *s.as_bytes().get(3).unwrap_or(&b' ') != b'*') ||
               s.starts_with("/*!")) && s.len() >= 5;
    debug!("is {:?} a doc comment? {}", s, res);
    res
}

/// Determine whether `c` is a valid start for an ident.
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

#[inline]
fn char_at(s: &str, byte: usize) -> char {
    s[byte..].chars().next().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ast::CrateConfig;
    use crate::symbol::Symbol;
    use crate::source_map::{SourceMap, FilePathMapping};
    use crate::feature_gate::UnstableFeatures;
    use crate::parse::token;
    use crate::diagnostics::plugin::ErrorMap;
    use crate::with_default_globals;
    use std::io;
    use std::path::PathBuf;
    use syntax_pos::{BytePos, Span, NO_EXPANSION, edition::Edition};
    use rustc_data_structures::fx::{FxHashSet, FxHashMap};
    use rustc_data_structures::sync::Lock;

    fn mk_sess(sm: Lrc<SourceMap>) -> ParseSess {
        let emitter = errors::emitter::EmitterWriter::new(Box::new(io::sink()),
                                                          Some(sm.clone()),
                                                          false,
                                                          false,
                                                          false);
        ParseSess {
            span_diagnostic: errors::Handler::with_emitter(true, None, Box::new(emitter)),
            unstable_features: UnstableFeatures::from_environment(),
            config: CrateConfig::default(),
            included_mod_stack: Lock::new(Vec::new()),
            source_map: sm,
            missing_fragment_specifiers: Lock::new(FxHashSet::default()),
            raw_identifier_spans: Lock::new(Vec::new()),
            registered_diagnostics: Lock::new(ErrorMap::new()),
            buffered_lints: Lock::new(vec![]),
            edition: Edition::from_session(),
            ambiguous_block_expr_parse: Lock::new(FxHashMap::default()),
            param_attr_spans: Lock::new(Vec::new()),
            let_chains_spans: Lock::new(Vec::new()),
            async_closure_spans: Lock::new(Vec::new()),
        }
    }

    // open a string reader for the given string
    fn setup<'a>(sm: &SourceMap,
                 sess: &'a ParseSess,
                 teststr: String)
                 -> StringReader<'a> {
        let sf = sm.new_source_file(PathBuf::from(teststr.clone()).into(), teststr);
        let mut sr = StringReader::new_raw(sess, sf, None);
        if sr.advance_token().is_err() {
            sr.emit_fatal_errors();
            FatalError.raise();
        }
        sr
    }

    #[test]
    fn t1() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            let mut string_reader = setup(&sm,
                                        &sh,
                                        "/* my source file */ fn main() { println!(\"zebra\"); }\n"
                                            .to_string());
            assert_eq!(string_reader.next_token(), token::Comment);
            assert_eq!(string_reader.next_token(), token::Whitespace);
            let tok1 = string_reader.next_token();
            let tok2 = Token::new(
                mk_ident("fn"),
                Span::new(BytePos(21), BytePos(23), NO_EXPANSION),
            );
            assert_eq!(tok1.kind, tok2.kind);
            assert_eq!(tok1.span, tok2.span);
            assert_eq!(string_reader.next_token(), token::Whitespace);
            // the 'main' id is already read:
            assert_eq!(string_reader.pos.clone(), BytePos(28));
            // read another token:
            let tok3 = string_reader.next_token();
            let tok4 = Token::new(
                mk_ident("main"),
                Span::new(BytePos(24), BytePos(28), NO_EXPANSION),
            );
            assert_eq!(tok3.kind, tok4.kind);
            assert_eq!(tok3.span, tok4.span);
            // the lparen is already read:
            assert_eq!(string_reader.pos.clone(), BytePos(29))
        })
    }

    // check that the given reader produces the desired stream
    // of tokens (stop checking after exhausting the expected vec)
    fn check_tokenization(mut string_reader: StringReader<'_>, expected: Vec<TokenKind>) {
        for expected_tok in &expected {
            assert_eq!(&string_reader.next_token(), expected_tok);
        }
    }

    // make the identifier by looking up the string in the interner
    fn mk_ident(id: &str) -> TokenKind {
        token::Ident(Symbol::intern(id), false)
    }

    fn mk_lit(kind: token::LitKind, symbol: &str, suffix: Option<&str>) -> TokenKind {
        TokenKind::lit(kind, Symbol::intern(symbol), suffix.map(Symbol::intern))
    }

    #[test]
    fn doublecolonparsing() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a b".to_string()),
                            vec![mk_ident("a"), token::Whitespace, mk_ident("b")]);
        })
    }

    #[test]
    fn dcparsing_2() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a::b".to_string()),
                            vec![mk_ident("a"), token::ModSep, mk_ident("b")]);
        })
    }

    #[test]
    fn dcparsing_3() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a ::b".to_string()),
                            vec![mk_ident("a"), token::Whitespace, token::ModSep, mk_ident("b")]);
        })
    }

    #[test]
    fn dcparsing_4() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            check_tokenization(setup(&sm, &sh, "a:: b".to_string()),
                            vec![mk_ident("a"), token::ModSep, token::Whitespace, mk_ident("b")]);
        })
    }

    #[test]
    fn character_a() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "'a'".to_string()).next_token(),
                       mk_lit(token::Char, "a", None));
        })
    }

    #[test]
    fn character_space() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "' '".to_string()).next_token(),
                       mk_lit(token::Char, " ", None));
        })
    }

    #[test]
    fn character_escaped() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "'\\n'".to_string()).next_token(),
                       mk_lit(token::Char, "\\n", None));
        })
    }

    #[test]
    fn lifetime_name() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "'abc".to_string()).next_token(),
                       token::Lifetime(Symbol::intern("'abc")));
        })
    }

    #[test]
    fn raw_string() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            assert_eq!(setup(&sm, &sh, "r###\"\"#a\\b\x00c\"\"###".to_string()).next_token(),
                       mk_lit(token::StrRaw(3), "\"#a\\b\x00c\"", None));
        })
    }

    #[test]
    fn literal_suffixes() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            macro_rules! test {
                ($input: expr, $tok_type: ident, $tok_contents: expr) => {{
                    assert_eq!(setup(&sm, &sh, format!("{}suffix", $input)).next_token(),
                               mk_lit(token::$tok_type, $tok_contents, Some("suffix")));
                    // with a whitespace separator:
                    assert_eq!(setup(&sm, &sh, format!("{} suffix", $input)).next_token(),
                               mk_lit(token::$tok_type, $tok_contents, None));
                }}
            }

            test!("'a'", Char, "a");
            test!("b'a'", Byte, "a");
            test!("\"a\"", Str, "a");
            test!("b\"a\"", ByteStr, "a");
            test!("1234", Integer, "1234");
            test!("0b101", Integer, "0b101");
            test!("0xABC", Integer, "0xABC");
            test!("1.0", Float, "1.0");
            test!("1.0e10", Float, "1.0e10");

            assert_eq!(setup(&sm, &sh, "2us".to_string()).next_token(),
                       mk_lit(token::Integer, "2", Some("us")));
            assert_eq!(setup(&sm, &sh, "r###\"raw\"###suffix".to_string()).next_token(),
                       mk_lit(token::StrRaw(3), "raw", Some("suffix")));
            assert_eq!(setup(&sm, &sh, "br###\"raw\"###suffix".to_string()).next_token(),
                       mk_lit(token::ByteStrRaw(3), "raw", Some("suffix")));
        })
    }

    #[test]
    fn line_doc_comments() {
        assert!(is_doc_comment("///"));
        assert!(is_doc_comment("/// blah"));
        assert!(!is_doc_comment("////"));
    }

    #[test]
    fn nested_block_comments() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            let mut lexer = setup(&sm, &sh, "/* /* */ */'a'".to_string());
            assert_eq!(lexer.next_token(), token::Comment);
            assert_eq!(lexer.next_token(), mk_lit(token::Char, "a", None));
        })
    }

    #[test]
    fn crlf_comments() {
        with_default_globals(|| {
            let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
            let sh = mk_sess(sm.clone());
            let mut lexer = setup(&sm, &sh, "// test\r\n/// test\r\n".to_string());
            let comment = lexer.next_token();
            assert_eq!(comment.kind, token::Comment);
            assert_eq!((comment.span.lo(), comment.span.hi()), (BytePos(0), BytePos(7)));
            assert_eq!(lexer.next_token(), token::Whitespace);
            assert_eq!(lexer.next_token(), token::DocComment(Symbol::intern("/// test")));
        })
    }
}
