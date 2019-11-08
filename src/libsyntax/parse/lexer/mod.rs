use crate::sess::ParseSess;
use crate::symbol::{sym, Symbol};
use crate::token::{self, Token, TokenKind};
use crate::util::comments;

use errors::{DiagnosticBuilder, FatalError};
use syntax_pos::{BytePos, Pos, Span};

use log::debug;
use rustc_data_structures::sync::Lrc;
use std::char;
use std::convert::TryInto;

use unescape_error_reporting::push_escaped_char;

#[cfg(test)]
mod tests;

mod literal_validation;
mod tokentrees;
mod unescape_error_reporting;
mod unicode_chars;
mod verification;

#[derive(Clone, Debug)]
pub struct UnmatchedBrace {
    pub expected_delim: token::DelimToken,
    pub found_delim: Option<token::DelimToken>,
    pub found_span: Span,
    pub unclosed_span: Option<Span>,
    pub candidate_span: Option<Span>,
}

pub struct StringReader<'a> {
    sess: &'a ParseSess,
    /// Initial position, read-only.
    start_pos: BytePos,
    /// The absolute offset within the source_map of the current character.
    pos: BytePos,
    /// Stop reading src at this index.
    end_src_index: usize,
    /// Source text to tokenize.
    src: Lrc<String>,
    override_span: Option<Span>,
}

impl<'a> StringReader<'a> {
    pub fn new(
        sess: &'a ParseSess,
        source_file: Lrc<syntax_pos::SourceFile>,
        override_span: Option<Span>,
    ) -> Self {
        if source_file.src.is_none() {
            sess.span_diagnostic
                .bug(&format!("cannot lex `source_file` without source: {}", source_file.name));
        }

        let src = (*source_file.src.as_ref().unwrap()).clone();

        StringReader {
            sess,
            start_pos: source_file.start_pos,
            pos: source_file.start_pos,
            end_src_index: src.len(),
            src,
            override_span,
        }
    }

    pub fn retokenize(sess: &'a ParseSess, mut span: Span) -> Self {
        let begin = sess.source_map().lookup_byte_offset(span.lo());
        let end = sess.source_map().lookup_byte_offset(span.hi());

        // Make the range zero-length if the span is invalid.
        if begin.sf.start_pos != end.sf.start_pos {
            span = span.shrink_to_lo();
        }

        let mut sr = StringReader::new(sess, begin.sf, None);

        // Seek the lexer to the right byte range.
        sr.end_src_index = sr.src_index(span.hi());

        sr
    }

    fn mk_sp(&self, lo: BytePos, hi: BytePos) -> Span {
        self.override_span.unwrap_or_else(|| Span::with_root_ctxt(lo, hi))
    }

    /// Returns the next token, including trivia like whitespace or comments.
    ///
    /// `Err(())` means that some errors were encountered, which can be
    /// retrieved using `buffer_fatal_errors`.
    pub fn next_token(&mut self) -> Token {
        let start_src_index = self.src_index(self.pos);
        let text: &str = &self.src[start_src_index..self.end_src_index];

        if text.is_empty() {
            let span = self.mk_sp(self.pos, self.pos);
            return Token::new(token::Eof, span);
        }

        {
            let is_beginning_of_file = self.pos == self.start_pos;
            if is_beginning_of_file {
                if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
                    let start = self.pos;
                    self.pos = self.pos + BytePos::from_usize(shebang_len);

                    let sym = self.symbol_from(start + BytePos::from_usize("#!".len()));
                    let kind = token::Shebang(sym);

                    let span = self.mk_sp(start, self.pos);
                    return Token::new(kind, span);
                }
            }
        }

        let token = rustc_lexer::first_token(text);

        let start = self.pos;
        self.pos = self.pos + BytePos::from_usize(token.len);

        debug!("try_next_token: {:?}({:?})", token.kind, self.str_from(start));

        // This could use `?`, but that makes code significantly (10-20%) slower.
        // https://github.com/rust-lang/rust/issues/37939
        let kind = self.cook_lexer_token(token.kind, start);

        let span = self.mk_sp(start, self.pos);
        Token::new(kind, span)
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

    fn struct_span_fatal(
        &self,
        from_pos: BytePos,
        to_pos: BytePos,
        m: &str,
    ) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), m)
    }

    fn struct_fatal_span_char(
        &self,
        from_pos: BytePos,
        to_pos: BytePos,
        m: &str,
        c: char,
    ) -> DiagnosticBuilder<'a> {
        let mut m = m.to_string();
        m.push_str(": ");
        push_escaped_char(&mut m, c);

        self.sess.span_diagnostic.struct_span_fatal(self.mk_sp(from_pos, to_pos), &m[..])
    }

    /// Turns simple `rustc_lexer::TokenKind` enum into a rich
    /// `libsyntax::TokenKind`. This turns strings into interned
    /// symbols and runs additional validation.
    fn cook_lexer_token(&self, token: rustc_lexer::TokenKind, start: BytePos) -> TokenKind {
        match token {
            rustc_lexer::TokenKind::LineComment => {
                let string = self.str_from(start);
                // Comments with more than three "/"s are not doc comments.
                let tok = if comments::is_line_doc_comment(string) {
                    let is_block_comment = false;
                    self.verify_doc_comment_contents(start, string, is_block_comment);
                    token::DocComment(Symbol::intern(string))
                } else {
                    token::Comment
                };

                tok
            }
            rustc_lexer::TokenKind::BlockComment { terminated } => {
                let string = self.str_from(start);
                // Block comments starting with "/**" or "/*!" are doc-comments,
                // but comments with only "*"s between two "/"s are not.
                let is_doc_comment = comments::is_block_doc_comment(string);
                self.verify_doc_comment_terminated(start, terminated, is_doc_comment);

                let tok = if is_doc_comment {
                    let is_block_comment = true;
                    self.verify_doc_comment_contents(start, string, is_block_comment);
                    token::DocComment(Symbol::intern(string))
                } else {
                    token::Comment
                };

                tok
            }
            rustc_lexer::TokenKind::Whitespace => token::Whitespace,
            rustc_lexer::TokenKind::Ident => {
                let is_raw_ident = false;

                // FIXME: perform NFKC normalization here. (Issue #2253)
                let sym = self.symbol_from(start);
                token::Ident(sym, is_raw_ident)
            }
            rustc_lexer::TokenKind::RawIdent => {
                let is_raw_ident = true;
                let ident_start = start + BytePos(2);

                // FIXME: perform NFKC normalization here. (Issue #2253)
                let sym = self.symbol_from(ident_start);
                let span = self.mk_sp(start, self.pos);
                self.verify_raw_symbol(&sym, span);
                self.sess.raw_identifier_spans.borrow_mut().push(span);

                token::Ident(sym, is_raw_ident)
            }
            rustc_lexer::TokenKind::Literal { kind, suffix_start } => {
                let suffix_start = start + BytePos(suffix_start as u32);
                let (kind, symbol) = self.cook_lexer_literal(start, suffix_start, kind);
                let suffix = if suffix_start < self.pos {
                    let string = self.str_from(suffix_start);
                    if self.verify_no_underscore_literal_suffix(suffix_start, string).is_ok() {
                        Some(Symbol::intern(string))
                    } else {
                        None
                    }
                } else {
                    None
                };
                token::Literal(token::Lit { kind, symbol, suffix })
            }
            rustc_lexer::TokenKind::Lifetime { starts_with_number } => {
                // Include the leading `'` in the real identifier, for macro
                // expansion purposes. See #12512 for the gory details of why
                // this is necessary.
                let lifetime_name = self.str_from(start);
                self.verify_lifetime(start, starts_with_number);

                let ident = Symbol::intern(lifetime_name);
                token::Lifetime(ident)
            }
            rustc_lexer::TokenKind::Semi => token::Semi,
            rustc_lexer::TokenKind::Comma => token::Comma,
            rustc_lexer::TokenKind::Dot => token::Dot,
            rustc_lexer::TokenKind::OpenParen => token::OpenDelim(token::Paren),
            rustc_lexer::TokenKind::CloseParen => token::CloseDelim(token::Paren),
            rustc_lexer::TokenKind::OpenBrace => token::OpenDelim(token::Brace),
            rustc_lexer::TokenKind::CloseBrace => token::CloseDelim(token::Brace),
            rustc_lexer::TokenKind::OpenBracket => token::OpenDelim(token::Bracket),
            rustc_lexer::TokenKind::CloseBracket => token::CloseDelim(token::Bracket),
            rustc_lexer::TokenKind::At => token::At,
            rustc_lexer::TokenKind::Pound => token::Pound,
            rustc_lexer::TokenKind::Tilde => token::Tilde,
            rustc_lexer::TokenKind::Question => token::Question,
            rustc_lexer::TokenKind::Colon => token::Colon,
            rustc_lexer::TokenKind::Dollar => token::Dollar,
            rustc_lexer::TokenKind::Eq => token::Eq,
            rustc_lexer::TokenKind::Not => token::Not,
            rustc_lexer::TokenKind::Lt => token::Lt,
            rustc_lexer::TokenKind::Gt => token::Gt,
            rustc_lexer::TokenKind::Minus => token::BinOp(token::Minus),
            rustc_lexer::TokenKind::And => token::BinOp(token::And),
            rustc_lexer::TokenKind::Or => token::BinOp(token::Or),
            rustc_lexer::TokenKind::Plus => token::BinOp(token::Plus),
            rustc_lexer::TokenKind::Star => token::BinOp(token::Star),
            rustc_lexer::TokenKind::Slash => token::BinOp(token::Slash),
            rustc_lexer::TokenKind::Caret => token::BinOp(token::Caret),
            rustc_lexer::TokenKind::Percent => token::BinOp(token::Percent),

            rustc_lexer::TokenKind::Unknown => {
                // Report an error about unknown token.
                let c = self.str_from(start).chars().next().unwrap();
                let mut err =
                    self.struct_fatal_span_char(start, self.pos, "unknown start of token", c);
                // FIXME: the lexer could be used to turn the ASCII version of unicode homoglyphs,
                // instead of keeping a table in `check_for_substitution`into the token. Ideally,
                // this should be inside `rustc_lexer`. However, we should first remove compound
                // tokens like `<<` from `rustc_lexer`, and then add fancier error recovery to it,
                // as there will be less overall work to do this way.
                let token = unicode_chars::check_for_substitution(self, start, c, &mut err)
                    .unwrap_or_else(|| token::Unknown(self.symbol_from(start)));
                err.emit();
                token
            }
        }
    }

    fn cook_lexer_literal(
        &self,
        start: BytePos,
        suffix_start: BytePos,
        kind: rustc_lexer::LiteralKind,
    ) -> (token::LitKind, Symbol) {
        match kind {
            rustc_lexer::LiteralKind::Char { .. } => {
                self.verify_literal_enclosed(start, suffix_start, kind);
                let content_start = start + BytePos(1);
                let content_end = suffix_start - BytePos(1);
                self.validate_char_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Char, id)
            }
            rustc_lexer::LiteralKind::Byte { .. } => {
                self.verify_literal_enclosed(start, suffix_start, kind);
                let content_start = start + BytePos(2);
                let content_end = suffix_start - BytePos(1);
                self.validate_byte_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Byte, id)
            }
            rustc_lexer::LiteralKind::Str { .. } => {
                self.verify_literal_enclosed(start, suffix_start, kind);
                let content_start = start + BytePos(1);
                let content_end = suffix_start - BytePos(1);
                self.validate_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::Str, id)
            }
            rustc_lexer::LiteralKind::ByteStr { .. } => {
                self.verify_literal_enclosed(start, suffix_start, kind);
                let content_start = start + BytePos(2);
                let content_end = suffix_start - BytePos(1);
                self.validate_byte_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::ByteStr, id)
            }
            rustc_lexer::LiteralKind::RawStr { n_hashes, .. } => {
                self.verify_literal_enclosed(start, suffix_start, kind);
                let n_hashes: u16 = self.restrict_n_hashes(start, n_hashes);
                let content_start = start + BytePos(2 + n_hashes as u32);
                let content_end = suffix_start - BytePos(1 + n_hashes as u32);
                self.validate_raw_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::StrRaw(n_hashes), id)
            }
            rustc_lexer::LiteralKind::RawByteStr { n_hashes, .. } => {
                self.verify_literal_enclosed(start, suffix_start, kind);
                let n_hashes: u16 = self.restrict_n_hashes(start, n_hashes);
                let content_start = start + BytePos(3 + n_hashes as u32);
                let content_end = suffix_start - BytePos(1 + n_hashes as u32);
                self.validate_raw_byte_str_escape(content_start, content_end);
                let id = self.symbol_from_to(content_start, content_end);
                (token::ByteStrRaw(n_hashes), id)
            }
            rustc_lexer::LiteralKind::Int { base, empty_int } => {
                if self.verify_int_not_empty(start, suffix_start, empty_int).is_ok() {
                    self.validate_int_literal(base, start, suffix_start);
                    (token::Integer, self.symbol_from_to(start, suffix_start))
                } else {
                    (token::Integer, sym::integer(0))
                }
            }
            rustc_lexer::LiteralKind::Float { base, empty_exponent } => {
                self.verify_float_exponent_not_empty(start, empty_exponent);
                self.verify_float_base(start, suffix_start, base);

                let id = self.symbol_from_to(start, suffix_start);
                (token::Float, id)
            }
        }
    }

    #[inline]
    fn src_index(&self, pos: BytePos) -> usize {
        (pos - self.start_pos).to_usize()
    }

    /// Slice of the source text from `start` up to but excluding `self.pos`,
    /// meaning the slice does not include the character `self.ch`.
    fn str_from(&self, start: BytePos) -> &str {
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
    fn str_from_to(&self, start: BytePos, end: BytePos) -> &str {
        &self.src[self.src_index(start)..self.src_index(end)]
    }

    fn restrict_n_hashes(&self, start: BytePos, n_hashes: usize) -> u16 {
        match n_hashes.try_into() {
            Ok(n_hashes) => n_hashes,
            Err(_) => {
                self.fatal_span_(
                    start,
                    self.pos,
                    "too many `#` symbols: raw strings may be \
                     delimited by up to 65535 `#` symbols",
                )
                .raise();
            }
        }
    }
}
