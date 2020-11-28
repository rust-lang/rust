pub mod attr;
mod diagnostics;
mod expr;
mod generics;
mod item;
mod nonterminal;
mod pat;
mod path;
mod stmt;
mod ty;

use crate::lexer::UnmatchedBrace;
pub use diagnostics::AttemptLocalParseRecovery;
use diagnostics::Error;
pub use path::PathStyle;

use rustc_ast::ptr::P;
use rustc_ast::token::{self, DelimToken, Token, TokenKind};
use rustc_ast::tokenstream::{self, DelimSpan, LazyTokenStream, Spacing};
use rustc_ast::tokenstream::{CreateTokenStream, TokenStream, TokenTree};
use rustc_ast::DUMMY_NODE_ID;
use rustc_ast::{self as ast, AnonConst, AttrStyle, AttrVec, Const, CrateSugar, Extern, Unsafe};
use rustc_ast::{Async, Expr, ExprKind, MacArgs, MacDelimiter, Mutability, StrLit};
use rustc_ast::{Visibility, VisibilityKind};
use rustc_ast_pretty::pprust;
use rustc_errors::PResult;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, FatalError};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{Span, DUMMY_SP};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use tracing::debug;

use std::{cmp, mem, slice};

bitflags::bitflags! {
    struct Restrictions: u8 {
        const STMT_EXPR         = 1 << 0;
        const NO_STRUCT_LITERAL = 1 << 1;
        const CONST_EXPR        = 1 << 2;
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum SemiColonMode {
    Break,
    Ignore,
    Comma,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum BlockMode {
    Break,
    Ignore,
}

/// Like `maybe_whole_expr`, but for things other than expressions.
#[macro_export]
macro_rules! maybe_whole {
    ($p:expr, $constructor:ident, |$x:ident| $e:expr) => {
        if let token::Interpolated(nt) = &$p.token.kind {
            if let token::$constructor(x) = &**nt {
                let $x = x.clone();
                $p.bump();
                return Ok($e);
            }
        }
    };
}

/// If the next tokens are ill-formed `$ty::` recover them as `<$ty>::`.
#[macro_export]
macro_rules! maybe_recover_from_interpolated_ty_qpath {
    ($self: expr, $allow_qpath_recovery: expr) => {
        if $allow_qpath_recovery && $self.look_ahead(1, |t| t == &token::ModSep) {
            if let token::Interpolated(nt) = &$self.token.kind {
                if let token::NtTy(ty) = &**nt {
                    let ty = ty.clone();
                    $self.bump();
                    return $self.maybe_recover_from_bad_qpath_stage_2($self.prev_token.span, ty);
                }
            }
        }
    };
}

#[derive(Clone)]
pub struct Parser<'a> {
    pub sess: &'a ParseSess,
    /// The current token.
    pub token: Token,
    /// The spacing for the current token
    pub token_spacing: Spacing,
    /// The previous token.
    pub prev_token: Token,
    restrictions: Restrictions,
    expected_tokens: Vec<TokenType>,
    // Important: This must only be advanced from `next_tok`
    // to ensure that `token_cursor.num_next_calls` is updated properly
    token_cursor: TokenCursor,
    desugar_doc_comments: bool,
    /// This field is used to keep track of how many left angle brackets we have seen. This is
    /// required in order to detect extra leading left angle brackets (`<` characters) and error
    /// appropriately.
    ///
    /// See the comments in the `parse_path_segment` function for more details.
    unmatched_angle_bracket_count: u32,
    max_angle_bracket_count: u32,
    /// A list of all unclosed delimiters found by the lexer. If an entry is used for error recovery
    /// it gets removed from here. Every entry left at the end gets emitted as an independent
    /// error.
    pub(super) unclosed_delims: Vec<UnmatchedBrace>,
    last_unexpected_token_span: Option<Span>,
    /// Span pointing at the `:` for the last type ascription the parser has seen, and whether it
    /// looked like it could have been a mistyped path or literal `Option:Some(42)`).
    pub last_type_ascription: Option<(Span, bool /* likely path typo */)>,
    /// If present, this `Parser` is not parsing Rust code but rather a macro call.
    subparser_name: Option<&'static str>,
}

impl<'a> Drop for Parser<'a> {
    fn drop(&mut self) {
        emit_unclosed_delims(&mut self.unclosed_delims, &self.sess);
    }
}

#[derive(Clone)]
struct TokenCursor {
    frame: TokenCursorFrame,
    stack: Vec<TokenCursorFrame>,
    desugar_doc_comments: bool,
    // Counts the number of calls to `next` or `next_desugared`,
    // depending on whether `desugar_doc_comments` is set.
    num_next_calls: usize,
}

#[derive(Clone)]
struct TokenCursorFrame {
    delim: token::DelimToken,
    span: DelimSpan,
    open_delim: bool,
    tree_cursor: tokenstream::Cursor,
    close_delim: bool,
}

impl TokenCursorFrame {
    fn new(span: DelimSpan, delim: DelimToken, tts: TokenStream) -> Self {
        TokenCursorFrame {
            delim,
            span,
            open_delim: delim == token::NoDelim,
            tree_cursor: tts.into_trees(),
            close_delim: delim == token::NoDelim,
        }
    }
}

impl TokenCursor {
    fn next(&mut self) -> (Token, Spacing) {
        loop {
            let (tree, spacing) = if !self.frame.open_delim {
                self.frame.open_delim = true;
                TokenTree::open_tt(self.frame.span, self.frame.delim).into()
            } else if let Some(tree) = self.frame.tree_cursor.next_with_spacing() {
                tree
            } else if !self.frame.close_delim {
                self.frame.close_delim = true;
                TokenTree::close_tt(self.frame.span, self.frame.delim).into()
            } else if let Some(frame) = self.stack.pop() {
                self.frame = frame;
                continue;
            } else {
                (TokenTree::Token(Token::new(token::Eof, DUMMY_SP)), Spacing::Alone)
            };

            match tree {
                TokenTree::Token(token) => {
                    return (token, spacing);
                }
                TokenTree::Delimited(sp, delim, tts) => {
                    let frame = TokenCursorFrame::new(sp, delim, tts);
                    self.stack.push(mem::replace(&mut self.frame, frame));
                }
            }
        }
    }

    fn next_desugared(&mut self) -> (Token, Spacing) {
        let (data, attr_style, sp) = match self.next() {
            (Token { kind: token::DocComment(_, attr_style, data), span }, _) => {
                (data, attr_style, span)
            }
            tok => return tok,
        };

        // Searches for the occurrences of `"#*` and returns the minimum number of `#`s
        // required to wrap the text.
        let mut num_of_hashes = 0;
        let mut count = 0;
        for ch in data.as_str().chars() {
            count = match ch {
                '"' => 1,
                '#' if count > 0 => count + 1,
                _ => 0,
            };
            num_of_hashes = cmp::max(num_of_hashes, count);
        }

        let delim_span = DelimSpan::from_single(sp);
        let body = TokenTree::Delimited(
            delim_span,
            token::Bracket,
            [
                TokenTree::token(token::Ident(sym::doc, false), sp),
                TokenTree::token(token::Eq, sp),
                TokenTree::token(TokenKind::lit(token::StrRaw(num_of_hashes), data, None), sp),
            ]
            .iter()
            .cloned()
            .collect::<TokenStream>(),
        );

        self.stack.push(mem::replace(
            &mut self.frame,
            TokenCursorFrame::new(
                delim_span,
                token::NoDelim,
                if attr_style == AttrStyle::Inner {
                    [TokenTree::token(token::Pound, sp), TokenTree::token(token::Not, sp), body]
                        .iter()
                        .cloned()
                        .collect::<TokenStream>()
                } else {
                    [TokenTree::token(token::Pound, sp), body]
                        .iter()
                        .cloned()
                        .collect::<TokenStream>()
                },
            ),
        ));

        self.next()
    }
}

#[derive(Clone, PartialEq)]
enum TokenType {
    Token(TokenKind),
    Keyword(Symbol),
    Operator,
    Lifetime,
    Ident,
    Path,
    Type,
    Const,
}

impl TokenType {
    fn to_string(&self) -> String {
        match *self {
            TokenType::Token(ref t) => format!("`{}`", pprust::token_kind_to_string(t)),
            TokenType::Keyword(kw) => format!("`{}`", kw),
            TokenType::Operator => "an operator".to_string(),
            TokenType::Lifetime => "lifetime".to_string(),
            TokenType::Ident => "identifier".to_string(),
            TokenType::Path => "path".to_string(),
            TokenType::Type => "type".to_string(),
            TokenType::Const => "const".to_string(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum TokenExpectType {
    Expect,
    NoExpect,
}

/// A sequence separator.
struct SeqSep {
    /// The separator token.
    sep: Option<TokenKind>,
    /// `true` if a trailing separator is allowed.
    trailing_sep_allowed: bool,
}

impl SeqSep {
    fn trailing_allowed(t: TokenKind) -> SeqSep {
        SeqSep { sep: Some(t), trailing_sep_allowed: true }
    }

    fn none() -> SeqSep {
        SeqSep { sep: None, trailing_sep_allowed: false }
    }
}

pub enum FollowedByType {
    Yes,
    No,
}

fn token_descr_opt(token: &Token) -> Option<&'static str> {
    Some(match token.kind {
        _ if token.is_special_ident() => "reserved identifier",
        _ if token.is_used_keyword() => "keyword",
        _ if token.is_unused_keyword() => "reserved keyword",
        token::DocComment(..) => "doc comment",
        _ => return None,
    })
}

pub(super) fn token_descr(token: &Token) -> String {
    let token_str = pprust::token_to_string(token);
    match token_descr_opt(token) {
        Some(prefix) => format!("{} `{}`", prefix, token_str),
        _ => format!("`{}`", token_str),
    }
}

impl<'a> Parser<'a> {
    pub fn new(
        sess: &'a ParseSess,
        tokens: TokenStream,
        desugar_doc_comments: bool,
        subparser_name: Option<&'static str>,
    ) -> Self {
        let mut parser = Parser {
            sess,
            token: Token::dummy(),
            token_spacing: Spacing::Alone,
            prev_token: Token::dummy(),
            restrictions: Restrictions::empty(),
            expected_tokens: Vec::new(),
            token_cursor: TokenCursor {
                frame: TokenCursorFrame::new(DelimSpan::dummy(), token::NoDelim, tokens),
                stack: Vec::new(),
                num_next_calls: 0,
                desugar_doc_comments,
            },
            desugar_doc_comments,
            unmatched_angle_bracket_count: 0,
            max_angle_bracket_count: 0,
            unclosed_delims: Vec::new(),
            last_unexpected_token_span: None,
            last_type_ascription: None,
            subparser_name,
        };

        // Make parser point to the first token.
        parser.bump();

        parser
    }

    fn next_tok(&mut self, fallback_span: Span) -> (Token, Spacing) {
        let (mut next, spacing) = if self.desugar_doc_comments {
            self.token_cursor.next_desugared()
        } else {
            self.token_cursor.next()
        };
        self.token_cursor.num_next_calls += 1;
        if next.span.is_dummy() {
            // Tweak the location for better diagnostics, but keep syntactic context intact.
            next.span = fallback_span.with_ctxt(next.span.ctxt());
        }
        (next, spacing)
    }

    pub fn unexpected<T>(&mut self) -> PResult<'a, T> {
        match self.expect_one_of(&[], &[]) {
            Err(e) => Err(e),
            // We can get `Ok(true)` from `recover_closing_delimiter`
            // which is called in `expected_one_of_not_found`.
            Ok(_) => FatalError.raise(),
        }
    }

    /// Expects and consumes the token `t`. Signals an error if the next token is not `t`.
    pub fn expect(&mut self, t: &TokenKind) -> PResult<'a, bool /* recovered */> {
        if self.expected_tokens.is_empty() {
            if self.token == *t {
                self.bump();
                Ok(false)
            } else {
                self.unexpected_try_recover(t)
            }
        } else {
            self.expect_one_of(slice::from_ref(t), &[])
        }
    }

    /// Expect next token to be edible or inedible token.  If edible,
    /// then consume it; if inedible, then return without consuming
    /// anything.  Signal a fatal error if next token is unexpected.
    pub fn expect_one_of(
        &mut self,
        edible: &[TokenKind],
        inedible: &[TokenKind],
    ) -> PResult<'a, bool /* recovered */> {
        if edible.contains(&self.token.kind) {
            self.bump();
            Ok(false)
        } else if inedible.contains(&self.token.kind) {
            // leave it in the input
            Ok(false)
        } else if self.last_unexpected_token_span == Some(self.token.span) {
            FatalError.raise();
        } else {
            self.expected_one_of_not_found(edible, inedible)
        }
    }

    // Public for rustfmt usage.
    pub fn parse_ident(&mut self) -> PResult<'a, Ident> {
        self.parse_ident_common(true)
    }

    fn parse_ident_common(&mut self, recover: bool) -> PResult<'a, Ident> {
        match self.token.ident() {
            Some((ident, is_raw)) => {
                if !is_raw && ident.is_reserved() {
                    let mut err = self.expected_ident_found();
                    if recover {
                        err.emit();
                    } else {
                        return Err(err);
                    }
                }
                self.bump();
                Ok(ident)
            }
            _ => Err(match self.prev_token.kind {
                TokenKind::DocComment(..) => {
                    self.span_fatal_err(self.prev_token.span, Error::UselessDocComment)
                }
                _ => self.expected_ident_found(),
            }),
        }
    }

    /// Checks if the next token is `tok`, and returns `true` if so.
    ///
    /// This method will automatically add `tok` to `expected_tokens` if `tok` is not
    /// encountered.
    fn check(&mut self, tok: &TokenKind) -> bool {
        let is_present = self.token == *tok;
        if !is_present {
            self.expected_tokens.push(TokenType::Token(tok.clone()));
        }
        is_present
    }

    /// Consumes a token 'tok' if it exists. Returns whether the given token was present.
    pub fn eat(&mut self, tok: &TokenKind) -> bool {
        let is_present = self.check(tok);
        if is_present {
            self.bump()
        }
        is_present
    }

    /// If the next token is the given keyword, returns `true` without eating it.
    /// An expectation is also added for diagnostics purposes.
    fn check_keyword(&mut self, kw: Symbol) -> bool {
        self.expected_tokens.push(TokenType::Keyword(kw));
        self.token.is_keyword(kw)
    }

    /// If the next token is the given keyword, eats it and returns `true`.
    /// Otherwise, returns `false`. An expectation is also added for diagnostics purposes.
    // Public for rustfmt usage.
    pub fn eat_keyword(&mut self, kw: Symbol) -> bool {
        if self.check_keyword(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_keyword_noexpect(&mut self, kw: Symbol) -> bool {
        if self.token.is_keyword(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// If the given word is not a keyword, signals an error.
    /// If the next token is not the given word, signals an error.
    /// Otherwise, eats it.
    fn expect_keyword(&mut self, kw: Symbol) -> PResult<'a, ()> {
        if !self.eat_keyword(kw) { self.unexpected() } else { Ok(()) }
    }

    /// Is the given keyword `kw` followed by a non-reserved identifier?
    fn is_kw_followed_by_ident(&self, kw: Symbol) -> bool {
        self.token.is_keyword(kw) && self.look_ahead(1, |t| t.is_ident() && !t.is_reserved_ident())
    }

    fn check_or_expected(&mut self, ok: bool, typ: TokenType) -> bool {
        if ok {
            true
        } else {
            self.expected_tokens.push(typ);
            false
        }
    }

    fn check_ident(&mut self) -> bool {
        self.check_or_expected(self.token.is_ident(), TokenType::Ident)
    }

    fn check_path(&mut self) -> bool {
        self.check_or_expected(self.token.is_path_start(), TokenType::Path)
    }

    fn check_type(&mut self) -> bool {
        self.check_or_expected(self.token.can_begin_type(), TokenType::Type)
    }

    fn check_const_arg(&mut self) -> bool {
        self.check_or_expected(self.token.can_begin_const_arg(), TokenType::Const)
    }

    fn check_inline_const(&self, dist: usize) -> bool {
        self.is_keyword_ahead(dist, &[kw::Const])
            && self.look_ahead(dist + 1, |t| match t.kind {
                token::Interpolated(ref nt) => matches!(**nt, token::NtBlock(..)),
                token::OpenDelim(DelimToken::Brace) => true,
                _ => false,
            })
    }

    /// Checks to see if the next token is either `+` or `+=`.
    /// Otherwise returns `false`.
    fn check_plus(&mut self) -> bool {
        self.check_or_expected(
            self.token.is_like_plus(),
            TokenType::Token(token::BinOp(token::Plus)),
        )
    }

    /// Eats the expected token if it's present possibly breaking
    /// compound tokens like multi-character operators in process.
    /// Returns `true` if the token was eaten.
    fn break_and_eat(&mut self, expected: TokenKind) -> bool {
        if self.token.kind == expected {
            self.bump();
            return true;
        }
        match self.token.kind.break_two_token_op() {
            Some((first, second)) if first == expected => {
                let first_span = self.sess.source_map().start_point(self.token.span);
                let second_span = self.token.span.with_lo(first_span.hi());
                self.token = Token::new(first, first_span);
                // Use the spacing of the glued token as the spacing
                // of the unglued second token.
                self.bump_with((Token::new(second, second_span), self.token_spacing));
                true
            }
            _ => {
                self.expected_tokens.push(TokenType::Token(expected));
                false
            }
        }
    }

    /// Eats `+` possibly breaking tokens like `+=` in process.
    fn eat_plus(&mut self) -> bool {
        self.break_and_eat(token::BinOp(token::Plus))
    }

    /// Eats `&` possibly breaking tokens like `&&` in process.
    /// Signals an error if `&` is not eaten.
    fn expect_and(&mut self) -> PResult<'a, ()> {
        if self.break_and_eat(token::BinOp(token::And)) { Ok(()) } else { self.unexpected() }
    }

    /// Eats `|` possibly breaking tokens like `||` in process.
    /// Signals an error if `|` was not eaten.
    fn expect_or(&mut self) -> PResult<'a, ()> {
        if self.break_and_eat(token::BinOp(token::Or)) { Ok(()) } else { self.unexpected() }
    }

    /// Eats `<` possibly breaking tokens like `<<` in process.
    fn eat_lt(&mut self) -> bool {
        let ate = self.break_and_eat(token::Lt);
        if ate {
            // See doc comment for `unmatched_angle_bracket_count`.
            self.unmatched_angle_bracket_count += 1;
            self.max_angle_bracket_count += 1;
            debug!("eat_lt: (increment) count={:?}", self.unmatched_angle_bracket_count);
        }
        ate
    }

    /// Eats `<` possibly breaking tokens like `<<` in process.
    /// Signals an error if `<` was not eaten.
    fn expect_lt(&mut self) -> PResult<'a, ()> {
        if self.eat_lt() { Ok(()) } else { self.unexpected() }
    }

    /// Eats `>` possibly breaking tokens like `>>` in process.
    /// Signals an error if `>` was not eaten.
    fn expect_gt(&mut self) -> PResult<'a, ()> {
        if self.break_and_eat(token::Gt) {
            // See doc comment for `unmatched_angle_bracket_count`.
            if self.unmatched_angle_bracket_count > 0 {
                self.unmatched_angle_bracket_count -= 1;
                debug!("expect_gt: (decrement) count={:?}", self.unmatched_angle_bracket_count);
            }
            Ok(())
        } else {
            self.unexpected()
        }
    }

    fn expect_any_with_type(&mut self, kets: &[&TokenKind], expect: TokenExpectType) -> bool {
        kets.iter().any(|k| match expect {
            TokenExpectType::Expect => self.check(k),
            TokenExpectType::NoExpect => self.token == **k,
        })
    }

    fn parse_seq_to_before_tokens<T>(
        &mut self,
        kets: &[&TokenKind],
        sep: SeqSep,
        expect: TokenExpectType,
        mut f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool /* trailing */, bool /* recovered */)> {
        let mut first = true;
        let mut recovered = false;
        let mut trailing = false;
        let mut v = vec![];
        while !self.expect_any_with_type(kets, expect) {
            if let token::CloseDelim(..) | token::Eof = self.token.kind {
                break;
            }
            if let Some(ref t) = sep.sep {
                if first {
                    first = false;
                } else {
                    match self.expect(t) {
                        Ok(false) => {}
                        Ok(true) => {
                            recovered = true;
                            break;
                        }
                        Err(mut expect_err) => {
                            let sp = self.prev_token.span.shrink_to_hi();
                            let token_str = pprust::token_kind_to_string(t);

                            // Attempt to keep parsing if it was a similar separator.
                            if let Some(ref tokens) = t.similar_tokens() {
                                if tokens.contains(&self.token.kind) {
                                    self.bump();
                                }
                            }

                            // If this was a missing `@` in a binding pattern
                            // bail with a suggestion
                            // https://github.com/rust-lang/rust/issues/72373
                            if self.prev_token.is_ident() && self.token.kind == token::DotDot {
                                let msg = format!(
                                    "if you meant to bind the contents of \
                                    the rest of the array pattern into `{}`, use `@`",
                                    pprust::token_to_string(&self.prev_token)
                                );
                                expect_err
                                    .span_suggestion_verbose(
                                        self.prev_token.span.shrink_to_hi().until(self.token.span),
                                        &msg,
                                        " @ ".to_string(),
                                        Applicability::MaybeIncorrect,
                                    )
                                    .emit();
                                break;
                            }

                            // Attempt to keep parsing if it was an omitted separator.
                            match f(self) {
                                Ok(t) => {
                                    // Parsed successfully, therefore most probably the code only
                                    // misses a separator.
                                    let mut exp_span = self.sess.source_map().next_point(sp);
                                    if self.sess.source_map().is_multiline(exp_span) {
                                        exp_span = sp;
                                    }
                                    expect_err
                                        .span_suggestion_short(
                                            exp_span,
                                            &format!("missing `{}`", token_str),
                                            token_str,
                                            Applicability::MaybeIncorrect,
                                        )
                                        .emit();

                                    v.push(t);
                                    continue;
                                }
                                Err(mut e) => {
                                    // Parsing failed, therefore it must be something more serious
                                    // than just a missing separator.
                                    expect_err.emit();

                                    e.cancel();
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if sep.trailing_sep_allowed && self.expect_any_with_type(kets, expect) {
                trailing = true;
                break;
            }

            let t = f(self)?;
            v.push(t);
        }

        Ok((v, trailing, recovered))
    }

    /// Parses a sequence, not including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_seq_to_before_end<T>(
        &mut self,
        ket: &TokenKind,
        sep: SeqSep,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool, bool)> {
        self.parse_seq_to_before_tokens(&[ket], sep, TokenExpectType::Expect, f)
    }

    /// Parses a sequence, including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_seq_to_end<T>(
        &mut self,
        ket: &TokenKind,
        sep: SeqSep,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool /* trailing */)> {
        let (val, trailing, recovered) = self.parse_seq_to_before_end(ket, sep, f)?;
        if !recovered {
            self.eat(ket);
        }
        Ok((val, trailing))
    }

    /// Parses a sequence, including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_unspanned_seq<T>(
        &mut self,
        bra: &TokenKind,
        ket: &TokenKind,
        sep: SeqSep,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool)> {
        self.expect(bra)?;
        self.parse_seq_to_end(ket, sep, f)
    }

    fn parse_delim_comma_seq<T>(
        &mut self,
        delim: DelimToken,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool)> {
        self.parse_unspanned_seq(
            &token::OpenDelim(delim),
            &token::CloseDelim(delim),
            SeqSep::trailing_allowed(token::Comma),
            f,
        )
    }

    fn parse_paren_comma_seq<T>(
        &mut self,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool)> {
        self.parse_delim_comma_seq(token::Paren, f)
    }

    /// Advance the parser by one token using provided token as the next one.
    fn bump_with(&mut self, (next_token, next_spacing): (Token, Spacing)) {
        // Bumping after EOF is a bad sign, usually an infinite loop.
        if self.prev_token.kind == TokenKind::Eof {
            let msg = "attempted to bump the parser past EOF (may be stuck in a loop)";
            self.span_bug(self.token.span, msg);
        }

        // Update the current and previous tokens.
        self.prev_token = mem::replace(&mut self.token, next_token);
        self.token_spacing = next_spacing;

        // Diagnostics.
        self.expected_tokens.clear();
    }

    /// Advance the parser by one token.
    pub fn bump(&mut self) {
        let next_token = self.next_tok(self.token.span);
        self.bump_with(next_token);
    }

    /// Look-ahead `dist` tokens of `self.token` and get access to that token there.
    /// When `dist == 0` then the current token is looked at.
    pub fn look_ahead<R>(&self, dist: usize, looker: impl FnOnce(&Token) -> R) -> R {
        if dist == 0 {
            return looker(&self.token);
        }

        let frame = &self.token_cursor.frame;
        match frame.tree_cursor.look_ahead(dist - 1) {
            Some(tree) => match tree {
                TokenTree::Token(token) => looker(token),
                TokenTree::Delimited(dspan, delim, _) => {
                    looker(&Token::new(token::OpenDelim(*delim), dspan.open))
                }
            },
            None => looker(&Token::new(token::CloseDelim(frame.delim), frame.span.close)),
        }
    }

    /// Returns whether any of the given keywords are `dist` tokens ahead of the current one.
    fn is_keyword_ahead(&self, dist: usize, kws: &[Symbol]) -> bool {
        self.look_ahead(dist, |t| kws.iter().any(|&kw| t.is_keyword(kw)))
    }

    /// Parses asyncness: `async` or nothing.
    fn parse_asyncness(&mut self) -> Async {
        if self.eat_keyword(kw::Async) {
            let span = self.prev_token.uninterpolated_span();
            Async::Yes { span, closure_id: DUMMY_NODE_ID, return_impl_trait_id: DUMMY_NODE_ID }
        } else {
            Async::No
        }
    }

    /// Parses unsafety: `unsafe` or nothing.
    fn parse_unsafety(&mut self) -> Unsafe {
        if self.eat_keyword(kw::Unsafe) {
            Unsafe::Yes(self.prev_token.uninterpolated_span())
        } else {
            Unsafe::No
        }
    }

    /// Parses constness: `const` or nothing.
    fn parse_constness(&mut self) -> Const {
        // Avoid const blocks to be parsed as const items
        if self.look_ahead(1, |t| t != &token::OpenDelim(DelimToken::Brace))
            && self.eat_keyword(kw::Const)
        {
            Const::Yes(self.prev_token.uninterpolated_span())
        } else {
            Const::No
        }
    }

    /// Parses inline const expressions.
    fn parse_const_block(&mut self, span: Span) -> PResult<'a, P<Expr>> {
        self.sess.gated_spans.gate(sym::inline_const, span);
        self.eat_keyword(kw::Const);
        let blk = self.parse_block()?;
        let anon_const = AnonConst {
            id: DUMMY_NODE_ID,
            value: self.mk_expr(blk.span, ExprKind::Block(blk, None), AttrVec::new()),
        };
        let blk_span = anon_const.value.span;
        Ok(self.mk_expr(span.to(blk_span), ExprKind::ConstBlock(anon_const), AttrVec::new()))
    }

    /// Parses mutability (`mut` or nothing).
    fn parse_mutability(&mut self) -> Mutability {
        if self.eat_keyword(kw::Mut) { Mutability::Mut } else { Mutability::Not }
    }

    /// Possibly parses mutability (`const` or `mut`).
    fn parse_const_or_mut(&mut self) -> Option<Mutability> {
        if self.eat_keyword(kw::Mut) {
            Some(Mutability::Mut)
        } else if self.eat_keyword(kw::Const) {
            Some(Mutability::Not)
        } else {
            None
        }
    }

    fn parse_field_name(&mut self) -> PResult<'a, Ident> {
        if let token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) = self.token.kind
        {
            self.expect_no_suffix(self.token.span, "a tuple index", suffix);
            self.bump();
            Ok(Ident::new(symbol, self.prev_token.span))
        } else {
            self.parse_ident_common(false)
        }
    }

    fn parse_mac_args(&mut self) -> PResult<'a, P<MacArgs>> {
        self.parse_mac_args_common(true).map(P)
    }

    fn parse_attr_args(&mut self) -> PResult<'a, MacArgs> {
        self.parse_mac_args_common(false)
    }

    fn parse_mac_args_common(&mut self, delimited_only: bool) -> PResult<'a, MacArgs> {
        Ok(
            if self.check(&token::OpenDelim(DelimToken::Paren))
                || self.check(&token::OpenDelim(DelimToken::Bracket))
                || self.check(&token::OpenDelim(DelimToken::Brace))
            {
                match self.parse_token_tree() {
                    TokenTree::Delimited(dspan, delim, tokens) =>
                    // We've confirmed above that there is a delimiter so unwrapping is OK.
                    {
                        MacArgs::Delimited(dspan, MacDelimiter::from_token(delim).unwrap(), tokens)
                    }
                    _ => unreachable!(),
                }
            } else if !delimited_only {
                if self.eat(&token::Eq) {
                    let eq_span = self.prev_token.span;
                    let mut is_interpolated_expr = false;
                    if let token::Interpolated(nt) = &self.token.kind {
                        if let token::NtExpr(..) = **nt {
                            is_interpolated_expr = true;
                        }
                    }
                    let token_tree = if is_interpolated_expr {
                        // We need to accept arbitrary interpolated expressions to continue
                        // supporting things like `doc = $expr` that work on stable.
                        // Non-literal interpolated expressions are rejected after expansion.
                        self.parse_token_tree()
                    } else {
                        self.parse_unsuffixed_lit()?.token_tree()
                    };

                    MacArgs::Eq(eq_span, token_tree.into())
                } else {
                    MacArgs::Empty
                }
            } else {
                return self.unexpected();
            },
        )
    }

    fn parse_or_use_outer_attributes(
        &mut self,
        already_parsed_attrs: Option<AttrVec>,
    ) -> PResult<'a, AttrVec> {
        if let Some(attrs) = already_parsed_attrs {
            Ok(attrs)
        } else {
            self.parse_outer_attributes().map(|a| a.into())
        }
    }

    /// Parses a single token tree from the input.
    pub(crate) fn parse_token_tree(&mut self) -> TokenTree {
        match self.token.kind {
            token::OpenDelim(..) => {
                let depth = self.token_cursor.stack.len();

                // We keep advancing the token cursor until we hit
                // the matching `CloseDelim` token.
                while !(depth == self.token_cursor.stack.len()
                    && matches!(self.token.kind, token::CloseDelim(_)))
                {
                    // Advance one token at a time, so `TokenCursor::next()`
                    // can capture these tokens if necessary.
                    self.bump();
                }
                // We are still inside the frame corresponding
                // to the delimited stream we captured, so grab
                // the tokens from this frame.
                let frame = &self.token_cursor.frame;
                let stream = frame.tree_cursor.stream.clone();
                let span = frame.span;
                let delim = frame.delim;
                // Consume close delimiter
                self.bump();
                TokenTree::Delimited(span, delim, stream)
            }
            token::CloseDelim(_) | token::Eof => unreachable!(),
            _ => {
                self.bump();
                TokenTree::Token(self.prev_token.clone())
            }
        }
    }

    /// Parses a stream of tokens into a list of `TokenTree`s, up to EOF.
    pub fn parse_all_token_trees(&mut self) -> PResult<'a, Vec<TokenTree>> {
        let mut tts = Vec::new();
        while self.token != token::Eof {
            tts.push(self.parse_token_tree());
        }
        Ok(tts)
    }

    pub fn parse_tokens(&mut self) -> TokenStream {
        let mut result = Vec::new();
        loop {
            match self.token.kind {
                token::Eof | token::CloseDelim(..) => break,
                _ => result.push(self.parse_token_tree().into()),
            }
        }
        TokenStream::new(result)
    }

    /// Evaluates the closure with restrictions in place.
    ///
    /// Afters the closure is evaluated, restrictions are reset.
    fn with_res<T>(&mut self, res: Restrictions, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = self.restrictions;
        self.restrictions = res;
        let res = f(self);
        self.restrictions = old;
        res
    }

    fn is_crate_vis(&self) -> bool {
        self.token.is_keyword(kw::Crate) && self.look_ahead(1, |t| t != &token::ModSep)
    }

    /// Parses `pub`, `pub(crate)` and `pub(in path)` plus shortcuts `crate` for `pub(crate)`,
    /// `pub(self)` for `pub(in self)` and `pub(super)` for `pub(in super)`.
    /// If the following element can't be a tuple (i.e., it's a function definition), then
    /// it's not a tuple struct field), and the contents within the parentheses isn't valid,
    /// so emit a proper diagnostic.
    // Public for rustfmt usage.
    pub fn parse_visibility(&mut self, fbt: FollowedByType) -> PResult<'a, Visibility> {
        maybe_whole!(self, NtVis, |x| x);

        self.expected_tokens.push(TokenType::Keyword(kw::Crate));
        if self.is_crate_vis() {
            self.bump(); // `crate`
            self.sess.gated_spans.gate(sym::crate_visibility_modifier, self.prev_token.span);
            return Ok(Visibility {
                span: self.prev_token.span,
                kind: VisibilityKind::Crate(CrateSugar::JustCrate),
                tokens: None,
            });
        }

        if !self.eat_keyword(kw::Pub) {
            // We need a span for our `Spanned<VisibilityKind>`, but there's inherently no
            // keyword to grab a span from for inherited visibility; an empty span at the
            // beginning of the current token would seem to be the "Schelling span".
            return Ok(Visibility {
                span: self.token.span.shrink_to_lo(),
                kind: VisibilityKind::Inherited,
                tokens: None,
            });
        }
        let lo = self.prev_token.span;

        if self.check(&token::OpenDelim(token::Paren)) {
            // We don't `self.bump()` the `(` yet because this might be a struct definition where
            // `()` or a tuple might be allowed. For example, `struct Struct(pub (), pub (usize));`.
            // Because of this, we only `bump` the `(` if we're assured it is appropriate to do so
            // by the following tokens.
            if self.is_keyword_ahead(1, &[kw::Crate]) && self.look_ahead(2, |t| t != &token::ModSep)
            // account for `pub(crate::foo)`
            {
                // Parse `pub(crate)`.
                self.bump(); // `(`
                self.bump(); // `crate`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = VisibilityKind::Crate(CrateSugar::PubCrate);
                return Ok(Visibility {
                    span: lo.to(self.prev_token.span),
                    kind: vis,
                    tokens: None,
                });
            } else if self.is_keyword_ahead(1, &[kw::In]) {
                // Parse `pub(in path)`.
                self.bump(); // `(`
                self.bump(); // `in`
                let path = self.parse_path(PathStyle::Mod)?; // `path`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = VisibilityKind::Restricted { path: P(path), id: ast::DUMMY_NODE_ID };
                return Ok(Visibility {
                    span: lo.to(self.prev_token.span),
                    kind: vis,
                    tokens: None,
                });
            } else if self.look_ahead(2, |t| t == &token::CloseDelim(token::Paren))
                && self.is_keyword_ahead(1, &[kw::Super, kw::SelfLower])
            {
                // Parse `pub(self)` or `pub(super)`.
                self.bump(); // `(`
                let path = self.parse_path(PathStyle::Mod)?; // `super`/`self`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = VisibilityKind::Restricted { path: P(path), id: ast::DUMMY_NODE_ID };
                return Ok(Visibility {
                    span: lo.to(self.prev_token.span),
                    kind: vis,
                    tokens: None,
                });
            } else if let FollowedByType::No = fbt {
                // Provide this diagnostic if a type cannot follow;
                // in particular, if this is not a tuple struct.
                self.recover_incorrect_vis_restriction()?;
                // Emit diagnostic, but continue with public visibility.
            }
        }

        Ok(Visibility { span: lo, kind: VisibilityKind::Public, tokens: None })
    }

    /// Recovery for e.g. `pub(something) fn ...` or `struct X { pub(something) y: Z }`
    fn recover_incorrect_vis_restriction(&mut self) -> PResult<'a, ()> {
        self.bump(); // `(`
        let path = self.parse_path(PathStyle::Mod)?;
        self.expect(&token::CloseDelim(token::Paren))?; // `)`

        let msg = "incorrect visibility restriction";
        let suggestion = r##"some possible visibility restrictions are:
`pub(crate)`: visible only on the current crate
`pub(super)`: visible only in the current module's parent
`pub(in path::to::module)`: visible only on the specified path"##;

        let path_str = pprust::path_to_string(&path);

        struct_span_err!(self.sess.span_diagnostic, path.span, E0704, "{}", msg)
            .help(suggestion)
            .span_suggestion(
                path.span,
                &format!("make this visible only to module `{}` with `in`", path_str),
                format!("in {}", path_str),
                Applicability::MachineApplicable,
            )
            .emit();

        Ok(())
    }

    /// Parses `extern string_literal?`.
    fn parse_extern(&mut self) -> PResult<'a, Extern> {
        Ok(if self.eat_keyword(kw::Extern) {
            Extern::from_abi(self.parse_abi())
        } else {
            Extern::None
        })
    }

    /// Parses a string literal as an ABI spec.
    fn parse_abi(&mut self) -> Option<StrLit> {
        match self.parse_str_lit() {
            Ok(str_lit) => Some(str_lit),
            Err(Some(lit)) => match lit.kind {
                ast::LitKind::Err(_) => None,
                _ => {
                    self.struct_span_err(lit.span, "non-string ABI literal")
                        .span_suggestion(
                            lit.span,
                            "specify the ABI with a string literal",
                            "\"C\"".to_string(),
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                    None
                }
            },
            Err(None) => None,
        }
    }

    /// Records all tokens consumed by the provided callback,
    /// including the current token. These tokens are collected
    /// into a `LazyTokenStream`, and returned along with the result
    /// of the callback.
    ///
    /// Note: If your callback consumes an opening delimiter
    /// (including the case where you call `collect_tokens`
    /// when the current token is an opening delimeter),
    /// you must also consume the corresponding closing delimiter.
    ///
    /// That is, you can consume
    /// `something ([{ }])` or `([{}])`, but not `([{}]`
    ///
    /// This restriction shouldn't be an issue in practice,
    /// since this function is used to record the tokens for
    /// a parsed AST item, which always has matching delimiters.
    pub fn collect_tokens<R>(
        &mut self,
        f: impl FnOnce(&mut Self) -> PResult<'a, R>,
    ) -> PResult<'a, (R, Option<LazyTokenStream>)> {
        let start_token = (self.token.clone(), self.token_spacing);
        let cursor_snapshot = self.token_cursor.clone();

        let ret = f(self)?;

        // Produces a `TokenStream` on-demand. Using `cursor_snapshot`
        // and `num_calls`, we can reconstruct the `TokenStream` seen
        // by the callback. This allows us to avoid producing a `TokenStream`
        // if it is never needed - for example, a captured `macro_rules!`
        // argument that is never passed to a proc macro.
        // In practice token stream creation happens rarely compared to
        // calls to `collect_tokens` (see some statistics in #78736),
        // so we are doing as little up-front work as possible.
        //
        // This also makes `Parser` very cheap to clone, since
        // there is no intermediate collection buffer to clone.
        #[derive(Clone)]
        struct LazyTokenStreamImpl {
            start_token: (Token, Spacing),
            cursor_snapshot: TokenCursor,
            num_calls: usize,
            desugar_doc_comments: bool,
            trailing_semi: bool,
        }
        impl CreateTokenStream for LazyTokenStreamImpl {
            fn create_token_stream(&self) -> TokenStream {
                let mut num_calls = self.num_calls;
                if self.trailing_semi {
                    num_calls += 1;
                }
                // The token produced by the final call to `next` or `next_desugared`
                // was not actually consumed by the callback. The combination
                // of chaining the initial token and using `take` produces the desired
                // result - we produce an empty `TokenStream` if no calls were made,
                // and omit the final token otherwise.
                let mut cursor_snapshot = self.cursor_snapshot.clone();
                let tokens = std::iter::once(self.start_token.clone())
                    .chain((0..num_calls).map(|_| {
                        if self.desugar_doc_comments {
                            cursor_snapshot.next_desugared()
                        } else {
                            cursor_snapshot.next()
                        }
                    }))
                    .take(num_calls);

                make_token_stream(tokens)
            }
            fn add_trailing_semi(&self) -> Box<dyn CreateTokenStream> {
                if self.trailing_semi {
                    panic!("Called `add_trailing_semi` twice!");
                }
                let mut new = self.clone();
                new.trailing_semi = true;
                Box::new(new)
            }
        }

        let lazy_impl = LazyTokenStreamImpl {
            start_token,
            num_calls: self.token_cursor.num_next_calls - cursor_snapshot.num_next_calls,
            cursor_snapshot,
            desugar_doc_comments: self.desugar_doc_comments,
            trailing_semi: false,
        };
        Ok((ret, Some(LazyTokenStream::new(lazy_impl))))
    }

    /// `::{` or `::*`
    fn is_import_coupler(&mut self) -> bool {
        self.check(&token::ModSep)
            && self.look_ahead(1, |t| {
                *t == token::OpenDelim(token::Brace) || *t == token::BinOp(token::Star)
            })
    }

    pub fn clear_expected_tokens(&mut self) {
        self.expected_tokens.clear();
    }
}

crate fn make_unclosed_delims_error(
    unmatched: UnmatchedBrace,
    sess: &ParseSess,
) -> Option<DiagnosticBuilder<'_>> {
    // `None` here means an `Eof` was found. We already emit those errors elsewhere, we add them to
    // `unmatched_braces` only for error recovery in the `Parser`.
    let found_delim = unmatched.found_delim?;
    let mut err = sess.span_diagnostic.struct_span_err(
        unmatched.found_span,
        &format!(
            "mismatched closing delimiter: `{}`",
            pprust::token_kind_to_string(&token::CloseDelim(found_delim)),
        ),
    );
    err.span_label(unmatched.found_span, "mismatched closing delimiter");
    if let Some(sp) = unmatched.candidate_span {
        err.span_label(sp, "closing delimiter possibly meant for this");
    }
    if let Some(sp) = unmatched.unclosed_span {
        err.span_label(sp, "unclosed delimiter");
    }
    Some(err)
}

pub fn emit_unclosed_delims(unclosed_delims: &mut Vec<UnmatchedBrace>, sess: &ParseSess) {
    *sess.reached_eof.borrow_mut() |=
        unclosed_delims.iter().any(|unmatched_delim| unmatched_delim.found_delim.is_none());
    for unmatched in unclosed_delims.drain(..) {
        if let Some(mut e) = make_unclosed_delims_error(unmatched, sess) {
            e.emit();
        }
    }
}

/// Converts a flattened iterator of tokens (including open and close delimiter tokens)
/// into a `TokenStream`, creating a `TokenTree::Delimited` for each matching pair
/// of open and close delims.
fn make_token_stream(tokens: impl Iterator<Item = (Token, Spacing)>) -> TokenStream {
    #[derive(Debug)]
    struct FrameData {
        open: Span,
        inner: Vec<(TokenTree, Spacing)>,
    }
    let mut stack = vec![FrameData { open: DUMMY_SP, inner: vec![] }];
    for (token, spacing) in tokens {
        match token {
            Token { kind: TokenKind::OpenDelim(_), span } => {
                stack.push(FrameData { open: span, inner: vec![] });
            }
            Token { kind: TokenKind::CloseDelim(delim), span } => {
                let frame_data = stack.pop().expect("Token stack was empty!");
                let dspan = DelimSpan::from_pair(frame_data.open, span);
                let stream = TokenStream::new(frame_data.inner);
                let delimited = TokenTree::Delimited(dspan, delim, stream);
                stack
                    .last_mut()
                    .unwrap_or_else(|| panic!("Bottom token frame is missing for tokens!"))
                    .inner
                    .push((delimited, Spacing::Alone));
            }
            token => stack
                .last_mut()
                .expect("Bottom token frame is missing!")
                .inner
                .push((TokenTree::Token(token), spacing)),
        }
    }
    let final_buf = stack.pop().expect("Missing final buf!");
    assert!(stack.is_empty(), "Stack should be empty: final_buf={:?} stack={:?}", final_buf, stack);
    TokenStream::new(final_buf.inner)
}
