pub mod attr;
mod attr_wrapper;
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
pub use attr_wrapper::AttrWrapper;
pub use diagnostics::AttemptLocalParseRecovery;
use diagnostics::Error;
pub(crate) use item::FnParseMode;
pub use pat::{CommaRecoveryMode, RecoverColon, RecoverComma};
pub use path::PathStyle;

use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::AttributesData;
use rustc_ast::tokenstream::{self, DelimSpan, Spacing};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::AttrId;
use rustc_ast::DUMMY_NODE_ID;
use rustc_ast::{self as ast, AnonConst, AttrStyle, AttrVec, Const, Extern};
use rustc_ast::{Async, Expr, ExprKind, MacArgs, MacArgsEq, MacDelimiter, Mutability, StrLit};
use rustc_ast::{HasAttrs, HasTokens, Unsafe, Visibility, VisibilityKind};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::PResult;
use rustc_errors::{
    struct_span_err, Applicability, DiagnosticBuilder, ErrorGuaranteed, FatalError, MultiSpan,
};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{Span, DUMMY_SP};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use tracing::debug;

use std::ops::Range;
use std::{cmp, mem, slice};

bitflags::bitflags! {
    struct Restrictions: u8 {
        const STMT_EXPR         = 1 << 0;
        const NO_STRUCT_LITERAL = 1 << 1;
        const CONST_EXPR        = 1 << 2;
        const ALLOW_LET         = 1 << 3;
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

/// Whether or not we should force collection of tokens for an AST node,
/// regardless of whether or not it has attributes
#[derive(Clone, Copy, PartialEq)]
pub enum ForceCollect {
    Yes,
    No,
}

#[derive(Debug, Eq, PartialEq)]
pub enum TrailingToken {
    None,
    Semi,
    /// If the trailing token is a comma, then capture it
    /// Otherwise, ignore the trailing token
    MaybeComma,
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
        if $allow_qpath_recovery
                    && $self.look_ahead(1, |t| t == &token::ModSep)
                    && let token::Interpolated(nt) = &$self.token.kind
                    && let token::NtTy(ty) = &**nt
                {
                    let ty = ty.clone();
                    $self.bump();
                    return $self.maybe_recover_from_bad_qpath_stage_2($self.prev_token.span, ty);
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
    pub capture_cfg: bool,
    restrictions: Restrictions,
    expected_tokens: Vec<TokenType>,
    // Important: This must only be advanced from `bump` to ensure that
    // `token_cursor.num_next_calls` is updated properly.
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
    capture_state: CaptureState,
    /// This allows us to recover when the user forget to add braces around
    /// multiple statements in the closure body.
    pub current_closure: Option<ClosureSpans>,
}

// This type is used a lot, e.g. it's cloned when matching many declarative macro rules. Make sure
// it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Parser<'_>, 328);

/// Stores span information about a closure.
#[derive(Clone)]
pub struct ClosureSpans {
    pub whole_closure: Span,
    pub closing_pipe: Span,
    pub body: Span,
}

/// Indicates a range of tokens that should be replaced by
/// the tokens in the provided vector. This is used in two
/// places during token collection:
///
/// 1. During the parsing of an AST node that may have a `#[derive]`
/// attribute, we parse a nested AST node that has `#[cfg]` or `#[cfg_attr]`
/// In this case, we use a `ReplaceRange` to replace the entire inner AST node
/// with `FlatToken::AttrTarget`, allowing us to perform eager cfg-expansion
/// on an `AttrAnnotatedTokenStream`
///
/// 2. When we parse an inner attribute while collecting tokens. We
/// remove inner attributes from the token stream entirely, and
/// instead track them through the `attrs` field on the AST node.
/// This allows us to easily manipulate them (for example, removing
/// the first macro inner attribute to invoke a proc-macro).
/// When create a `TokenStream`, the inner attributes get inserted
/// into the proper place in the token stream.
pub type ReplaceRange = (Range<u32>, Vec<(FlatToken, Spacing)>);

/// Controls how we capture tokens. Capturing can be expensive,
/// so we try to avoid performing capturing in cases where
/// we will never need an `AttrAnnotatedTokenStream`
#[derive(Copy, Clone)]
pub enum Capturing {
    /// We aren't performing any capturing - this is the default mode.
    No,
    /// We are capturing tokens
    Yes,
}

#[derive(Clone)]
struct CaptureState {
    capturing: Capturing,
    replace_ranges: Vec<ReplaceRange>,
    inner_attr_ranges: FxHashMap<AttrId, ReplaceRange>,
}

impl<'a> Drop for Parser<'a> {
    fn drop(&mut self) {
        emit_unclosed_delims(&mut self.unclosed_delims, &self.sess);
    }
}

#[derive(Clone)]
struct TokenCursor {
    // The current (innermost) frame. `frame` and `stack` could be combined,
    // but it's faster to have them separately to access `frame` directly
    // rather than via something like `stack.last().unwrap()` or
    // `stack[stack.len() - 1]`.
    frame: TokenCursorFrame,
    // Additional frames that enclose `frame`.
    stack: Vec<TokenCursorFrame>,
    desugar_doc_comments: bool,
    // Counts the number of calls to `{,inlined_}next`.
    num_next_calls: usize,
    // During parsing, we may sometimes need to 'unglue' a
    // glued token into two component tokens
    // (e.g. '>>' into '>' and '>), so that the parser
    // can consume them one at a time. This process
    // bypasses the normal capturing mechanism
    // (e.g. `num_next_calls` will not be incremented),
    // since the 'unglued' tokens due not exist in
    // the original `TokenStream`.
    //
    // If we end up consuming both unglued tokens,
    // then this is not an issue - we'll end up
    // capturing the single 'glued' token.
    //
    // However, in certain circumstances, we may
    // want to capture just the first 'unglued' token.
    // For example, capturing the `Vec<u8>`
    // in `Option<Vec<u8>>` requires us to unglue
    // the trailing `>>` token. The `break_last_token`
    // field is used to track this token - it gets
    // appended to the captured stream when
    // we evaluate a `LazyTokenStream`
    break_last_token: bool,
}

#[derive(Clone)]
struct TokenCursorFrame {
    delim_sp: Option<(Delimiter, DelimSpan)>,
    tree_cursor: tokenstream::Cursor,
}

impl TokenCursorFrame {
    fn new(delim_sp: Option<(Delimiter, DelimSpan)>, tts: TokenStream) -> Self {
        TokenCursorFrame { delim_sp, tree_cursor: tts.into_trees() }
    }
}

impl TokenCursor {
    fn next(&mut self, desugar_doc_comments: bool) -> (Token, Spacing) {
        self.inlined_next(desugar_doc_comments)
    }

    /// This always-inlined version should only be used on hot code paths.
    #[inline(always)]
    fn inlined_next(&mut self, desugar_doc_comments: bool) -> (Token, Spacing) {
        loop {
            // FIXME: we currently don't return `Delimiter` open/close delims. To fix #67062 we will
            // need to, whereupon the `delim != Delimiter::Invisible` conditions below can be
            // removed.
            if let Some(tree) = self.frame.tree_cursor.next_ref() {
                match tree {
                    &TokenTree::Token(ref token, spacing) => match (desugar_doc_comments, token) {
                        (true, &Token { kind: token::DocComment(_, attr_style, data), span }) => {
                            return self.desugar(attr_style, data, span);
                        }
                        _ => return (token.clone(), spacing),
                    },
                    &TokenTree::Delimited(sp, delim, ref tts) => {
                        // Set `open_delim` to true here because we deal with it immediately.
                        let frame = TokenCursorFrame::new(Some((delim, sp)), tts.clone());
                        self.stack.push(mem::replace(&mut self.frame, frame));
                        if delim != Delimiter::Invisible {
                            return (Token::new(token::OpenDelim(delim), sp.open), Spacing::Alone);
                        }
                        // No open delimeter to return; continue on to the next iteration.
                    }
                };
            } else if let Some(frame) = self.stack.pop() {
                if let Some((delim, span)) = self.frame.delim_sp && delim != Delimiter::Invisible {
                    self.frame = frame;
                    return (Token::new(token::CloseDelim(delim), span.close), Spacing::Alone);
                }
                self.frame = frame;
                // No close delimiter to return; continue on to the next iteration.
            } else {
                return (Token::new(token::Eof, DUMMY_SP), Spacing::Alone);
            }
        }
    }

    fn desugar(&mut self, attr_style: AttrStyle, data: Symbol, span: Span) -> (Token, Spacing) {
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

        let delim_span = DelimSpan::from_single(span);
        let body = TokenTree::Delimited(
            delim_span,
            Delimiter::Bracket,
            [
                TokenTree::token_alone(token::Ident(sym::doc, false), span),
                TokenTree::token_alone(token::Eq, span),
                TokenTree::token_alone(
                    TokenKind::lit(token::StrRaw(num_of_hashes), data, None),
                    span,
                ),
            ]
            .into_iter()
            .collect::<TokenStream>(),
        );

        self.stack.push(mem::replace(
            &mut self.frame,
            TokenCursorFrame::new(
                None,
                if attr_style == AttrStyle::Inner {
                    [
                        TokenTree::token_alone(token::Pound, span),
                        TokenTree::token_alone(token::Not, span),
                        body,
                    ]
                    .into_iter()
                    .collect::<TokenStream>()
                } else {
                    [TokenTree::token_alone(token::Pound, span), body]
                        .into_iter()
                        .collect::<TokenStream>()
                },
            ),
        ));

        self.next(/* desugar_doc_comments */ false)
    }
}

#[derive(Debug, Clone, PartialEq)]
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
            TokenType::Const => "a const expression".to_string(),
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
            capture_cfg: false,
            restrictions: Restrictions::empty(),
            expected_tokens: Vec::new(),
            token_cursor: TokenCursor {
                frame: TokenCursorFrame::new(None, tokens),
                stack: Vec::new(),
                num_next_calls: 0,
                desugar_doc_comments,
                break_last_token: false,
            },
            desugar_doc_comments,
            unmatched_angle_bracket_count: 0,
            max_angle_bracket_count: 0,
            unclosed_delims: Vec::new(),
            last_unexpected_token_span: None,
            last_type_ascription: None,
            subparser_name,
            capture_state: CaptureState {
                capturing: Capturing::No,
                replace_ranges: Vec::new(),
                inner_attr_ranges: Default::default(),
            },
            current_closure: None,
        };

        // Make parser point to the first token.
        parser.bump();

        parser
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

    fn ident_or_err(&mut self) -> PResult<'a, (Ident, /* is_raw */ bool)> {
        self.token.ident().ok_or_else(|| match self.prev_token.kind {
            TokenKind::DocComment(..) => {
                self.span_err(self.prev_token.span, Error::UselessDocComment)
            }
            _ => self.expected_ident_found(),
        })
    }

    fn parse_ident_common(&mut self, recover: bool) -> PResult<'a, Ident> {
        let (ident, is_raw) = self.ident_or_err()?;
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

    fn check_noexpect(&self, tok: &TokenKind) -> bool {
        self.token == *tok
    }

    /// Consumes a token 'tok' if it exists. Returns whether the given token was present.
    ///
    /// the main purpose of this function is to reduce the cluttering of the suggestions list
    /// which using the normal eat method could introduce in some cases.
    pub fn eat_noexpect(&mut self, tok: &TokenKind) -> bool {
        let is_present = self.check_noexpect(tok);
        if is_present {
            self.bump()
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
                token::OpenDelim(Delimiter::Brace) => true,
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
                // Keep track of this token - if we end token capturing now,
                // we'll want to append this token to the captured stream.
                //
                // If we consume any additional tokens, then this token
                // is not needed (we'll capture the entire 'glued' token),
                // and `bump` will set this field to `None`
                self.token_cursor.break_last_token = true;
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
        let unclosed_delims = !self.unclosed_delims.is_empty();

        while !self.expect_any_with_type(kets, expect) {
            if let token::CloseDelim(..) | token::Eof = self.token.kind {
                break;
            }
            if let Some(ref t) = sep.sep {
                if first {
                    first = false;
                } else {
                    match self.expect(t) {
                        Ok(false) => {
                            self.current_closure.take();
                        }
                        Ok(true) => {
                            self.current_closure.take();
                            recovered = true;
                            break;
                        }
                        Err(mut expect_err) => {
                            let sp = self.prev_token.span.shrink_to_hi();
                            let token_str = pprust::token_kind_to_string(t);

                            match self.current_closure.take() {
                                Some(closure_spans) if self.token.kind == TokenKind::Semi => {
                                    // Finding a semicolon instead of a comma
                                    // after a closure body indicates that the
                                    // closure body may be a block but the user
                                    // forgot to put braces around its
                                    // statements.

                                    self.recover_missing_braces_around_closure_body(
                                        closure_spans,
                                        expect_err,
                                    )?;

                                    continue;
                                }

                                _ => {
                                    // Attempt to keep parsing if it was a similar separator.
                                    if let Some(ref tokens) = t.similar_tokens() {
                                        if tokens.contains(&self.token.kind) && !unclosed_delims {
                                            self.bump();
                                        }
                                    }
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
                                        " @ ",
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
                                    expect_err
                                        .span_suggestion_short(
                                            sp,
                                            &format!("missing `{}`", token_str),
                                            token_str,
                                            Applicability::MaybeIncorrect,
                                        )
                                        .emit();

                                    v.push(t);
                                    continue;
                                }
                                Err(e) => {
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

    fn recover_missing_braces_around_closure_body(
        &mut self,
        closure_spans: ClosureSpans,
        mut expect_err: DiagnosticBuilder<'_, ErrorGuaranteed>,
    ) -> PResult<'a, ()> {
        let initial_semicolon = self.token.span;

        while self.eat(&TokenKind::Semi) {
            let _ = self.parse_stmt(ForceCollect::Yes)?;
        }

        expect_err.set_primary_message(
            "closure bodies that contain statements must be surrounded by braces",
        );

        let preceding_pipe_span = closure_spans.closing_pipe;
        let following_token_span = self.token.span;

        let mut first_note = MultiSpan::from(vec![initial_semicolon]);
        first_note.push_span_label(
            initial_semicolon,
            "this `;` turns the preceding closure into a statement",
        );
        first_note.push_span_label(
            closure_spans.body,
            "this expression is a statement because of the trailing semicolon",
        );
        expect_err.span_note(first_note, "statement found outside of a block");

        let mut second_note = MultiSpan::from(vec![closure_spans.whole_closure]);
        second_note.push_span_label(closure_spans.whole_closure, "this is the parsed closure...");
        second_note.push_span_label(
            following_token_span,
            "...but likely you meant the closure to end here",
        );
        expect_err.span_note(second_note, "the closure body may be incorrectly delimited");

        expect_err.set_span(vec![preceding_pipe_span, following_token_span]);

        let opening_suggestion_str = " {".to_string();
        let closing_suggestion_str = "}".to_string();

        expect_err.multipart_suggestion(
            "try adding braces",
            vec![
                (preceding_pipe_span.shrink_to_hi(), opening_suggestion_str),
                (following_token_span.shrink_to_lo(), closing_suggestion_str),
            ],
            Applicability::MaybeIncorrect,
        );

        expect_err.emit();

        Ok(())
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
        delim: Delimiter,
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
        self.parse_delim_comma_seq(Delimiter::Parenthesis, f)
    }

    /// Advance the parser by one token using provided token as the next one.
    fn bump_with(&mut self, next: (Token, Spacing)) {
        self.inlined_bump_with(next)
    }

    /// This always-inlined version should only be used on hot code paths.
    #[inline(always)]
    fn inlined_bump_with(&mut self, (next_token, next_spacing): (Token, Spacing)) {
        // Update the current and previous tokens.
        self.prev_token = mem::replace(&mut self.token, next_token);
        self.token_spacing = next_spacing;

        // Diagnostics.
        self.expected_tokens.clear();
    }

    /// Advance the parser by one token.
    pub fn bump(&mut self) {
        // Note: destructuring here would give nicer code, but it was found in #96210 to be slower
        // than `.0`/`.1` access.
        let mut next = self.token_cursor.inlined_next(self.desugar_doc_comments);
        self.token_cursor.num_next_calls += 1;
        // We've retrieved an token from the underlying
        // cursor, so we no longer need to worry about
        // an unglued token. See `break_and_eat` for more details
        self.token_cursor.break_last_token = false;
        if next.0.span.is_dummy() {
            // Tweak the location for better diagnostics, but keep syntactic context intact.
            let fallback_span = self.token.span;
            next.0.span = fallback_span.with_ctxt(next.0.span.ctxt());
        }
        debug_assert!(!matches!(
            next.0.kind,
            token::OpenDelim(Delimiter::Invisible) | token::CloseDelim(Delimiter::Invisible)
        ));
        self.inlined_bump_with(next)
    }

    /// Look-ahead `dist` tokens of `self.token` and get access to that token there.
    /// When `dist == 0` then the current token is looked at.
    pub fn look_ahead<R>(&self, dist: usize, looker: impl FnOnce(&Token) -> R) -> R {
        if dist == 0 {
            return looker(&self.token);
        }

        let frame = &self.token_cursor.frame;
        if let Some((delim, span)) = frame.delim_sp && delim != Delimiter::Invisible {
            let all_normal = (0..dist).all(|i| {
                let token = frame.tree_cursor.look_ahead(i);
                !matches!(token, Some(TokenTree::Delimited(_, Delimiter::Invisible, _)))
            });
            if all_normal {
                return match frame.tree_cursor.look_ahead(dist - 1) {
                    Some(tree) => match tree {
                        TokenTree::Token(token, _) => looker(token),
                        TokenTree::Delimited(dspan, delim, _) => {
                            looker(&Token::new(token::OpenDelim(*delim), dspan.open))
                        }
                    },
                    None => looker(&Token::new(token::CloseDelim(delim), span.close)),
                };
            }
        }

        let mut cursor = self.token_cursor.clone();
        let mut i = 0;
        let mut token = Token::dummy();
        while i < dist {
            token = cursor.next(/* desugar_doc_comments */ false).0;
            if matches!(
                token.kind,
                token::OpenDelim(Delimiter::Invisible) | token::CloseDelim(Delimiter::Invisible)
            ) {
                continue;
            }
            i += 1;
        }
        return looker(&token);
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
        if self.look_ahead(1, |t| t != &token::OpenDelim(Delimiter::Brace))
            && self.eat_keyword(kw::Const)
        {
            Const::Yes(self.prev_token.uninterpolated_span())
        } else {
            Const::No
        }
    }

    /// Parses inline const expressions.
    fn parse_const_block(&mut self, span: Span, pat: bool) -> PResult<'a, P<Expr>> {
        if pat {
            self.sess.gated_spans.gate(sym::inline_const_pat, span);
        } else {
            self.sess.gated_spans.gate(sym::inline_const, span);
        }
        self.eat_keyword(kw::Const);
        let (attrs, blk) = self.parse_inner_attrs_and_block()?;
        let anon_const = AnonConst {
            id: DUMMY_NODE_ID,
            value: self.mk_expr(blk.span, ExprKind::Block(blk, None)),
        };
        let blk_span = anon_const.value.span;
        Ok(self.mk_expr_with_attrs(
            span.to(blk_span),
            ExprKind::ConstBlock(anon_const),
            AttrVec::from(attrs),
        ))
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
            self.parse_ident_common(true)
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
            if self.check(&token::OpenDelim(Delimiter::Parenthesis))
                || self.check(&token::OpenDelim(Delimiter::Bracket))
                || self.check(&token::OpenDelim(Delimiter::Brace))
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
                    MacArgs::Eq(eq_span, MacArgsEq::Ast(self.parse_expr_force_collect()?))
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
        already_parsed_attrs: Option<AttrWrapper>,
    ) -> PResult<'a, AttrWrapper> {
        if let Some(attrs) = already_parsed_attrs {
            Ok(attrs)
        } else {
            self.parse_outer_attributes()
        }
    }

    /// Parses a single token tree from the input.
    pub(crate) fn parse_token_tree(&mut self) -> TokenTree {
        match self.token.kind {
            token::OpenDelim(..) => {
                // Grab the tokens from this frame.
                let frame = &self.token_cursor.frame;
                let stream = frame.tree_cursor.stream.clone();
                let (delim, span) = frame.delim_sp.unwrap();

                // Advance the token cursor through the entire delimited
                // sequence. After getting the `OpenDelim` we are *within* the
                // delimited sequence, i.e. at depth `d`. After getting the
                // matching `CloseDelim` we are *after* the delimited sequence,
                // i.e. at depth `d - 1`.
                let target_depth = self.token_cursor.stack.len() - 1;
                loop {
                    // Advance one token at a time, so `TokenCursor::next()`
                    // can capture these tokens if necessary.
                    self.bump();
                    if self.token_cursor.stack.len() == target_depth {
                        debug_assert!(matches!(self.token.kind, token::CloseDelim(_)));
                        break;
                    }
                }

                // Consume close delimiter
                self.bump();
                TokenTree::Delimited(span, delim, stream)
            }
            token::CloseDelim(_) | token::Eof => unreachable!(),
            _ => {
                self.bump();
                TokenTree::Token(self.prev_token.clone(), Spacing::Alone)
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
                _ => result.push(self.parse_token_tree()),
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

    /// Parses `pub` and `pub(in path)` plus shortcuts `pub(crate)` for `pub(in crate)`, `pub(self)`
    /// for `pub(in self)` and `pub(super)` for `pub(in super)`.
    /// If the following element can't be a tuple (i.e., it's a function definition), then
    /// it's not a tuple struct field), and the contents within the parentheses aren't valid,
    /// so emit a proper diagnostic.
    // Public for rustfmt usage.
    pub fn parse_visibility(&mut self, fbt: FollowedByType) -> PResult<'a, Visibility> {
        maybe_whole!(self, NtVis, |x| x.into_inner());

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

        if self.check(&token::OpenDelim(Delimiter::Parenthesis)) {
            // We don't `self.bump()` the `(` yet because this might be a struct definition where
            // `()` or a tuple might be allowed. For example, `struct Struct(pub (), pub (usize));`.
            // Because of this, we only `bump` the `(` if we're assured it is appropriate to do so
            // by the following tokens.
            if self.is_keyword_ahead(1, &[kw::In]) {
                // Parse `pub(in path)`.
                self.bump(); // `(`
                self.bump(); // `in`
                let path = self.parse_path(PathStyle::Mod)?; // `path`
                self.expect(&token::CloseDelim(Delimiter::Parenthesis))?; // `)`
                let vis = VisibilityKind::Restricted {
                    path: P(path),
                    id: ast::DUMMY_NODE_ID,
                    shorthand: false,
                };
                return Ok(Visibility {
                    span: lo.to(self.prev_token.span),
                    kind: vis,
                    tokens: None,
                });
            } else if self.look_ahead(2, |t| t == &token::CloseDelim(Delimiter::Parenthesis))
                && self.is_keyword_ahead(1, &[kw::Crate, kw::Super, kw::SelfLower])
            {
                // Parse `pub(crate)`, `pub(self)`, or `pub(super)`.
                self.bump(); // `(`
                let path = self.parse_path(PathStyle::Mod)?; // `crate`/`super`/`self`
                self.expect(&token::CloseDelim(Delimiter::Parenthesis))?; // `)`
                let vis = VisibilityKind::Restricted {
                    path: P(path),
                    id: ast::DUMMY_NODE_ID,
                    shorthand: true,
                };
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
        self.expect(&token::CloseDelim(Delimiter::Parenthesis))?; // `)`

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
    fn parse_extern(&mut self) -> Extern {
        if self.eat_keyword(kw::Extern) {
            let mut extern_span = self.prev_token.span;
            let abi = self.parse_abi();
            if let Some(abi) = abi {
                extern_span = extern_span.to(abi.span);
            }
            Extern::from_abi(abi, extern_span)
        } else {
            Extern::None
        }
    }

    /// Parses a string literal as an ABI spec.
    fn parse_abi(&mut self) -> Option<StrLit> {
        match self.parse_str_lit() {
            Ok(str_lit) => Some(str_lit),
            Err(Some(lit)) => match lit.kind {
                ast::LitKind::Err => None,
                _ => {
                    self.struct_span_err(lit.span, "non-string ABI literal")
                        .span_suggestion(
                            lit.span,
                            "specify the ABI with a string literal",
                            "\"C\"",
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                    None
                }
            },
            Err(None) => None,
        }
    }

    pub fn collect_tokens_no_attrs<R: HasAttrs + HasTokens>(
        &mut self,
        f: impl FnOnce(&mut Self) -> PResult<'a, R>,
    ) -> PResult<'a, R> {
        // The only reason to call `collect_tokens_no_attrs` is if you want tokens, so use
        // `ForceCollect::Yes`
        self.collect_tokens_trailing_token(
            AttrWrapper::empty(),
            ForceCollect::Yes,
            |this, _attrs| Ok((f(this)?, TrailingToken::None)),
        )
    }

    /// `::{` or `::*`
    fn is_import_coupler(&mut self) -> bool {
        self.check(&token::ModSep)
            && self.look_ahead(1, |t| {
                *t == token::OpenDelim(Delimiter::Brace) || *t == token::BinOp(token::Star)
            })
    }

    pub fn clear_expected_tokens(&mut self) {
        self.expected_tokens.clear();
    }
}

pub(crate) fn make_unclosed_delims_error(
    unmatched: UnmatchedBrace,
    sess: &ParseSess,
) -> Option<DiagnosticBuilder<'_, ErrorGuaranteed>> {
    // `None` here means an `Eof` was found. We already emit those errors elsewhere, we add them to
    // `unmatched_braces` only for error recovery in the `Parser`.
    let found_delim = unmatched.found_delim?;
    let span: MultiSpan = if let Some(sp) = unmatched.unclosed_span {
        vec![unmatched.found_span, sp].into()
    } else {
        unmatched.found_span.into()
    };
    let mut err = sess.span_diagnostic.struct_span_err(
        span,
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

/// A helper struct used when building an `AttrAnnotatedTokenStream` from
/// a `LazyTokenStream`. Both delimiter and non-delimited tokens
/// are stored as `FlatToken::Token`. A vector of `FlatToken`s
/// is then 'parsed' to build up an `AttrAnnotatedTokenStream` with nested
/// `AttrAnnotatedTokenTree::Delimited` tokens
#[derive(Debug, Clone)]
pub enum FlatToken {
    /// A token - this holds both delimiter (e.g. '{' and '}')
    /// and non-delimiter tokens
    Token(Token),
    /// Holds the `AttributesData` for an AST node. The
    /// `AttributesData` is inserted directly into the
    /// constructed `AttrAnnotatedTokenStream` as
    /// an `AttrAnnotatedTokenTree::Attributes`
    AttrTarget(AttributesData),
    /// A special 'empty' token that is ignored during the conversion
    /// to an `AttrAnnotatedTokenStream`. This is used to simplify the
    /// handling of replace ranges.
    Empty,
}

#[derive(Debug)]
pub enum NtOrTt {
    Nt(Nonterminal),
    Tt(TokenTree),
}
