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

use crate::lexer::UnmatchedDelim;
pub use attr_wrapper::AttrWrapper;
pub use diagnostics::AttemptLocalParseRecovery;
pub(crate) use item::FnParseMode;
pub use pat::{CommaRecoveryMode, RecoverColon, RecoverComma};
pub use path::PathStyle;

use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{AttributesData, DelimSpan, Spacing};
use rustc_ast::tokenstream::{TokenStream, TokenTree, TokenTreeCursor};
use rustc_ast::util::case::Case;
use rustc_ast::AttrId;
use rustc_ast::DUMMY_NODE_ID;
use rustc_ast::{self as ast, AnonConst, AttrStyle, Const, DelimArgs, Extern};
use rustc_ast::{Async, AttrArgs, AttrArgsEq, Expr, ExprKind, MacDelimiter, Mutability, StrLit};
use rustc_ast::{HasAttrs, HasTokens, Unsafe, Visibility, VisibilityKind};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::PResult;
use rustc_errors::{
    Applicability, DiagnosticBuilder, ErrorGuaranteed, FatalError, IntoDiagnostic, MultiSpan,
};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{Span, DUMMY_SP};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use std::ops::Range;
use std::{cmp, mem, slice};
use thin_vec::ThinVec;
use tracing::debug;

use crate::errors::{
    DocCommentDoesNotDocumentAnything, IncorrectVisibilityRestriction, MismatchedClosingDelimiter,
    NonStringAbiLiteral,
};

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
    Gt,
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
                    && $self.may_recover()
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

#[derive(Clone, Copy)]
pub enum Recovery {
    Allowed,
    Forbidden,
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
    pub(super) unclosed_delims: Vec<UnmatchedDelim>,
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
    /// Whether the parser is allowed to do recovery.
    /// This is disabled when parsing macro arguments, see #103534
    pub recovery: Recovery,
}

// This type is used a lot, e.g. it's cloned when matching many declarative macro rules with nonterminals. Make sure
// it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(Parser<'_>, 312);

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
/// on an `AttrTokenStream`.
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
/// we will never need an `AttrTokenStream`.
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

/// Iterator over a `TokenStream` that produces `Token`s. It's a bit odd that
/// we (a) lex tokens into a nice tree structure (`TokenStream`), and then (b)
/// use this type to emit them as a linear sequence. But a linear sequence is
/// what the parser expects, for the most part.
#[derive(Clone)]
struct TokenCursor {
    // Cursor for the current (innermost) token stream. The delimiters for this
    // token stream are found in `self.stack.last()`; when that is `None` then
    // we are in the outermost token stream which never has delimiters.
    tree_cursor: TokenTreeCursor,

    // Token streams surrounding the current one. The delimiters for stack[n]'s
    // tokens are in `stack[n-1]`. `stack[0]` (when present) has no delimiters
    // because it's the outermost token stream which never has delimiters.
    stack: Vec<(TokenTreeCursor, Delimiter, DelimSpan)>,

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
    // we evaluate a `LazyAttrTokenStream`.
    break_last_token: bool,
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
            if let Some(tree) = self.tree_cursor.next_ref() {
                match tree {
                    &TokenTree::Token(ref token, spacing) => match (desugar_doc_comments, token) {
                        (true, &Token { kind: token::DocComment(_, attr_style, data), span }) => {
                            let desugared = self.desugar(attr_style, data, span);
                            self.tree_cursor.replace_prev_and_rewind(desugared);
                            // Continue to get the first token of the desugared doc comment.
                        }
                        _ => {
                            debug_assert!(!matches!(
                                token.kind,
                                token::OpenDelim(_) | token::CloseDelim(_)
                            ));
                            return (token.clone(), spacing);
                        }
                    },
                    &TokenTree::Delimited(sp, delim, ref tts) => {
                        let trees = tts.clone().into_trees();
                        self.stack.push((mem::replace(&mut self.tree_cursor, trees), delim, sp));
                        if delim != Delimiter::Invisible {
                            return (Token::new(token::OpenDelim(delim), sp.open), Spacing::Alone);
                        }
                        // No open delimiter to return; continue on to the next iteration.
                    }
                };
            } else if let Some((tree_cursor, delim, span)) = self.stack.pop() {
                // We have exhausted this token stream. Move back to its parent token stream.
                self.tree_cursor = tree_cursor;
                if delim != Delimiter::Invisible {
                    return (Token::new(token::CloseDelim(delim), span.close), Spacing::Alone);
                }
                // No close delimiter to return; continue on to the next iteration.
            } else {
                // We have exhausted the outermost token stream.
                return (Token::new(token::Eof, DUMMY_SP), Spacing::Alone);
            }
        }
    }

    // Desugar a doc comment into something like `#[doc = r"foo"]`.
    fn desugar(&mut self, attr_style: AttrStyle, data: Symbol, span: Span) -> Vec<TokenTree> {
        // Searches for the occurrences of `"#*` and returns the minimum number of `#`s
        // required to wrap the text. E.g.
        // - `abc d` is wrapped as `r"abc d"` (num_of_hashes = 0)
        // - `abc "d"` is wrapped as `r#"abc "d""#` (num_of_hashes = 1)
        // - `abc "##d##"` is wrapped as `r###"abc ##"d"##"###` (num_of_hashes = 3)
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

        // `/// foo` becomes `doc = r"foo".
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

        if attr_style == AttrStyle::Inner {
            vec![
                TokenTree::token_alone(token::Pound, span),
                TokenTree::token_alone(token::Not, span),
                body,
            ]
        } else {
            vec![TokenTree::token_alone(token::Pound, span), body]
        }
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
        match self {
            TokenType::Token(t) => format!("`{}`", pprust::token_kind_to_string(t)),
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TokenDescription {
    ReservedIdentifier,
    Keyword,
    ReservedKeyword,
    DocComment,
}

impl TokenDescription {
    pub fn from_token(token: &Token) -> Option<Self> {
        match token.kind {
            _ if token.is_special_ident() => Some(TokenDescription::ReservedIdentifier),
            _ if token.is_used_keyword() => Some(TokenDescription::Keyword),
            _ if token.is_unused_keyword() => Some(TokenDescription::ReservedKeyword),
            token::DocComment(..) => Some(TokenDescription::DocComment),
            _ => None,
        }
    }
}

pub(super) fn token_descr(token: &Token) -> String {
    let name = pprust::token_to_string(token).to_string();

    let kind = TokenDescription::from_token(token).map(|kind| match kind {
        TokenDescription::ReservedIdentifier => "reserved identifier",
        TokenDescription::Keyword => "keyword",
        TokenDescription::ReservedKeyword => "reserved keyword",
        TokenDescription::DocComment => "doc comment",
    });

    if let Some(kind) = kind { format!("{} `{}`", kind, name) } else { format!("`{}`", name) }
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
                tree_cursor: tokens.into_trees(),
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
            recovery: Recovery::Allowed,
        };

        // Make parser point to the first token.
        parser.bump();

        parser
    }

    pub fn recovery(mut self, recovery: Recovery) -> Self {
        self.recovery = recovery;
        self
    }

    /// Whether the parser is allowed to recover from broken code.
    ///
    /// If this returns false, recovering broken code into valid code (especially if this recovery does lookahead)
    /// is not allowed. All recovery done by the parser must be gated behind this check.
    ///
    /// Technically, this only needs to restrict eager recovery by doing lookahead at more tokens.
    /// But making the distinction is very subtle, and simply forbidding all recovery is a lot simpler to uphold.
    fn may_recover(&self) -> bool {
        matches!(self.recovery, Recovery::Allowed)
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

    /// Expect next token to be edible or inedible token. If edible,
    /// then consume it; if inedible, then return without consuming
    /// anything. Signal a fatal error if next token is unexpected.
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
            TokenKind::DocComment(..) => DocCommentDoesNotDocumentAnything {
                span: self.prev_token.span,
                missing_comma: None,
            }
            .into_diagnostic(&self.sess.span_diagnostic),
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

    fn check_keyword_case(&mut self, kw: Symbol, case: Case) -> bool {
        if self.check_keyword(kw) {
            return true;
        }

        if case == Case::Insensitive
        && let Some((ident, /* is_raw */ false)) = self.token.ident()
        && ident.as_str().to_lowercase() == kw.as_str().to_lowercase() {
            true
        } else {
            false
        }
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

    /// Eats a keyword, optionally ignoring the case.
    /// If the case differs (and is ignored) an error is issued.
    /// This is useful for recovery.
    fn eat_keyword_case(&mut self, kw: Symbol, case: Case) -> bool {
        if self.eat_keyword(kw) {
            return true;
        }

        if case == Case::Insensitive
        && let Some((ident, /* is_raw */ false)) = self.token.ident()
        && ident.as_str().to_lowercase() == kw.as_str().to_lowercase() {
            self
                .struct_span_err(ident.span, format!("keyword `{kw}` is written in a wrong case"))
                .span_suggestion(
                    ident.span,
                    "write it in the correct case",
                    kw,
                    Applicability::MachineApplicable
                ).emit();

            self.bump();
            return true;
        }

        false
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

    fn check_const_closure(&self) -> bool {
        self.is_keyword_ahead(0, &[kw::Const])
            && self.look_ahead(1, |t| match &t.kind {
                // async closures do not work with const closures, so we do not parse that here.
                token::Ident(kw::Move | kw::Static, _) | token::OrOr | token::BinOp(token::Or) => {
                    true
                }
                _ => false,
            })
    }

    fn check_inline_const(&self, dist: usize) -> bool {
        self.is_keyword_ahead(dist, &[kw::Const])
            && self.look_ahead(dist + 1, |t| match &t.kind {
                token::Interpolated(nt) => matches!(**nt, token::NtBlock(..)),
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
    ) -> PResult<'a, (ThinVec<T>, bool /* trailing */, bool /* recovered */)> {
        let mut first = true;
        let mut recovered = false;
        let mut trailing = false;
        let mut v = ThinVec::new();
        let unclosed_delims = !self.unclosed_delims.is_empty();

        while !self.expect_any_with_type(kets, expect) {
            if let token::CloseDelim(..) | token::Eof = self.token.kind {
                break;
            }
            if let Some(t) = &sep.sep {
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
                                    if let Some(tokens) = t.similar_tokens() {
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
                                    for xx in &e.children {
                                        // propagate the help message from sub error 'e' to main error 'expect_err;
                                        expect_err.children.push(xx.clone());
                                    }
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
            let _ =
                self.parse_stmt_without_recovery(false, ForceCollect::Yes).unwrap_or_else(|e| {
                    e.cancel();
                    None
                });
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
    ) -> PResult<'a, (ThinVec<T>, bool, bool)> {
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
    ) -> PResult<'a, (ThinVec<T>, bool /* trailing */)> {
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
    ) -> PResult<'a, (ThinVec<T>, bool)> {
        self.expect(bra)?;
        self.parse_seq_to_end(ket, sep, f)
    }

    fn parse_delim_comma_seq<T>(
        &mut self,
        delim: Delimiter,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, bool)> {
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
    ) -> PResult<'a, (ThinVec<T>, bool)> {
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

        let tree_cursor = &self.token_cursor.tree_cursor;
        if let Some(&(_, delim, span)) = self.token_cursor.stack.last()
            && delim != Delimiter::Invisible
        {
            let all_normal = (0..dist).all(|i| {
                let token = tree_cursor.look_ahead(i);
                !matches!(token, Some(TokenTree::Delimited(_, Delimiter::Invisible, _)))
            });
            if all_normal {
                return match tree_cursor.look_ahead(dist - 1) {
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
    fn parse_asyncness(&mut self, case: Case) -> Async {
        if self.eat_keyword_case(kw::Async, case) {
            let span = self.prev_token.uninterpolated_span();
            Async::Yes { span, closure_id: DUMMY_NODE_ID, return_impl_trait_id: DUMMY_NODE_ID }
        } else {
            Async::No
        }
    }

    /// Parses unsafety: `unsafe` or nothing.
    fn parse_unsafety(&mut self, case: Case) -> Unsafe {
        if self.eat_keyword_case(kw::Unsafe, case) {
            Unsafe::Yes(self.prev_token.uninterpolated_span())
        } else {
            Unsafe::No
        }
    }

    /// Parses constness: `const` or nothing.
    fn parse_constness(&mut self, case: Case) -> Const {
        self.parse_constness_(case, false)
    }

    /// Parses constness for closures
    fn parse_closure_constness(&mut self, case: Case) -> Const {
        self.parse_constness_(case, true)
    }

    fn parse_constness_(&mut self, case: Case, is_closure: bool) -> Const {
        // Avoid const blocks and const closures to be parsed as const items
        if (self.check_const_closure() == is_closure)
            && self.look_ahead(1, |t| t != &token::OpenDelim(Delimiter::Brace))
            && self.eat_keyword_case(kw::Const, case)
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
        Ok(self.mk_expr_with_attrs(span.to(blk_span), ExprKind::ConstBlock(anon_const), attrs))
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
            if let Some(suffix) = suffix {
                self.expect_no_tuple_index_suffix(self.token.span, suffix);
            }
            self.bump();
            Ok(Ident::new(symbol, self.prev_token.span))
        } else {
            self.parse_ident_common(true)
        }
    }

    fn parse_delim_args(&mut self) -> PResult<'a, P<DelimArgs>> {
        if let Some(args) = self.parse_delim_args_inner() { Ok(P(args)) } else { self.unexpected() }
    }

    fn parse_attr_args(&mut self) -> PResult<'a, AttrArgs> {
        Ok(if let Some(args) = self.parse_delim_args_inner() {
            AttrArgs::Delimited(args)
        } else {
            if self.eat(&token::Eq) {
                let eq_span = self.prev_token.span;
                AttrArgs::Eq(eq_span, AttrArgsEq::Ast(self.parse_expr_force_collect()?))
            } else {
                AttrArgs::Empty
            }
        })
    }

    fn parse_delim_args_inner(&mut self) -> Option<DelimArgs> {
        let delimited = self.check(&token::OpenDelim(Delimiter::Parenthesis))
            || self.check(&token::OpenDelim(Delimiter::Bracket))
            || self.check(&token::OpenDelim(Delimiter::Brace));

        delimited.then(|| {
            // We've confirmed above that there is a delimiter so unwrapping is OK.
            let TokenTree::Delimited(dspan, delim, tokens) = self.parse_token_tree() else { unreachable!() };

            DelimArgs { dspan, delim: MacDelimiter::from_token(delim).unwrap(), tokens }
        })
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
                // Grab the tokens within the delimiters.
                let tree_cursor = &self.token_cursor.tree_cursor;
                let stream = tree_cursor.stream.clone();
                let (_, delim, span) = *self.token_cursor.stack.last().unwrap();

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

        let path_str = pprust::path_to_string(&path);
        self.sess.emit_err(IncorrectVisibilityRestriction { span: path.span, inner_str: path_str });

        Ok(())
    }

    /// Parses `extern string_literal?`.
    fn parse_extern(&mut self, case: Case) -> Extern {
        if self.eat_keyword_case(kw::Extern, case) {
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
                    self.sess.emit_err(NonStringAbiLiteral { span: lit.span });
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

    pub fn approx_token_stream_pos(&self) -> usize {
        self.token_cursor.num_next_calls
    }
}

pub(crate) fn make_unclosed_delims_error(
    unmatched: UnmatchedDelim,
    sess: &ParseSess,
) -> Option<DiagnosticBuilder<'_, ErrorGuaranteed>> {
    // `None` here means an `Eof` was found. We already emit those errors elsewhere, we add them to
    // `unmatched_delims` only for error recovery in the `Parser`.
    let found_delim = unmatched.found_delim?;
    let mut spans = vec![unmatched.found_span];
    if let Some(sp) = unmatched.unclosed_span {
        spans.push(sp);
    };
    let err = MismatchedClosingDelimiter {
        spans,
        delimiter: pprust::token_kind_to_string(&token::CloseDelim(found_delim)).to_string(),
        unmatched: unmatched.found_span,
        opening_candidate: unmatched.candidate_span,
        unclosed: unmatched.unclosed_span,
    }
    .into_diagnostic(&sess.span_diagnostic);
    Some(err)
}

pub fn emit_unclosed_delims(unclosed_delims: &mut Vec<UnmatchedDelim>, sess: &ParseSess) {
    *sess.reached_eof.borrow_mut() |=
        unclosed_delims.iter().any(|unmatched_delim| unmatched_delim.found_delim.is_none());
    for unmatched in unclosed_delims.drain(..) {
        if let Some(mut e) = make_unclosed_delims_error(unmatched, sess) {
            e.emit();
        }
    }
}

/// A helper struct used when building an `AttrTokenStream` from
/// a `LazyAttrTokenStream`. Both delimiter and non-delimited tokens
/// are stored as `FlatToken::Token`. A vector of `FlatToken`s
/// is then 'parsed' to build up an `AttrTokenStream` with nested
/// `AttrTokenTree::Delimited` tokens.
#[derive(Debug, Clone)]
pub enum FlatToken {
    /// A token - this holds both delimiter (e.g. '{' and '}')
    /// and non-delimiter tokens
    Token(Token),
    /// Holds the `AttributesData` for an AST node. The
    /// `AttributesData` is inserted directly into the
    /// constructed `AttrTokenStream` as
    /// an `AttrTokenTree::Attributes`.
    AttrTarget(AttributesData),
    /// A special 'empty' token that is ignored during the conversion
    /// to an `AttrTokenStream`. This is used to simplify the
    /// handling of replace ranges.
    Empty,
}

#[derive(Debug)]
pub enum NtOrTt {
    Nt(Nonterminal),
    Tt(TokenTree),
}
