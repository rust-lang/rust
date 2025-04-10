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
pub mod token_type;
mod ty;

use std::assert_matches::debug_assert_matches;
use std::ops::Range;
use std::sync::Arc;
use std::{fmt, mem, slice};

use attr_wrapper::{AttrWrapper, UsePreAttrPos};
pub use diagnostics::AttemptLocalParseRecovery;
pub(crate) use expr::ForbiddenLetReason;
pub(crate) use item::FnParseMode;
pub use pat::{CommaRecoveryMode, RecoverColon, RecoverComma};
use path::PathStyle;
use rustc_ast::ptr::P;
use rustc_ast::token::{
    self, Delimiter, IdentIsRaw, InvisibleOrigin, MetaVarKind, Nonterminal, NtExprKind, NtPatKind,
    Token, TokenKind,
};
use rustc_ast::tokenstream::{AttrsTarget, Spacing, TokenStream, TokenTree};
use rustc_ast::util::case::Case;
use rustc_ast::{
    self as ast, AnonConst, AttrArgs, AttrId, ByRef, Const, CoroutineKind, DUMMY_NODE_ID,
    DelimArgs, Expr, ExprKind, Extern, HasAttrs, HasTokens, Mutability, Recovered, Safety, StrLit,
    Visibility, VisibilityKind,
};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Applicability, Diag, FatalError, MultiSpan, PResult};
use rustc_index::interval::IntervalSet;
use rustc_session::parse::ParseSess;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};
use thin_vec::ThinVec;
use token_type::TokenTypeSet;
pub use token_type::{ExpKeywordPair, ExpTokenPair, TokenType};
use tracing::debug;

use crate::errors::{
    self, IncorrectVisibilityRestriction, MismatchedClosingDelimiter, NonStringAbiLiteral,
};
use crate::exp;
use crate::lexer::UnmatchedDelim;

#[cfg(test)]
mod tests;

// Ideally, these tests would be in `rustc_ast`. But they depend on having a
// parser, so they are here.
#[cfg(test)]
mod tokenstream {
    mod tests;
}
#[cfg(test)]
mod mut_visit {
    mod tests;
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug)]
    struct Restrictions: u8 {
        const STMT_EXPR         = 1 << 0;
        const NO_STRUCT_LITERAL = 1 << 1;
        const CONST_EXPR        = 1 << 2;
        const ALLOW_LET         = 1 << 3;
        const IN_IF_GUARD       = 1 << 4;
        const IS_PAT            = 1 << 5;
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ForceCollect {
    Yes,
    No,
}

#[macro_export]
macro_rules! maybe_whole {
    ($p:expr, $constructor:ident, |$x:ident| $e:expr) => {
        #[allow(irrefutable_let_patterns)] // FIXME: temporary
        if let token::Interpolated(nt) = &$p.token.kind
            && let token::$constructor(x) = &**nt
        {
            #[allow(unused_mut)]
            let mut $x = x.clone();
            $p.bump();
            return Ok($e);
        }
    };
}

/// If the next tokens are ill-formed `$ty::` recover them as `<$ty>::`.
#[macro_export]
macro_rules! maybe_recover_from_interpolated_ty_qpath {
    ($self: expr, $allow_qpath_recovery: expr) => {
        if $allow_qpath_recovery
            && $self.may_recover()
            && let Some(mv_kind) = $self.token.is_metavar_seq()
            && let token::MetaVarKind::Ty { .. } = mv_kind
            && $self.check_noexpect_past_close_delim(&token::PathSep)
        {
            // Reparse the type, then move to recovery.
            let ty = $self
                .eat_metavar_seq(mv_kind, |this| this.parse_ty_no_question_mark_recover())
                .expect("metavar seq ty");

            return $self.maybe_recover_from_bad_qpath_stage_2($self.prev_token.span, ty);
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub enum Recovery {
    Allowed,
    Forbidden,
}

#[derive(Clone)]
pub struct Parser<'a> {
    pub psess: &'a ParseSess,
    /// The current token.
    pub token: Token,
    /// The spacing for the current token.
    token_spacing: Spacing,
    /// The previous token.
    pub prev_token: Token,
    pub capture_cfg: bool,
    restrictions: Restrictions,
    expected_token_types: TokenTypeSet,
    token_cursor: TokenCursor,
    // The number of calls to `bump`, i.e. the position in the token stream.
    num_bump_calls: u32,
    // During parsing we may sometimes need to "unglue" a glued token into two
    // or three component tokens (e.g. `>>` into `>` and `>`, or `>>=` into `>`
    // and `>` and `=`), so the parser can consume them one at a time. This
    // process bypasses the normal capturing mechanism (e.g. `num_bump_calls`
    // will not be incremented), since the "unglued" tokens due not exist in
    // the original `TokenStream`.
    //
    // If we end up consuming all the component tokens, this is not an issue,
    // because we'll end up capturing the single "glued" token.
    //
    // However, sometimes we may want to capture not all of the original
    // token. For example, capturing the `Vec<u8>` in `Option<Vec<u8>>`
    // requires us to unglue the trailing `>>` token. The `break_last_token`
    // field is used to track these tokens. They get appended to the captured
    // stream when we evaluate a `LazyAttrTokenStream`.
    //
    // This value is always 0, 1, or 2. It can only reach 2 when splitting
    // `>>=` or `<<=`.
    break_last_token: u32,
    /// This field is used to keep track of how many left angle brackets we have seen. This is
    /// required in order to detect extra leading left angle brackets (`<` characters) and error
    /// appropriately.
    ///
    /// See the comments in the `parse_path_segment` function for more details.
    unmatched_angle_bracket_count: u16,
    angle_bracket_nesting: u16,

    last_unexpected_token_span: Option<Span>,
    /// If present, this `Parser` is not parsing Rust code but rather a macro call.
    subparser_name: Option<&'static str>,
    capture_state: CaptureState,
    /// This allows us to recover when the user forget to add braces around
    /// multiple statements in the closure body.
    current_closure: Option<ClosureSpans>,
    /// Whether the parser is allowed to do recovery.
    /// This is disabled when parsing macro arguments, see #103534
    recovery: Recovery,
}

// This type is used a lot, e.g. it's cloned when matching many declarative macro rules with
// nonterminals. Make sure it doesn't unintentionally get bigger. We only check a few arches
// though, because `TokenTypeSet(u128)` alignment varies on others, changing the total size.
#[cfg(all(target_pointer_width = "64", any(target_arch = "aarch64", target_arch = "x86_64")))]
rustc_data_structures::static_assert_size!(Parser<'_>, 288);

/// Stores span information about a closure.
#[derive(Clone, Debug)]
struct ClosureSpans {
    whole_closure: Span,
    closing_pipe: Span,
    body: Span,
}

/// A token range within a `Parser`'s full token stream.
#[derive(Clone, Debug)]
struct ParserRange(Range<u32>);

/// A token range within an individual AST node's (lazy) token stream, i.e.
/// relative to that node's first token. Distinct from `ParserRange` so the two
/// kinds of range can't be mixed up.
#[derive(Clone, Debug)]
struct NodeRange(Range<u32>);

/// Indicates a range of tokens that should be replaced by an `AttrsTarget`
/// (replacement) or be replaced by nothing (deletion). This is used in two
/// places during token collection.
///
/// 1. Replacement. During the parsing of an AST node that may have a
///    `#[derive]` attribute, when we parse a nested AST node that has `#[cfg]`
///    or `#[cfg_attr]`, we replace the entire inner AST node with
///    `FlatToken::AttrsTarget`. This lets us perform eager cfg-expansion on an
///    `AttrTokenStream`.
///
/// 2. Deletion. We delete inner attributes from all collected token streams,
///    and instead track them through the `attrs` field on the AST node. This
///    lets us manipulate them similarly to outer attributes. When we create a
///    `TokenStream`, the inner attributes are inserted into the proper place
///    in the token stream.
///
/// Each replacement starts off in `ParserReplacement` form but is converted to
/// `NodeReplacement` form when it is attached to a single AST node, via
/// `LazyAttrTokenStreamImpl`.
type ParserReplacement = (ParserRange, Option<AttrsTarget>);

/// See the comment on `ParserReplacement`.
type NodeReplacement = (NodeRange, Option<AttrsTarget>);

impl NodeRange {
    // Converts a range within a parser's tokens to a range within a
    // node's tokens beginning at `start_pos`.
    //
    // For example, imagine a parser with 50 tokens in its token stream, a
    // function that spans `ParserRange(20..40)` and an inner attribute within
    // that function that spans `ParserRange(30..35)`. We would find the inner
    // attribute's range within the function's tokens by subtracting 20, which
    // is the position of the function's start token. This gives
    // `NodeRange(10..15)`.
    fn new(ParserRange(parser_range): ParserRange, start_pos: u32) -> NodeRange {
        assert!(!parser_range.is_empty());
        assert!(parser_range.start >= start_pos);
        NodeRange((parser_range.start - start_pos)..(parser_range.end - start_pos))
    }
}

/// Controls how we capture tokens. Capturing can be expensive,
/// so we try to avoid performing capturing in cases where
/// we will never need an `AttrTokenStream`.
#[derive(Copy, Clone, Debug)]
enum Capturing {
    /// We aren't performing any capturing - this is the default mode.
    No,
    /// We are capturing tokens
    Yes,
}

// This state is used by `Parser::collect_tokens`.
#[derive(Clone, Debug)]
struct CaptureState {
    capturing: Capturing,
    parser_replacements: Vec<ParserReplacement>,
    inner_attr_parser_ranges: FxHashMap<AttrId, ParserRange>,
    // `IntervalSet` is good for perf because attrs are mostly added to this
    // set in contiguous ranges.
    seen_attrs: IntervalSet<AttrId>,
}

#[derive(Clone, Debug)]
struct TokenTreeCursor {
    stream: TokenStream,
    /// Points to the current token tree in the stream. In `TokenCursor::curr`,
    /// this can be any token tree. In `TokenCursor::stack`, this is always a
    /// `TokenTree::Delimited`.
    index: usize,
}

impl TokenTreeCursor {
    #[inline]
    fn new(stream: TokenStream) -> Self {
        TokenTreeCursor { stream, index: 0 }
    }

    #[inline]
    fn curr(&self) -> Option<&TokenTree> {
        self.stream.get(self.index)
    }

    fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
        self.stream.get(self.index + n)
    }

    #[inline]
    fn bump(&mut self) {
        self.index += 1;
    }
}

/// A `TokenStream` cursor that produces `Token`s. It's a bit odd that
/// we (a) lex tokens into a nice tree structure (`TokenStream`), and then (b)
/// use this type to emit them as a linear sequence. But a linear sequence is
/// what the parser expects, for the most part.
#[derive(Clone, Debug)]
struct TokenCursor {
    // Cursor for the current (innermost) token stream. The index within the
    // cursor can point to any token tree in the stream (or one past the end).
    // The delimiters for this token stream are found in `self.stack.last()`;
    // if that is `None` we are in the outermost token stream which never has
    // delimiters.
    curr: TokenTreeCursor,

    // Token streams surrounding the current one. The index within each cursor
    // always points to a `TokenTree::Delimited`.
    stack: Vec<TokenTreeCursor>,
}

impl TokenCursor {
    fn next(&mut self) -> (Token, Spacing) {
        self.inlined_next()
    }

    /// This always-inlined version should only be used on hot code paths.
    #[inline(always)]
    fn inlined_next(&mut self) -> (Token, Spacing) {
        loop {
            // FIXME: we currently don't return `Delimiter::Invisible` open/close delims. To fix
            // #67062 we will need to, whereupon the `delim != Delimiter::Invisible` conditions
            // below can be removed.
            if let Some(tree) = self.curr.curr() {
                match tree {
                    &TokenTree::Token(ref token, spacing) => {
                        debug_assert!(!matches!(
                            token.kind,
                            token::OpenDelim(_) | token::CloseDelim(_)
                        ));
                        let res = (token.clone(), spacing);
                        self.curr.bump();
                        return res;
                    }
                    &TokenTree::Delimited(sp, spacing, delim, ref tts) => {
                        let trees = TokenTreeCursor::new(tts.clone());
                        self.stack.push(mem::replace(&mut self.curr, trees));
                        if !delim.skip() {
                            return (Token::new(token::OpenDelim(delim), sp.open), spacing.open);
                        }
                        // No open delimiter to return; continue on to the next iteration.
                    }
                };
            } else if let Some(parent) = self.stack.pop() {
                // We have exhausted this token stream. Move back to its parent token stream.
                let Some(&TokenTree::Delimited(span, spacing, delim, _)) = parent.curr() else {
                    panic!("parent should be Delimited")
                };
                self.curr = parent;
                self.curr.bump(); // move past the `Delimited`
                if !delim.skip() {
                    return (Token::new(token::CloseDelim(delim), span.close), spacing.close);
                }
                // No close delimiter to return; continue on to the next iteration.
            } else {
                // We have exhausted the outermost token stream. The use of
                // `Spacing::Alone` is arbitrary and immaterial, because the
                // `Eof` token's spacing is never used.
                return (Token::new(token::Eof, DUMMY_SP), Spacing::Alone);
            }
        }
    }
}

/// A sequence separator.
#[derive(Debug)]
struct SeqSep<'a> {
    /// The separator token.
    sep: Option<ExpTokenPair<'a>>,
    /// `true` if a trailing separator is allowed.
    trailing_sep_allowed: bool,
}

impl<'a> SeqSep<'a> {
    fn trailing_allowed(sep: ExpTokenPair<'a>) -> SeqSep<'a> {
        SeqSep { sep: Some(sep), trailing_sep_allowed: true }
    }

    fn none() -> SeqSep<'a> {
        SeqSep { sep: None, trailing_sep_allowed: false }
    }
}

#[derive(Debug)]
pub enum FollowedByType {
    Yes,
    No,
}

#[derive(Copy, Clone, Debug)]
enum Trailing {
    No,
    Yes,
}

impl From<bool> for Trailing {
    fn from(b: bool) -> Trailing {
        if b { Trailing::Yes } else { Trailing::No }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TokenDescription {
    ReservedIdentifier,
    Keyword,
    ReservedKeyword,
    DocComment,

    // Expanded metavariables are wrapped in invisible delimiters which aren't
    // pretty-printed. In error messages we must handle these specially
    // otherwise we get confusing things in messages like "expected `(`, found
    // ``". It's better to say e.g. "expected `(`, found type metavariable".
    MetaVar(MetaVarKind),
}

impl TokenDescription {
    pub(super) fn from_token(token: &Token) -> Option<Self> {
        match token.kind {
            _ if token.is_special_ident() => Some(TokenDescription::ReservedIdentifier),
            _ if token.is_used_keyword() => Some(TokenDescription::Keyword),
            _ if token.is_unused_keyword() => Some(TokenDescription::ReservedKeyword),
            token::DocComment(..) => Some(TokenDescription::DocComment),
            token::OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(kind))) => {
                Some(TokenDescription::MetaVar(kind))
            }
            _ => None,
        }
    }
}

pub fn token_descr(token: &Token) -> String {
    let s = pprust::token_to_string(token).to_string();

    match (TokenDescription::from_token(token), &token.kind) {
        (Some(TokenDescription::ReservedIdentifier), _) => format!("reserved identifier `{s}`"),
        (Some(TokenDescription::Keyword), _) => format!("keyword `{s}`"),
        (Some(TokenDescription::ReservedKeyword), _) => format!("reserved keyword `{s}`"),
        (Some(TokenDescription::DocComment), _) => format!("doc comment `{s}`"),
        // Deliberately doesn't print `s`, which is empty.
        (Some(TokenDescription::MetaVar(kind)), _) => format!("`{kind}` metavariable"),
        (None, TokenKind::NtIdent(..)) => format!("identifier `{s}`"),
        (None, TokenKind::NtLifetime(..)) => format!("lifetime `{s}`"),
        (None, TokenKind::Interpolated(node)) => format!("{} `{s}`", node.descr()),
        (None, _) => format!("`{s}`"),
    }
}

impl<'a> Parser<'a> {
    pub fn new(
        psess: &'a ParseSess,
        stream: TokenStream,
        subparser_name: Option<&'static str>,
    ) -> Self {
        let mut parser = Parser {
            psess,
            token: Token::dummy(),
            token_spacing: Spacing::Alone,
            prev_token: Token::dummy(),
            capture_cfg: false,
            restrictions: Restrictions::empty(),
            expected_token_types: TokenTypeSet::new(),
            token_cursor: TokenCursor { curr: TokenTreeCursor::new(stream), stack: Vec::new() },
            num_bump_calls: 0,
            break_last_token: 0,
            unmatched_angle_bracket_count: 0,
            angle_bracket_nesting: 0,
            last_unexpected_token_span: None,
            subparser_name,
            capture_state: CaptureState {
                capturing: Capturing::No,
                parser_replacements: Vec::new(),
                inner_attr_parser_ranges: Default::default(),
                seen_attrs: IntervalSet::new(u32::MAX as usize),
            },
            current_closure: None,
            recovery: Recovery::Allowed,
        };

        // Make parser point to the first token.
        parser.bump();

        // Change this from 1 back to 0 after the bump. This eases debugging of
        // `Parser::collect_tokens` because 0-indexed token positions are nicer
        // than 1-indexed token positions.
        parser.num_bump_calls = 0;

        parser
    }

    #[inline]
    pub fn recovery(mut self, recovery: Recovery) -> Self {
        self.recovery = recovery;
        self
    }

    #[inline]
    fn with_recovery<T>(&mut self, recovery: Recovery, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = mem::replace(&mut self.recovery, recovery);
        let res = f(self);
        self.recovery = old;
        res
    }

    /// Whether the parser is allowed to recover from broken code.
    ///
    /// If this returns false, recovering broken code into valid code (especially if this recovery does lookahead)
    /// is not allowed. All recovery done by the parser must be gated behind this check.
    ///
    /// Technically, this only needs to restrict eager recovery by doing lookahead at more tokens.
    /// But making the distinction is very subtle, and simply forbidding all recovery is a lot simpler to uphold.
    #[inline]
    fn may_recover(&self) -> bool {
        matches!(self.recovery, Recovery::Allowed)
    }

    /// Version of [`unexpected`](Parser::unexpected) that "returns" any type in the `Ok`
    /// (both those functions never return "Ok", and so can lie like that in the type).
    pub fn unexpected_any<T>(&mut self) -> PResult<'a, T> {
        match self.expect_one_of(&[], &[]) {
            Err(e) => Err(e),
            // We can get `Ok(true)` from `recover_closing_delimiter`
            // which is called in `expected_one_of_not_found`.
            Ok(_) => FatalError.raise(),
        }
    }

    pub fn unexpected(&mut self) -> PResult<'a, ()> {
        self.unexpected_any()
    }

    /// Expects and consumes the token `t`. Signals an error if the next token is not `t`.
    pub fn expect(&mut self, exp: ExpTokenPair<'_>) -> PResult<'a, Recovered> {
        if self.expected_token_types.is_empty() {
            if self.token == *exp.tok {
                self.bump();
                Ok(Recovered::No)
            } else {
                self.unexpected_try_recover(exp.tok)
            }
        } else {
            self.expect_one_of(slice::from_ref(&exp), &[])
        }
    }

    /// Expect next token to be edible or inedible token. If edible,
    /// then consume it; if inedible, then return without consuming
    /// anything. Signal a fatal error if next token is unexpected.
    fn expect_one_of(
        &mut self,
        edible: &[ExpTokenPair<'_>],
        inedible: &[ExpTokenPair<'_>],
    ) -> PResult<'a, Recovered> {
        if edible.iter().any(|exp| exp.tok == &self.token.kind) {
            self.bump();
            Ok(Recovered::No)
        } else if inedible.iter().any(|exp| exp.tok == &self.token.kind) {
            // leave it in the input
            Ok(Recovered::No)
        } else if self.token != token::Eof
            && self.last_unexpected_token_span == Some(self.token.span)
        {
            FatalError.raise();
        } else {
            self.expected_one_of_not_found(edible, inedible)
                .map(|error_guaranteed| Recovered::Yes(error_guaranteed))
        }
    }

    // Public for rustfmt usage.
    pub fn parse_ident(&mut self) -> PResult<'a, Ident> {
        self.parse_ident_common(true)
    }

    fn parse_ident_common(&mut self, recover: bool) -> PResult<'a, Ident> {
        let (ident, is_raw) = self.ident_or_err(recover)?;

        if matches!(is_raw, IdentIsRaw::No) && ident.is_reserved() {
            let err = self.expected_ident_found_err();
            if recover {
                err.emit();
            } else {
                return Err(err);
            }
        }
        self.bump();
        Ok(ident)
    }

    fn ident_or_err(&mut self, recover: bool) -> PResult<'a, (Ident, IdentIsRaw)> {
        match self.token.ident() {
            Some(ident) => Ok(ident),
            None => self.expected_ident_found(recover),
        }
    }

    /// Checks if the next token is `tok`, and returns `true` if so.
    ///
    /// This method will automatically add `tok` to `expected_token_types` if `tok` is not
    /// encountered.
    #[inline]
    fn check(&mut self, exp: ExpTokenPair<'_>) -> bool {
        let is_present = self.token == *exp.tok;
        if !is_present {
            self.expected_token_types.insert(exp.token_type);
        }
        is_present
    }

    #[inline]
    #[must_use]
    fn check_noexpect(&self, tok: &TokenKind) -> bool {
        self.token == *tok
    }

    // Check the first token after the delimiter that closes the current
    // delimited sequence. (Panics if used in the outermost token stream, which
    // has no delimiters.) It uses a clone of the relevant tree cursor to skip
    // past the entire `TokenTree::Delimited` in a single step, avoiding the
    // need for unbounded token lookahead.
    //
    // Primarily used when `self.token` matches
    // `OpenDelim(Delimiter::Invisible(_))`, to look ahead through the current
    // metavar expansion.
    fn check_noexpect_past_close_delim(&self, tok: &TokenKind) -> bool {
        let mut tree_cursor = self.token_cursor.stack.last().unwrap().clone();
        tree_cursor.bump();
        matches!(
            tree_cursor.curr(),
            Some(TokenTree::Token(token::Token { kind, .. }, _)) if kind == tok
        )
    }

    /// Consumes a token 'tok' if it exists. Returns whether the given token was present.
    ///
    /// the main purpose of this function is to reduce the cluttering of the suggestions list
    /// which using the normal eat method could introduce in some cases.
    #[inline]
    #[must_use]
    fn eat_noexpect(&mut self, tok: &TokenKind) -> bool {
        let is_present = self.check_noexpect(tok);
        if is_present {
            self.bump()
        }
        is_present
    }

    /// Consumes a token 'tok' if it exists. Returns whether the given token was present.
    #[inline]
    #[must_use]
    pub fn eat(&mut self, exp: ExpTokenPair<'_>) -> bool {
        let is_present = self.check(exp);
        if is_present {
            self.bump()
        }
        is_present
    }

    /// If the next token is the given keyword, returns `true` without eating it.
    /// An expectation is also added for diagnostics purposes.
    #[inline]
    #[must_use]
    fn check_keyword(&mut self, exp: ExpKeywordPair) -> bool {
        let is_keyword = self.token.is_keyword(exp.kw);
        if !is_keyword {
            self.expected_token_types.insert(exp.token_type);
        }
        is_keyword
    }

    #[inline]
    #[must_use]
    fn check_keyword_case(&mut self, exp: ExpKeywordPair, case: Case) -> bool {
        if self.check_keyword(exp) {
            true
        } else if case == Case::Insensitive
            && let Some((ident, IdentIsRaw::No)) = self.token.ident()
            // Do an ASCII case-insensitive match, because all keywords are ASCII.
            && ident.as_str().eq_ignore_ascii_case(exp.kw.as_str())
        {
            true
        } else {
            false
        }
    }

    /// If the next token is the given keyword, eats it and returns `true`.
    /// Otherwise, returns `false`. An expectation is also added for diagnostics purposes.
    // Public for rustc_builtin_macros and rustfmt usage.
    #[inline]
    #[must_use]
    pub fn eat_keyword(&mut self, exp: ExpKeywordPair) -> bool {
        let is_keyword = self.check_keyword(exp);
        if is_keyword {
            self.bump();
        }
        is_keyword
    }

    /// Eats a keyword, optionally ignoring the case.
    /// If the case differs (and is ignored) an error is issued.
    /// This is useful for recovery.
    #[inline]
    #[must_use]
    fn eat_keyword_case(&mut self, exp: ExpKeywordPair, case: Case) -> bool {
        if self.eat_keyword(exp) {
            true
        } else if case == Case::Insensitive
            && let Some((ident, IdentIsRaw::No)) = self.token.ident()
            // Do an ASCII case-insensitive match, because all keywords are ASCII.
            && ident.as_str().eq_ignore_ascii_case(exp.kw.as_str())
        {
            self.dcx().emit_err(errors::KwBadCase { span: ident.span, kw: exp.kw.as_str() });
            self.bump();
            true
        } else {
            false
        }
    }

    /// If the next token is the given keyword, eats it and returns `true`.
    /// Otherwise, returns `false`. No expectation is added.
    // Public for rustc_builtin_macros usage.
    #[inline]
    #[must_use]
    pub fn eat_keyword_noexpect(&mut self, kw: Symbol) -> bool {
        let is_keyword = self.token.is_keyword(kw);
        if is_keyword {
            self.bump();
        }
        is_keyword
    }

    /// If the given word is not a keyword, signals an error.
    /// If the next token is not the given word, signals an error.
    /// Otherwise, eats it.
    pub fn expect_keyword(&mut self, exp: ExpKeywordPair) -> PResult<'a, ()> {
        if !self.eat_keyword(exp) { self.unexpected() } else { Ok(()) }
    }

    /// Consume a sequence produced by a metavar expansion, if present.
    fn eat_metavar_seq<T>(
        &mut self,
        mv_kind: MetaVarKind,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> Option<T> {
        self.eat_metavar_seq_with_matcher(|mvk| mvk == mv_kind, f)
    }

    /// A slightly more general form of `eat_metavar_seq`, for use with the
    /// `MetaVarKind` variants that have parameters, where an exact match isn't
    /// desired.
    fn eat_metavar_seq_with_matcher<T>(
        &mut self,
        match_mv_kind: impl Fn(MetaVarKind) -> bool,
        mut f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> Option<T> {
        if let token::OpenDelim(delim) = self.token.kind
            && let Delimiter::Invisible(InvisibleOrigin::MetaVar(mv_kind)) = delim
            && match_mv_kind(mv_kind)
        {
            self.bump();

            // Recovery is disabled when parsing macro arguments, so it must
            // also be disabled when reparsing pasted macro arguments,
            // otherwise we get inconsistent results (e.g. #137874).
            let res = self.with_recovery(Recovery::Forbidden, |this| f(this));

            let res = match res {
                Ok(res) => res,
                Err(err) => {
                    // This can occur in unusual error cases, e.g. #139445.
                    err.delay_as_bug();
                    return None;
                }
            };

            if let token::CloseDelim(delim) = self.token.kind
                && let Delimiter::Invisible(InvisibleOrigin::MetaVar(mv_kind)) = delim
                && match_mv_kind(mv_kind)
            {
                self.bump();
                Some(res)
            } else {
                // This can occur when invalid syntax is passed to a decl macro. E.g. see #139248,
                // where the reparse attempt of an invalid expr consumed the trailing invisible
                // delimiter.
                self.dcx()
                    .span_delayed_bug(self.token.span, "no close delim with reparsing {mv_kind:?}");
                None
            }
        } else {
            None
        }
    }

    /// Is the given keyword `kw` followed by a non-reserved identifier?
    fn is_kw_followed_by_ident(&self, kw: Symbol) -> bool {
        self.token.is_keyword(kw) && self.look_ahead(1, |t| t.is_ident() && !t.is_reserved_ident())
    }

    #[inline]
    fn check_or_expected(&mut self, ok: bool, token_type: TokenType) -> bool {
        if !ok {
            self.expected_token_types.insert(token_type);
        }
        ok
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
                token::Ident(kw::Move | kw::Use | kw::Static, IdentIsRaw::No)
                | token::OrOr
                | token::Or => true,
                _ => false,
            })
    }

    fn check_inline_const(&self, dist: usize) -> bool {
        self.is_keyword_ahead(dist, &[kw::Const])
            && self.look_ahead(dist + 1, |t| match &t.kind {
                token::Interpolated(nt) => matches!(&**nt, token::NtBlock(..)),
                token::OpenDelim(Delimiter::Brace) => true,
                _ => false,
            })
    }

    /// Checks to see if the next token is either `+` or `+=`.
    /// Otherwise returns `false`.
    #[inline]
    fn check_plus(&mut self) -> bool {
        self.check_or_expected(self.token.is_like_plus(), TokenType::Plus)
    }

    /// Eats the expected token if it's present possibly breaking
    /// compound tokens like multi-character operators in process.
    /// Returns `true` if the token was eaten.
    fn break_and_eat(&mut self, exp: ExpTokenPair<'_>) -> bool {
        if self.token == *exp.tok {
            self.bump();
            return true;
        }
        match self.token.kind.break_two_token_op(1) {
            Some((first, second)) if first == *exp.tok => {
                let first_span = self.psess.source_map().start_point(self.token.span);
                let second_span = self.token.span.with_lo(first_span.hi());
                self.token = Token::new(first, first_span);
                // Keep track of this token - if we end token capturing now,
                // we'll want to append this token to the captured stream.
                //
                // If we consume any additional tokens, then this token
                // is not needed (we'll capture the entire 'glued' token),
                // and `bump` will set this field to 0.
                self.break_last_token += 1;
                // Use the spacing of the glued token as the spacing of the
                // unglued second token.
                self.bump_with((Token::new(second, second_span), self.token_spacing));
                true
            }
            _ => {
                self.expected_token_types.insert(exp.token_type);
                false
            }
        }
    }

    /// Eats `+` possibly breaking tokens like `+=` in process.
    fn eat_plus(&mut self) -> bool {
        self.break_and_eat(exp!(Plus))
    }

    /// Eats `&` possibly breaking tokens like `&&` in process.
    /// Signals an error if `&` is not eaten.
    fn expect_and(&mut self) -> PResult<'a, ()> {
        if self.break_and_eat(exp!(And)) { Ok(()) } else { self.unexpected() }
    }

    /// Eats `|` possibly breaking tokens like `||` in process.
    /// Signals an error if `|` was not eaten.
    fn expect_or(&mut self) -> PResult<'a, ()> {
        if self.break_and_eat(exp!(Or)) { Ok(()) } else { self.unexpected() }
    }

    /// Eats `<` possibly breaking tokens like `<<` in process.
    fn eat_lt(&mut self) -> bool {
        let ate = self.break_and_eat(exp!(Lt));
        if ate {
            // See doc comment for `unmatched_angle_bracket_count`.
            self.unmatched_angle_bracket_count += 1;
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
        if self.break_and_eat(exp!(Gt)) {
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

    /// Checks if the next token is contained within `closes`, and returns `true` if so.
    fn expect_any_with_type(
        &mut self,
        closes_expected: &[ExpTokenPair<'_>],
        closes_not_expected: &[&TokenKind],
    ) -> bool {
        closes_expected.iter().any(|&close| self.check(close))
            || closes_not_expected.iter().any(|k| self.check_noexpect(k))
    }

    /// Parses a sequence until the specified delimiters. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_seq_to_before_tokens<T>(
        &mut self,
        closes_expected: &[ExpTokenPair<'_>],
        closes_not_expected: &[&TokenKind],
        sep: SeqSep<'_>,
        mut f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, Trailing, Recovered)> {
        let mut first = true;
        let mut recovered = Recovered::No;
        let mut trailing = Trailing::No;
        let mut v = ThinVec::new();

        while !self.expect_any_with_type(closes_expected, closes_not_expected) {
            if let token::CloseDelim(..) | token::Eof = self.token.kind {
                break;
            }
            if let Some(exp) = sep.sep {
                if first {
                    // no separator for the first element
                    first = false;
                } else {
                    // check for separator
                    match self.expect(exp) {
                        Ok(Recovered::No) => {
                            self.current_closure.take();
                        }
                        Ok(Recovered::Yes(guar)) => {
                            self.current_closure.take();
                            recovered = Recovered::Yes(guar);
                            break;
                        }
                        Err(mut expect_err) => {
                            let sp = self.prev_token.span.shrink_to_hi();
                            let token_str = pprust::token_kind_to_string(exp.tok);

                            match self.current_closure.take() {
                                Some(closure_spans) if self.token == TokenKind::Semi => {
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
                                    if exp.tok.similar_tokens().contains(&self.token.kind) {
                                        self.bump();
                                    }
                                }
                            }

                            // If this was a missing `@` in a binding pattern
                            // bail with a suggestion
                            // https://github.com/rust-lang/rust/issues/72373
                            if self.prev_token.is_ident() && self.token == token::DotDot {
                                let msg = format!(
                                    "if you meant to bind the contents of the rest of the array \
                                     pattern into `{}`, use `@`",
                                    pprust::token_to_string(&self.prev_token)
                                );
                                expect_err
                                    .with_span_suggestion_verbose(
                                        self.prev_token.span.shrink_to_hi().until(self.token.span),
                                        msg,
                                        " @ ",
                                        Applicability::MaybeIncorrect,
                                    )
                                    .emit();
                                break;
                            }

                            // Attempt to keep parsing if it was an omitted separator.
                            self.last_unexpected_token_span = None;
                            match f(self) {
                                Ok(t) => {
                                    // Parsed successfully, therefore most probably the code only
                                    // misses a separator.
                                    expect_err
                                        .with_span_suggestion_short(
                                            sp,
                                            format!("missing `{token_str}`"),
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
                                        // Propagate the help message from sub error `e` to main
                                        // error `expect_err`.
                                        expect_err.children.push(xx.clone());
                                    }
                                    e.cancel();
                                    if self.token == token::Colon {
                                        // We will try to recover in
                                        // `maybe_recover_struct_lit_bad_delims`.
                                        return Err(expect_err);
                                    } else if let [exp] = closes_expected
                                        && exp.token_type == TokenType::CloseParen
                                    {
                                        return Err(expect_err);
                                    } else {
                                        expect_err.emit();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if sep.trailing_sep_allowed
                && self.expect_any_with_type(closes_expected, closes_not_expected)
            {
                trailing = Trailing::Yes;
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
        mut expect_err: Diag<'_>,
    ) -> PResult<'a, ()> {
        let initial_semicolon = self.token.span;

        while self.eat(exp!(Semi)) {
            let _ = self
                .parse_stmt_without_recovery(false, ForceCollect::No, false)
                .unwrap_or_else(|e| {
                    e.cancel();
                    None
                });
        }

        expect_err
            .primary_message("closure bodies that contain statements must be surrounded by braces");

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

        expect_err.span(vec![preceding_pipe_span, following_token_span]);

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

    /// Parses a sequence, not including the delimiters. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_seq_to_before_end<T>(
        &mut self,
        close: ExpTokenPair<'_>,
        sep: SeqSep<'_>,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, Trailing, Recovered)> {
        self.parse_seq_to_before_tokens(&[close], &[], sep, f)
    }

    /// Parses a sequence, including only the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_seq_to_end<T>(
        &mut self,
        close: ExpTokenPair<'_>,
        sep: SeqSep<'_>,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, Trailing)> {
        let (val, trailing, recovered) = self.parse_seq_to_before_end(close, sep, f)?;
        if matches!(recovered, Recovered::No) && !self.eat(close) {
            self.dcx().span_delayed_bug(
                self.token.span,
                "recovered but `parse_seq_to_before_end` did not give us the close token",
            );
        }
        Ok((val, trailing))
    }

    /// Parses a sequence, including both delimiters. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_unspanned_seq<T>(
        &mut self,
        open: ExpTokenPair<'_>,
        close: ExpTokenPair<'_>,
        sep: SeqSep<'_>,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, Trailing)> {
        self.expect(open)?;
        self.parse_seq_to_end(close, sep, f)
    }

    /// Parses a comma-separated sequence, including both delimiters.
    /// The function `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_delim_comma_seq<T>(
        &mut self,
        open: ExpTokenPair<'_>,
        close: ExpTokenPair<'_>,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, Trailing)> {
        self.parse_unspanned_seq(open, close, SeqSep::trailing_allowed(exp!(Comma)), f)
    }

    /// Parses a comma-separated sequence delimited by parentheses (e.g. `(x, y)`).
    /// The function `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_paren_comma_seq<T>(
        &mut self,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (ThinVec<T>, Trailing)> {
        self.parse_delim_comma_seq(exp!(OpenParen), exp!(CloseParen), f)
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
        self.expected_token_types.clear();
    }

    /// Advance the parser by one token.
    pub fn bump(&mut self) {
        // Note: destructuring here would give nicer code, but it was found in #96210 to be slower
        // than `.0`/`.1` access.
        let mut next = self.token_cursor.inlined_next();
        self.num_bump_calls += 1;
        // We got a token from the underlying cursor and no longer need to
        // worry about an unglued token. See `break_and_eat` for more details.
        self.break_last_token = 0;
        if next.0.span.is_dummy() {
            // Tweak the location for better diagnostics, but keep syntactic context intact.
            let fallback_span = self.token.span;
            next.0.span = fallback_span.with_ctxt(next.0.span.ctxt());
        }
        debug_assert!(!matches!(
            next.0.kind,
            token::OpenDelim(delim) | token::CloseDelim(delim) if delim.skip()
        ));
        self.inlined_bump_with(next)
    }

    /// Look-ahead `dist` tokens of `self.token` and get access to that token there.
    /// When `dist == 0` then the current token is looked at. `Eof` will be
    /// returned if the look-ahead is any distance past the end of the tokens.
    pub fn look_ahead<R>(&self, dist: usize, looker: impl FnOnce(&Token) -> R) -> R {
        if dist == 0 {
            return looker(&self.token);
        }

        // Typically around 98% of the `dist > 0` cases have `dist == 1`, so we
        // have a fast special case for that.
        if dist == 1 {
            // The index is zero because the tree cursor's index always points
            // to the next token to be gotten.
            match self.token_cursor.curr.curr() {
                Some(tree) => {
                    // Indexing stayed within the current token tree.
                    match tree {
                        TokenTree::Token(token, _) => return looker(token),
                        &TokenTree::Delimited(dspan, _, delim, _) => {
                            if !delim.skip() {
                                return looker(&Token::new(token::OpenDelim(delim), dspan.open));
                            }
                        }
                    }
                }
                None => {
                    // The tree cursor lookahead went (one) past the end of the
                    // current token tree. Try to return a close delimiter.
                    if let Some(last) = self.token_cursor.stack.last()
                        && let Some(&TokenTree::Delimited(span, _, delim, _)) = last.curr()
                        && !delim.skip()
                    {
                        // We are not in the outermost token stream, so we have
                        // delimiters. Also, those delimiters are not skipped.
                        return looker(&Token::new(token::CloseDelim(delim), span.close));
                    }
                }
            }
        }

        // Just clone the token cursor and use `next`, skipping delimiters as
        // necessary. Slow but simple.
        let mut cursor = self.token_cursor.clone();
        let mut i = 0;
        let mut token = Token::dummy();
        while i < dist {
            token = cursor.next().0;
            if matches!(
                token.kind,
                token::OpenDelim(delim) | token::CloseDelim(delim) if delim.skip()
            ) {
                continue;
            }
            i += 1;
        }
        looker(&token)
    }

    /// Like `lookahead`, but skips over token trees rather than tokens. Useful
    /// when looking past possible metavariable pasting sites.
    pub fn tree_look_ahead<R>(
        &self,
        dist: usize,
        looker: impl FnOnce(&TokenTree) -> R,
    ) -> Option<R> {
        assert_ne!(dist, 0);
        self.token_cursor.curr.look_ahead(dist - 1).map(looker)
    }

    /// Returns whether any of the given keywords are `dist` tokens ahead of the current one.
    pub(crate) fn is_keyword_ahead(&self, dist: usize, kws: &[Symbol]) -> bool {
        self.look_ahead(dist, |t| kws.iter().any(|&kw| t.is_keyword(kw)))
    }

    /// Parses asyncness: `async` or nothing.
    fn parse_coroutine_kind(&mut self, case: Case) -> Option<CoroutineKind> {
        let span = self.token_uninterpolated_span();
        if self.eat_keyword_case(exp!(Async), case) {
            // FIXME(gen_blocks): Do we want to unconditionally parse `gen` and then
            // error if edition <= 2024, like we do with async and edition <= 2018?
            if self.token_uninterpolated_span().at_least_rust_2024()
                && self.eat_keyword_case(exp!(Gen), case)
            {
                let gen_span = self.prev_token_uninterpolated_span();
                Some(CoroutineKind::AsyncGen {
                    span: span.to(gen_span),
                    closure_id: DUMMY_NODE_ID,
                    return_impl_trait_id: DUMMY_NODE_ID,
                })
            } else {
                Some(CoroutineKind::Async {
                    span,
                    closure_id: DUMMY_NODE_ID,
                    return_impl_trait_id: DUMMY_NODE_ID,
                })
            }
        } else if self.token_uninterpolated_span().at_least_rust_2024()
            && self.eat_keyword_case(exp!(Gen), case)
        {
            Some(CoroutineKind::Gen {
                span,
                closure_id: DUMMY_NODE_ID,
                return_impl_trait_id: DUMMY_NODE_ID,
            })
        } else {
            None
        }
    }

    /// Parses fn unsafety: `unsafe`, `safe` or nothing.
    fn parse_safety(&mut self, case: Case) -> Safety {
        if self.eat_keyword_case(exp!(Unsafe), case) {
            Safety::Unsafe(self.prev_token_uninterpolated_span())
        } else if self.eat_keyword_case(exp!(Safe), case) {
            Safety::Safe(self.prev_token_uninterpolated_span())
        } else {
            Safety::Default
        }
    }

    /// Parses constness: `const` or nothing.
    fn parse_constness(&mut self, case: Case) -> Const {
        self.parse_constness_(case, false)
    }

    /// Parses constness for closures (case sensitive, feature-gated)
    fn parse_closure_constness(&mut self) -> Const {
        let constness = self.parse_constness_(Case::Sensitive, true);
        if let Const::Yes(span) = constness {
            self.psess.gated_spans.gate(sym::const_closures, span);
        }
        constness
    }

    fn parse_constness_(&mut self, case: Case, is_closure: bool) -> Const {
        // Avoid const blocks and const closures to be parsed as const items
        if (self.check_const_closure() == is_closure)
            && !self
                .look_ahead(1, |t| *t == token::OpenDelim(Delimiter::Brace) || t.is_whole_block())
            && self.eat_keyword_case(exp!(Const), case)
        {
            Const::Yes(self.prev_token_uninterpolated_span())
        } else {
            Const::No
        }
    }

    /// Parses inline const expressions.
    fn parse_const_block(&mut self, span: Span, pat: bool) -> PResult<'a, P<Expr>> {
        self.expect_keyword(exp!(Const))?;
        let (attrs, blk) = self.parse_inner_attrs_and_block(None)?;
        let anon_const = AnonConst {
            id: DUMMY_NODE_ID,
            value: self.mk_expr(blk.span, ExprKind::Block(blk, None)),
        };
        let blk_span = anon_const.value.span;
        let kind = if pat {
            let guar = self
                .dcx()
                .struct_span_err(blk_span, "`inline_const_pat` has been removed")
                .with_help("use a named `const`-item or an `if`-guard instead")
                .emit();
            ExprKind::Err(guar)
        } else {
            ExprKind::ConstBlock(anon_const)
        };
        Ok(self.mk_expr_with_attrs(span.to(blk_span), kind, attrs))
    }

    /// Parses mutability (`mut` or nothing).
    fn parse_mutability(&mut self) -> Mutability {
        if self.eat_keyword(exp!(Mut)) { Mutability::Mut } else { Mutability::Not }
    }

    /// Parses reference binding mode (`ref`, `ref mut`, or nothing).
    fn parse_byref(&mut self) -> ByRef {
        if self.eat_keyword(exp!(Ref)) { ByRef::Yes(self.parse_mutability()) } else { ByRef::No }
    }

    /// Possibly parses mutability (`const` or `mut`).
    fn parse_const_or_mut(&mut self) -> Option<Mutability> {
        if self.eat_keyword(exp!(Mut)) {
            Some(Mutability::Mut)
        } else if self.eat_keyword(exp!(Const)) {
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
        if let Some(args) = self.parse_delim_args_inner() {
            Ok(P(args))
        } else {
            self.unexpected_any()
        }
    }

    fn parse_attr_args(&mut self) -> PResult<'a, AttrArgs> {
        Ok(if let Some(args) = self.parse_delim_args_inner() {
            AttrArgs::Delimited(args)
        } else if self.eat(exp!(Eq)) {
            let eq_span = self.prev_token.span;
            AttrArgs::Eq { eq_span, expr: self.parse_expr_force_collect()? }
        } else {
            AttrArgs::Empty
        })
    }

    fn parse_delim_args_inner(&mut self) -> Option<DelimArgs> {
        let delimited = self.check(exp!(OpenParen))
            || self.check(exp!(OpenBracket))
            || self.check(exp!(OpenBrace));

        delimited.then(|| {
            let TokenTree::Delimited(dspan, _, delim, tokens) = self.parse_token_tree() else {
                unreachable!()
            };
            DelimArgs { dspan, delim, tokens }
        })
    }

    /// Parses a single token tree from the input.
    pub fn parse_token_tree(&mut self) -> TokenTree {
        match self.token.kind {
            token::OpenDelim(..) => {
                // Clone the `TokenTree::Delimited` that we are currently
                // within. That's what we are going to return.
                let tree = self.token_cursor.stack.last().unwrap().curr().unwrap().clone();
                debug_assert_matches!(tree, TokenTree::Delimited(..));

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
                        debug_assert_matches!(self.token.kind, token::CloseDelim(_));
                        break;
                    }
                }

                // Consume close delimiter
                self.bump();
                tree
            }
            token::CloseDelim(_) | token::Eof => unreachable!(),
            _ => {
                let prev_spacing = self.token_spacing;
                self.bump();
                TokenTree::Token(self.prev_token.clone(), prev_spacing)
            }
        }
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
        if let Some(vis) = self
            .eat_metavar_seq(MetaVarKind::Vis, |this| this.parse_visibility(FollowedByType::Yes))
        {
            return Ok(vis);
        }

        if !self.eat_keyword(exp!(Pub)) {
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

        if self.check(exp!(OpenParen)) {
            // We don't `self.bump()` the `(` yet because this might be a struct definition where
            // `()` or a tuple might be allowed. For example, `struct Struct(pub (), pub (usize));`.
            // Because of this, we only `bump` the `(` if we're assured it is appropriate to do so
            // by the following tokens.
            if self.is_keyword_ahead(1, &[kw::In]) {
                // Parse `pub(in path)`.
                self.bump(); // `(`
                self.bump(); // `in`
                let path = self.parse_path(PathStyle::Mod)?; // `path`
                self.expect(exp!(CloseParen))?; // `)`
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
                self.expect(exp!(CloseParen))?; // `)`
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
        self.expect(exp!(CloseParen))?; // `)`

        let path_str = pprust::path_to_string(&path);
        self.dcx()
            .emit_err(IncorrectVisibilityRestriction { span: path.span, inner_str: path_str });

        Ok(())
    }

    /// Parses `extern string_literal?`.
    fn parse_extern(&mut self, case: Case) -> Extern {
        if self.eat_keyword_case(exp!(Extern), case) {
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
                ast::LitKind::Err(_) => None,
                _ => {
                    self.dcx().emit_err(NonStringAbiLiteral { span: lit.span });
                    None
                }
            },
            Err(None) => None,
        }
    }

    fn collect_tokens_no_attrs<R: HasAttrs + HasTokens>(
        &mut self,
        f: impl FnOnce(&mut Self) -> PResult<'a, R>,
    ) -> PResult<'a, R> {
        // The only reason to call `collect_tokens_no_attrs` is if you want tokens, so use
        // `ForceCollect::Yes`
        self.collect_tokens(None, AttrWrapper::empty(), ForceCollect::Yes, |this, _attrs| {
            Ok((f(this)?, Trailing::No, UsePreAttrPos::No))
        })
    }

    /// Checks for `::` or, potentially, `:::` and then look ahead after it.
    fn check_path_sep_and_look_ahead(&mut self, looker: impl Fn(&Token) -> bool) -> bool {
        if self.check(exp!(PathSep)) {
            if self.may_recover() && self.look_ahead(1, |t| t.kind == token::Colon) {
                debug_assert!(!self.look_ahead(1, &looker), "Looker must not match on colon");
                self.look_ahead(2, looker)
            } else {
                self.look_ahead(1, looker)
            }
        } else {
            false
        }
    }

    /// `::{` or `::*`
    fn is_import_coupler(&mut self) -> bool {
        self.check_path_sep_and_look_ahead(|t| {
            matches!(t.kind, token::OpenDelim(Delimiter::Brace) | token::Star)
        })
    }

    // Debug view of the parser's token stream, up to `{lookahead}` tokens.
    // Only used when debugging.
    #[allow(unused)]
    pub(crate) fn debug_lookahead(&self, lookahead: usize) -> impl fmt::Debug {
        fmt::from_fn(move |f| {
            let mut dbg_fmt = f.debug_struct("Parser"); // or at least, one view of

            // we don't need N spans, but we want at least one, so print all of prev_token
            dbg_fmt.field("prev_token", &self.prev_token);
            let mut tokens = vec![];
            for i in 0..lookahead {
                let tok = self.look_ahead(i, |tok| tok.kind.clone());
                let is_eof = tok == TokenKind::Eof;
                tokens.push(tok);
                if is_eof {
                    // Don't look ahead past EOF.
                    break;
                }
            }
            dbg_fmt.field_with("tokens", |field| field.debug_list().entries(tokens).finish());
            dbg_fmt.field("approx_token_stream_pos", &self.num_bump_calls);

            // some fields are interesting for certain values, as they relate to macro parsing
            if let Some(subparser) = self.subparser_name {
                dbg_fmt.field("subparser_name", &subparser);
            }
            if let Recovery::Forbidden = self.recovery {
                dbg_fmt.field("recovery", &self.recovery);
            }

            // imply there's "more to know" than this view
            dbg_fmt.finish_non_exhaustive()
        })
    }

    pub fn clear_expected_token_types(&mut self) {
        self.expected_token_types.clear();
    }

    pub fn approx_token_stream_pos(&self) -> u32 {
        self.num_bump_calls
    }

    /// For interpolated `self.token`, returns a span of the fragment to which
    /// the interpolated token refers. For all other tokens this is just a
    /// regular span. It is particularly important to use this for identifiers
    /// and lifetimes for which spans affect name resolution and edition
    /// checks. Note that keywords are also identifiers, so they should use
    /// this if they keep spans or perform edition checks.
    pub fn token_uninterpolated_span(&self) -> Span {
        match &self.token.kind {
            token::NtIdent(ident, _) | token::NtLifetime(ident, _) => ident.span,
            token::Interpolated(nt) => nt.use_span(),
            token::OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(_))) => {
                self.look_ahead(1, |t| t.span)
            }
            _ => self.token.span,
        }
    }

    /// Like `token_uninterpolated_span`, but works on `self.prev_token`.
    pub fn prev_token_uninterpolated_span(&self) -> Span {
        match &self.prev_token.kind {
            token::NtIdent(ident, _) | token::NtLifetime(ident, _) => ident.span,
            token::Interpolated(nt) => nt.use_span(),
            token::OpenDelim(Delimiter::Invisible(InvisibleOrigin::MetaVar(_))) => {
                self.look_ahead(0, |t| t.span)
            }
            _ => self.prev_token.span,
        }
    }
}

pub(crate) fn make_unclosed_delims_error(
    unmatched: UnmatchedDelim,
    psess: &ParseSess,
) -> Option<Diag<'_>> {
    // `None` here means an `Eof` was found. We already emit those errors elsewhere, we add them to
    // `unmatched_delims` only for error recovery in the `Parser`.
    let found_delim = unmatched.found_delim?;
    let mut spans = vec![unmatched.found_span];
    if let Some(sp) = unmatched.unclosed_span {
        spans.push(sp);
    };
    let err = psess.dcx().create_err(MismatchedClosingDelimiter {
        spans,
        delimiter: pprust::token_kind_to_string(&token::CloseDelim(found_delim)).to_string(),
        unmatched: unmatched.found_span,
        opening_candidate: unmatched.candidate_span,
        unclosed: unmatched.unclosed_span,
    });
    Some(err)
}

/// A helper struct used when building an `AttrTokenStream` from
/// a `LazyAttrTokenStream`. Both delimiter and non-delimited tokens
/// are stored as `FlatToken::Token`. A vector of `FlatToken`s
/// is then 'parsed' to build up an `AttrTokenStream` with nested
/// `AttrTokenTree::Delimited` tokens.
#[derive(Debug, Clone)]
enum FlatToken {
    /// A token - this holds both delimiter (e.g. '{' and '}')
    /// and non-delimiter tokens
    Token((Token, Spacing)),
    /// Holds the `AttrsTarget` for an AST node. The `AttrsTarget` is inserted
    /// directly into the constructed `AttrTokenStream` as an
    /// `AttrTokenTree::AttrsTarget`.
    AttrsTarget(AttrsTarget),
    /// A special 'empty' token that is ignored during the conversion
    /// to an `AttrTokenStream`. This is used to simplify the
    /// handling of replace ranges.
    Empty,
}

// Metavar captures of various kinds.
#[derive(Clone, Debug)]
pub enum ParseNtResult {
    Tt(TokenTree),
    Ident(Ident, IdentIsRaw),
    Lifetime(Ident, IdentIsRaw),
    Item(P<ast::Item>),
    Stmt(P<ast::Stmt>),
    Pat(P<ast::Pat>, NtPatKind),
    Expr(P<ast::Expr>, NtExprKind),
    Literal(P<ast::Expr>),
    Ty(P<ast::Ty>),
    Meta(P<ast::AttrItem>),
    Path(P<ast::Path>),
    Vis(P<ast::Visibility>),

    /// This variant will eventually be removed, along with `Token::Interpolate`.
    Nt(Arc<Nonterminal>),
}
