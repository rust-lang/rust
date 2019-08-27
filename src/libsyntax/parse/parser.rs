mod expr;
mod pat;
mod item;
pub use item::AliasKind;
mod module;
pub use module::{ModulePath, ModulePathSuccess};
mod ty;
mod path;
pub use path::PathStyle;
mod stmt;
mod generics;

use crate::ast::{self, AttrStyle, Attribute, Arg, BindingMode, StrStyle, SelfKind};
use crate::ast::{FnDecl, Ident, IsAsync, MacDelimiter, Mutability, TyKind};
use crate::ast::{Visibility, VisibilityKind, Unsafety, CrateSugar};
use crate::source_map::{self, respan};
use crate::parse::{SeqSep, literal, token};
use crate::parse::lexer::UnmatchedBrace;
use crate::parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use crate::parse::token::{Token, TokenKind, DelimToken};
use crate::parse::{ParseSess, Directory, DirectoryOwnership};
use crate::print::pprust;
use crate::ptr::P;
use crate::parse::PResult;
use crate::ThinVec;
use crate::tokenstream::{self, DelimSpan, TokenTree, TokenStream, TreeAndJoint};
use crate::symbol::{kw, sym, Symbol};
use crate::parse::diagnostics::{Error, dummy_arg};

use errors::{Applicability, DiagnosticId, FatalError};
use rustc_target::spec::abi::{self, Abi};
use syntax_pos::{Span, BytePos, DUMMY_SP, FileName};
use log::debug;

use std::borrow::Cow;
use std::{cmp, mem, slice};
use std::path::PathBuf;

bitflags::bitflags! {
    struct Restrictions: u8 {
        const STMT_EXPR         = 1 << 0;
        const NO_STRUCT_LITERAL = 1 << 1;
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
crate enum SemiColonMode {
    Break,
    Ignore,
    Comma,
}

#[derive(Clone, Copy, PartialEq, Debug)]
crate enum BlockMode {
    Break,
    Ignore,
}

/// As maybe_whole_expr, but for things other than expressions
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
                    return $self.maybe_recover_from_bad_qpath_stage_2($self.prev_span, ty);
                }
            }
        }
    }
}

fn maybe_append(mut lhs: Vec<Attribute>, mut rhs: Option<Vec<Attribute>>) -> Vec<Attribute> {
    if let Some(ref mut rhs) = rhs {
        lhs.append(rhs);
    }
    lhs
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PrevTokenKind {
    DocComment,
    Comma,
    Plus,
    Interpolated,
    Eof,
    Ident,
    BitOr,
    Other,
}

// NOTE: `Ident`s are handled by `common.rs`.

#[derive(Clone)]
pub struct Parser<'a> {
    pub sess: &'a ParseSess,
    /// The current normalized token.
    /// "Normalized" means that some interpolated tokens
    /// (`$i: ident` and `$l: lifetime` meta-variables) are replaced
    /// with non-interpolated identifier and lifetime tokens they refer to.
    /// Perhaps the normalized / non-normalized setup can be simplified somehow.
    pub token: Token,
    /// Span of the current non-normalized token.
    meta_var_span: Option<Span>,
    /// Span of the previous non-normalized token.
    pub prev_span: Span,
    /// Kind of the previous normalized token (in simplified form).
    prev_token_kind: PrevTokenKind,
    restrictions: Restrictions,
    /// Used to determine the path to externally loaded source files.
    crate directory: Directory<'a>,
    /// `true` to parse sub-modules in other files.
    pub recurse_into_file_modules: bool,
    /// Name of the root module this parser originated from. If `None`, then the
    /// name is not known. This does not change while the parser is descending
    /// into modules, and sub-parsers have new values for this name.
    pub root_module_name: Option<String>,
    crate expected_tokens: Vec<TokenType>,
    crate token_cursor: TokenCursor,
    desugar_doc_comments: bool,
    /// `true` we should configure out of line modules as we parse.
    pub cfg_mods: bool,
    /// This field is used to keep track of how many left angle brackets we have seen. This is
    /// required in order to detect extra leading left angle brackets (`<` characters) and error
    /// appropriately.
    ///
    /// See the comments in the `parse_path_segment` function for more details.
    crate unmatched_angle_bracket_count: u32,
    crate max_angle_bracket_count: u32,
    /// List of all unclosed delimiters found by the lexer. If an entry is used for error recovery
    /// it gets removed from here. Every entry left at the end gets emitted as an independent
    /// error.
    crate unclosed_delims: Vec<UnmatchedBrace>,
    crate last_unexpected_token_span: Option<Span>,
    crate last_type_ascription: Option<(Span, bool /* likely path typo */)>,
    /// If present, this `Parser` is not parsing Rust code but rather a macro call.
    crate subparser_name: Option<&'static str>,
}

impl<'a> Drop for Parser<'a> {
    fn drop(&mut self) {
        let diag = self.diagnostic();
        emit_unclosed_delims(&mut self.unclosed_delims, diag);
    }
}

#[derive(Clone)]
crate struct TokenCursor {
    crate frame: TokenCursorFrame,
    crate stack: Vec<TokenCursorFrame>,
}

#[derive(Clone)]
crate struct TokenCursorFrame {
    crate delim: token::DelimToken,
    crate span: DelimSpan,
    crate open_delim: bool,
    crate tree_cursor: tokenstream::Cursor,
    crate close_delim: bool,
    crate last_token: LastToken,
}

/// This is used in `TokenCursorFrame` above to track tokens that are consumed
/// by the parser, and then that's transitively used to record the tokens that
/// each parse AST item is created with.
///
/// Right now this has two states, either collecting tokens or not collecting
/// tokens. If we're collecting tokens we just save everything off into a local
/// `Vec`. This should eventually though likely save tokens from the original
/// token stream and just use slicing of token streams to avoid creation of a
/// whole new vector.
///
/// The second state is where we're passively not recording tokens, but the last
/// token is still tracked for when we want to start recording tokens. This
/// "last token" means that when we start recording tokens we'll want to ensure
/// that this, the first token, is included in the output.
///
/// You can find some more example usage of this in the `collect_tokens` method
/// on the parser.
#[derive(Clone)]
crate enum LastToken {
    Collecting(Vec<TreeAndJoint>),
    Was(Option<TreeAndJoint>),
}

impl TokenCursorFrame {
    fn new(span: DelimSpan, delim: DelimToken, tts: &TokenStream) -> Self {
        TokenCursorFrame {
            delim,
            span,
            open_delim: delim == token::NoDelim,
            tree_cursor: tts.clone().into_trees(),
            close_delim: delim == token::NoDelim,
            last_token: LastToken::Was(None),
        }
    }
}

impl TokenCursor {
    fn next(&mut self) -> Token {
        loop {
            let tree = if !self.frame.open_delim {
                self.frame.open_delim = true;
                TokenTree::open_tt(self.frame.span.open, self.frame.delim)
            } else if let Some(tree) = self.frame.tree_cursor.next() {
                tree
            } else if !self.frame.close_delim {
                self.frame.close_delim = true;
                TokenTree::close_tt(self.frame.span.close, self.frame.delim)
            } else if let Some(frame) = self.stack.pop() {
                self.frame = frame;
                continue
            } else {
                return Token::new(token::Eof, DUMMY_SP);
            };

            match self.frame.last_token {
                LastToken::Collecting(ref mut v) => v.push(tree.clone().into()),
                LastToken::Was(ref mut t) => *t = Some(tree.clone().into()),
            }

            match tree {
                TokenTree::Token(token) => return token,
                TokenTree::Delimited(sp, delim, tts) => {
                    let frame = TokenCursorFrame::new(sp, delim, &tts);
                    self.stack.push(mem::replace(&mut self.frame, frame));
                }
            }
        }
    }

    fn next_desugared(&mut self) -> Token {
        let (name, sp) = match self.next() {
            Token { kind: token::DocComment(name), span } => (name, span),
            tok => return tok,
        };

        let stripped = strip_doc_comment_decoration(&name.as_str());

        // Searches for the occurrences of `"#*` and returns the minimum number of `#`s
        // required to wrap the text.
        let mut num_of_hashes = 0;
        let mut count = 0;
        for ch in stripped.chars() {
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
                TokenTree::token(TokenKind::lit(
                    token::StrRaw(num_of_hashes), Symbol::intern(&stripped), None
                ), sp),
            ]
            .iter().cloned().collect::<TokenStream>().into(),
        );

        self.stack.push(mem::replace(&mut self.frame, TokenCursorFrame::new(
            delim_span,
            token::NoDelim,
            &if doc_comment_style(&name.as_str()) == AttrStyle::Inner {
                [TokenTree::token(token::Pound, sp), TokenTree::token(token::Not, sp), body]
                    .iter().cloned().collect::<TokenStream>().into()
            } else {
                [TokenTree::token(token::Pound, sp), body]
                    .iter().cloned().collect::<TokenStream>().into()
            },
        )));

        self.next()
    }
}

#[derive(Clone, PartialEq)]
crate enum TokenType {
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
    crate fn to_string(&self) -> String {
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
crate enum TokenExpectType {
    Expect,
    NoExpect,
}

impl<'a> Parser<'a> {
    pub fn new(
        sess: &'a ParseSess,
        tokens: TokenStream,
        directory: Option<Directory<'a>>,
        recurse_into_file_modules: bool,
        desugar_doc_comments: bool,
        subparser_name: Option<&'static str>,
    ) -> Self {
        let mut parser = Parser {
            sess,
            token: Token::dummy(),
            prev_span: DUMMY_SP,
            meta_var_span: None,
            prev_token_kind: PrevTokenKind::Other,
            restrictions: Restrictions::empty(),
            recurse_into_file_modules,
            directory: Directory {
                path: Cow::from(PathBuf::new()),
                ownership: DirectoryOwnership::Owned { relative: None }
            },
            root_module_name: None,
            expected_tokens: Vec::new(),
            token_cursor: TokenCursor {
                frame: TokenCursorFrame::new(
                    DelimSpan::dummy(),
                    token::NoDelim,
                    &tokens.into(),
                ),
                stack: Vec::new(),
            },
            desugar_doc_comments,
            cfg_mods: true,
            unmatched_angle_bracket_count: 0,
            max_angle_bracket_count: 0,
            unclosed_delims: Vec::new(),
            last_unexpected_token_span: None,
            last_type_ascription: None,
            subparser_name,
        };

        parser.token = parser.next_tok();

        if let Some(directory) = directory {
            parser.directory = directory;
        } else if !parser.token.span.is_dummy() {
            if let Some(FileName::Real(path)) =
                    &sess.source_map().lookup_char_pos(parser.token.span.lo()).file.unmapped_path {
                if let Some(directory_path) = path.parent() {
                    parser.directory.path = Cow::from(directory_path.to_path_buf());
                }
            }
        }

        parser.process_potential_macro_variable();
        parser
    }

    fn next_tok(&mut self) -> Token {
        let mut next = if self.desugar_doc_comments {
            self.token_cursor.next_desugared()
        } else {
            self.token_cursor.next()
        };
        if next.span.is_dummy() {
            // Tweak the location for better diagnostics, but keep syntactic context intact.
            next.span = self.prev_span.with_ctxt(next.span.ctxt());
        }
        next
    }

    /// Converts the current token to a string using `self`'s reader.
    pub fn this_token_to_string(&self) -> String {
        pprust::token_to_string(&self.token)
    }

    crate fn token_descr(&self) -> Option<&'static str> {
        Some(match &self.token.kind {
            _ if self.token.is_special_ident() => "reserved identifier",
            _ if self.token.is_used_keyword() => "keyword",
            _ if self.token.is_unused_keyword() => "reserved keyword",
            token::DocComment(..) => "doc comment",
            _ => return None,
        })
    }

    crate fn this_token_descr(&self) -> String {
        if let Some(prefix) = self.token_descr() {
            format!("{} `{}`", prefix, self.this_token_to_string())
        } else {
            format!("`{}`", self.this_token_to_string())
        }
    }

    crate fn unexpected<T>(&mut self) -> PResult<'a, T> {
        match self.expect_one_of(&[], &[]) {
            Err(e) => Err(e),
            Ok(_) => unreachable!(),
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

    pub fn parse_ident(&mut self) -> PResult<'a, ast::Ident> {
        self.parse_ident_common(true)
    }

    fn parse_ident_common(&mut self, recover: bool) -> PResult<'a, ast::Ident> {
        match self.token.kind {
            token::Ident(name, _) => {
                if self.token.is_reserved_ident() {
                    let mut err = self.expected_ident_found();
                    if recover {
                        err.emit();
                    } else {
                        return Err(err);
                    }
                }
                let span = self.token.span;
                self.bump();
                Ok(Ident::new(name, span))
            }
            _ => {
                Err(if self.prev_token_kind == PrevTokenKind::DocComment {
                    self.span_fatal_err(self.prev_span, Error::UselessDocComment)
                } else {
                    self.expected_ident_found()
                })
            }
        }
    }

    /// Checks if the next token is `tok`, and returns `true` if so.
    ///
    /// This method will automatically add `tok` to `expected_tokens` if `tok` is not
    /// encountered.
    crate fn check(&mut self, tok: &TokenKind) -> bool {
        let is_present = self.token == *tok;
        if !is_present { self.expected_tokens.push(TokenType::Token(tok.clone())); }
        is_present
    }

    /// Consumes a token 'tok' if it exists. Returns whether the given token was present.
    pub fn eat(&mut self, tok: &TokenKind) -> bool {
        let is_present = self.check(tok);
        if is_present { self.bump() }
        is_present
    }

    fn check_keyword(&mut self, kw: Symbol) -> bool {
        self.expected_tokens.push(TokenType::Keyword(kw));
        self.token.is_keyword(kw)
    }

    /// If the next token is the given keyword, eats it and returns
    /// `true`. Otherwise, returns `false`.
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
        if !self.eat_keyword(kw) {
            self.unexpected()
        } else {
            Ok(())
        }
    }

    crate fn check_ident(&mut self) -> bool {
        if self.token.is_ident() {
            true
        } else {
            self.expected_tokens.push(TokenType::Ident);
            false
        }
    }

    fn check_path(&mut self) -> bool {
        if self.token.is_path_start() {
            true
        } else {
            self.expected_tokens.push(TokenType::Path);
            false
        }
    }

    fn check_type(&mut self) -> bool {
        if self.token.can_begin_type() {
            true
        } else {
            self.expected_tokens.push(TokenType::Type);
            false
        }
    }

    fn check_const_arg(&mut self) -> bool {
        if self.token.can_begin_const_arg() {
            true
        } else {
            self.expected_tokens.push(TokenType::Const);
            false
        }
    }

    /// Expects and consumes a `+`. if `+=` is seen, replaces it with a `=`
    /// and continues. If a `+` is not seen, returns `false`.
    ///
    /// This is used when token-splitting `+=` into `+`.
    /// See issue #47856 for an example of when this may occur.
    fn eat_plus(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Token(token::BinOp(token::Plus)));
        match self.token.kind {
            token::BinOp(token::Plus) => {
                self.bump();
                true
            }
            token::BinOpEq(token::Plus) => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                self.bump_with(token::Eq, span);
                true
            }
            _ => false,
        }
    }

    /// Checks to see if the next token is either `+` or `+=`.
    /// Otherwise returns `false`.
    fn check_plus(&mut self) -> bool {
        if self.token.is_like_plus() {
            true
        }
        else {
            self.expected_tokens.push(TokenType::Token(token::BinOp(token::Plus)));
            false
        }
    }

    /// Expects and consumes an `&`. If `&&` is seen, replaces it with a single
    /// `&` and continues. If an `&` is not seen, signals an error.
    fn expect_and(&mut self) -> PResult<'a, ()> {
        self.expected_tokens.push(TokenType::Token(token::BinOp(token::And)));
        match self.token.kind {
            token::BinOp(token::And) => {
                self.bump();
                Ok(())
            }
            token::AndAnd => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                Ok(self.bump_with(token::BinOp(token::And), span))
            }
            _ => self.unexpected()
        }
    }

    /// Expects and consumes an `|`. If `||` is seen, replaces it with a single
    /// `|` and continues. If an `|` is not seen, signals an error.
    fn expect_or(&mut self) -> PResult<'a, ()> {
        self.expected_tokens.push(TokenType::Token(token::BinOp(token::Or)));
        match self.token.kind {
            token::BinOp(token::Or) => {
                self.bump();
                Ok(())
            }
            token::OrOr => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                Ok(self.bump_with(token::BinOp(token::Or), span))
            }
            _ => self.unexpected()
        }
    }

    fn expect_no_suffix(&self, sp: Span, kind: &str, suffix: Option<ast::Name>) {
        literal::expect_no_suffix(&self.sess.span_diagnostic, sp, kind, suffix)
    }

    /// Attempts to consume a `<`. If `<<` is seen, replaces it with a single
    /// `<` and continue. If `<-` is seen, replaces it with a single `<`
    /// and continue. If a `<` is not seen, returns false.
    ///
    /// This is meant to be used when parsing generics on a path to get the
    /// starting token.
    fn eat_lt(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Token(token::Lt));
        let ate = match self.token.kind {
            token::Lt => {
                self.bump();
                true
            }
            token::BinOp(token::Shl) => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                self.bump_with(token::Lt, span);
                true
            }
            token::LArrow => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                self.bump_with(token::BinOp(token::Minus), span);
                true
            }
            _ => false,
        };

        if ate {
            // See doc comment for `unmatched_angle_bracket_count`.
            self.unmatched_angle_bracket_count += 1;
            self.max_angle_bracket_count += 1;
            debug!("eat_lt: (increment) count={:?}", self.unmatched_angle_bracket_count);
        }

        ate
    }

    fn expect_lt(&mut self) -> PResult<'a, ()> {
        if !self.eat_lt() {
            self.unexpected()
        } else {
            Ok(())
        }
    }

    /// Expects and consumes a single `>` token. if a `>>` is seen, replaces it
    /// with a single `>` and continues. If a `>` is not seen, signals an error.
    fn expect_gt(&mut self) -> PResult<'a, ()> {
        self.expected_tokens.push(TokenType::Token(token::Gt));
        let ate = match self.token.kind {
            token::Gt => {
                self.bump();
                Some(())
            }
            token::BinOp(token::Shr) => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                Some(self.bump_with(token::Gt, span))
            }
            token::BinOpEq(token::Shr) => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                Some(self.bump_with(token::Ge, span))
            }
            token::Ge => {
                let span = self.token.span.with_lo(self.token.span.lo() + BytePos(1));
                Some(self.bump_with(token::Eq, span))
            }
            _ => None,
        };

        match ate {
            Some(_) => {
                // See doc comment for `unmatched_angle_bracket_count`.
                if self.unmatched_angle_bracket_count > 0 {
                    self.unmatched_angle_bracket_count -= 1;
                    debug!("expect_gt: (decrement) count={:?}", self.unmatched_angle_bracket_count);
                }

                Ok(())
            },
            None => self.unexpected(),
        }
    }

    /// Parses a sequence, including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_end<T>(
        &mut self,
        ket: &TokenKind,
        sep: SeqSep,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a,  T>,
    ) -> PResult<'a, Vec<T>> {
        let (val, _, recovered) = self.parse_seq_to_before_end(ket, sep, f)?;
        if !recovered {
            self.bump();
        }
        Ok(val)
    }

    /// Parses a sequence, not including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_before_end<T>(
        &mut self,
        ket: &TokenKind,
        sep: SeqSep,
        f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    ) -> PResult<'a, (Vec<T>, bool, bool)> {
        self.parse_seq_to_before_tokens(&[ket], sep, TokenExpectType::Expect, f)
    }

    fn expect_any_with_type(&mut self, kets: &[&TokenKind], expect: TokenExpectType) -> bool {
        kets.iter().any(|k| {
            match expect {
                TokenExpectType::Expect => self.check(k),
                TokenExpectType::NoExpect => self.token == **k,
            }
        })
    }

    crate fn parse_seq_to_before_tokens<T>(
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
                break
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
                        Err(mut e) => {
                            // Attempt to keep parsing if it was a similar separator
                            if let Some(ref tokens) = t.similar_tokens() {
                                if tokens.contains(&self.token.kind) {
                                    self.bump();
                                }
                            }
                            e.emit();
                            // Attempt to keep parsing if it was an omitted separator
                            match f(self) {
                                Ok(t) => {
                                    v.push(t);
                                    continue;
                                },
                                Err(mut e) => {
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
        let (result, trailing, recovered) = self.parse_seq_to_before_end(ket, sep, f)?;
        if !recovered {
            self.eat(ket);
        }
        Ok((result, trailing))
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

    /// Advance the parser by one token
    pub fn bump(&mut self) {
        if self.prev_token_kind == PrevTokenKind::Eof {
            // Bumping after EOF is a bad sign, usually an infinite loop.
            self.bug("attempted to bump the parser past EOF (may be stuck in a loop)");
        }

        self.prev_span = self.meta_var_span.take().unwrap_or(self.token.span);

        // Record last token kind for possible error recovery.
        self.prev_token_kind = match self.token.kind {
            token::DocComment(..) => PrevTokenKind::DocComment,
            token::Comma => PrevTokenKind::Comma,
            token::BinOp(token::Plus) => PrevTokenKind::Plus,
            token::BinOp(token::Or) => PrevTokenKind::BitOr,
            token::Interpolated(..) => PrevTokenKind::Interpolated,
            token::Eof => PrevTokenKind::Eof,
            token::Ident(..) => PrevTokenKind::Ident,
            _ => PrevTokenKind::Other,
        };

        self.token = self.next_tok();
        self.expected_tokens.clear();
        // check after each token
        self.process_potential_macro_variable();
    }

    /// Advance the parser using provided token as a next one. Use this when
    /// consuming a part of a token. For example a single `<` from `<<`.
    fn bump_with(&mut self, next: TokenKind, span: Span) {
        self.prev_span = self.token.span.with_hi(span.lo());
        // It would be incorrect to record the kind of the current token, but
        // fortunately for tokens currently using `bump_with`, the
        // prev_token_kind will be of no use anyway.
        self.prev_token_kind = PrevTokenKind::Other;
        self.token = Token::new(next, span);
        self.expected_tokens.clear();
    }

    pub fn look_ahead<R, F>(&self, dist: usize, f: F) -> R where
        F: FnOnce(&Token) -> R,
    {
        if dist == 0 {
            return f(&self.token);
        }

        let frame = &self.token_cursor.frame;
        f(&match frame.tree_cursor.look_ahead(dist - 1) {
            Some(tree) => match tree {
                TokenTree::Token(token) => token,
                TokenTree::Delimited(dspan, delim, _) =>
                    Token::new(token::OpenDelim(delim), dspan.open),
            }
            None => Token::new(token::CloseDelim(frame.delim), frame.span.close)
        })
    }

    /// Returns whether any of the given keywords are `dist` tokens ahead of the current one.
    fn is_keyword_ahead(&self, dist: usize, kws: &[Symbol]) -> bool {
        self.look_ahead(dist, |t| kws.iter().any(|&kw| t.is_keyword(kw)))
    }

    /// Parses asyncness: `async` or nothing.
    fn parse_asyncness(&mut self) -> IsAsync {
        if self.eat_keyword(kw::Async) {
            IsAsync::Async {
                closure_id: ast::DUMMY_NODE_ID,
                return_impl_trait_id: ast::DUMMY_NODE_ID,
            }
        } else {
            IsAsync::NotAsync
        }
    }

    /// Parses unsafety: `unsafe` or nothing.
    fn parse_unsafety(&mut self) -> Unsafety {
        if self.eat_keyword(kw::Unsafe) {
            Unsafety::Unsafe
        } else {
            Unsafety::Normal
        }
    }

    fn is_named_argument(&self) -> bool {
        let offset = match self.token.kind {
            token::Interpolated(ref nt) => match **nt {
                token::NtPat(..) => return self.look_ahead(1, |t| t == &token::Colon),
                _ => 0,
            }
            token::BinOp(token::And) | token::AndAnd => 1,
            _ if self.token.is_keyword(kw::Mut) => 1,
            _ => 0,
        };

        self.look_ahead(offset, |t| t.is_ident()) &&
        self.look_ahead(offset + 1, |t| t == &token::Colon)
    }

    /// Skips unexpected attributes and doc comments in this position and emits an appropriate
    /// error.
    /// This version of parse arg doesn't necessarily require identifier names.
    fn parse_arg_general(
        &mut self,
        is_trait_item: bool,
        allow_c_variadic: bool,
        is_name_required: impl Fn(&token::Token) -> bool,
    ) -> PResult<'a, Arg> {
        let lo = self.token.span;
        let attrs = self.parse_arg_attributes()?;
        if let Some(mut arg) = self.parse_self_arg()? {
            arg.attrs = attrs.into();
            return self.recover_bad_self_arg(arg, is_trait_item);
        }

        let is_name_required = is_name_required(&self.token);
        let (pat, ty) = if is_name_required || self.is_named_argument() {
            debug!("parse_arg_general parse_pat (is_name_required:{})", is_name_required);

            let pat = self.parse_fn_param_pat()?;
            if let Err(mut err) = self.expect(&token::Colon) {
                if let Some(ident) = self.argument_without_type(
                    &mut err,
                    pat,
                    is_name_required,
                    is_trait_item,
                ) {
                    err.emit();
                    return Ok(dummy_arg(ident));
                } else {
                    return Err(err);
                }
            }

            self.eat_incorrect_doc_comment_for_arg_type();
            (pat, self.parse_ty_common(true, true, allow_c_variadic)?)
        } else {
            debug!("parse_arg_general ident_to_pat");
            let parser_snapshot_before_ty = self.clone();
            self.eat_incorrect_doc_comment_for_arg_type();
            let mut ty = self.parse_ty_common(true, true, allow_c_variadic);
            if ty.is_ok() && self.token != token::Comma &&
               self.token != token::CloseDelim(token::Paren) {
                // This wasn't actually a type, but a pattern looking like a type,
                // so we are going to rollback and re-parse for recovery.
                ty = self.unexpected();
            }
            match ty {
                Ok(ty) => {
                    let ident = Ident::new(kw::Invalid, self.prev_span);
                    let bm = BindingMode::ByValue(Mutability::Immutable);
                    let pat = self.mk_pat_ident(ty.span, bm, ident);
                    (pat, ty)
                }
                Err(mut err) => {
                    // If this is a C-variadic argument and we hit an error, return the
                    // error.
                    if self.token == token::DotDotDot {
                        return Err(err);
                    }
                    // Recover from attempting to parse the argument as a type without pattern.
                    err.cancel();
                    mem::replace(self, parser_snapshot_before_ty);
                    self.recover_arg_parse()?
                }
            }
        };

        let span = lo.to(self.token.span);

        Ok(Arg { attrs: attrs.into(), id: ast::DUMMY_NODE_ID, pat, span, ty })
    }

    /// Parses mutability (`mut` or nothing).
    fn parse_mutability(&mut self) -> Mutability {
        if self.eat_keyword(kw::Mut) {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        }
    }

    fn parse_field_name(&mut self) -> PResult<'a, Ident> {
        if let token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) =
                self.token.kind {
            self.expect_no_suffix(self.token.span, "a tuple index", suffix);
            self.bump();
            Ok(Ident::new(symbol, self.prev_span))
        } else {
            self.parse_ident_common(false)
        }
    }

    fn expect_delimited_token_tree(&mut self) -> PResult<'a, (MacDelimiter, TokenStream)> {
        let delim = match self.token.kind {
            token::OpenDelim(delim) => delim,
            _ => {
                let msg = "expected open delimiter";
                let mut err = self.fatal(msg);
                err.span_label(self.token.span, msg);
                return Err(err)
            }
        };
        let tts = match self.parse_token_tree() {
            TokenTree::Delimited(_, _, tts) => tts,
            _ => unreachable!(),
        };
        let delim = match delim {
            token::Paren => MacDelimiter::Parenthesis,
            token::Bracket => MacDelimiter::Bracket,
            token::Brace => MacDelimiter::Brace,
            token::NoDelim => self.bug("unexpected no delimiter"),
        };
        Ok((delim, tts.into()))
    }

    fn parse_or_use_outer_attributes(&mut self,
                                     already_parsed_attrs: Option<ThinVec<Attribute>>)
                                     -> PResult<'a, ThinVec<Attribute>> {
        if let Some(attrs) = already_parsed_attrs {
            Ok(attrs)
        } else {
            self.parse_outer_attributes().map(|a| a.into())
        }
    }

    crate fn process_potential_macro_variable(&mut self) {
        self.token = match self.token.kind {
            token::Dollar if self.token.span.from_expansion() &&
                             self.look_ahead(1, |t| t.is_ident()) => {
                self.bump();
                let name = match self.token.kind {
                    token::Ident(name, _) => name,
                    _ => unreachable!()
                };
                let span = self.prev_span.to(self.token.span);
                self.diagnostic()
                    .struct_span_fatal(span, &format!("unknown macro variable `{}`", name))
                    .span_label(span, "unknown macro variable")
                    .emit();
                self.bump();
                return
            }
            token::Interpolated(ref nt) => {
                self.meta_var_span = Some(self.token.span);
                // Interpolated identifier and lifetime tokens are replaced with usual identifier
                // and lifetime tokens, so the former are never encountered during normal parsing.
                match **nt {
                    token::NtIdent(ident, is_raw) =>
                        Token::new(token::Ident(ident.name, is_raw), ident.span),
                    token::NtLifetime(ident) =>
                        Token::new(token::Lifetime(ident.name), ident.span),
                    _ => return,
                }
            }
            _ => return,
        };
    }

    /// Parses a single token tree from the input.
    crate fn parse_token_tree(&mut self) -> TokenTree {
        match self.token.kind {
            token::OpenDelim(..) => {
                let frame = mem::replace(&mut self.token_cursor.frame,
                                         self.token_cursor.stack.pop().unwrap());
                self.token.span = frame.span.entire();
                self.bump();
                TokenTree::Delimited(
                    frame.span,
                    frame.delim,
                    frame.tree_cursor.stream.into(),
                )
            },
            token::CloseDelim(_) | token::Eof => unreachable!(),
            _ => {
                let token = self.token.take();
                self.bump();
                TokenTree::Token(token)
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
    fn with_res<F, T>(&mut self, r: Restrictions, f: F) -> T
        where F: FnOnce(&mut Self) -> T
    {
        let old = self.restrictions;
        self.restrictions = r;
        let r = f(self);
        self.restrictions = old;
        return r;

    }

    fn parse_fn_args(&mut self, named_args: bool, allow_c_variadic: bool)
                     -> PResult<'a, (Vec<Arg> , bool)> {
        let sp = self.token.span;
        let mut c_variadic = false;
        let (args, _): (Vec<Option<Arg>>, _) = self.parse_paren_comma_seq(|p| {
            let do_not_enforce_named_arguments_for_c_variadic =
                |token: &token::Token| -> bool {
                    if token == &token::DotDotDot {
                        false
                    } else {
                        named_args
                    }
                };
            match p.parse_arg_general(
                false,
                allow_c_variadic,
                do_not_enforce_named_arguments_for_c_variadic
            ) {
                Ok(arg) => {
                    if let TyKind::CVarArgs = arg.ty.node {
                        c_variadic = true;
                        if p.token != token::CloseDelim(token::Paren) {
                            let span = p.token.span;
                            p.span_err(span,
                                "`...` must be the last argument of a C-variadic function");
                            Ok(None)
                        } else {
                            Ok(Some(arg))
                        }
                    } else {
                        Ok(Some(arg))
                    }
                },
                Err(mut e) => {
                    e.emit();
                    let lo = p.prev_span;
                    // Skip every token until next possible arg or end.
                    p.eat_to_tokens(&[&token::Comma, &token::CloseDelim(token::Paren)]);
                    // Create a placeholder argument for proper arg count (issue #34264).
                    let span = lo.to(p.prev_span);
                    Ok(Some(dummy_arg(Ident::new(kw::Invalid, span))))
                }
            }
        })?;

        let args: Vec<_> = args.into_iter().filter_map(|x| x).collect();

        if c_variadic && args.len() <= 1 {
            self.span_err(sp,
                          "C-variadic function must be declared with at least one named argument");
        }

        Ok((args, c_variadic))
    }

    /// Returns the parsed optional self argument and whether a self shortcut was used.
    ///
    /// See `parse_self_arg_with_attrs` to collect attributes.
    fn parse_self_arg(&mut self) -> PResult<'a, Option<Arg>> {
        let expect_ident = |this: &mut Self| match this.token.kind {
            // Preserve hygienic context.
            token::Ident(name, _) =>
                { let span = this.token.span; this.bump(); Ident::new(name, span) }
            _ => unreachable!()
        };
        let isolated_self = |this: &mut Self, n| {
            this.look_ahead(n, |t| t.is_keyword(kw::SelfLower)) &&
            this.look_ahead(n + 1, |t| t != &token::ModSep)
        };

        // Parse optional `self` parameter of a method.
        // Only a limited set of initial token sequences is considered `self` parameters; anything
        // else is parsed as a normal function parameter list, so some lookahead is required.
        let eself_lo = self.token.span;
        let (eself, eself_ident, eself_hi) = match self.token.kind {
            token::BinOp(token::And) => {
                // `&self`
                // `&mut self`
                // `&'lt self`
                // `&'lt mut self`
                // `&not_self`
                (if isolated_self(self, 1) {
                    self.bump();
                    SelfKind::Region(None, Mutability::Immutable)
                } else if self.is_keyword_ahead(1, &[kw::Mut]) &&
                          isolated_self(self, 2) {
                    self.bump();
                    self.bump();
                    SelfKind::Region(None, Mutability::Mutable)
                } else if self.look_ahead(1, |t| t.is_lifetime()) &&
                          isolated_self(self, 2) {
                    self.bump();
                    let lt = self.expect_lifetime();
                    SelfKind::Region(Some(lt), Mutability::Immutable)
                } else if self.look_ahead(1, |t| t.is_lifetime()) &&
                          self.is_keyword_ahead(2, &[kw::Mut]) &&
                          isolated_self(self, 3) {
                    self.bump();
                    let lt = self.expect_lifetime();
                    self.bump();
                    SelfKind::Region(Some(lt), Mutability::Mutable)
                } else {
                    return Ok(None);
                }, expect_ident(self), self.prev_span)
            }
            token::BinOp(token::Star) => {
                // `*self`
                // `*const self`
                // `*mut self`
                // `*not_self`
                // Emit special error for `self` cases.
                let msg = "cannot pass `self` by raw pointer";
                (if isolated_self(self, 1) {
                    self.bump();
                    self.struct_span_err(self.token.span, msg)
                        .span_label(self.token.span, msg)
                        .emit();
                    SelfKind::Value(Mutability::Immutable)
                } else if self.look_ahead(1, |t| t.is_mutability()) &&
                          isolated_self(self, 2) {
                    self.bump();
                    self.bump();
                    self.struct_span_err(self.token.span, msg)
                        .span_label(self.token.span, msg)
                        .emit();
                    SelfKind::Value(Mutability::Immutable)
                } else {
                    return Ok(None);
                }, expect_ident(self), self.prev_span)
            }
            token::Ident(..) => {
                if isolated_self(self, 0) {
                    // `self`
                    // `self: TYPE`
                    let eself_ident = expect_ident(self);
                    let eself_hi = self.prev_span;
                    (if self.eat(&token::Colon) {
                        let ty = self.parse_ty()?;
                        SelfKind::Explicit(ty, Mutability::Immutable)
                    } else {
                        SelfKind::Value(Mutability::Immutable)
                    }, eself_ident, eself_hi)
                } else if self.token.is_keyword(kw::Mut) &&
                          isolated_self(self, 1) {
                    // `mut self`
                    // `mut self: TYPE`
                    self.bump();
                    let eself_ident = expect_ident(self);
                    let eself_hi = self.prev_span;
                    (if self.eat(&token::Colon) {
                        let ty = self.parse_ty()?;
                        SelfKind::Explicit(ty, Mutability::Mutable)
                    } else {
                        SelfKind::Value(Mutability::Mutable)
                    }, eself_ident, eself_hi)
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        let eself = source_map::respan(eself_lo.to(eself_hi), eself);
        Ok(Some(Arg::from_self(ThinVec::default(), eself, eself_ident)))
    }

    /// Returns the parsed optional self argument with attributes and whether a self
    /// shortcut was used.
    fn parse_self_arg_with_attrs(&mut self) -> PResult<'a, Option<Arg>> {
        let attrs = self.parse_arg_attributes()?;
        let arg_opt = self.parse_self_arg()?;
        Ok(arg_opt.map(|mut arg| {
            arg.attrs = attrs.into();
            arg
        }))
    }

    /// Parses the parameter list and result type of a function that may have a `self` parameter.
    fn parse_fn_decl_with_self<F>(&mut self, parse_arg_fn: F) -> PResult<'a, P<FnDecl>>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a,  Arg>,
    {
        self.expect(&token::OpenDelim(token::Paren))?;

        // Parse optional self argument.
        let self_arg = self.parse_self_arg_with_attrs()?;

        // Parse the rest of the function parameter list.
        let sep = SeqSep::trailing_allowed(token::Comma);
        let (mut fn_inputs, recovered) = if let Some(self_arg) = self_arg {
            if self.check(&token::CloseDelim(token::Paren)) {
                (vec![self_arg], false)
            } else if self.eat(&token::Comma) {
                let mut fn_inputs = vec![self_arg];
                let (mut input, _, recovered) = self.parse_seq_to_before_end(
                    &token::CloseDelim(token::Paren), sep, parse_arg_fn)?;
                fn_inputs.append(&mut input);
                (fn_inputs, recovered)
            } else {
                match self.expect_one_of(&[], &[]) {
                    Err(err) => return Err(err),
                    Ok(recovered) => (vec![self_arg], recovered),
                }
            }
        } else {
            let (input, _, recovered) =
                self.parse_seq_to_before_end(&token::CloseDelim(token::Paren), sep, parse_arg_fn)?;
            (input, recovered)
        };

        if !recovered {
            // Parse closing paren and return type.
            self.expect(&token::CloseDelim(token::Paren))?;
        }
        // Replace duplicated recovered arguments with `_` pattern to avoid unecessary errors.
        self.deduplicate_recovered_arg_names(&mut fn_inputs);

        Ok(P(FnDecl {
            inputs: fn_inputs,
            output: self.parse_ret_ty(true)?,
            c_variadic: false
        }))
    }

    fn is_crate_vis(&self) -> bool {
        self.token.is_keyword(kw::Crate) && self.look_ahead(1, |t| t != &token::ModSep)
    }

    /// Parses `pub`, `pub(crate)` and `pub(in path)` plus shortcuts `crate` for `pub(crate)`,
    /// `pub(self)` for `pub(in self)` and `pub(super)` for `pub(in super)`.
    /// If the following element can't be a tuple (i.e., it's a function definition), then
    /// it's not a tuple struct field), and the contents within the parentheses isn't valid,
    /// so emit a proper diagnostic.
    pub fn parse_visibility(&mut self, can_take_tuple: bool) -> PResult<'a, Visibility> {
        maybe_whole!(self, NtVis, |x| x);

        self.expected_tokens.push(TokenType::Keyword(kw::Crate));
        if self.is_crate_vis() {
            self.bump(); // `crate`
            return Ok(respan(self.prev_span, VisibilityKind::Crate(CrateSugar::JustCrate)));
        }

        if !self.eat_keyword(kw::Pub) {
            // We need a span for our `Spanned<VisibilityKind>`, but there's inherently no
            // keyword to grab a span from for inherited visibility; an empty span at the
            // beginning of the current token would seem to be the "Schelling span".
            return Ok(respan(self.token.span.shrink_to_lo(), VisibilityKind::Inherited))
        }
        let lo = self.prev_span;

        if self.check(&token::OpenDelim(token::Paren)) {
            // We don't `self.bump()` the `(` yet because this might be a struct definition where
            // `()` or a tuple might be allowed. For example, `struct Struct(pub (), pub (usize));`.
            // Because of this, we only `bump` the `(` if we're assured it is appropriate to do so
            // by the following tokens.
            if self.is_keyword_ahead(1, &[kw::Crate]) &&
                self.look_ahead(2, |t| t != &token::ModSep) // account for `pub(crate::foo)`
            {
                // `pub(crate)`
                self.bump(); // `(`
                self.bump(); // `crate`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = respan(
                    lo.to(self.prev_span),
                    VisibilityKind::Crate(CrateSugar::PubCrate),
                );
                return Ok(vis)
            } else if self.is_keyword_ahead(1, &[kw::In]) {
                // `pub(in path)`
                self.bump(); // `(`
                self.bump(); // `in`
                let path = self.parse_path(PathStyle::Mod)?; // `path`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = respan(lo.to(self.prev_span), VisibilityKind::Restricted {
                    path: P(path),
                    id: ast::DUMMY_NODE_ID,
                });
                return Ok(vis)
            } else if self.look_ahead(2, |t| t == &token::CloseDelim(token::Paren)) &&
                      self.is_keyword_ahead(1, &[kw::Super, kw::SelfLower])
            {
                // `pub(self)` or `pub(super)`
                self.bump(); // `(`
                let path = self.parse_path(PathStyle::Mod)?; // `super`/`self`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = respan(lo.to(self.prev_span), VisibilityKind::Restricted {
                    path: P(path),
                    id: ast::DUMMY_NODE_ID,
                });
                return Ok(vis)
            } else if !can_take_tuple {  // Provide this diagnostic if this is not a tuple struct
                // `pub(something) fn ...` or `struct X { pub(something) y: Z }`
                self.bump(); // `(`
                let msg = "incorrect visibility restriction";
                let suggestion = r##"some possible visibility restrictions are:
`pub(crate)`: visible only on the current crate
`pub(super)`: visible only in the current module's parent
`pub(in path::to::module)`: visible only on the specified path"##;
                let path = self.parse_path(PathStyle::Mod)?;
                let sp = path.span;
                let help_msg = format!("make this visible only to module `{}` with `in`", path);
                self.expect(&token::CloseDelim(token::Paren))?;  // `)`
                struct_span_err!(self.sess.span_diagnostic, sp, E0704, "{}", msg)
                    .help(suggestion)
                    .span_suggestion(
                        sp,
                        &help_msg,
                        format!("in {}", path),
                        Applicability::MachineApplicable,
                    )
                    .emit();  // emit diagnostic, but continue with public visibility
            }
        }

        Ok(respan(lo, VisibilityKind::Public))
    }

    /// Parses a string as an ABI spec on an extern type or module. Consumes
    /// the `extern` keyword, if one is found.
    fn parse_opt_abi(&mut self) -> PResult<'a, Option<Abi>> {
        match self.token.kind {
            token::Literal(token::Lit { kind: token::Str, symbol, suffix }) |
            token::Literal(token::Lit { kind: token::StrRaw(..), symbol, suffix }) => {
                let sp = self.token.span;
                self.expect_no_suffix(sp, "an ABI spec", suffix);
                self.bump();
                match abi::lookup(&symbol.as_str()) {
                    Some(abi) => Ok(Some(abi)),
                    None => {
                        let prev_span = self.prev_span;
                        struct_span_err!(
                            self.sess.span_diagnostic,
                            prev_span,
                            E0703,
                            "invalid ABI: found `{}`",
                            symbol
                        )
                        .span_label(prev_span, "invalid ABI")
                        .help(&format!("valid ABIs: {}", abi::all_names().join(", ")))
                        .emit();
                        Ok(None)
                    }
                }
            }

            _ => Ok(None),
        }
    }

    /// We are parsing `async fn`. If we are on Rust 2015, emit an error.
    fn ban_async_in_2015(&self, async_span: Span) {
        if async_span.rust_2015() {
            self.diagnostic()
                .struct_span_err_with_code(
                    async_span,
                    "`async fn` is not permitted in the 2015 edition",
                    DiagnosticId::Error("E0670".into())
                )
                .emit();
        }
    }

    fn collect_tokens<F, R>(&mut self, f: F) -> PResult<'a, (R, TokenStream)>
        where F: FnOnce(&mut Self) -> PResult<'a, R>
    {
        // Record all tokens we parse when parsing this item.
        let mut tokens = Vec::new();
        let prev_collecting = match self.token_cursor.frame.last_token {
            LastToken::Collecting(ref mut list) => {
                Some(mem::take(list))
            }
            LastToken::Was(ref mut last) => {
                tokens.extend(last.take());
                None
            }
        };
        self.token_cursor.frame.last_token = LastToken::Collecting(tokens);
        let prev = self.token_cursor.stack.len();
        let ret = f(self);
        let last_token = if self.token_cursor.stack.len() == prev {
            &mut self.token_cursor.frame.last_token
        } else if self.token_cursor.stack.get(prev).is_none() {
            // This can happen due to a bad interaction of two unrelated recovery mechanisms with
            // mismatched delimiters *and* recovery lookahead on the likely typo `pub ident(`
            // (#62881).
            return Ok((ret?, TokenStream::new(vec![])));
        } else {
            &mut self.token_cursor.stack[prev].last_token
        };

        // Pull out the tokens that we've collected from the call to `f` above.
        let mut collected_tokens = match *last_token {
            LastToken::Collecting(ref mut v) => mem::take(v),
            LastToken::Was(ref was) => {
                let msg = format!("our vector went away? - found Was({:?})", was);
                debug!("collect_tokens: {}", msg);
                self.sess.span_diagnostic.delay_span_bug(self.token.span, &msg);
                // This can happen due to a bad interaction of two unrelated recovery mechanisms
                // with mismatched delimiters *and* recovery lookahead on the likely typo
                // `pub ident(` (#62895, different but similar to the case above).
                return Ok((ret?, TokenStream::new(vec![])));
            }
        };

        // If we're not at EOF our current token wasn't actually consumed by
        // `f`, but it'll still be in our list that we pulled out. In that case
        // put it back.
        let extra_token = if self.token != token::Eof {
            collected_tokens.pop()
        } else {
            None
        };

        // If we were previously collecting tokens, then this was a recursive
        // call. In that case we need to record all the tokens we collected in
        // our parent list as well. To do that we push a clone of our stream
        // onto the previous list.
        match prev_collecting {
            Some(mut list) => {
                list.extend(collected_tokens.iter().cloned());
                list.extend(extra_token);
                *last_token = LastToken::Collecting(list);
            }
            None => {
                *last_token = LastToken::Was(extra_token);
            }
        }

        Ok((ret?, TokenStream::new(collected_tokens)))
    }

    /// `::{` or `::*`
    fn is_import_coupler(&mut self) -> bool {
        self.check(&token::ModSep) &&
            self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace) ||
                                   *t == token::BinOp(token::Star))
    }

    pub fn parse_optional_str(&mut self) -> Option<(Symbol, ast::StrStyle, Option<ast::Name>)> {
        let ret = match self.token.kind {
            token::Literal(token::Lit { kind: token::Str, symbol, suffix }) =>
                (symbol, ast::StrStyle::Cooked, suffix),
            token::Literal(token::Lit { kind: token::StrRaw(n), symbol, suffix }) =>
                (symbol, ast::StrStyle::Raw(n), suffix),
            _ => return None
        };
        self.bump();
        Some(ret)
    }

    pub fn parse_str(&mut self) -> PResult<'a, (Symbol, StrStyle)> {
        match self.parse_optional_str() {
            Some((s, style, suf)) => {
                let sp = self.prev_span;
                self.expect_no_suffix(sp, "a string literal", suf);
                Ok((s, style))
            }
            _ => {
                let msg = "expected string literal";
                let mut err = self.fatal(msg);
                err.span_label(self.token.span, msg);
                Err(err)
            }
        }
    }

    fn report_invalid_macro_expansion_item(&self) {
        self.struct_span_err(
            self.prev_span,
            "macros that expand to items must be delimited with braces or followed by a semicolon",
        ).multipart_suggestion(
            "change the delimiters to curly braces",
            vec![
                (self.prev_span.with_hi(self.prev_span.lo() + BytePos(1)), String::from(" {")),
                (self.prev_span.with_lo(self.prev_span.hi() - BytePos(1)), '}'.to_string()),
            ],
            Applicability::MaybeIncorrect,
        ).span_suggestion(
            self.sess.source_map.next_point(self.prev_span),
            "add a semicolon",
            ';'.to_string(),
            Applicability::MaybeIncorrect,
        ).emit();
    }
}

pub fn emit_unclosed_delims(unclosed_delims: &mut Vec<UnmatchedBrace>, handler: &errors::Handler) {
    for unmatched in unclosed_delims.iter() {
        let mut err = handler.struct_span_err(unmatched.found_span, &format!(
            "incorrect close delimiter: `{}`",
            pprust::token_kind_to_string(&token::CloseDelim(unmatched.found_delim)),
        ));
        err.span_label(unmatched.found_span, "incorrect close delimiter");
        if let Some(sp) = unmatched.candidate_span {
            err.span_label(sp, "close delimiter possibly meant for this");
        }
        if let Some(sp) = unmatched.unclosed_span {
            err.span_label(sp, "un-closed delimiter");
        }
        err.emit();
    }
    unclosed_delims.clear();
}
