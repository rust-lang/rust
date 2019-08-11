// ignore-tidy-filelength

mod expr;
use expr::LhsExpr;
mod pat;
mod item;
pub use item::AliasKind;
mod module;
pub use module::{ModulePath, ModulePathSuccess};

use crate::ast::{self, AngleBracketedArgs, ParenthesizedArgs, AttrStyle, BareFnTy};
use crate::ast::{AnonConst, Arg, Attribute, BindingMode};
use crate::ast::{Block, BlockCheckMode, Expr, ExprKind, Stmt, StmtKind};
use crate::ast::{FnDecl, FunctionRetTy};
use crate::ast::{Ident, IsAsync, Local, Lifetime};
use crate::ast::{MacStmtStyle, Mac_, MacDelimiter};
use crate::ast::{MutTy, Mutability};
use crate::ast::{PolyTraitRef, QSelf, PathSegment};
use crate::ast::StrStyle;
use crate::ast::SelfKind;
use crate::ast::{GenericBound, TraitBoundModifier, TraitObjectSyntax};
use crate::ast::{GenericParam, GenericParamKind, GenericArg, WhereClause};
use crate::ast::{Ty, TyKind, AssocTyConstraint, AssocTyConstraintKind, GenericBounds};
use crate::ast::{Visibility, VisibilityKind, Unsafety, CrateSugar};
use crate::ext::base::DummyResult;
use crate::ext::hygiene::SyntaxContext;
use crate::source_map::{self, respan};
use crate::parse::{SeqSep, classify, literal, token};
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

/// Specifies how to parse a path.
#[derive(Copy, Clone, PartialEq)]
pub enum PathStyle {
    /// In some contexts, notably in expressions, paths with generic arguments are ambiguous
    /// with something else. For example, in expressions `segment < ....` can be interpreted
    /// as a comparison and `segment ( ....` can be interpreted as a function call.
    /// In all such contexts the non-path interpretation is preferred by default for practical
    /// reasons, but the path interpretation can be forced by the disambiguator `::`, e.g.
    /// `x<y>` - comparisons, `x::<y>` - unambiguously a path.
    Expr,
    /// In other contexts, notably in types, no ambiguity exists and paths can be written
    /// without the disambiguator, e.g., `x<y>` - unambiguously a path.
    /// Paths with disambiguators are still accepted, `x::<Y>` - unambiguously a path too.
    Type,
    /// A path with generic arguments disallowed, e.g., `foo::bar::Baz`, used in imports,
    /// visibilities or attributes.
    /// Technically, this variant is unnecessary and e.g., `Expr` can be used instead
    /// (paths in "mod" contexts have to be checked later for absence of generic arguments
    /// anyway, due to macros), but it is used to avoid weird suggestions about expected
    /// tokens when something goes wrong.
    Mod,
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

/// Returns `true` if `IDENT t` can start a type -- `IDENT::a::b`, `IDENT<u8, u8>`,
/// `IDENT<<u8 as Trait>::AssocTy>`.
///
/// Types can also be of the form `IDENT(u8, u8) -> u8`, however this assumes
/// that `IDENT` is not the ident of a fn trait.
fn can_continue_type_after_non_fn_ident(t: &Token) -> bool {
    t == &token::ModSep || t == &token::Lt ||
    t == &token::BinOp(token::Shl)
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
            if let FileName::Real(mut path) =
                    sess.source_map().span_to_unmapped_path(parser.token.span) {
                path.pop();
                parser.directory.path = Cow::from(path);
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

    /// Is the current token one of the keywords that signals a bare function type?
    fn token_is_bare_fn_keyword(&mut self) -> bool {
        self.check_keyword(kw::Fn) ||
            self.check_keyword(kw::Unsafe) ||
            self.check_keyword(kw::Extern)
    }

    /// Parses a `TyKind::BareFn` type.
    fn parse_ty_bare_fn(&mut self, generic_params: Vec<GenericParam>) -> PResult<'a, TyKind> {
        /*

        [unsafe] [extern "ABI"] fn (S) -> T
         ^~~~^           ^~~~^     ^~^    ^
           |               |        |     |
           |               |        |   Return type
           |               |      Argument types
           |               |
           |              ABI
        Function Style
        */

        let unsafety = self.parse_unsafety();
        let abi = if self.eat_keyword(kw::Extern) {
            self.parse_opt_abi()?.unwrap_or(Abi::C)
        } else {
            Abi::Rust
        };

        self.expect_keyword(kw::Fn)?;
        let (inputs, c_variadic) = self.parse_fn_args(false, true)?;
        let ret_ty = self.parse_ret_ty(false)?;
        let decl = P(FnDecl {
            inputs,
            output: ret_ty,
            c_variadic,
        });
        Ok(TyKind::BareFn(P(BareFnTy {
            abi,
            unsafety,
            generic_params,
            decl,
        })))
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

    /// Parses an optional return type `[ -> TY ]` in a function declaration.
    fn parse_ret_ty(&mut self, allow_plus: bool) -> PResult<'a, FunctionRetTy> {
        if self.eat(&token::RArrow) {
            Ok(FunctionRetTy::Ty(self.parse_ty_common(allow_plus, true, false)?))
        } else {
            Ok(FunctionRetTy::Default(self.token.span.shrink_to_lo()))
        }
    }

    /// Parses a type.
    pub fn parse_ty(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(true, true, false)
    }

    /// Parses a type in restricted contexts where `+` is not permitted.
    ///
    /// Example 1: `&'a TYPE`
    ///     `+` is prohibited to maintain operator priority (P(+) < P(&)).
    /// Example 2: `value1 as TYPE + value2`
    ///     `+` is prohibited to avoid interactions with expression grammar.
    fn parse_ty_no_plus(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(false, true, false)
    }

    fn parse_ty_common(&mut self, allow_plus: bool, allow_qpath_recovery: bool,
                       allow_c_variadic: bool) -> PResult<'a, P<Ty>> {
        maybe_recover_from_interpolated_ty_qpath!(self, allow_qpath_recovery);
        maybe_whole!(self, NtTy, |x| x);

        let lo = self.token.span;
        let mut impl_dyn_multi = false;
        let node = if self.eat(&token::OpenDelim(token::Paren)) {
            // `(TYPE)` is a parenthesized type.
            // `(TYPE,)` is a tuple with a single field of type TYPE.
            let mut ts = vec![];
            let mut last_comma = false;
            while self.token != token::CloseDelim(token::Paren) {
                ts.push(self.parse_ty()?);
                if self.eat(&token::Comma) {
                    last_comma = true;
                } else {
                    last_comma = false;
                    break;
                }
            }
            let trailing_plus = self.prev_token_kind == PrevTokenKind::Plus;
            self.expect(&token::CloseDelim(token::Paren))?;

            if ts.len() == 1 && !last_comma {
                let ty = ts.into_iter().nth(0).unwrap().into_inner();
                let maybe_bounds = allow_plus && self.token.is_like_plus();
                match ty.node {
                    // `(TY_BOUND_NOPAREN) + BOUND + ...`.
                    TyKind::Path(None, ref path) if maybe_bounds => {
                        self.parse_remaining_bounds(Vec::new(), path.clone(), lo, true)?
                    }
                    TyKind::TraitObject(ref bounds, TraitObjectSyntax::None)
                            if maybe_bounds && bounds.len() == 1 && !trailing_plus => {
                        let path = match bounds[0] {
                            GenericBound::Trait(ref pt, ..) => pt.trait_ref.path.clone(),
                            GenericBound::Outlives(..) => self.bug("unexpected lifetime bound"),
                        };
                        self.parse_remaining_bounds(Vec::new(), path, lo, true)?
                    }
                    // `(TYPE)`
                    _ => TyKind::Paren(P(ty))
                }
            } else {
                TyKind::Tup(ts)
            }
        } else if self.eat(&token::Not) {
            // Never type `!`
            TyKind::Never
        } else if self.eat(&token::BinOp(token::Star)) {
            // Raw pointer
            TyKind::Ptr(self.parse_ptr()?)
        } else if self.eat(&token::OpenDelim(token::Bracket)) {
            // Array or slice
            let t = self.parse_ty()?;
            // Parse optional `; EXPR` in `[TYPE; EXPR]`
            let t = match self.maybe_parse_fixed_length_of_vec()? {
                None => TyKind::Slice(t),
                Some(length) => TyKind::Array(t, AnonConst {
                    id: ast::DUMMY_NODE_ID,
                    value: length,
                }),
            };
            self.expect(&token::CloseDelim(token::Bracket))?;
            t
        } else if self.check(&token::BinOp(token::And)) || self.check(&token::AndAnd) {
            // Reference
            self.expect_and()?;
            self.parse_borrowed_pointee()?
        } else if self.eat_keyword_noexpect(kw::Typeof) {
            // `typeof(EXPR)`
            // In order to not be ambiguous, the type must be surrounded by parens.
            self.expect(&token::OpenDelim(token::Paren))?;
            let e = AnonConst {
                id: ast::DUMMY_NODE_ID,
                value: self.parse_expr()?,
            };
            self.expect(&token::CloseDelim(token::Paren))?;
            TyKind::Typeof(e)
        } else if self.eat_keyword(kw::Underscore) {
            // A type to be inferred `_`
            TyKind::Infer
        } else if self.token_is_bare_fn_keyword() {
            // Function pointer type
            self.parse_ty_bare_fn(Vec::new())?
        } else if self.check_keyword(kw::For) {
            // Function pointer type or bound list (trait object type) starting with a poly-trait.
            //   `for<'lt> [unsafe] [extern "ABI"] fn (&'lt S) -> T`
            //   `for<'lt> Trait1<'lt> + Trait2 + 'a`
            let lo = self.token.span;
            let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
            if self.token_is_bare_fn_keyword() {
                self.parse_ty_bare_fn(lifetime_defs)?
            } else {
                let path = self.parse_path(PathStyle::Type)?;
                let parse_plus = allow_plus && self.check_plus();
                self.parse_remaining_bounds(lifetime_defs, path, lo, parse_plus)?
            }
        } else if self.eat_keyword(kw::Impl) {
            // Always parse bounds greedily for better error recovery.
            let bounds = self.parse_generic_bounds(None)?;
            impl_dyn_multi = bounds.len() > 1 || self.prev_token_kind == PrevTokenKind::Plus;
            TyKind::ImplTrait(ast::DUMMY_NODE_ID, bounds)
        } else if self.check_keyword(kw::Dyn) &&
                  (self.token.span.rust_2018() ||
                   self.look_ahead(1, |t| t.can_begin_bound() &&
                                          !can_continue_type_after_non_fn_ident(t))) {
            self.bump(); // `dyn`
            // Always parse bounds greedily for better error recovery.
            let bounds = self.parse_generic_bounds(None)?;
            impl_dyn_multi = bounds.len() > 1 || self.prev_token_kind == PrevTokenKind::Plus;
            TyKind::TraitObject(bounds, TraitObjectSyntax::Dyn)
        } else if self.check(&token::Question) ||
                  self.check_lifetime() && self.look_ahead(1, |t| t.is_like_plus()) {
            // Bound list (trait object type)
            TyKind::TraitObject(self.parse_generic_bounds_common(allow_plus, None)?,
                                TraitObjectSyntax::None)
        } else if self.eat_lt() {
            // Qualified path
            let (qself, path) = self.parse_qpath(PathStyle::Type)?;
            TyKind::Path(Some(qself), path)
        } else if self.token.is_path_start() {
            // Simple path
            let path = self.parse_path(PathStyle::Type)?;
            if self.eat(&token::Not) {
                // Macro invocation in type position
                let (delim, tts) = self.expect_delimited_token_tree()?;
                let node = Mac_ {
                    path,
                    tts,
                    delim,
                    prior_type_ascription: self.last_type_ascription,
                };
                TyKind::Mac(respan(lo.to(self.prev_span), node))
            } else {
                // Just a type path or bound list (trait object type) starting with a trait.
                //   `Type`
                //   `Trait1 + Trait2 + 'a`
                if allow_plus && self.check_plus() {
                    self.parse_remaining_bounds(Vec::new(), path, lo, true)?
                } else {
                    TyKind::Path(None, path)
                }
            }
        } else if self.check(&token::DotDotDot) {
            if allow_c_variadic {
                self.eat(&token::DotDotDot);
                TyKind::CVarArgs
            } else {
                return Err(self.fatal(
                    "only foreign functions are allowed to be C-variadic"
                ));
            }
        } else {
            let msg = format!("expected type, found {}", self.this_token_descr());
            let mut err = self.fatal(&msg);
            err.span_label(self.token.span, "expected type");
            self.maybe_annotate_with_ascription(&mut err, true);
            return Err(err);
        };

        let span = lo.to(self.prev_span);
        let ty = P(Ty { node, span, id: ast::DUMMY_NODE_ID });

        // Try to recover from use of `+` with incorrect priority.
        self.maybe_report_ambiguous_plus(allow_plus, impl_dyn_multi, &ty);
        self.maybe_recover_from_bad_type_plus(allow_plus, &ty)?;
        self.maybe_recover_from_bad_qpath(ty, allow_qpath_recovery)
    }

    fn parse_remaining_bounds(&mut self, generic_params: Vec<GenericParam>, path: ast::Path,
                              lo: Span, parse_plus: bool) -> PResult<'a, TyKind> {
        let poly_trait_ref = PolyTraitRef::new(generic_params, path, lo.to(self.prev_span));
        let mut bounds = vec![GenericBound::Trait(poly_trait_ref, TraitBoundModifier::None)];
        if parse_plus {
            self.eat_plus(); // `+`, or `+=` gets split and `+` is discarded
            bounds.append(&mut self.parse_generic_bounds(Some(self.prev_span))?);
        }
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::None))
    }

    fn parse_borrowed_pointee(&mut self) -> PResult<'a, TyKind> {
        let opt_lifetime = if self.check_lifetime() { Some(self.expect_lifetime()) } else { None };
        let mutbl = self.parse_mutability();
        let ty = self.parse_ty_no_plus()?;
        return Ok(TyKind::Rptr(opt_lifetime, MutTy { ty, mutbl }));
    }

    fn parse_ptr(&mut self) -> PResult<'a, MutTy> {
        let mutbl = if self.eat_keyword(kw::Mut) {
            Mutability::Mutable
        } else if self.eat_keyword(kw::Const) {
            Mutability::Immutable
        } else {
            let span = self.prev_span;
            let msg = "expected mut or const in raw pointer type";
            self.struct_span_err(span, msg)
                .span_label(span, msg)
                .help("use `*mut T` or `*const T` as appropriate")
                .emit();
            Mutability::Immutable
        };
        let t = self.parse_ty_no_plus()?;
        Ok(MutTy { ty: t, mutbl })
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
    fn parse_arg_general<F>(
        &mut self,
        is_trait_item: bool,
        allow_c_variadic: bool,
        is_name_required: F,
    ) -> PResult<'a, Arg>
    where
        F: Fn(&token::Token) -> bool
    {
        let lo = self.token.span;
        let attrs = self.parse_arg_attributes()?;
        if let Some(mut arg) = self.parse_self_arg()? {
            arg.attrs = attrs.into();
            return self.recover_bad_self_arg(arg, is_trait_item);
        }

        let is_name_required = is_name_required(&self.token);
        let (pat, ty) = if is_name_required || self.is_named_argument() {
            debug!("parse_arg_general parse_pat (is_name_required:{})", is_name_required);

            let pat = self.parse_pat(Some("argument name"))?;
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

    /// Parses an argument in a lambda header (e.g., `|arg, arg|`).
    fn parse_fn_block_arg(&mut self) -> PResult<'a, Arg> {
        let lo = self.token.span;
        let attrs = self.parse_arg_attributes()?;
        let pat = self.parse_pat(Some("argument name"))?;
        let t = if self.eat(&token::Colon) {
            self.parse_ty()?
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyKind::Infer,
                span: self.prev_span,
            })
        };
        let span = lo.to(self.token.span);
        Ok(Arg {
            attrs: attrs.into(),
            ty: t,
            pat,
            span,
            id: ast::DUMMY_NODE_ID
        })
    }

    fn maybe_parse_fixed_length_of_vec(&mut self) -> PResult<'a, Option<P<ast::Expr>>> {
        if self.eat(&token::Semi) {
            Ok(Some(self.parse_expr()?))
        } else {
            Ok(None)
        }
    }

    fn parse_path_segment_ident(&mut self) -> PResult<'a, ast::Ident> {
        match self.token.kind {
            token::Ident(name, _) if name.is_path_segment_keyword() => {
                let span = self.token.span;
                self.bump();
                Ok(Ident::new(name, span))
            }
            _ => self.parse_ident(),
        }
    }

    fn parse_ident_or_underscore(&mut self) -> PResult<'a, ast::Ident> {
        match self.token.kind {
            token::Ident(name, false) if name == kw::Underscore => {
                let span = self.token.span;
                self.bump();
                Ok(Ident::new(name, span))
            }
            _ => self.parse_ident(),
        }
    }

    /// Parses a qualified path.
    /// Assumes that the leading `<` has been parsed already.
    ///
    /// `qualified_path = <type [as trait_ref]>::path`
    ///
    /// # Examples
    /// `<T>::default`
    /// `<T as U>::a`
    /// `<T as U>::F::a<S>` (without disambiguator)
    /// `<T as U>::F::a::<S>` (with disambiguator)
    fn parse_qpath(&mut self, style: PathStyle) -> PResult<'a, (QSelf, ast::Path)> {
        let lo = self.prev_span;
        let ty = self.parse_ty()?;

        // `path` will contain the prefix of the path up to the `>`,
        // if any (e.g., `U` in the `<T as U>::*` examples
        // above). `path_span` has the span of that path, or an empty
        // span in the case of something like `<T>::Bar`.
        let (mut path, path_span);
        if self.eat_keyword(kw::As) {
            let path_lo = self.token.span;
            path = self.parse_path(PathStyle::Type)?;
            path_span = path_lo.to(self.prev_span);
        } else {
            path_span = self.token.span.to(self.token.span);
            path = ast::Path { segments: Vec::new(), span: path_span };
        }

        // See doc comment for `unmatched_angle_bracket_count`.
        self.expect(&token::Gt)?;
        if self.unmatched_angle_bracket_count > 0 {
            self.unmatched_angle_bracket_count -= 1;
            debug!("parse_qpath: (decrement) count={:?}", self.unmatched_angle_bracket_count);
        }

        self.expect(&token::ModSep)?;

        let qself = QSelf { ty, path_span, position: path.segments.len() };
        self.parse_path_segments(&mut path.segments, style)?;

        Ok((qself, ast::Path { segments: path.segments, span: lo.to(self.prev_span) }))
    }

    /// Parses simple paths.
    ///
    /// `path = [::] segment+`
    /// `segment = ident | ident[::]<args> | ident[::](args) [-> type]`
    ///
    /// # Examples
    /// `a::b::C<D>` (without disambiguator)
    /// `a::b::C::<D>` (with disambiguator)
    /// `Fn(Args)` (without disambiguator)
    /// `Fn::(Args)` (with disambiguator)
    pub fn parse_path(&mut self, style: PathStyle) -> PResult<'a, ast::Path> {
        maybe_whole!(self, NtPath, |path| {
            if style == PathStyle::Mod &&
               path.segments.iter().any(|segment| segment.args.is_some()) {
                self.diagnostic().span_err(path.span, "unexpected generic arguments in path");
            }
            path
        });

        let lo = self.meta_var_span.unwrap_or(self.token.span);
        let mut segments = Vec::new();
        let mod_sep_ctxt = self.token.span.ctxt();
        if self.eat(&token::ModSep) {
            segments.push(PathSegment::path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt)));
        }
        self.parse_path_segments(&mut segments, style)?;

        Ok(ast::Path { segments, span: lo.to(self.prev_span) })
    }

    /// Like `parse_path`, but also supports parsing `Word` meta items into paths for
    /// backwards-compatibility. This is used when parsing derive macro paths in `#[derive]`
    /// attributes.
    pub fn parse_path_allowing_meta(&mut self, style: PathStyle) -> PResult<'a, ast::Path> {
        let meta_ident = match self.token.kind {
            token::Interpolated(ref nt) => match **nt {
                token::NtMeta(ref meta) => match meta.node {
                    ast::MetaItemKind::Word => Some(meta.path.clone()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        if let Some(path) = meta_ident {
            self.bump();
            return Ok(path);
        }
        self.parse_path(style)
    }

    crate fn parse_path_segments(&mut self,
                           segments: &mut Vec<PathSegment>,
                           style: PathStyle)
                           -> PResult<'a, ()> {
        loop {
            let segment = self.parse_path_segment(style)?;
            if style == PathStyle::Expr {
                // In order to check for trailing angle brackets, we must have finished
                // recursing (`parse_path_segment` can indirectly call this function),
                // that is, the next token must be the highlighted part of the below example:
                //
                // `Foo::<Bar as Baz<T>>::Qux`
                //                      ^ here
                //
                // As opposed to the below highlight (if we had only finished the first
                // recursion):
                //
                // `Foo::<Bar as Baz<T>>::Qux`
                //                     ^ here
                //
                // `PathStyle::Expr` is only provided at the root invocation and never in
                // `parse_path_segment` to recurse and therefore can be checked to maintain
                // this invariant.
                self.check_trailing_angle_brackets(&segment, token::ModSep);
            }
            segments.push(segment);

            if self.is_import_coupler() || !self.eat(&token::ModSep) {
                return Ok(());
            }
        }
    }

    fn parse_path_segment(&mut self, style: PathStyle) -> PResult<'a, PathSegment> {
        let ident = self.parse_path_segment_ident()?;

        let is_args_start = |token: &Token| match token.kind {
            token::Lt | token::BinOp(token::Shl) | token::OpenDelim(token::Paren)
            | token::LArrow => true,
            _ => false,
        };
        let check_args_start = |this: &mut Self| {
            this.expected_tokens.extend_from_slice(
                &[TokenType::Token(token::Lt), TokenType::Token(token::OpenDelim(token::Paren))]
            );
            is_args_start(&this.token)
        };

        Ok(if style == PathStyle::Type && check_args_start(self) ||
              style != PathStyle::Mod && self.check(&token::ModSep)
                                      && self.look_ahead(1, |t| is_args_start(t)) {
            // We use `style == PathStyle::Expr` to check if this is in a recursion or not. If
            // it isn't, then we reset the unmatched angle bracket count as we're about to start
            // parsing a new path.
            if style == PathStyle::Expr {
                self.unmatched_angle_bracket_count = 0;
                self.max_angle_bracket_count = 0;
            }

            // Generic arguments are found - `<`, `(`, `::<` or `::(`.
            self.eat(&token::ModSep);
            let lo = self.token.span;
            let args = if self.eat_lt() {
                // `<'a, T, A = U>`
                let (args, constraints) =
                    self.parse_generic_args_with_leaning_angle_bracket_recovery(style, lo)?;
                self.expect_gt()?;
                let span = lo.to(self.prev_span);
                AngleBracketedArgs { args, constraints, span }.into()
            } else {
                // `(T, U) -> R`
                let (inputs, _) = self.parse_paren_comma_seq(|p| p.parse_ty())?;
                let span = lo.to(self.prev_span);
                let output = if self.eat(&token::RArrow) {
                    Some(self.parse_ty_common(false, false, false)?)
                } else {
                    None
                };
                ParenthesizedArgs { inputs, output, span }.into()
            };

            PathSegment { ident, args, id: ast::DUMMY_NODE_ID }
        } else {
            // Generic arguments are not found.
            PathSegment::from_ident(ident)
        })
    }

    crate fn check_lifetime(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Lifetime);
        self.token.is_lifetime()
    }

    /// Parses a single lifetime `'a` or panics.
    crate fn expect_lifetime(&mut self) -> Lifetime {
        if let Some(ident) = self.token.lifetime() {
            let span = self.token.span;
            self.bump();
            Lifetime { ident: Ident::new(ident.name, span), id: ast::DUMMY_NODE_ID }
        } else {
            self.span_bug(self.token.span, "not a lifetime")
        }
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
            token::Dollar if self.token.span.ctxt() != SyntaxContext::empty() &&
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

    /// Parses the RHS of a local variable declaration (e.g., '= 14;').
    fn parse_initializer(&mut self, skip_eq: bool) -> PResult<'a, Option<P<Expr>>> {
        if self.eat(&token::Eq) {
            Ok(Some(self.parse_expr()?))
        } else if skip_eq {
            Ok(Some(self.parse_expr()?))
        } else {
            Ok(None)
        }
    }

    /// Parses a local variable declaration.
    fn parse_local(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Local>> {
        let lo = self.prev_span;
        let pat = self.parse_top_level_pat()?;

        let (err, ty) = if self.eat(&token::Colon) {
            // Save the state of the parser before parsing type normally, in case there is a `:`
            // instead of an `=` typo.
            let parser_snapshot_before_type = self.clone();
            let colon_sp = self.prev_span;
            match self.parse_ty() {
                Ok(ty) => (None, Some(ty)),
                Err(mut err) => {
                    // Rewind to before attempting to parse the type and continue parsing
                    let parser_snapshot_after_type = self.clone();
                    mem::replace(self, parser_snapshot_before_type);

                    let snippet = self.span_to_snippet(pat.span).unwrap();
                    err.span_label(pat.span, format!("while parsing the type for `{}`", snippet));
                    (Some((parser_snapshot_after_type, colon_sp, err)), None)
                }
            }
        } else {
            (None, None)
        };
        let init = match (self.parse_initializer(err.is_some()), err) {
            (Ok(init), None) => {  // init parsed, ty parsed
                init
            }
            (Ok(init), Some((_, colon_sp, mut err))) => {  // init parsed, ty error
                // Could parse the type as if it were the initializer, it is likely there was a
                // typo in the code: `:` instead of `=`. Add suggestion and emit the error.
                err.span_suggestion_short(
                    colon_sp,
                    "use `=` if you meant to assign",
                    "=".to_string(),
                    Applicability::MachineApplicable
                );
                err.emit();
                // As this was parsed successfully, continue as if the code has been fixed for the
                // rest of the file. It will still fail due to the emitted error, but we avoid
                // extra noise.
                init
            }
            (Err(mut init_err), Some((snapshot, _, ty_err))) => {  // init error, ty error
                init_err.cancel();
                // Couldn't parse the type nor the initializer, only raise the type error and
                // return to the parser state before parsing the type as the initializer.
                // let x: <parse_error>;
                mem::replace(self, snapshot);
                return Err(ty_err);
            }
            (Err(err), None) => {  // init error, ty parsed
                // Couldn't parse the initializer and we're not attempting to recover a failed
                // parse of the type, return the error.
                return Err(err);
            }
        };
        let hi = if self.token == token::Semi {
            self.token.span
        } else {
            self.prev_span
        };
        Ok(P(ast::Local {
            ty,
            pat,
            init,
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            attrs,
        }))
    }

    /// Parse a statement. This stops just before trailing semicolons on everything but items.
    /// e.g., a `StmtKind::Semi` parses to a `StmtKind::Expr`, leaving the trailing `;` unconsumed.
    pub fn parse_stmt(&mut self) -> PResult<'a, Option<Stmt>> {
        Ok(self.parse_stmt_(true))
    }

    fn parse_stmt_(&mut self, macro_legacy_warnings: bool) -> Option<Stmt> {
        self.parse_stmt_without_recovery(macro_legacy_warnings).unwrap_or_else(|mut e| {
            e.emit();
            self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
            None
        })
    }

    fn is_async_fn(&self) -> bool {
        self.token.is_keyword(kw::Async) &&
            self.is_keyword_ahead(1, &[kw::Fn])
    }

    fn is_crate_vis(&self) -> bool {
        self.token.is_keyword(kw::Crate) && self.look_ahead(1, |t| t != &token::ModSep)
    }

    fn is_auto_trait_item(&self) -> bool {
        // auto trait
        (self.token.is_keyword(kw::Auto) &&
            self.is_keyword_ahead(1, &[kw::Trait]))
        || // unsafe auto trait
        (self.token.is_keyword(kw::Unsafe) &&
         self.is_keyword_ahead(1, &[kw::Auto]) &&
         self.is_keyword_ahead(2, &[kw::Trait]))
    }

    fn parse_stmt_without_recovery(
        &mut self,
        macro_legacy_warnings: bool,
    ) -> PResult<'a, Option<Stmt>> {
        maybe_whole!(self, NtStmt, |x| Some(x));

        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        Ok(Some(if self.eat_keyword(kw::Let) {
            Stmt {
                id: ast::DUMMY_NODE_ID,
                node: StmtKind::Local(self.parse_local(attrs.into())?),
                span: lo.to(self.prev_span),
            }
        } else if let Some(macro_def) = self.eat_macro_def(
            &attrs,
            &source_map::respan(lo, VisibilityKind::Inherited),
            lo,
        )? {
            Stmt {
                id: ast::DUMMY_NODE_ID,
                node: StmtKind::Item(macro_def),
                span: lo.to(self.prev_span),
            }
        // Starts like a simple path, being careful to avoid contextual keywords
        // such as a union items, item with `crate` visibility or auto trait items.
        // Our goal here is to parse an arbitrary path `a::b::c` but not something that starts
        // like a path (1 token), but it fact not a path.
        // `union::b::c` - path, `union U { ... }` - not a path.
        // `crate::b::c` - path, `crate struct S;` - not a path.
        } else if self.token.is_path_start() &&
                  !self.token.is_qpath_start() &&
                  !self.is_union_item() &&
                  !self.is_crate_vis() &&
                  !self.is_auto_trait_item() &&
                  !self.is_async_fn() {
            let path = self.parse_path(PathStyle::Expr)?;

            if !self.eat(&token::Not) {
                let expr = if self.check(&token::OpenDelim(token::Brace)) {
                    self.parse_struct_expr(lo, path, ThinVec::new())?
                } else {
                    let hi = self.prev_span;
                    self.mk_expr(lo.to(hi), ExprKind::Path(None, path), ThinVec::new())
                };

                let expr = self.with_res(Restrictions::STMT_EXPR, |this| {
                    let expr = this.parse_dot_or_call_expr_with(expr, lo, attrs.into())?;
                    this.parse_assoc_expr_with(0, LhsExpr::AlreadyParsed(expr))
                })?;

                return Ok(Some(Stmt {
                    id: ast::DUMMY_NODE_ID,
                    node: StmtKind::Expr(expr),
                    span: lo.to(self.prev_span),
                }));
            }

            let (delim, tts) = self.expect_delimited_token_tree()?;
            let hi = self.prev_span;

            let style = if delim == MacDelimiter::Brace {
                MacStmtStyle::Braces
            } else {
                MacStmtStyle::NoBraces
            };

            let mac = respan(lo.to(hi), Mac_ {
                path,
                tts,
                delim,
                prior_type_ascription: self.last_type_ascription,
            });
            let node = if delim == MacDelimiter::Brace ||
                          self.token == token::Semi || self.token == token::Eof {
                StmtKind::Mac(P((mac, style, attrs.into())))
            }
            // We used to incorrectly stop parsing macro-expanded statements here.
            // If the next token will be an error anyway but could have parsed with the
            // earlier behavior, stop parsing here and emit a warning to avoid breakage.
            else if macro_legacy_warnings &&
                    self.token.can_begin_expr() &&
                    match self.token.kind {
                // These can continue an expression, so we can't stop parsing and warn.
                token::OpenDelim(token::Paren) | token::OpenDelim(token::Bracket) |
                token::BinOp(token::Minus) | token::BinOp(token::Star) |
                token::BinOp(token::And) | token::BinOp(token::Or) |
                token::AndAnd | token::OrOr |
                token::DotDot | token::DotDotDot | token::DotDotEq => false,
                _ => true,
            } {
                self.warn_missing_semicolon();
                StmtKind::Mac(P((mac, style, attrs.into())))
            } else {
                let e = self.mk_expr(mac.span, ExprKind::Mac(mac), ThinVec::new());
                let e = self.maybe_recover_from_bad_qpath(e, true)?;
                let e = self.parse_dot_or_call_expr_with(e, lo, attrs.into())?;
                let e = self.parse_assoc_expr_with(0, LhsExpr::AlreadyParsed(e))?;
                StmtKind::Expr(e)
            };
            Stmt {
                id: ast::DUMMY_NODE_ID,
                span: lo.to(hi),
                node,
            }
        } else {
            // FIXME: Bad copy of attrs
            let old_directory_ownership =
                mem::replace(&mut self.directory.ownership, DirectoryOwnership::UnownedViaBlock);
            let item = self.parse_item_(attrs.clone(), false, true)?;
            self.directory.ownership = old_directory_ownership;

            match item {
                Some(i) => Stmt {
                    id: ast::DUMMY_NODE_ID,
                    span: lo.to(i.span),
                    node: StmtKind::Item(i),
                },
                None => {
                    let unused_attrs = |attrs: &[Attribute], s: &mut Self| {
                        if !attrs.is_empty() {
                            if s.prev_token_kind == PrevTokenKind::DocComment {
                                s.span_fatal_err(s.prev_span, Error::UselessDocComment).emit();
                            } else if attrs.iter().any(|a| a.style == AttrStyle::Outer) {
                                s.span_err(
                                    s.token.span, "expected statement after outer attribute"
                                );
                            }
                        }
                    };

                    // Do not attempt to parse an expression if we're done here.
                    if self.token == token::Semi {
                        unused_attrs(&attrs, self);
                        self.bump();
                        return Ok(None);
                    }

                    if self.token == token::CloseDelim(token::Brace) {
                        unused_attrs(&attrs, self);
                        return Ok(None);
                    }

                    // Remainder are line-expr stmts.
                    let e = self.parse_expr_res(
                        Restrictions::STMT_EXPR, Some(attrs.into()))?;
                    Stmt {
                        id: ast::DUMMY_NODE_ID,
                        span: lo.to(e.span),
                        node: StmtKind::Expr(e),
                    }
                }
            }
        }))
    }

    /// Parses a block. No inner attributes are allowed.
    pub fn parse_block(&mut self) -> PResult<'a, P<Block>> {
        maybe_whole!(self, NtBlock, |x| x);

        let lo = self.token.span;

        if !self.eat(&token::OpenDelim(token::Brace)) {
            let sp = self.token.span;
            let tok = self.this_token_descr();
            let mut e = self.span_fatal(sp, &format!("expected `{{`, found {}", tok));
            let do_not_suggest_help =
                self.token.is_keyword(kw::In) || self.token == token::Colon;

            if self.token.is_ident_named(sym::and) {
                e.span_suggestion_short(
                    self.token.span,
                    "use `&&` instead of `and` for the boolean operator",
                    "&&".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            if self.token.is_ident_named(sym::or) {
                e.span_suggestion_short(
                    self.token.span,
                    "use `||` instead of `or` for the boolean operator",
                    "||".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }

            // Check to see if the user has written something like
            //
            //    if (cond)
            //      bar;
            //
            // Which is valid in other languages, but not Rust.
            match self.parse_stmt_without_recovery(false) {
                Ok(Some(stmt)) => {
                    if self.look_ahead(1, |t| t == &token::OpenDelim(token::Brace))
                        || do_not_suggest_help {
                        // if the next token is an open brace (e.g., `if a b {`), the place-
                        // inside-a-block suggestion would be more likely wrong than right
                        e.span_label(sp, "expected `{`");
                        return Err(e);
                    }
                    let mut stmt_span = stmt.span;
                    // expand the span to include the semicolon, if it exists
                    if self.eat(&token::Semi) {
                        stmt_span = stmt_span.with_hi(self.prev_span.hi());
                    }
                    if let Ok(snippet) = self.span_to_snippet(stmt_span) {
                        e.span_suggestion(
                            stmt_span,
                            "try placing this code inside a block",
                            format!("{{ {} }}", snippet),
                            // speculative, has been misleading in the past (#46836)
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
                Err(mut e) => {
                    self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
                    self.cancel(&mut e);
                }
                _ => ()
            }
            e.span_label(sp, "expected `{`");
            return Err(e);
        }

        self.parse_block_tail(lo, BlockCheckMode::Default)
    }

    /// Parses a block. Inner attributes are allowed.
    crate fn parse_inner_attrs_and_block(&mut self) -> PResult<'a, (Vec<Attribute>, P<Block>)> {
        maybe_whole!(self, NtBlock, |x| (Vec::new(), x));

        let lo = self.token.span;
        self.expect(&token::OpenDelim(token::Brace))?;
        Ok((self.parse_inner_attributes()?,
            self.parse_block_tail(lo, BlockCheckMode::Default)?))
    }

    /// Parses the rest of a block expression or function body.
    /// Precondition: already parsed the '{'.
    fn parse_block_tail(&mut self, lo: Span, s: BlockCheckMode) -> PResult<'a, P<Block>> {
        let mut stmts = vec![];
        while !self.eat(&token::CloseDelim(token::Brace)) {
            if self.token == token::Eof {
                break;
            }
            let stmt = match self.parse_full_stmt(false) {
                Err(mut err) => {
                    err.emit();
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    Some(Stmt {
                        id: ast::DUMMY_NODE_ID,
                        node: StmtKind::Expr(DummyResult::raw_expr(self.token.span, true)),
                        span: self.token.span,
                    })
                }
                Ok(stmt) => stmt,
            };
            if let Some(stmt) = stmt {
                stmts.push(stmt);
            } else {
                // Found only `;` or `}`.
                continue;
            };
        }
        Ok(P(ast::Block {
            stmts,
            id: ast::DUMMY_NODE_ID,
            rules: s,
            span: lo.to(self.prev_span),
        }))
    }

    /// Parses a statement, including the trailing semicolon.
    crate fn parse_full_stmt(&mut self, macro_legacy_warnings: bool) -> PResult<'a, Option<Stmt>> {
        // skip looking for a trailing semicolon when we have an interpolated statement
        maybe_whole!(self, NtStmt, |x| Some(x));

        let mut stmt = match self.parse_stmt_without_recovery(macro_legacy_warnings)? {
            Some(stmt) => stmt,
            None => return Ok(None),
        };

        match stmt.node {
            StmtKind::Expr(ref expr) if self.token != token::Eof => {
                // expression without semicolon
                if classify::expr_requires_semi_to_be_stmt(expr) {
                    // Just check for errors and recover; do not eat semicolon yet.
                    if let Err(mut e) =
                        self.expect_one_of(&[], &[token::Semi, token::CloseDelim(token::Brace)])
                    {
                        e.emit();
                        self.recover_stmt();
                        // Don't complain about type errors in body tail after parse error (#57383).
                        let sp = expr.span.to(self.prev_span);
                        stmt.node = StmtKind::Expr(DummyResult::raw_expr(sp, true));
                    }
                }
            }
            StmtKind::Local(..) => {
                // We used to incorrectly allow a macro-expanded let statement to lack a semicolon.
                if macro_legacy_warnings && self.token != token::Semi {
                    self.warn_missing_semicolon();
                } else {
                    self.expect_one_of(&[], &[token::Semi])?;
                }
            }
            _ => {}
        }

        if self.eat(&token::Semi) {
            stmt = stmt.add_trailing_semicolon();
        }
        stmt.span = stmt.span.to(self.prev_span);
        Ok(Some(stmt))
    }

    fn warn_missing_semicolon(&self) {
        self.diagnostic().struct_span_warn(self.token.span, {
            &format!("expected `;`, found {}", self.this_token_descr())
        }).note({
            "This was erroneously allowed and will become a hard error in a future release"
        }).emit();
    }

    /// Parses bounds of a type parameter `BOUND + BOUND + ...`, possibly with trailing `+`.
    ///
    /// ```
    /// BOUND = TY_BOUND | LT_BOUND
    /// LT_BOUND = LIFETIME (e.g., `'a`)
    /// TY_BOUND = TY_BOUND_NOPAREN | (TY_BOUND_NOPAREN)
    /// TY_BOUND_NOPAREN = [?] [for<LT_PARAM_DEFS>] SIMPLE_PATH (e.g., `?for<'a: 'b> m::Trait<'a>`)
    /// ```
    fn parse_generic_bounds_common(&mut self,
                                   allow_plus: bool,
                                   colon_span: Option<Span>) -> PResult<'a, GenericBounds> {
        let mut bounds = Vec::new();
        let mut negative_bounds = Vec::new();
        let mut last_plus_span = None;
        let mut was_negative = false;
        loop {
            // This needs to be synchronized with `TokenKind::can_begin_bound`.
            let is_bound_start = self.check_path() || self.check_lifetime() ||
                                 self.check(&token::Not) || // used for error reporting only
                                 self.check(&token::Question) ||
                                 self.check_keyword(kw::For) ||
                                 self.check(&token::OpenDelim(token::Paren));
            if is_bound_start {
                let lo = self.token.span;
                let has_parens = self.eat(&token::OpenDelim(token::Paren));
                let inner_lo = self.token.span;
                let is_negative = self.eat(&token::Not);
                let question = if self.eat(&token::Question) { Some(self.prev_span) } else { None };
                if self.token.is_lifetime() {
                    if let Some(question_span) = question {
                        self.span_err(question_span,
                                      "`?` may only modify trait bounds, not lifetime bounds");
                    }
                    bounds.push(GenericBound::Outlives(self.expect_lifetime()));
                    if has_parens {
                        let inner_span = inner_lo.to(self.prev_span);
                        self.expect(&token::CloseDelim(token::Paren))?;
                        let mut err = self.struct_span_err(
                            lo.to(self.prev_span),
                            "parenthesized lifetime bounds are not supported"
                        );
                        if let Ok(snippet) = self.span_to_snippet(inner_span) {
                            err.span_suggestion_short(
                                lo.to(self.prev_span),
                                "remove the parentheses",
                                snippet.to_owned(),
                                Applicability::MachineApplicable
                            );
                        }
                        err.emit();
                    }
                } else {
                    let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
                    let path = self.parse_path(PathStyle::Type)?;
                    if has_parens {
                        self.expect(&token::CloseDelim(token::Paren))?;
                    }
                    let poly_span = lo.to(self.prev_span);
                    if is_negative {
                        was_negative = true;
                        if let Some(sp) = last_plus_span.or(colon_span) {
                            negative_bounds.push(sp.to(poly_span));
                        }
                    } else {
                        let poly_trait = PolyTraitRef::new(lifetime_defs, path, poly_span);
                        let modifier = if question.is_some() {
                            TraitBoundModifier::Maybe
                        } else {
                            TraitBoundModifier::None
                        };
                        bounds.push(GenericBound::Trait(poly_trait, modifier));
                    }
                }
            } else {
                break
            }

            if !allow_plus || !self.eat_plus() {
                break
            } else {
                last_plus_span = Some(self.prev_span);
            }
        }

        if !negative_bounds.is_empty() || was_negative {
            let plural = negative_bounds.len() > 1;
            let last_span = negative_bounds.last().map(|sp| *sp);
            let mut err = self.struct_span_err(
                negative_bounds,
                "negative trait bounds are not supported",
            );
            if let Some(sp) = last_span {
                err.span_label(sp, "negative trait bounds are not supported");
            }
            if let Some(bound_list) = colon_span {
                let bound_list = bound_list.to(self.prev_span);
                let mut new_bound_list = String::new();
                if !bounds.is_empty() {
                    let mut snippets = bounds.iter().map(|bound| bound.span())
                        .map(|span| self.span_to_snippet(span));
                    while let Some(Ok(snippet)) = snippets.next() {
                        new_bound_list.push_str(" + ");
                        new_bound_list.push_str(&snippet);
                    }
                    new_bound_list = new_bound_list.replacen(" +", ":", 1);
                }
                err.span_suggestion_hidden(
                    bound_list,
                    &format!("remove the trait bound{}", if plural { "s" } else { "" }),
                    new_bound_list,
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        }

        return Ok(bounds);
    }

    crate fn parse_generic_bounds(&mut self,
                                  colon_span: Option<Span>) -> PResult<'a, GenericBounds> {
        self.parse_generic_bounds_common(true, colon_span)
    }

    /// Parses bounds of a lifetime parameter `BOUND + BOUND + BOUND`, possibly with trailing `+`.
    ///
    /// ```
    /// BOUND = LT_BOUND (e.g., `'a`)
    /// ```
    fn parse_lt_param_bounds(&mut self) -> GenericBounds {
        let mut lifetimes = Vec::new();
        while self.check_lifetime() {
            lifetimes.push(ast::GenericBound::Outlives(self.expect_lifetime()));

            if !self.eat_plus() {
                break
            }
        }
        lifetimes
    }

    /// Matches `typaram = IDENT (`?` unbound)? optbounds ( EQ ty )?`.
    fn parse_ty_param(&mut self,
                      preceding_attrs: Vec<Attribute>)
                      -> PResult<'a, GenericParam> {
        let ident = self.parse_ident()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_generic_bounds(Some(self.prev_span))?
        } else {
            Vec::new()
        };

        let default = if self.eat(&token::Eq) {
            Some(self.parse_ty()?)
        } else {
            None
        };

        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs.into(),
            bounds,
            kind: GenericParamKind::Type {
                default,
            }
        })
    }

    fn parse_const_param(&mut self, preceding_attrs: Vec<Attribute>) -> PResult<'a, GenericParam> {
        self.expect_keyword(kw::Const)?;
        let ident = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;

        Ok(GenericParam {
            ident,
            id: ast::DUMMY_NODE_ID,
            attrs: preceding_attrs.into(),
            bounds: Vec::new(),
            kind: GenericParamKind::Const {
                ty,
            }
        })
    }

    /// Parses a (possibly empty) list of lifetime and type parameters, possibly including
    /// a trailing comma and erroneous trailing attributes.
    crate fn parse_generic_params(&mut self) -> PResult<'a, Vec<ast::GenericParam>> {
        let mut params = Vec::new();
        loop {
            let attrs = self.parse_outer_attributes()?;
            if self.check_lifetime() {
                let lifetime = self.expect_lifetime();
                // Parse lifetime parameter.
                let bounds = if self.eat(&token::Colon) {
                    self.parse_lt_param_bounds()
                } else {
                    Vec::new()
                };
                params.push(ast::GenericParam {
                    ident: lifetime.ident,
                    id: lifetime.id,
                    attrs: attrs.into(),
                    bounds,
                    kind: ast::GenericParamKind::Lifetime,
                });
            } else if self.check_keyword(kw::Const) {
                // Parse const parameter.
                params.push(self.parse_const_param(attrs)?);
            } else if self.check_ident() {
                // Parse type parameter.
                params.push(self.parse_ty_param(attrs)?);
            } else {
                // Check for trailing attributes and stop parsing.
                if !attrs.is_empty() {
                    if !params.is_empty() {
                        self.struct_span_err(
                            attrs[0].span,
                            &format!("trailing attribute after generic parameter"),
                        )
                        .span_label(attrs[0].span, "attributes must go before parameters")
                        .emit();
                    } else {
                        self.struct_span_err(
                            attrs[0].span,
                            &format!("attribute without generic parameters"),
                        )
                        .span_label(
                            attrs[0].span,
                            "attributes are only permitted when preceding parameters",
                        )
                        .emit();
                    }
                }
                break
            }

            if !self.eat(&token::Comma) {
                break
            }
        }
        Ok(params)
    }

    /// Parses a set of optional generic type parameter declarations. Where
    /// clauses are not parsed here, and must be added later via
    /// `parse_where_clause()`.
    ///
    /// matches generics = ( ) | ( < > ) | ( < typaramseq ( , )? > ) | ( < lifetimes ( , )? > )
    ///                  | ( < lifetimes , typaramseq ( , )? > )
    /// where   typaramseq = ( typaram ) | ( typaram , typaramseq )
    fn parse_generics(&mut self) -> PResult<'a, ast::Generics> {
        let span_lo = self.token.span;
        let (params, span) = if self.eat_lt() {
            let params = self.parse_generic_params()?;
            self.expect_gt()?;
            (params, span_lo.to(self.prev_span))
        } else {
            (vec![], self.prev_span.between(self.token.span))
        };
        Ok(ast::Generics {
            params,
            where_clause: WhereClause {
                predicates: Vec::new(),
                span: DUMMY_SP,
            },
            span,
        })
    }

    /// Parses generic args (within a path segment) with recovery for extra leading angle brackets.
    /// For the purposes of understanding the parsing logic of generic arguments, this function
    /// can be thought of being the same as just calling `self.parse_generic_args()` if the source
    /// had the correct amount of leading angle brackets.
    ///
    /// ```ignore (diagnostics)
    /// bar::<<<<T as Foo>::Output>();
    ///      ^^ help: remove extra angle brackets
    /// ```
    fn parse_generic_args_with_leaning_angle_bracket_recovery(
        &mut self,
        style: PathStyle,
        lo: Span,
    ) -> PResult<'a, (Vec<GenericArg>, Vec<AssocTyConstraint>)> {
        // We need to detect whether there are extra leading left angle brackets and produce an
        // appropriate error and suggestion. This cannot be implemented by looking ahead at
        // upcoming tokens for a matching `>` character - if there are unmatched `<` tokens
        // then there won't be matching `>` tokens to find.
        //
        // To explain how this detection works, consider the following example:
        //
        // ```ignore (diagnostics)
        // bar::<<<<T as Foo>::Output>();
        //      ^^ help: remove extra angle brackets
        // ```
        //
        // Parsing of the left angle brackets starts in this function. We start by parsing the
        // `<` token (incrementing the counter of unmatched angle brackets on `Parser` via
        // `eat_lt`):
        //
        // *Upcoming tokens:* `<<<<T as Foo>::Output>;`
        // *Unmatched count:* 1
        // *`parse_path_segment` calls deep:* 0
        //
        // This has the effect of recursing as this function is called if a `<` character
        // is found within the expected generic arguments:
        //
        // *Upcoming tokens:* `<<<T as Foo>::Output>;`
        // *Unmatched count:* 2
        // *`parse_path_segment` calls deep:* 1
        //
        // Eventually we will have recursed until having consumed all of the `<` tokens and
        // this will be reflected in the count:
        //
        // *Upcoming tokens:* `T as Foo>::Output>;`
        // *Unmatched count:* 4
        // `parse_path_segment` calls deep:* 3
        //
        // The parser will continue until reaching the first `>` - this will decrement the
        // unmatched angle bracket count and return to the parent invocation of this function
        // having succeeded in parsing:
        //
        // *Upcoming tokens:* `::Output>;`
        // *Unmatched count:* 3
        // *`parse_path_segment` calls deep:* 2
        //
        // This will continue until the next `>` character which will also return successfully
        // to the parent invocation of this function and decrement the count:
        //
        // *Upcoming tokens:* `;`
        // *Unmatched count:* 2
        // *`parse_path_segment` calls deep:* 1
        //
        // At this point, this function will expect to find another matching `>` character but
        // won't be able to and will return an error. This will continue all the way up the
        // call stack until the first invocation:
        //
        // *Upcoming tokens:* `;`
        // *Unmatched count:* 2
        // *`parse_path_segment` calls deep:* 0
        //
        // In doing this, we have managed to work out how many unmatched leading left angle
        // brackets there are, but we cannot recover as the unmatched angle brackets have
        // already been consumed. To remedy this, we keep a snapshot of the parser state
        // before we do the above. We can then inspect whether we ended up with a parsing error
        // and unmatched left angle brackets and if so, restore the parser state before we
        // consumed any `<` characters to emit an error and consume the erroneous tokens to
        // recover by attempting to parse again.
        //
        // In practice, the recursion of this function is indirect and there will be other
        // locations that consume some `<` characters - as long as we update the count when
        // this happens, it isn't an issue.

        let is_first_invocation = style == PathStyle::Expr;
        // Take a snapshot before attempting to parse - we can restore this later.
        let snapshot = if is_first_invocation {
            Some(self.clone())
        } else {
            None
        };

        debug!("parse_generic_args_with_leading_angle_bracket_recovery: (snapshotting)");
        match self.parse_generic_args() {
            Ok(value) => Ok(value),
            Err(ref mut e) if is_first_invocation && self.unmatched_angle_bracket_count > 0 => {
                // Cancel error from being unable to find `>`. We know the error
                // must have been this due to a non-zero unmatched angle bracket
                // count.
                e.cancel();

                // Swap `self` with our backup of the parser state before attempting to parse
                // generic arguments.
                let snapshot = mem::replace(self, snapshot.unwrap());

                debug!(
                    "parse_generic_args_with_leading_angle_bracket_recovery: (snapshot failure) \
                     snapshot.count={:?}",
                    snapshot.unmatched_angle_bracket_count,
                );

                // Eat the unmatched angle brackets.
                for _ in 0..snapshot.unmatched_angle_bracket_count {
                    self.eat_lt();
                }

                // Make a span over ${unmatched angle bracket count} characters.
                let span = lo.with_hi(
                    lo.lo() + BytePos(snapshot.unmatched_angle_bracket_count)
                );
                let plural = snapshot.unmatched_angle_bracket_count > 1;
                self.diagnostic()
                    .struct_span_err(
                        span,
                        &format!(
                            "unmatched angle bracket{}",
                            if plural { "s" } else { "" }
                        ),
                    )
                    .span_suggestion(
                        span,
                        &format!(
                            "remove extra angle bracket{}",
                            if plural { "s" } else { "" }
                        ),
                        String::new(),
                        Applicability::MachineApplicable,
                    )
                    .emit();

                // Try again without unmatched angle bracket characters.
                self.parse_generic_args()
            },
            Err(e) => Err(e),
        }
    }

    /// Parses (possibly empty) list of lifetime and type arguments and associated type bindings,
    /// possibly including trailing comma.
    fn parse_generic_args(&mut self) -> PResult<'a, (Vec<GenericArg>, Vec<AssocTyConstraint>)> {
        let mut args = Vec::new();
        let mut constraints = Vec::new();
        let mut misplaced_assoc_ty_constraints: Vec<Span> = Vec::new();
        let mut assoc_ty_constraints: Vec<Span> = Vec::new();

        let args_lo = self.token.span;

        loop {
            if self.check_lifetime() && self.look_ahead(1, |t| !t.is_like_plus()) {
                // Parse lifetime argument.
                args.push(GenericArg::Lifetime(self.expect_lifetime()));
                misplaced_assoc_ty_constraints.append(&mut assoc_ty_constraints);
            } else if self.check_ident() && self.look_ahead(1,
                    |t| t == &token::Eq || t == &token::Colon) {
                // Parse associated type constraint.
                let lo = self.token.span;
                let ident = self.parse_ident()?;
                let kind = if self.eat(&token::Eq) {
                    AssocTyConstraintKind::Equality {
                        ty: self.parse_ty()?,
                    }
                } else if self.eat(&token::Colon) {
                    AssocTyConstraintKind::Bound {
                        bounds: self.parse_generic_bounds(Some(self.prev_span))?,
                    }
                } else {
                    unreachable!();
                };
                let span = lo.to(self.prev_span);
                constraints.push(AssocTyConstraint {
                    id: ast::DUMMY_NODE_ID,
                    ident,
                    kind,
                    span,
                });
                assoc_ty_constraints.push(span);
            } else if self.check_const_arg() {
                // Parse const argument.
                let expr = if let token::OpenDelim(token::Brace) = self.token.kind {
                    self.parse_block_expr(
                        None, self.token.span, BlockCheckMode::Default, ThinVec::new()
                    )?
                } else if self.token.is_ident() {
                    // FIXME(const_generics): to distinguish between idents for types and consts,
                    // we should introduce a GenericArg::Ident in the AST and distinguish when
                    // lowering to the HIR. For now, idents for const args are not permitted.
                    if self.token.is_keyword(kw::True) || self.token.is_keyword(kw::False) {
                        self.parse_literal_maybe_minus()?
                    } else {
                        return Err(
                            self.fatal("identifiers may currently not be used for const generics")
                        );
                    }
                } else {
                    self.parse_literal_maybe_minus()?
                };
                let value = AnonConst {
                    id: ast::DUMMY_NODE_ID,
                    value: expr,
                };
                args.push(GenericArg::Const(value));
                misplaced_assoc_ty_constraints.append(&mut assoc_ty_constraints);
            } else if self.check_type() {
                // Parse type argument.
                args.push(GenericArg::Type(self.parse_ty()?));
                misplaced_assoc_ty_constraints.append(&mut assoc_ty_constraints);
            } else {
                break
            }

            if !self.eat(&token::Comma) {
                break
            }
        }

        // FIXME: we would like to report this in ast_validation instead, but we currently do not
        // preserve ordering of generic parameters with respect to associated type binding, so we
        // lose that information after parsing.
        if misplaced_assoc_ty_constraints.len() > 0 {
            let mut err = self.struct_span_err(
                args_lo.to(self.prev_span),
                "associated type bindings must be declared after generic parameters",
            );
            for span in misplaced_assoc_ty_constraints {
                err.span_label(
                    span,
                    "this associated type binding should be moved after the generic parameters",
                );
            }
            err.emit();
        }

        Ok((args, constraints))
    }

    /// Parses an optional where-clause and places it in `generics`.
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// where T : Trait<U, V> + 'b, 'a : 'b
    /// ```
    fn parse_where_clause(&mut self) -> PResult<'a, WhereClause> {
        let mut where_clause = WhereClause {
            predicates: Vec::new(),
            span: self.prev_span.to(self.prev_span),
        };

        if !self.eat_keyword(kw::Where) {
            return Ok(where_clause);
        }
        let lo = self.prev_span;

        // We are considering adding generics to the `where` keyword as an alternative higher-rank
        // parameter syntax (as in `where<'a>` or `where<T>`. To avoid that being a breaking
        // change we parse those generics now, but report an error.
        if self.choose_generics_over_qpath() {
            let generics = self.parse_generics()?;
            self.struct_span_err(
                generics.span,
                "generic parameters on `where` clauses are reserved for future use",
            )
                .span_label(generics.span, "currently unsupported")
                .emit();
        }

        loop {
            let lo = self.token.span;
            if self.check_lifetime() && self.look_ahead(1, |t| !t.is_like_plus()) {
                let lifetime = self.expect_lifetime();
                // Bounds starting with a colon are mandatory, but possibly empty.
                self.expect(&token::Colon)?;
                let bounds = self.parse_lt_param_bounds();
                where_clause.predicates.push(ast::WherePredicate::RegionPredicate(
                    ast::WhereRegionPredicate {
                        span: lo.to(self.prev_span),
                        lifetime,
                        bounds,
                    }
                ));
            } else if self.check_type() {
                // Parse optional `for<'a, 'b>`.
                // This `for` is parsed greedily and applies to the whole predicate,
                // the bounded type can have its own `for` applying only to it.
                // Examples:
                // * `for<'a> Trait1<'a>: Trait2<'a /* ok */>`
                // * `(for<'a> Trait1<'a>): Trait2<'a /* not ok */>`
                // * `for<'a> for<'b> Trait1<'a, 'b>: Trait2<'a /* ok */, 'b /* not ok */>`
                let lifetime_defs = self.parse_late_bound_lifetime_defs()?;

                // Parse type with mandatory colon and (possibly empty) bounds,
                // or with mandatory equality sign and the second type.
                let ty = self.parse_ty()?;
                if self.eat(&token::Colon) {
                    let bounds = self.parse_generic_bounds(Some(self.prev_span))?;
                    where_clause.predicates.push(ast::WherePredicate::BoundPredicate(
                        ast::WhereBoundPredicate {
                            span: lo.to(self.prev_span),
                            bound_generic_params: lifetime_defs,
                            bounded_ty: ty,
                            bounds,
                        }
                    ));
                // FIXME: Decide what should be used here, `=` or `==`.
                // FIXME: We are just dropping the binders in lifetime_defs on the floor here.
                } else if self.eat(&token::Eq) || self.eat(&token::EqEq) {
                    let rhs_ty = self.parse_ty()?;
                    where_clause.predicates.push(ast::WherePredicate::EqPredicate(
                        ast::WhereEqPredicate {
                            span: lo.to(self.prev_span),
                            lhs_ty: ty,
                            rhs_ty,
                            id: ast::DUMMY_NODE_ID,
                        }
                    ));
                } else {
                    return self.unexpected();
                }
            } else {
                break
            }

            if !self.eat(&token::Comma) {
                break
            }
        }

        where_clause.span = lo.to(self.prev_span);
        Ok(where_clause)
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

        if c_variadic && args.is_empty() {
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

    /// Parses the `|arg, arg|` header of a closure.
    fn parse_fn_block_decl(&mut self) -> PResult<'a, P<FnDecl>> {
        let inputs_captures = {
            if self.eat(&token::OrOr) {
                Vec::new()
            } else {
                self.expect(&token::BinOp(token::Or))?;
                let args = self.parse_seq_to_before_tokens(
                    &[&token::BinOp(token::Or), &token::OrOr],
                    SeqSep::trailing_allowed(token::Comma),
                    TokenExpectType::NoExpect,
                    |p| p.parse_fn_block_arg()
                )?.0;
                self.expect_or()?;
                args
            }
        };
        let output = self.parse_ret_ty(true)?;

        Ok(P(FnDecl {
            inputs: inputs_captures,
            output,
            c_variadic: false
        }))
    }

    fn choose_generics_over_qpath(&self) -> bool {
        // There's an ambiguity between generic parameters and qualified paths in impls.
        // If we see `<` it may start both, so we have to inspect some following tokens.
        // The following combinations can only start generics,
        // but not qualified paths (with one exception):
        //     `<` `>` - empty generic parameters
        //     `<` `#` - generic parameters with attributes
        //     `<` (LIFETIME|IDENT) `>` - single generic parameter
        //     `<` (LIFETIME|IDENT) `,` - first generic parameter in a list
        //     `<` (LIFETIME|IDENT) `:` - generic parameter with bounds
        //     `<` (LIFETIME|IDENT) `=` - generic parameter with a default
        //     `<` const                - generic const parameter
        // The only truly ambiguous case is
        //     `<` IDENT `>` `::` IDENT ...
        // we disambiguate it in favor of generics (`impl<T> ::absolute::Path<T> { ... }`)
        // because this is what almost always expected in practice, qualified paths in impls
        // (`impl <Type>::AssocTy { ... }`) aren't even allowed by type checker at the moment.
        self.token == token::Lt &&
            (self.look_ahead(1, |t| t == &token::Pound || t == &token::Gt) ||
             self.look_ahead(1, |t| t.is_lifetime() || t.is_ident()) &&
                self.look_ahead(2, |t| t == &token::Gt || t == &token::Comma ||
                                       t == &token::Colon || t == &token::Eq) ||
            self.is_keyword_ahead(1, &[kw::Const]))
    }

    fn parse_late_bound_lifetime_defs(&mut self) -> PResult<'a, Vec<GenericParam>> {
        if self.eat_keyword(kw::For) {
            self.expect_lt()?;
            let params = self.parse_generic_params()?;
            self.expect_gt()?;
            // We rely on AST validation to rule out invalid cases: There must not be type
            // parameters, and the lifetime parameters must not have bounds.
            Ok(params)
        } else {
            Ok(Vec::new())
        }
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
