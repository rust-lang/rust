// ignore-tidy-filelength

use crate::ast::{AngleBracketedArgs, ParenthesizedArgs, AttrStyle, BareFnTy};
use crate::ast::{GenericBound, TraitBoundModifier};
use crate::ast::Unsafety;
use crate::ast::{Mod, AnonConst, Arg, Arm, Attribute, BindingMode, TraitItemKind};
use crate::ast::Block;
use crate::ast::{BlockCheckMode, CaptureBy, Movability};
use crate::ast::{Constness, Crate};
use crate::ast::Defaultness;
use crate::ast::EnumDef;
use crate::ast::{Expr, ExprKind, RangeLimits};
use crate::ast::{Field, FnDecl, FnHeader};
use crate::ast::{ForeignItem, ForeignItemKind, FunctionRetTy};
use crate::ast::{GenericParam, GenericParamKind};
use crate::ast::GenericArg;
use crate::ast::{Ident, ImplItem, IsAsync, IsAuto, Item, ItemKind};
use crate::ast::{Label, Lifetime};
use crate::ast::Local;
use crate::ast::MacStmtStyle;
use crate::ast::{Mac, Mac_, MacDelimiter};
use crate::ast::{MutTy, Mutability};
use crate::ast::{Pat, PatKind, PathSegment};
use crate::ast::{PolyTraitRef, QSelf};
use crate::ast::{Stmt, StmtKind};
use crate::ast::{VariantData, StructField};
use crate::ast::StrStyle;
use crate::ast::SelfKind;
use crate::ast::{TraitItem, TraitRef, TraitObjectSyntax};
use crate::ast::{Ty, TyKind, AssocTyConstraint, AssocTyConstraintKind, GenericBounds};
use crate::ast::{Visibility, VisibilityKind, WhereClause, CrateSugar};
use crate::ast::{UseTree, UseTreeKind};
use crate::ast::{BinOpKind, UnOp};
use crate::ast::{RangeEnd, RangeSyntax};
use crate::{ast, attr};
use crate::ext::base::DummyResult;
use crate::ext::hygiene::SyntaxContext;
use crate::source_map::{self, SourceMap, Spanned, respan};
use crate::parse::{SeqSep, classify, literal, token};
use crate::parse::lexer::UnmatchedBrace;
use crate::parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use crate::parse::token::{Token, TokenKind, DelimToken};
use crate::parse::{new_sub_parser_from_file, ParseSess, Directory, DirectoryOwnership};
use crate::util::parser::{AssocOp, Fixity, prec_let_scrutinee_needs_par};
use crate::print::pprust;
use crate::ptr::P;
use crate::parse::PResult;
use crate::ThinVec;
use crate::tokenstream::{self, DelimSpan, TokenTree, TokenStream, TreeAndJoint};
use crate::symbol::{kw, sym, Symbol};
use crate::parse::diagnostics::{Error, dummy_arg};

use errors::{Applicability, DiagnosticBuilder, DiagnosticId, FatalError};
use rustc_target::spec::abi::{self, Abi};
use syntax_pos::{Span, BytePos, DUMMY_SP, FileName};
use log::debug;

use std::borrow::Cow;
use std::cmp;
use std::mem;
use std::path::{self, Path, PathBuf};
use std::slice;

#[derive(Debug)]
/// Whether the type alias or associated type is a concrete type or an existential type
pub enum AliasKind {
    /// Just a new name for the same type
    Weak(P<Ty>),
    /// Only trait impls of the type will be usable, not the actual type itself
    Existential(GenericBounds),
}

bitflags::bitflags! {
    struct Restrictions: u8 {
        const STMT_EXPR         = 1 << 0;
        const NO_STRUCT_LITERAL = 1 << 1;
    }
}

type ItemInfo = (Ident, ItemKind, Option<Vec<Attribute>>);

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

/// Possibly accepts an `token::Interpolated` expression (a pre-parsed expression
/// dropped into the token stream, which happens while parsing the result of
/// macro expansion). Placement of these is not as complex as I feared it would
/// be. The important thing is to make sure that lookahead doesn't balk at
/// `token::Interpolated` tokens.
macro_rules! maybe_whole_expr {
    ($p:expr) => {
        if let token::Interpolated(nt) = &$p.token.kind {
            match &**nt {
                token::NtExpr(e) | token::NtLiteral(e) => {
                    let e = e.clone();
                    $p.bump();
                    return Ok(e);
                }
                token::NtPath(path) => {
                    let path = path.clone();
                    $p.bump();
                    return Ok($p.mk_expr(
                        $p.token.span, ExprKind::Path(None, path), ThinVec::new()
                    ));
                }
                token::NtBlock(block) => {
                    let block = block.clone();
                    $p.bump();
                    return Ok($p.mk_expr(
                        $p.token.span, ExprKind::Block(block, None), ThinVec::new()
                    ));
                }
                _ => {},
            };
        }
    }
}

/// As maybe_whole_expr, but for things other than expressions
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

/// Information about the path to a module.
pub struct ModulePath {
    name: String,
    path_exists: bool,
    pub result: Result<ModulePathSuccess, Error>,
}

pub struct ModulePathSuccess {
    pub path: PathBuf,
    pub directory_ownership: DirectoryOwnership,
    warn: bool,
}

#[derive(Debug)]
enum LhsExpr {
    NotYetParsed,
    AttributesParsed(ThinVec<Attribute>),
    AlreadyParsed(P<Expr>),
}

impl From<Option<ThinVec<Attribute>>> for LhsExpr {
    fn from(o: Option<ThinVec<Attribute>>) -> Self {
        if let Some(attrs) = o {
            LhsExpr::AttributesParsed(attrs)
        } else {
            LhsExpr::NotYetParsed
        }
    }
}

impl From<P<Expr>> for LhsExpr {
    fn from(expr: P<Expr>) -> Self {
        LhsExpr::AlreadyParsed(expr)
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

    /// Returns the span of expr, if it was not interpolated or the span of the interpolated token.
    fn interpolated_or_expr_span(
        &self,
        expr: PResult<'a, P<Expr>>,
    ) -> PResult<'a, (Span, P<Expr>)> {
        expr.map(|e| {
            if self.prev_token_kind == PrevTokenKind::Interpolated {
                (self.prev_span, e)
            } else {
                (e.span, e)
            }
        })
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
    pub fn parse_seq_to_end<T, F>(&mut self,
                                  ket: &TokenKind,
                                  sep: SeqSep,
                                  f: F)
                                  -> PResult<'a, Vec<T>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a,  T>,
    {
        let (val, recovered) = self.parse_seq_to_before_end(ket, sep, f)?;
        if !recovered {
            self.bump();
        }
        Ok(val)
    }

    /// Parses a sequence, not including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_before_end<T, F>(
        &mut self,
        ket: &TokenKind,
        sep: SeqSep,
        f: F,
    ) -> PResult<'a, (Vec<T>, bool)>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, T>
    {
        self.parse_seq_to_before_tokens(&[ket], sep, TokenExpectType::Expect, f)
    }

    crate fn parse_seq_to_before_tokens<T, F>(
        &mut self,
        kets: &[&TokenKind],
        sep: SeqSep,
        expect: TokenExpectType,
        mut f: F,
    ) -> PResult<'a, (Vec<T>, bool /* recovered */)>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, T>
    {
        let mut first = true;
        let mut recovered = false;
        let mut v = vec![];
        while !kets.iter().any(|k| {
                match expect {
                    TokenExpectType::Expect => self.check(k),
                    TokenExpectType::NoExpect => self.token == **k,
                }
            }) {
            match self.token.kind {
                token::CloseDelim(..) | token::Eof => break,
                _ => {}
            };
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
            if sep.trailing_sep_allowed && kets.iter().any(|k| {
                match expect {
                    TokenExpectType::Expect => self.check(k),
                    TokenExpectType::NoExpect => self.token == **k,
                }
            }) {
                break;
            }

            let t = f(self)?;
            v.push(t);
        }

        Ok((v, recovered))
    }

    /// Parses a sequence, including the closing delimiter. The function
    /// `f` must consume tokens until reaching the next separator or
    /// closing bracket.
    fn parse_unspanned_seq<T, F>(
        &mut self,
        bra: &TokenKind,
        ket: &TokenKind,
        sep: SeqSep,
        f: F,
    ) -> PResult<'a, Vec<T>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        self.expect(bra)?;
        let (result, recovered) = self.parse_seq_to_before_end(ket, sep, f)?;
        if !recovered {
            self.eat(ket);
        }
        Ok(result)
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

    /// Parses the items in a trait declaration.
    pub fn parse_trait_item(&mut self, at_end: &mut bool) -> PResult<'a, TraitItem> {
        maybe_whole!(self, NtTraitItem, |x| x);
        let attrs = self.parse_outer_attributes()?;
        let mut unclosed_delims = vec![];
        let (mut item, tokens) = self.collect_tokens(|this| {
            let item = this.parse_trait_item_(at_end, attrs);
            unclosed_delims.append(&mut this.unclosed_delims);
            item
        })?;
        self.unclosed_delims.append(&mut unclosed_delims);
        // See `parse_item` for why this clause is here.
        if !item.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
            item.tokens = Some(tokens);
        }
        Ok(item)
    }

    fn parse_trait_item_(&mut self,
                         at_end: &mut bool,
                         mut attrs: Vec<Attribute>) -> PResult<'a, TraitItem> {
        let lo = self.token.span;
        self.eat_bad_pub();
        let (name, node, generics) = if self.eat_keyword(kw::Type) {
            self.parse_trait_item_assoc_ty()?
        } else if self.is_const_item() {
            self.expect_keyword(kw::Const)?;
            let ident = self.parse_ident()?;
            self.expect(&token::Colon)?;
            let ty = self.parse_ty()?;
            let default = if self.eat(&token::Eq) {
                let expr = self.parse_expr()?;
                self.expect(&token::Semi)?;
                Some(expr)
            } else {
                self.expect(&token::Semi)?;
                None
            };
            (ident, TraitItemKind::Const(ty, default), ast::Generics::default())
        } else if let Some(mac) = self.parse_assoc_macro_invoc("trait", None, &mut false)? {
            // trait item macro.
            (Ident::invalid(), ast::TraitItemKind::Macro(mac), ast::Generics::default())
        } else {
            let (constness, unsafety, asyncness, abi) = self.parse_fn_front_matter()?;

            let ident = self.parse_ident()?;
            let mut generics = self.parse_generics()?;

            let decl = self.parse_fn_decl_with_self(|p: &mut Parser<'a>| {
                // This is somewhat dubious; We don't want to allow
                // argument names to be left off if there is a
                // definition...

                // We don't allow argument names to be left off in edition 2018.
                let is_name_required = p.token.span.rust_2018();
                p.parse_arg_general(true, false, |_| is_name_required)
            })?;
            generics.where_clause = self.parse_where_clause()?;

            let sig = ast::MethodSig {
                header: FnHeader {
                    unsafety,
                    constness,
                    abi,
                    asyncness,
                },
                decl,
            };

            let body = match self.token.kind {
                token::Semi => {
                    self.bump();
                    *at_end = true;
                    debug!("parse_trait_methods(): parsing required method");
                    None
                }
                token::OpenDelim(token::Brace) => {
                    debug!("parse_trait_methods(): parsing provided method");
                    *at_end = true;
                    let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
                    attrs.extend(inner_attrs.iter().cloned());
                    Some(body)
                }
                token::Interpolated(ref nt) => {
                    match **nt {
                        token::NtBlock(..) => {
                            *at_end = true;
                            let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
                            attrs.extend(inner_attrs.iter().cloned());
                            Some(body)
                        }
                        _ => {
                            return self.expected_semi_or_open_brace();
                        }
                    }
                }
                _ => {
                    return self.expected_semi_or_open_brace();
                }
            };
            (ident, ast::TraitItemKind::Method(sig, body), generics)
        };

        Ok(TraitItem {
            id: ast::DUMMY_NODE_ID,
            ident: name,
            attrs,
            generics,
            node,
            span: lo.to(self.prev_span),
            tokens: None,
        })
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
                let node = Mac_ { path, tts, delim };
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
            return Err(self.fatal(&msg));
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
        let attrs = self.parse_arg_attributes()?;
        if let Ok(Some(mut arg)) = self.parse_self_arg() {
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
                    let pat = P(Pat {
                        id: ast::DUMMY_NODE_ID,
                        node: PatKind::Ident(
                            BindingMode::ByValue(Mutability::Immutable), ident, None),
                        span: ty.span,
                    });
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

        Ok(Arg { attrs: attrs.into(), id: ast::DUMMY_NODE_ID, pat, ty })
    }

    /// Parses an argument in a lambda header (e.g., `|arg, arg|`).
    fn parse_fn_block_arg(&mut self) -> PResult<'a, Arg> {
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
        Ok(Arg {
            attrs: attrs.into(),
            ty: t,
            pat,
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

    /// Matches `'-' lit | lit` (cf. `ast_validation::AstValidator::check_expr_within_pat`).
    crate fn parse_literal_maybe_minus(&mut self) -> PResult<'a, P<Expr>> {
        maybe_whole_expr!(self);

        let minus_lo = self.token.span;
        let minus_present = self.eat(&token::BinOp(token::Minus));
        let lo = self.token.span;
        let literal = self.parse_lit()?;
        let hi = self.prev_span;
        let expr = self.mk_expr(lo.to(hi), ExprKind::Lit(literal), ThinVec::new());

        if minus_present {
            let minus_hi = self.prev_span;
            let unary = self.mk_unary(UnOp::Neg, expr);
            Ok(self.mk_expr(minus_lo.to(minus_hi), unary, ThinVec::new()))
        } else {
            Ok(expr)
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
                self.bump(); // `(`
                let (inputs, recovered) = self.parse_seq_to_before_tokens(
                    &[&token::CloseDelim(token::Paren)],
                    SeqSep::trailing_allowed(token::Comma),
                    TokenExpectType::Expect,
                    |p| p.parse_ty())?;
                if !recovered {
                    self.bump(); // `)`
                }
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

    fn eat_label(&mut self) -> Option<Label> {
        if let Some(ident) = self.token.lifetime() {
            let span = self.token.span;
            self.bump();
            Some(Label { ident: Ident::new(ident.name, span) })
        } else {
            None
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

    /// Parse ident (COLON expr)?
    fn parse_field(&mut self) -> PResult<'a, Field> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;

        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let (fieldname, expr, is_shorthand) = if self.look_ahead(1, |t| {
            t == &token::Colon || t == &token::Eq
        }) {
            let fieldname = self.parse_field_name()?;

            // Check for an equals token. This means the source incorrectly attempts to
            // initialize a field with an eq rather than a colon.
            if self.token == token::Eq {
                self.diagnostic()
                    .struct_span_err(self.token.span, "expected `:`, found `=`")
                    .span_suggestion(
                        fieldname.span.shrink_to_hi().to(self.token.span),
                        "replace equals symbol with a colon",
                        ":".to_string(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
            }
            self.bump(); // `:`
            (fieldname, self.parse_expr()?, false)
        } else {
            let fieldname = self.parse_ident_common(false)?;

            // Mimic `x: x` for the `x` field shorthand.
            let path = ast::Path::from_ident(fieldname);
            let expr = self.mk_expr(fieldname.span, ExprKind::Path(None, path), ThinVec::new());
            (fieldname, expr, true)
        };
        Ok(ast::Field {
            ident: fieldname,
            span: lo.to(expr.span),
            expr,
            is_shorthand,
            attrs: attrs.into(),
        })
    }

    crate fn mk_expr(&self, span: Span, node: ExprKind, attrs: ThinVec<Attribute>) -> P<Expr> {
        P(Expr { node, span, attrs, id: ast::DUMMY_NODE_ID })
    }

    fn mk_unary(&self, unop: ast::UnOp, expr: P<Expr>) -> ast::ExprKind {
        ExprKind::Unary(unop, expr)
    }

    fn mk_binary(&self, binop: ast::BinOp, lhs: P<Expr>, rhs: P<Expr>) -> ast::ExprKind {
        ExprKind::Binary(binop, lhs, rhs)
    }

    fn mk_call(&self, f: P<Expr>, args: Vec<P<Expr>>) -> ast::ExprKind {
        ExprKind::Call(f, args)
    }

    fn mk_index(&self, expr: P<Expr>, idx: P<Expr>) -> ast::ExprKind {
        ExprKind::Index(expr, idx)
    }

    fn mk_range(&self,
                    start: Option<P<Expr>>,
                    end: Option<P<Expr>>,
                    limits: RangeLimits)
                    -> PResult<'a, ast::ExprKind> {
        if end.is_none() && limits == RangeLimits::Closed {
            Err(self.span_fatal_err(self.token.span, Error::InclusiveRangeWithNoEnd))
        } else {
            Ok(ExprKind::Range(start, end, limits))
        }
    }

    fn mk_assign_op(&self, binop: ast::BinOp,
                        lhs: P<Expr>, rhs: P<Expr>) -> ast::ExprKind {
        ExprKind::AssignOp(binop, lhs, rhs)
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

    /// At the bottom (top?) of the precedence hierarchy,
    /// Parses things like parenthesized exprs, macros, `return`, etc.
    ///
    /// N.B., this does not parse outer attributes, and is private because it only works
    /// correctly if called from `parse_dot_or_call_expr()`.
    fn parse_bottom_expr(&mut self) -> PResult<'a, P<Expr>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);
        maybe_whole_expr!(self);

        // Outer attributes are already parsed and will be
        // added to the return value after the fact.
        //
        // Therefore, prevent sub-parser from parsing
        // attributes by giving them a empty "already parsed" list.
        let mut attrs = ThinVec::new();

        let lo = self.token.span;
        let mut hi = self.token.span;

        let ex: ExprKind;

        macro_rules! parse_lit {
            () => {
                match self.parse_lit() {
                    Ok(literal) => {
                        hi = self.prev_span;
                        ex = ExprKind::Lit(literal);
                    }
                    Err(mut err) => {
                        self.cancel(&mut err);
                        return Err(self.expected_expression_found());
                    }
                }
            }
        }

        // Note: when adding new syntax here, don't forget to adjust TokenKind::can_begin_expr().
        match self.token.kind {
            // This match arm is a special-case of the `_` match arm below and
            // could be removed without changing functionality, but it's faster
            // to have it here, especially for programs with large constants.
            token::Literal(_) => {
                parse_lit!()
            }
            token::OpenDelim(token::Paren) => {
                self.bump();

                attrs.extend(self.parse_inner_attributes()?);

                // (e) is parenthesized e
                // (e,) is a tuple with only one field, e
                let mut es = vec![];
                let mut trailing_comma = false;
                let mut recovered = false;
                while self.token != token::CloseDelim(token::Paren) {
                    es.push(match self.parse_expr() {
                        Ok(es) => es,
                        Err(err) => {
                            // recover from parse error in tuple list
                            return Ok(self.recover_seq_parse_error(token::Paren, lo, Err(err)));
                        }
                    });
                    recovered = self.expect_one_of(
                        &[],
                        &[token::Comma, token::CloseDelim(token::Paren)],
                    )?;
                    if self.eat(&token::Comma) {
                        trailing_comma = true;
                    } else {
                        trailing_comma = false;
                        break;
                    }
                }
                if !recovered {
                    self.bump();
                }

                hi = self.prev_span;
                ex = if es.len() == 1 && !trailing_comma {
                    ExprKind::Paren(es.into_iter().nth(0).unwrap())
                } else {
                    ExprKind::Tup(es)
                };
            }
            token::OpenDelim(token::Brace) => {
                return self.parse_block_expr(None, lo, BlockCheckMode::Default, attrs);
            }
            token::BinOp(token::Or) | token::OrOr => {
                return self.parse_lambda_expr(attrs);
            }
            token::OpenDelim(token::Bracket) => {
                self.bump();

                attrs.extend(self.parse_inner_attributes()?);

                if self.eat(&token::CloseDelim(token::Bracket)) {
                    // Empty vector.
                    ex = ExprKind::Array(Vec::new());
                } else {
                    // Nonempty vector.
                    let first_expr = self.parse_expr()?;
                    if self.eat(&token::Semi) {
                        // Repeating array syntax: [ 0; 512 ]
                        let count = AnonConst {
                            id: ast::DUMMY_NODE_ID,
                            value: self.parse_expr()?,
                        };
                        self.expect(&token::CloseDelim(token::Bracket))?;
                        ex = ExprKind::Repeat(first_expr, count);
                    } else if self.eat(&token::Comma) {
                        // Vector with two or more elements.
                        let remaining_exprs = self.parse_seq_to_end(
                            &token::CloseDelim(token::Bracket),
                            SeqSep::trailing_allowed(token::Comma),
                            |p| Ok(p.parse_expr()?)
                        )?;
                        let mut exprs = vec![first_expr];
                        exprs.extend(remaining_exprs);
                        ex = ExprKind::Array(exprs);
                    } else {
                        // Vector with one element.
                        self.expect(&token::CloseDelim(token::Bracket))?;
                        ex = ExprKind::Array(vec![first_expr]);
                    }
                }
                hi = self.prev_span;
            }
            _ => {
                if self.eat_lt() {
                    let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                    hi = path.span;
                    return Ok(self.mk_expr(lo.to(hi), ExprKind::Path(Some(qself), path), attrs));
                }
                if self.check_keyword(kw::Move) || self.check_keyword(kw::Static) {
                    return self.parse_lambda_expr(attrs);
                }
                if self.eat_keyword(kw::If) {
                    return self.parse_if_expr(attrs);
                }
                if self.eat_keyword(kw::For) {
                    let lo = self.prev_span;
                    return self.parse_for_expr(None, lo, attrs);
                }
                if self.eat_keyword(kw::While) {
                    let lo = self.prev_span;
                    return self.parse_while_expr(None, lo, attrs);
                }
                if let Some(label) = self.eat_label() {
                    let lo = label.ident.span;
                    self.expect(&token::Colon)?;
                    if self.eat_keyword(kw::While) {
                        return self.parse_while_expr(Some(label), lo, attrs)
                    }
                    if self.eat_keyword(kw::For) {
                        return self.parse_for_expr(Some(label), lo, attrs)
                    }
                    if self.eat_keyword(kw::Loop) {
                        return self.parse_loop_expr(Some(label), lo, attrs)
                    }
                    if self.token == token::OpenDelim(token::Brace) {
                        return self.parse_block_expr(Some(label),
                                                     lo,
                                                     BlockCheckMode::Default,
                                                     attrs);
                    }
                    let msg = "expected `while`, `for`, `loop` or `{` after a label";
                    let mut err = self.fatal(msg);
                    err.span_label(self.token.span, msg);
                    return Err(err);
                }
                if self.eat_keyword(kw::Loop) {
                    let lo = self.prev_span;
                    return self.parse_loop_expr(None, lo, attrs);
                }
                if self.eat_keyword(kw::Continue) {
                    let label = self.eat_label();
                    let ex = ExprKind::Continue(label);
                    let hi = self.prev_span;
                    return Ok(self.mk_expr(lo.to(hi), ex, attrs));
                }
                if self.eat_keyword(kw::Match) {
                    let match_sp = self.prev_span;
                    return self.parse_match_expr(attrs).map_err(|mut err| {
                        err.span_label(match_sp, "while parsing this match expression");
                        err
                    });
                }
                if self.eat_keyword(kw::Unsafe) {
                    return self.parse_block_expr(
                        None,
                        lo,
                        BlockCheckMode::Unsafe(ast::UserProvided),
                        attrs);
                }
                if self.is_do_catch_block() {
                    let mut db = self.fatal("found removed `do catch` syntax");
                    db.help("Following RFC #2388, the new non-placeholder syntax is `try`");
                    return Err(db);
                }
                if self.is_try_block() {
                    let lo = self.token.span;
                    assert!(self.eat_keyword(kw::Try));
                    return self.parse_try_block(lo, attrs);
                }

                // Span::rust_2018() is somewhat expensive; don't get it repeatedly.
                let is_span_rust_2018 = self.token.span.rust_2018();
                if is_span_rust_2018 && self.check_keyword(kw::Async) {
                    return if self.is_async_block() { // check for `async {` and `async move {`
                        self.parse_async_block(attrs)
                    } else {
                        self.parse_lambda_expr(attrs)
                    };
                }
                if self.eat_keyword(kw::Return) {
                    if self.token.can_begin_expr() {
                        let e = self.parse_expr()?;
                        hi = e.span;
                        ex = ExprKind::Ret(Some(e));
                    } else {
                        ex = ExprKind::Ret(None);
                    }
                } else if self.eat_keyword(kw::Break) {
                    let label = self.eat_label();
                    let e = if self.token.can_begin_expr()
                               && !(self.token == token::OpenDelim(token::Brace)
                                    && self.restrictions.contains(
                                           Restrictions::NO_STRUCT_LITERAL)) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    ex = ExprKind::Break(label, e);
                    hi = self.prev_span;
                } else if self.eat_keyword(kw::Yield) {
                    if self.token.can_begin_expr() {
                        let e = self.parse_expr()?;
                        hi = e.span;
                        ex = ExprKind::Yield(Some(e));
                    } else {
                        ex = ExprKind::Yield(None);
                    }
                } else if self.eat_keyword(kw::Let) {
                    return self.parse_let_expr(attrs);
                } else if is_span_rust_2018 && self.eat_keyword(kw::Await) {
                    let (await_hi, e_kind) = self.parse_await_macro_or_alt(lo, self.prev_span)?;
                    hi = await_hi;
                    ex = e_kind;
                } else if self.token.is_path_start() {
                    let path = self.parse_path(PathStyle::Expr)?;

                    // `!`, as an operator, is prefix, so we know this isn't that
                    if self.eat(&token::Not) {
                        // MACRO INVOCATION expression
                        let (delim, tts) = self.expect_delimited_token_tree()?;
                        hi = self.prev_span;
                        ex = ExprKind::Mac(respan(lo.to(hi), Mac_ { path, tts, delim }));
                    } else if self.check(&token::OpenDelim(token::Brace)) {
                        if let Some(expr) = self.maybe_parse_struct_expr(lo, &path, &attrs) {
                            return expr;
                        } else {
                            hi = path.span;
                            ex = ExprKind::Path(None, path);
                        }
                    } else {
                        hi = path.span;
                        ex = ExprKind::Path(None, path);
                    }
                } else {
                    if !self.unclosed_delims.is_empty() && self.check(&token::Semi) {
                        // Don't complain about bare semicolons after unclosed braces
                        // recovery in order to keep the error count down. Fixing the
                        // delimiters will possibly also fix the bare semicolon found in
                        // expression context. For example, silence the following error:
                        // ```
                        // error: expected expression, found `;`
                        //  --> file.rs:2:13
                        //   |
                        // 2 |     foo(bar(;
                        //   |             ^ expected expression
                        // ```
                        self.bump();
                        return Ok(self.mk_expr(self.token.span, ExprKind::Err, ThinVec::new()));
                    }
                    parse_lit!()
                }
            }
        }

        let expr = self.mk_expr(lo.to(hi), ex, attrs);
        self.maybe_recover_from_bad_qpath(expr, true)
    }

    /// Parse `await!(<expr>)` calls, or alternatively recover from incorrect but reasonable
    /// alternative syntaxes `await <expr>`, `await? <expr>`, `await(<expr>)` and
    /// `await { <expr> }`.
    fn parse_await_macro_or_alt(
        &mut self,
        lo: Span,
        await_sp: Span,
    ) -> PResult<'a, (Span, ExprKind)> {
        if self.token == token::Not {
            // Handle correct `await!(<expr>)`.
            // FIXME: make this an error when `await!` is no longer supported
            // https://github.com/rust-lang/rust/issues/60610
            self.expect(&token::Not)?;
            self.expect(&token::OpenDelim(token::Paren))?;
            let expr = self.parse_expr().map_err(|mut err| {
                err.span_label(await_sp, "while parsing this await macro call");
                err
            })?;
            self.expect(&token::CloseDelim(token::Paren))?;
            Ok((self.prev_span, ExprKind::Await(ast::AwaitOrigin::MacroLike, expr)))
        } else { // Handle `await <expr>`.
            self.parse_incorrect_await_syntax(lo, await_sp)
        }
    }

    fn maybe_parse_struct_expr(
        &mut self,
        lo: Span,
        path: &ast::Path,
        attrs: &ThinVec<Attribute>,
    ) -> Option<PResult<'a, P<Expr>>> {
        let struct_allowed = !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
        let certainly_not_a_block = || self.look_ahead(1, |t| t.is_ident()) && (
            // `{ ident, ` cannot start a block
            self.look_ahead(2, |t| t == &token::Comma) ||
            self.look_ahead(2, |t| t == &token::Colon) && (
                // `{ ident: token, ` cannot start a block
                self.look_ahead(4, |t| t == &token::Comma) ||
                // `{ ident: ` cannot start a block unless it's a type ascription `ident: Type`
                self.look_ahead(3, |t| !t.can_begin_type())
            )
        );

        if struct_allowed || certainly_not_a_block() {
            // This is a struct literal, but we don't can't accept them here
            let expr = self.parse_struct_expr(lo, path.clone(), attrs.clone());
            if let (Ok(expr), false) = (&expr, struct_allowed) {
                let mut err = self.diagnostic().struct_span_err(
                    expr.span,
                    "struct literals are not allowed here",
                );
                err.multipart_suggestion(
                    "surround the struct literal with parentheses",
                    vec![
                        (lo.shrink_to_lo(), "(".to_string()),
                        (expr.span.shrink_to_hi(), ")".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
                err.emit();
            }
            return Some(expr);
        }
        None
    }

    fn parse_struct_expr(&mut self, lo: Span, pth: ast::Path, mut attrs: ThinVec<Attribute>)
                         -> PResult<'a, P<Expr>> {
        let struct_sp = lo.to(self.prev_span);
        self.bump();
        let mut fields = Vec::new();
        let mut base = None;

        attrs.extend(self.parse_inner_attributes()?);

        while self.token != token::CloseDelim(token::Brace) {
            if self.eat(&token::DotDot) {
                let exp_span = self.prev_span;
                match self.parse_expr() {
                    Ok(e) => {
                        base = Some(e);
                    }
                    Err(mut e) => {
                        e.emit();
                        self.recover_stmt();
                    }
                }
                if self.token == token::Comma {
                    let mut err = self.sess.span_diagnostic.mut_span_err(
                        exp_span.to(self.prev_span),
                        "cannot use a comma after the base struct",
                    );
                    err.span_suggestion_short(
                        self.token.span,
                        "remove this comma",
                        String::new(),
                        Applicability::MachineApplicable
                    );
                    err.note("the base struct must always be the last field");
                    err.emit();
                    self.recover_stmt();
                }
                break;
            }

            let mut recovery_field = None;
            if let token::Ident(name, _) = self.token.kind {
                if !self.token.is_reserved_ident() && self.look_ahead(1, |t| *t == token::Colon) {
                    // Use in case of error after field-looking code: `S { foo: () with a }`
                    recovery_field = Some(ast::Field {
                        ident: Ident::new(name, self.token.span),
                        span: self.token.span,
                        expr: self.mk_expr(self.token.span, ExprKind::Err, ThinVec::new()),
                        is_shorthand: false,
                        attrs: ThinVec::new(),
                    });
                }
            }
            let mut parsed_field = None;
            match self.parse_field() {
                Ok(f) => parsed_field = Some(f),
                Err(mut e) => {
                    e.span_label(struct_sp, "while parsing this struct");
                    e.emit();

                    // If the next token is a comma, then try to parse
                    // what comes next as additional fields, rather than
                    // bailing out until next `}`.
                    if self.token != token::Comma {
                        self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                        if self.token != token::Comma {
                            break;
                        }
                    }
                }
            }

            match self.expect_one_of(&[token::Comma],
                                     &[token::CloseDelim(token::Brace)]) {
                Ok(_) => if let Some(f) = parsed_field.or(recovery_field) {
                    // only include the field if there's no parse error for the field name
                    fields.push(f);
                }
                Err(mut e) => {
                    if let Some(f) = recovery_field {
                        fields.push(f);
                    }
                    e.span_label(struct_sp, "while parsing this struct");
                    e.emit();
                    self.recover_stmt_(SemiColonMode::Comma, BlockMode::Ignore);
                    self.eat(&token::Comma);
                }
            }
        }

        let span = lo.to(self.token.span);
        self.expect(&token::CloseDelim(token::Brace))?;
        return Ok(self.mk_expr(span, ExprKind::Struct(pth, fields, base), attrs));
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

    /// Parses a block or unsafe block.
    crate fn parse_block_expr(
        &mut self,
        opt_label: Option<Label>,
        lo: Span,
        blk_mode: BlockCheckMode,
        outer_attrs: ThinVec<Attribute>,
    ) -> PResult<'a, P<Expr>> {
        self.expect(&token::OpenDelim(token::Brace))?;

        let mut attrs = outer_attrs;
        attrs.extend(self.parse_inner_attributes()?);

        let blk = self.parse_block_tail(lo, blk_mode)?;
        return Ok(self.mk_expr(blk.span, ExprKind::Block(blk, opt_label), attrs));
    }

    /// Parses `a.b` or `a(13)` or `a[4]` or just `a`.
    fn parse_dot_or_call_expr(&mut self,
                                  already_parsed_attrs: Option<ThinVec<Attribute>>)
                                  -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;

        let b = self.parse_bottom_expr();
        let (span, b) = self.interpolated_or_expr_span(b)?;
        self.parse_dot_or_call_expr_with(b, span, attrs)
    }

    fn parse_dot_or_call_expr_with(&mut self,
                                       e0: P<Expr>,
                                       lo: Span,
                                       mut attrs: ThinVec<Attribute>)
                                       -> PResult<'a, P<Expr>> {
        // Stitch the list of outer attributes onto the return value.
        // A little bit ugly, but the best way given the current code
        // structure
        self.parse_dot_or_call_expr_with_(e0, lo)
        .map(|expr|
            expr.map(|mut expr| {
                attrs.extend::<Vec<_>>(expr.attrs.into());
                expr.attrs = attrs;
                match expr.node {
                    ExprKind::If(..) if !expr.attrs.is_empty() => {
                        // Just point to the first attribute in there...
                        let span = expr.attrs[0].span;

                        self.span_err(span,
                            "attributes are not yet allowed on `if` \
                            expressions");
                    }
                    _ => {}
                }
                expr
            })
        )
    }

    // Assuming we have just parsed `.`, continue parsing into an expression.
    fn parse_dot_suffix(&mut self, self_arg: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        if self.token.span.rust_2018() && self.eat_keyword(kw::Await) {
            let span = lo.to(self.prev_span);
            let await_expr = self.mk_expr(
                span,
                ExprKind::Await(ast::AwaitOrigin::FieldLike, self_arg),
                ThinVec::new(),
            );
            self.recover_from_await_method_call();
            return Ok(await_expr);
        }
        let segment = self.parse_path_segment(PathStyle::Expr)?;
        self.check_trailing_angle_brackets(&segment, token::OpenDelim(token::Paren));

        Ok(match self.token.kind {
            token::OpenDelim(token::Paren) => {
                // Method call `expr.f()`
                let mut args = self.parse_unspanned_seq(
                    &token::OpenDelim(token::Paren),
                    &token::CloseDelim(token::Paren),
                    SeqSep::trailing_allowed(token::Comma),
                    |p| Ok(p.parse_expr()?)
                )?;
                args.insert(0, self_arg);

                let span = lo.to(self.prev_span);
                self.mk_expr(span, ExprKind::MethodCall(segment, args), ThinVec::new())
            }
            _ => {
                // Field access `expr.f`
                if let Some(args) = segment.args {
                    self.span_err(args.span(),
                                  "field expressions may not have generic arguments");
                }

                let span = lo.to(self.prev_span);
                self.mk_expr(span, ExprKind::Field(self_arg, segment.ident), ThinVec::new())
            }
        })
    }

    fn parse_dot_or_call_expr_with_(&mut self, e0: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        let mut e = e0;
        let mut hi;
        loop {
            // expr?
            while self.eat(&token::Question) {
                let hi = self.prev_span;
                e = self.mk_expr(lo.to(hi), ExprKind::Try(e), ThinVec::new());
            }

            // expr.f
            if self.eat(&token::Dot) {
                match self.token.kind {
                    token::Ident(..) => {
                        e = self.parse_dot_suffix(e, lo)?;
                    }
                    token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) => {
                        let span = self.token.span;
                        self.bump();
                        let field = ExprKind::Field(e, Ident::new(symbol, span));
                        e = self.mk_expr(lo.to(span), field, ThinVec::new());

                        self.expect_no_suffix(span, "a tuple index", suffix);
                    }
                    token::Literal(token::Lit { kind: token::Float, symbol, .. }) => {
                      self.bump();
                      let fstr = symbol.as_str();
                      let msg = format!("unexpected token: `{}`", symbol);
                      let mut err = self.diagnostic().struct_span_err(self.prev_span, &msg);
                      err.span_label(self.prev_span, "unexpected token");
                      if fstr.chars().all(|x| "0123456789.".contains(x)) {
                          let float = match fstr.parse::<f64>().ok() {
                              Some(f) => f,
                              None => continue,
                          };
                          let sugg = pprust::to_string(|s| {
                              use crate::print::pprust::PrintState;
                              s.popen();
                              s.print_expr(&e);
                              s.s.word( ".");
                              s.print_usize(float.trunc() as usize);
                              s.pclose();
                              s.s.word(".");
                              s.s.word(fstr.splitn(2, ".").last().unwrap().to_string())
                          });
                          err.span_suggestion(
                              lo.to(self.prev_span),
                              "try parenthesizing the first index",
                              sugg,
                              Applicability::MachineApplicable
                          );
                      }
                      return Err(err);

                    }
                    _ => {
                        // FIXME Could factor this out into non_fatal_unexpected or something.
                        let actual = self.this_token_to_string();
                        self.span_err(self.token.span, &format!("unexpected token: `{}`", actual));
                    }
                }
                continue;
            }
            if self.expr_is_complete(&e) { break; }
            match self.token.kind {
                // expr(...)
                token::OpenDelim(token::Paren) => {
                    let seq = self.parse_unspanned_seq(
                        &token::OpenDelim(token::Paren),
                        &token::CloseDelim(token::Paren),
                        SeqSep::trailing_allowed(token::Comma),
                        |p| Ok(p.parse_expr()?)
                    ).map(|es| {
                        let nd = self.mk_call(e, es);
                        let hi = self.prev_span;
                        self.mk_expr(lo.to(hi), nd, ThinVec::new())
                    });
                    e = self.recover_seq_parse_error(token::Paren, lo, seq);
                }

                // expr[...]
                // Could be either an index expression or a slicing expression.
                token::OpenDelim(token::Bracket) => {
                    self.bump();
                    let ix = self.parse_expr()?;
                    hi = self.token.span;
                    self.expect(&token::CloseDelim(token::Bracket))?;
                    let index = self.mk_index(e, ix);
                    e = self.mk_expr(lo.to(hi), index, ThinVec::new())
                }
                _ => return Ok(e)
            }
        }
        return Ok(e);
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

    /// Parse a prefix-unary-operator expr
    fn parse_prefix_expr(&mut self,
                             already_parsed_attrs: Option<ThinVec<Attribute>>)
                             -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;
        let lo = self.token.span;
        // Note: when adding new unary operators, don't forget to adjust TokenKind::can_begin_expr()
        let (hi, ex) = match self.token.kind {
            token::Not => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), self.mk_unary(UnOp::Not, e))
            }
            // Suggest `!` for bitwise negation when encountering a `~`
            token::Tilde => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                let span_of_tilde = lo;
                let mut err = self.diagnostic()
                    .struct_span_err(span_of_tilde, "`~` cannot be used as a unary operator");
                err.span_suggestion_short(
                    span_of_tilde,
                    "use `!` to perform bitwise negation",
                    "!".to_owned(),
                    Applicability::MachineApplicable
                );
                err.emit();
                (lo.to(span), self.mk_unary(UnOp::Not, e))
            }
            token::BinOp(token::Minus) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), self.mk_unary(UnOp::Neg, e))
            }
            token::BinOp(token::Star) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), self.mk_unary(UnOp::Deref, e))
            }
            token::BinOp(token::And) | token::AndAnd => {
                self.expect_and()?;
                let m = self.parse_mutability();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), ExprKind::AddrOf(m, e))
            }
            token::Ident(..) if self.token.is_keyword(kw::Box) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), ExprKind::Box(e))
            }
            token::Ident(..) if self.token.is_ident_named(sym::not) => {
                // `not` is just an ordinary identifier in Rust-the-language,
                // but as `rustc`-the-compiler, we can issue clever diagnostics
                // for confused users who really want to say `!`
                let token_cannot_continue_expr = |t: &Token| match t.kind {
                    // These tokens can start an expression after `!`, but
                    // can't continue an expression after an ident
                    token::Ident(name, is_raw) => token::ident_can_begin_expr(name, t.span, is_raw),
                    token::Literal(..) | token::Pound => true,
                    token::Interpolated(ref nt) => match **nt {
                        token::NtIdent(..) | token::NtExpr(..) |
                        token::NtBlock(..) | token::NtPath(..) => true,
                        _ => false,
                    },
                    _ => false
                };
                let cannot_continue_expr = self.look_ahead(1, token_cannot_continue_expr);
                if cannot_continue_expr {
                    self.bump();
                    // Emit the error ...
                    let mut err = self.diagnostic()
                        .struct_span_err(self.token.span,
                                         &format!("unexpected {} after identifier",
                                                  self.this_token_descr()));
                    // span the `not` plus trailing whitespace to avoid
                    // trailing whitespace after the `!` in our suggestion
                    let to_replace = self.sess.source_map()
                        .span_until_non_whitespace(lo.to(self.token.span));
                    err.span_suggestion_short(
                        to_replace,
                        "use `!` to perform logical negation",
                        "!".to_owned(),
                        Applicability::MachineApplicable
                    );
                    err.emit();
                    // —and recover! (just as if we were in the block
                    // for the `token::Not` arm)
                    let e = self.parse_prefix_expr(None);
                    let (span, e) = self.interpolated_or_expr_span(e)?;
                    (lo.to(span), self.mk_unary(UnOp::Not, e))
                } else {
                    return self.parse_dot_or_call_expr(Some(attrs));
                }
            }
            _ => { return self.parse_dot_or_call_expr(Some(attrs)); }
        };
        return Ok(self.mk_expr(lo.to(hi), ex, attrs));
    }

    /// Parses an associative expression.
    ///
    /// This parses an expression accounting for associativity and precedence of the operators in
    /// the expression.
    #[inline]
    fn parse_assoc_expr(&mut self,
                            already_parsed_attrs: Option<ThinVec<Attribute>>)
                            -> PResult<'a, P<Expr>> {
        self.parse_assoc_expr_with(0, already_parsed_attrs.into())
    }

    /// Parses an associative expression with operators of at least `min_prec` precedence.
    fn parse_assoc_expr_with(&mut self,
                                 min_prec: usize,
                                 lhs: LhsExpr)
                                 -> PResult<'a, P<Expr>> {
        let mut lhs = if let LhsExpr::AlreadyParsed(expr) = lhs {
            expr
        } else {
            let attrs = match lhs {
                LhsExpr::AttributesParsed(attrs) => Some(attrs),
                _ => None,
            };
            if [token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token.kind) {
                return self.parse_prefix_range_expr(attrs);
            } else {
                self.parse_prefix_expr(attrs)?
            }
        };

        match (self.expr_is_complete(&lhs), AssocOp::from_token(&self.token)) {
            (true, None) => {
                // Semi-statement forms are odd. See https://github.com/rust-lang/rust/issues/29071
                return Ok(lhs);
            }
            (false, _) => {} // continue parsing the expression
            // An exhaustive check is done in the following block, but these are checked first
            // because they *are* ambiguous but also reasonable looking incorrect syntax, so we
            // want to keep their span info to improve diagnostics in these cases in a later stage.
            (true, Some(AssocOp::Multiply)) | // `{ 42 } *foo = bar;` or `{ 42 } * 3`
            (true, Some(AssocOp::Subtract)) | // `{ 42 } -5`
            (true, Some(AssocOp::LAnd)) | // `{ 42 } &&x` (#61475)
            (true, Some(AssocOp::Add)) // `{ 42 } + 42
            // If the next token is a keyword, then the tokens above *are* unambiguously incorrect:
            // `if x { a } else { b } && if y { c } else { d }`
            if !self.look_ahead(1, |t| t.is_reserved_ident()) => {
                // These cases are ambiguous and can't be identified in the parser alone
                let sp = self.sess.source_map().start_point(self.token.span);
                self.sess.ambiguous_block_expr_parse.borrow_mut().insert(sp, lhs.span);
                return Ok(lhs);
            }
            (true, Some(ref op)) if !op.can_continue_expr_unambiguously() => {
                return Ok(lhs);
            }
            (true, Some(_)) => {
                // We've found an expression that would be parsed as a statement, but the next
                // token implies this should be parsed as an expression.
                // For example: `if let Some(x) = x { x } else { 0 } / 2`
                let mut err = self.sess.span_diagnostic.struct_span_err(self.token.span, &format!(
                    "expected expression, found `{}`",
                    pprust::token_to_string(&self.token),
                ));
                err.span_label(self.token.span, "expected expression");
                self.sess.expr_parentheses_needed(
                    &mut err,
                    lhs.span,
                    Some(pprust::expr_to_string(&lhs),
                ));
                err.emit();
            }
        }
        self.expected_tokens.push(TokenType::Operator);
        while let Some(op) = AssocOp::from_token(&self.token) {

            // Adjust the span for interpolated LHS to point to the `$lhs` token and not to what
            // it refers to. Interpolated identifiers are unwrapped early and never show up here
            // as `PrevTokenKind::Interpolated` so if LHS is a single identifier we always process
            // it as "interpolated", it doesn't change the answer for non-interpolated idents.
            let lhs_span = match (self.prev_token_kind, &lhs.node) {
                (PrevTokenKind::Interpolated, _) => self.prev_span,
                (PrevTokenKind::Ident, &ExprKind::Path(None, ref path))
                    if path.segments.len() == 1 => self.prev_span,
                _ => lhs.span,
            };

            let cur_op_span = self.token.span;
            let restrictions = if op.is_assign_like() {
                self.restrictions & Restrictions::NO_STRUCT_LITERAL
            } else {
                self.restrictions
            };
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            // Check for deprecated `...` syntax
            if self.token == token::DotDotDot && op == AssocOp::DotDotEq {
                self.err_dotdotdot_syntax(self.token.span);
            }

            self.bump();
            if op.is_comparison() {
                self.check_no_chained_comparison(&lhs, &op);
            }
            // Special cases:
            if op == AssocOp::As {
                lhs = self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Cast)?;
                continue
            } else if op == AssocOp::Colon {
                let maybe_path = self.could_ascription_be_path(&lhs.node);
                let next_sp = self.token.span;

                lhs = match self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Type) {
                    Ok(lhs) => lhs,
                    Err(mut err) => {
                        self.bad_type_ascription(
                            &mut err,
                            lhs_span,
                            cur_op_span,
                            next_sp,
                            maybe_path,
                        );
                        return Err(err);
                    }
                };
                continue
            } else if op == AssocOp::DotDot || op == AssocOp::DotDotEq {
                // If we didn’t have to handle `x..`/`x..=`, it would be pretty easy to
                // generalise it to the Fixity::None code.
                //
                // We have 2 alternatives here: `x..y`/`x..=y` and `x..`/`x..=` The other
                // two variants are handled with `parse_prefix_range_expr` call above.
                let rhs = if self.is_at_start_of_range_notation_rhs() {
                    Some(self.parse_assoc_expr_with(prec + 1, LhsExpr::NotYetParsed)?)
                } else {
                    None
                };
                let (lhs_span, rhs_span) = (lhs.span, if let Some(ref x) = rhs {
                    x.span
                } else {
                    cur_op_span
                });
                let limits = if op == AssocOp::DotDot {
                    RangeLimits::HalfOpen
                } else {
                    RangeLimits::Closed
                };

                let r = self.mk_range(Some(lhs), rhs, limits)?;
                lhs = self.mk_expr(lhs_span.to(rhs_span), r, ThinVec::new());
                break
            }

            let fixity = op.fixity();
            let prec_adjustment = match fixity {
                Fixity::Right => 0,
                Fixity::Left => 1,
                // We currently have no non-associative operators that are not handled above by
                // the special cases. The code is here only for future convenience.
                Fixity::None => 1,
            };
            let rhs = self.with_res(
                restrictions - Restrictions::STMT_EXPR,
                |this| this.parse_assoc_expr_with(prec + prec_adjustment, LhsExpr::NotYetParsed)
            )?;

            // Make sure that the span of the parent node is larger than the span of lhs and rhs,
            // including the attributes.
            let lhs_span = lhs
                .attrs
                .iter()
                .filter(|a| a.style == AttrStyle::Outer)
                .next()
                .map_or(lhs_span, |a| a.span);
            let span = lhs_span.to(rhs.span);
            lhs = match op {
                AssocOp::Add | AssocOp::Subtract | AssocOp::Multiply | AssocOp::Divide |
                AssocOp::Modulus | AssocOp::LAnd | AssocOp::LOr | AssocOp::BitXor |
                AssocOp::BitAnd | AssocOp::BitOr | AssocOp::ShiftLeft | AssocOp::ShiftRight |
                AssocOp::Equal | AssocOp::Less | AssocOp::LessEqual | AssocOp::NotEqual |
                AssocOp::Greater | AssocOp::GreaterEqual => {
                    let ast_op = op.to_ast_binop().unwrap();
                    let binary = self.mk_binary(source_map::respan(cur_op_span, ast_op), lhs, rhs);
                    self.mk_expr(span, binary, ThinVec::new())
                }
                AssocOp::Assign => self.mk_expr(span, ExprKind::Assign(lhs, rhs), ThinVec::new()),
                AssocOp::AssignOp(k) => {
                    let aop = match k {
                        token::Plus =>    BinOpKind::Add,
                        token::Minus =>   BinOpKind::Sub,
                        token::Star =>    BinOpKind::Mul,
                        token::Slash =>   BinOpKind::Div,
                        token::Percent => BinOpKind::Rem,
                        token::Caret =>   BinOpKind::BitXor,
                        token::And =>     BinOpKind::BitAnd,
                        token::Or =>      BinOpKind::BitOr,
                        token::Shl =>     BinOpKind::Shl,
                        token::Shr =>     BinOpKind::Shr,
                    };
                    let aopexpr = self.mk_assign_op(source_map::respan(cur_op_span, aop), lhs, rhs);
                    self.mk_expr(span, aopexpr, ThinVec::new())
                }
                AssocOp::As | AssocOp::Colon | AssocOp::DotDot | AssocOp::DotDotEq => {
                    self.bug("AssocOp should have been handled by special case")
                }
            };

            if let Fixity::None = fixity { break }
        }
        Ok(lhs)
    }

    fn parse_assoc_op_cast(&mut self, lhs: P<Expr>, lhs_span: Span,
                           expr_kind: fn(P<Expr>, P<Ty>) -> ExprKind)
                           -> PResult<'a, P<Expr>> {
        let mk_expr = |this: &mut Self, rhs: P<Ty>| {
            this.mk_expr(lhs_span.to(rhs.span), expr_kind(lhs, rhs), ThinVec::new())
        };

        // Save the state of the parser before parsing type normally, in case there is a
        // LessThan comparison after this cast.
        let parser_snapshot_before_type = self.clone();
        match self.parse_ty_no_plus() {
            Ok(rhs) => {
                Ok(mk_expr(self, rhs))
            }
            Err(mut type_err) => {
                // Rewind to before attempting to parse the type with generics, to recover
                // from situations like `x as usize < y` in which we first tried to parse
                // `usize < y` as a type with generic arguments.
                let parser_snapshot_after_type = self.clone();
                mem::replace(self, parser_snapshot_before_type);

                match self.parse_path(PathStyle::Expr) {
                    Ok(path) => {
                        let (op_noun, op_verb) = match self.token.kind {
                            token::Lt => ("comparison", "comparing"),
                            token::BinOp(token::Shl) => ("shift", "shifting"),
                            _ => {
                                // We can end up here even without `<` being the next token, for
                                // example because `parse_ty_no_plus` returns `Err` on keywords,
                                // but `parse_path` returns `Ok` on them due to error recovery.
                                // Return original error and parser state.
                                mem::replace(self, parser_snapshot_after_type);
                                return Err(type_err);
                            }
                        };

                        // Successfully parsed the type path leaving a `<` yet to parse.
                        type_err.cancel();

                        // Report non-fatal diagnostics, keep `x as usize` as an expression
                        // in AST and continue parsing.
                        let msg = format!("`<` is interpreted as a start of generic \
                                           arguments for `{}`, not a {}", path, op_noun);
                        let mut err =
                            self.sess.span_diagnostic.struct_span_err(self.token.span, &msg);
                        let span_after_type = parser_snapshot_after_type.token.span;
                        err.span_label(self.look_ahead(1, |t| t.span).to(span_after_type),
                                       "interpreted as generic arguments");
                        err.span_label(self.token.span, format!("not interpreted as {}", op_noun));

                        let expr = mk_expr(self, P(Ty {
                            span: path.span,
                            node: TyKind::Path(None, path),
                            id: ast::DUMMY_NODE_ID
                        }));

                        let expr_str = self.sess.source_map().span_to_snippet(expr.span)
                                                .unwrap_or_else(|_| pprust::expr_to_string(&expr));
                        err.span_suggestion(
                            expr.span,
                            &format!("try {} the cast value", op_verb),
                            format!("({})", expr_str),
                            Applicability::MachineApplicable
                        );
                        err.emit();

                        Ok(expr)
                    }
                    Err(mut path_err) => {
                        // Couldn't parse as a path, return original error and parser state.
                        path_err.cancel();
                        mem::replace(self, parser_snapshot_after_type);
                        Err(type_err)
                    }
                }
            }
        }
    }

    /// Parse prefix-forms of range notation: `..expr`, `..`, `..=expr`
    fn parse_prefix_range_expr(&mut self,
                               already_parsed_attrs: Option<ThinVec<Attribute>>)
                               -> PResult<'a, P<Expr>> {
        // Check for deprecated `...` syntax
        if self.token == token::DotDotDot {
            self.err_dotdotdot_syntax(self.token.span);
        }

        debug_assert!([token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token.kind),
                      "parse_prefix_range_expr: token {:?} is not DotDot/DotDotEq",
                      self.token);
        let tok = self.token.clone();
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;
        let lo = self.token.span;
        let mut hi = self.token.span;
        self.bump();
        let opt_end = if self.is_at_start_of_range_notation_rhs() {
            // RHS must be parsed with more associativity than the dots.
            let next_prec = AssocOp::from_token(&tok).unwrap().precedence() + 1;
            Some(self.parse_assoc_expr_with(next_prec,
                                            LhsExpr::NotYetParsed)
                .map(|x|{
                    hi = x.span;
                    x
                })?)
        } else {
            None
        };
        let limits = if tok == token::DotDot {
            RangeLimits::HalfOpen
        } else {
            RangeLimits::Closed
        };

        let r = self.mk_range(None, opt_end, limits)?;
        Ok(self.mk_expr(lo.to(hi), r, attrs))
    }

    fn is_at_start_of_range_notation_rhs(&self) -> bool {
        if self.token.can_begin_expr() {
            // parse `for i in 1.. { }` as infinite loop, not as `for i in (1..{})`.
            if self.token == token::OpenDelim(token::Brace) {
                return !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL);
            }
            true
        } else {
            false
        }
    }

    /// Parses an `if` expression (`if` token already eaten).
    fn parse_if_expr(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let lo = self.prev_span;
        let cond = self.parse_cond_expr()?;

        // Verify that the parsed `if` condition makes sense as a condition. If it is a block, then
        // verify that the last statement is either an implicit return (no `;`) or an explicit
        // return. This won't catch blocks with an explicit `return`, but that would be caught by
        // the dead code lint.
        if self.eat_keyword(kw::Else) || !cond.returns() {
            let sp = self.sess.source_map().next_point(lo);
            let mut err = self.diagnostic()
                .struct_span_err(sp, "missing condition for `if` statemement");
            err.span_label(sp, "expected if condition here");
            return Err(err)
        }
        let not_block = self.token != token::OpenDelim(token::Brace);
        let thn = self.parse_block().map_err(|mut err| {
            if not_block {
                err.span_label(lo, "this `if` statement has a condition, but no block");
            }
            err
        })?;
        let mut els: Option<P<Expr>> = None;
        let mut hi = thn.span;
        if self.eat_keyword(kw::Else) {
            let elexpr = self.parse_else_expr()?;
            hi = elexpr.span;
            els = Some(elexpr);
        }
        Ok(self.mk_expr(lo.to(hi), ExprKind::If(cond, thn, els), attrs))
    }

    /// Parse the condition of a `if`- or `while`-expression
    fn parse_cond_expr(&mut self) -> PResult<'a, P<Expr>> {
        let cond = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;

        if let ExprKind::Let(..) = cond.node {
            // Remove the last feature gating of a `let` expression since it's stable.
            let last = self.sess.let_chains_spans.borrow_mut().pop();
            debug_assert_eq!(cond.span, last.unwrap());
        }

        Ok(cond)
    }

    /// Parses a `let $pats = $expr` pseudo-expression.
    /// The `let` token has already been eaten.
    fn parse_let_expr(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let lo = self.prev_span;
        let pats = self.parse_pats()?;
        self.expect(&token::Eq)?;
        let expr = self.with_res(
            Restrictions::NO_STRUCT_LITERAL,
            |this| this.parse_assoc_expr_with(1 + prec_let_scrutinee_needs_par(), None.into())
        )?;
        let span = lo.to(expr.span);
        self.sess.let_chains_spans.borrow_mut().push(span);
        Ok(self.mk_expr(span, ExprKind::Let(pats, expr), attrs))
    }

    /// Parses `move |args| expr`.
    fn parse_lambda_expr(&mut self,
                             attrs: ThinVec<Attribute>)
                             -> PResult<'a, P<Expr>>
    {
        let lo = self.token.span;

        let movability = if self.eat_keyword(kw::Static) {
            Movability::Static
        } else {
            Movability::Movable
        };

        let asyncness = if self.token.span.rust_2018() {
            self.parse_asyncness()
        } else {
            IsAsync::NotAsync
        };
        if asyncness.is_async() {
            // Feature gate `async ||` closures.
            self.sess.async_closure_spans.borrow_mut().push(self.prev_span);
        }

        let capture_clause = self.parse_capture_clause();
        let decl = self.parse_fn_block_decl()?;
        let decl_hi = self.prev_span;
        let body = match decl.output {
            FunctionRetTy::Default(_) => {
                let restrictions = self.restrictions - Restrictions::STMT_EXPR;
                self.parse_expr_res(restrictions, None)?
            },
            _ => {
                // If an explicit return type is given, require a
                // block to appear (RFC 968).
                let body_lo = self.token.span;
                self.parse_block_expr(None, body_lo, BlockCheckMode::Default, ThinVec::new())?
            }
        };

        Ok(self.mk_expr(
            lo.to(body.span),
            ExprKind::Closure(capture_clause, asyncness, movability, decl, body, lo.to(decl_hi)),
            attrs))
    }

    /// `else` token already eaten
    fn parse_else_expr(&mut self) -> PResult<'a, P<Expr>> {
        if self.eat_keyword(kw::If) {
            return self.parse_if_expr(ThinVec::new());
        } else {
            let blk = self.parse_block()?;
            return Ok(self.mk_expr(blk.span, ExprKind::Block(blk, None), ThinVec::new()));
        }
    }

    /// Parse a 'for' .. 'in' expression ('for' token already eaten)
    fn parse_for_expr(&mut self, opt_label: Option<Label>,
                          span_lo: Span,
                          mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        // Parse: `for <src_pat> in <src_expr> <src_loop_block>`

        let pat = self.parse_top_level_pat()?;
        if !self.eat_keyword(kw::In) {
            let in_span = self.prev_span.between(self.token.span);
            let mut err = self.sess.span_diagnostic
                .struct_span_err(in_span, "missing `in` in `for` loop");
            err.span_suggestion_short(
                in_span, "try adding `in` here", " in ".into(),
                // has been misleading, at least in the past (closed Issue #48492)
                Applicability::MaybeIncorrect
            );
            err.emit();
        }
        let in_span = self.prev_span;
        self.check_for_for_in_in_typo(in_span);
        let expr = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        let (iattrs, loop_block) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);

        let hi = self.prev_span;
        Ok(self.mk_expr(span_lo.to(hi), ExprKind::ForLoop(pat, expr, loop_block, opt_label), attrs))
    }

    /// Parses a `while` or `while let` expression (`while` token already eaten).
    fn parse_while_expr(&mut self, opt_label: Option<Label>,
                            span_lo: Span,
                            mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let cond = self.parse_cond_expr()?;
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        Ok(self.mk_expr(span, ExprKind::While(cond, body, opt_label), attrs))
    }

    /// Parse `loop {...}`, `loop` token already eaten.
    fn parse_loop_expr(&mut self, opt_label: Option<Label>,
                           span_lo: Span,
                           mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        Ok(self.mk_expr(span, ExprKind::Loop(body, opt_label), attrs))
    }

    /// Parse an optional `move` prefix to a closure lke construct.
    fn parse_capture_clause(&mut self) -> CaptureBy {
        if self.eat_keyword(kw::Move) {
            CaptureBy::Value
        } else {
            CaptureBy::Ref
        }
    }

    /// Parses an `async move? {...}` expression.
    pub fn parse_async_block(&mut self, mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let span_lo = self.token.span;
        self.expect_keyword(kw::Async)?;
        let capture_clause = self.parse_capture_clause();
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        Ok(self.mk_expr(
            span_lo.to(body.span),
            ExprKind::Async(capture_clause, ast::DUMMY_NODE_ID, body), attrs))
    }

    /// Parses a `try {...}` expression (`try` token already eaten).
    fn parse_try_block(&mut self, span_lo: Span, mut attrs: ThinVec<Attribute>)
        -> PResult<'a, P<Expr>>
    {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        if self.eat_keyword(kw::Catch) {
            let mut error = self.struct_span_err(self.prev_span,
                                                 "keyword `catch` cannot follow a `try` block");
            error.help("try using `match` on the result of the `try` block instead");
            error.emit();
            Err(error)
        } else {
            Ok(self.mk_expr(span_lo.to(body.span), ExprKind::TryBlock(body), attrs))
        }
    }

    // `match` token already eaten
    fn parse_match_expr(&mut self, mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let match_span = self.prev_span;
        let lo = self.prev_span;
        let discriminant = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL,
                                               None)?;
        if let Err(mut e) = self.expect(&token::OpenDelim(token::Brace)) {
            if self.token == token::Semi {
                e.span_suggestion_short(
                    match_span,
                    "try removing this `match`",
                    String::new(),
                    Applicability::MaybeIncorrect // speculative
                );
            }
            return Err(e)
        }
        attrs.extend(self.parse_inner_attributes()?);

        let mut arms: Vec<Arm> = Vec::new();
        while self.token != token::CloseDelim(token::Brace) {
            match self.parse_arm() {
                Ok(arm) => arms.push(arm),
                Err(mut e) => {
                    // Recover by skipping to the end of the block.
                    e.emit();
                    self.recover_stmt();
                    let span = lo.to(self.token.span);
                    if self.token == token::CloseDelim(token::Brace) {
                        self.bump();
                    }
                    return Ok(self.mk_expr(span, ExprKind::Match(discriminant, arms), attrs));
                }
            }
        }
        let hi = self.token.span;
        self.bump();
        return Ok(self.mk_expr(lo.to(hi), ExprKind::Match(discriminant, arms), attrs));
    }

    crate fn parse_arm(&mut self) -> PResult<'a, Arm> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;
        let pats = self.parse_pats()?;
        let guard = if self.eat_keyword(kw::If) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let arrow_span = self.token.span;
        self.expect(&token::FatArrow)?;
        let arm_start_span = self.token.span;

        let expr = self.parse_expr_res(Restrictions::STMT_EXPR, None)
            .map_err(|mut err| {
                err.span_label(arrow_span, "while parsing the `match` arm starting here");
                err
            })?;

        let require_comma = classify::expr_requires_semi_to_be_stmt(&expr)
            && self.token != token::CloseDelim(token::Brace);

        let hi = self.token.span;

        if require_comma {
            let cm = self.sess.source_map();
            self.expect_one_of(&[token::Comma], &[token::CloseDelim(token::Brace)])
                .map_err(|mut err| {
                    match (cm.span_to_lines(expr.span), cm.span_to_lines(arm_start_span)) {
                        (Ok(ref expr_lines), Ok(ref arm_start_lines))
                        if arm_start_lines.lines[0].end_col == expr_lines.lines[0].end_col
                            && expr_lines.lines.len() == 2
                            && self.token == token::FatArrow => {
                            // We check whether there's any trailing code in the parse span,
                            // if there isn't, we very likely have the following:
                            //
                            // X |     &Y => "y"
                            //   |        --    - missing comma
                            //   |        |
                            //   |        arrow_span
                            // X |     &X => "x"
                            //   |      - ^^ self.token.span
                            //   |      |
                            //   |      parsed until here as `"y" & X`
                            err.span_suggestion_short(
                                cm.next_point(arm_start_span),
                                "missing a comma here to end this `match` arm",
                                ",".to_owned(),
                                Applicability::MachineApplicable
                            );
                        }
                        _ => {
                            err.span_label(arrow_span,
                                           "while parsing the `match` arm starting here");
                        }
                    }
                    err
                })?;
        } else {
            self.eat(&token::Comma);
        }

        Ok(ast::Arm {
            attrs,
            pats,
            guard,
            body: expr,
            span: lo.to(hi),
        })
    }

    /// Parses an expression.
    #[inline]
    pub fn parse_expr(&mut self) -> PResult<'a, P<Expr>> {
        self.parse_expr_res(Restrictions::empty(), None)
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

    /// Parses an expression, subject to the given restrictions.
    #[inline]
    fn parse_expr_res(&mut self, r: Restrictions,
                          already_parsed_attrs: Option<ThinVec<Attribute>>)
                          -> PResult<'a, P<Expr>> {
        self.with_res(r, |this| this.parse_assoc_expr(already_parsed_attrs))
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

    /// Parses patterns, separated by '|' s.
    fn parse_pats(&mut self) -> PResult<'a, Vec<P<Pat>>> {
        // Allow a '|' before the pats (RFC 1925 + RFC 2530)
        self.eat(&token::BinOp(token::Or));

        let mut pats = Vec::new();
        loop {
            pats.push(self.parse_top_level_pat()?);

            if self.token == token::OrOr {
                let mut err = self.struct_span_err(self.token.span,
                                                   "unexpected token `||` after pattern");
                err.span_suggestion(
                    self.token.span,
                    "use a single `|` to specify multiple patterns",
                    "|".to_owned(),
                    Applicability::MachineApplicable
                );
                err.emit();
                self.bump();
            } else if self.eat(&token::BinOp(token::Or)) {
                // This is a No-op. Continue the loop to parse the next
                // pattern.
            } else {
                return Ok(pats);
            }
        };
    }

    // Parses a parenthesized list of patterns like
    // `()`, `(p)`, `(p,)`, `(p, q)`, or `(p, .., q)`. Returns:
    // - a vector of the patterns that were parsed
    // - an option indicating the index of the `..` element
    // - a boolean indicating whether a trailing comma was present.
    // Trailing commas are significant because (p) and (p,) are different patterns.
    fn parse_parenthesized_pat_list(&mut self) -> PResult<'a, (Vec<P<Pat>>, Option<usize>, bool)> {
        self.expect(&token::OpenDelim(token::Paren))?;
        let result = match self.parse_pat_list() {
            Ok(result) => result,
            Err(mut err) => { // recover from parse error in tuple pattern list
                err.emit();
                self.consume_block(token::Paren);
                return Ok((vec![], Some(0), false));
            }
        };
        self.expect(&token::CloseDelim(token::Paren))?;
        Ok(result)
    }

    fn parse_pat_list(&mut self) -> PResult<'a, (Vec<P<Pat>>, Option<usize>, bool)> {
        let mut fields = Vec::new();
        let mut ddpos = None;
        let mut prev_dd_sp = None;
        let mut trailing_comma = false;
        loop {
            if self.eat(&token::DotDot) {
                if ddpos.is_none() {
                    ddpos = Some(fields.len());
                    prev_dd_sp = Some(self.prev_span);
                } else {
                    // Emit a friendly error, ignore `..` and continue parsing
                    let mut err = self.struct_span_err(
                        self.prev_span,
                        "`..` can only be used once per tuple or tuple struct pattern",
                    );
                    err.span_label(self.prev_span, "can only be used once per pattern");
                    if let Some(sp) = prev_dd_sp {
                        err.span_label(sp, "previously present here");
                    }
                    err.emit();
                }
            } else if !self.check(&token::CloseDelim(token::Paren)) {
                fields.push(self.parse_pat(None)?);
            } else {
                break
            }

            trailing_comma = self.eat(&token::Comma);
            if !trailing_comma {
                break
            }
        }

        if ddpos == Some(fields.len()) && trailing_comma {
            // `..` needs to be followed by `)` or `, pat`, `..,)` is disallowed.
            let msg = "trailing comma is not permitted after `..`";
            self.struct_span_err(self.prev_span, msg)
                .span_label(self.prev_span, msg)
                .emit();
        }

        Ok((fields, ddpos, trailing_comma))
    }

    fn parse_pat_vec_elements(
        &mut self,
    ) -> PResult<'a, (Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>)> {
        let mut before = Vec::new();
        let mut slice = None;
        let mut after = Vec::new();
        let mut first = true;
        let mut before_slice = true;

        while self.token != token::CloseDelim(token::Bracket) {
            if first {
                first = false;
            } else {
                self.expect(&token::Comma)?;

                if self.token == token::CloseDelim(token::Bracket)
                        && (before_slice || !after.is_empty()) {
                    break
                }
            }

            if before_slice {
                if self.eat(&token::DotDot) {

                    if self.check(&token::Comma) ||
                            self.check(&token::CloseDelim(token::Bracket)) {
                        slice = Some(P(Pat {
                            id: ast::DUMMY_NODE_ID,
                            node: PatKind::Wild,
                            span: self.prev_span,
                        }));
                        before_slice = false;
                    }
                    continue
                }
            }

            let subpat = self.parse_pat(None)?;
            if before_slice && self.eat(&token::DotDot) {
                slice = Some(subpat);
                before_slice = false;
            } else if before_slice {
                before.push(subpat);
            } else {
                after.push(subpat);
            }
        }

        Ok((before, slice, after))
    }

    fn parse_pat_field(
        &mut self,
        lo: Span,
        attrs: Vec<Attribute>
    ) -> PResult<'a, source_map::Spanned<ast::FieldPat>> {
        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let hi;
        let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
            // Parsing a pattern of the form "fieldname: pat"
            let fieldname = self.parse_field_name()?;
            self.bump();
            let pat = self.parse_pat(None)?;
            hi = pat.span;
            (pat, fieldname, false)
        } else {
            // Parsing a pattern of the form "(box) (ref) (mut) fieldname"
            let is_box = self.eat_keyword(kw::Box);
            let boxed_span = self.token.span;
            let is_ref = self.eat_keyword(kw::Ref);
            let is_mut = self.eat_keyword(kw::Mut);
            let fieldname = self.parse_ident()?;
            hi = self.prev_span;

            let bind_type = match (is_ref, is_mut) {
                (true, true) => BindingMode::ByRef(Mutability::Mutable),
                (true, false) => BindingMode::ByRef(Mutability::Immutable),
                (false, true) => BindingMode::ByValue(Mutability::Mutable),
                (false, false) => BindingMode::ByValue(Mutability::Immutable),
            };
            let fieldpat = P(Pat {
                id: ast::DUMMY_NODE_ID,
                node: PatKind::Ident(bind_type, fieldname, None),
                span: boxed_span.to(hi),
            });

            let subpat = if is_box {
                P(Pat {
                    id: ast::DUMMY_NODE_ID,
                    node: PatKind::Box(fieldpat),
                    span: lo.to(hi),
                })
            } else {
                fieldpat
            };
            (subpat, fieldname, true)
        };

        Ok(source_map::Spanned {
            span: lo.to(hi),
            node: ast::FieldPat {
                ident: fieldname,
                pat: subpat,
                is_shorthand,
                attrs: attrs.into(),
           }
        })
    }

    /// Parses the fields of a struct-like pattern.
    fn parse_pat_fields(&mut self) -> PResult<'a, (Vec<source_map::Spanned<ast::FieldPat>>, bool)> {
        let mut fields = Vec::new();
        let mut etc = false;
        let mut ate_comma = true;
        let mut delayed_err: Option<DiagnosticBuilder<'a>> = None;
        let mut etc_span = None;

        while self.token != token::CloseDelim(token::Brace) {
            let attrs = self.parse_outer_attributes()?;
            let lo = self.token.span;

            // check that a comma comes after every field
            if !ate_comma {
                let err = self.struct_span_err(self.prev_span, "expected `,`");
                if let Some(mut delayed) = delayed_err {
                    delayed.emit();
                }
                return Err(err);
            }
            ate_comma = false;

            if self.check(&token::DotDot) || self.token == token::DotDotDot {
                etc = true;
                let mut etc_sp = self.token.span;

                if self.token == token::DotDotDot { // Issue #46718
                    // Accept `...` as if it were `..` to avoid further errors
                    let mut err = self.struct_span_err(self.token.span,
                                                       "expected field pattern, found `...`");
                    err.span_suggestion(
                        self.token.span,
                        "to omit remaining fields, use one fewer `.`",
                        "..".to_owned(),
                        Applicability::MachineApplicable
                    );
                    err.emit();
                }
                self.bump();  // `..` || `...`

                if self.token == token::CloseDelim(token::Brace) {
                    etc_span = Some(etc_sp);
                    break;
                }
                let token_str = self.this_token_descr();
                let mut err = self.fatal(&format!("expected `}}`, found {}", token_str));

                err.span_label(self.token.span, "expected `}`");
                let mut comma_sp = None;
                if self.token == token::Comma { // Issue #49257
                    let nw_span = self.sess.source_map().span_until_non_whitespace(self.token.span);
                    etc_sp = etc_sp.to(nw_span);
                    err.span_label(etc_sp,
                                   "`..` must be at the end and cannot have a trailing comma");
                    comma_sp = Some(self.token.span);
                    self.bump();
                    ate_comma = true;
                }

                etc_span = Some(etc_sp.until(self.token.span));
                if self.token == token::CloseDelim(token::Brace) {
                    // If the struct looks otherwise well formed, recover and continue.
                    if let Some(sp) = comma_sp {
                        err.span_suggestion_short(
                            sp,
                            "remove this comma",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                    }
                    err.emit();
                    break;
                } else if self.token.is_ident() && ate_comma {
                    // Accept fields coming after `..,`.
                    // This way we avoid "pattern missing fields" errors afterwards.
                    // We delay this error until the end in order to have a span for a
                    // suggested fix.
                    if let Some(mut delayed_err) = delayed_err {
                        delayed_err.emit();
                        return Err(err);
                    } else {
                        delayed_err = Some(err);
                    }
                } else {
                    if let Some(mut err) = delayed_err {
                        err.emit();
                    }
                    return Err(err);
                }
            }

            fields.push(match self.parse_pat_field(lo, attrs) {
                Ok(field) => field,
                Err(err) => {
                    if let Some(mut delayed_err) = delayed_err {
                        delayed_err.emit();
                    }
                    return Err(err);
                }
            });
            ate_comma = self.eat(&token::Comma);
        }

        if let Some(mut err) = delayed_err {
            if let Some(etc_span) = etc_span {
                err.multipart_suggestion(
                    "move the `..` to the end of the field list",
                    vec![
                        (etc_span, String::new()),
                        (self.token.span, format!("{}.. }}", if ate_comma { "" } else { ", " })),
                    ],
                    Applicability::MachineApplicable,
                );
            }
            err.emit();
        }
        return Ok((fields, etc));
    }

    fn parse_pat_range_end(&mut self) -> PResult<'a, P<Expr>> {
        if self.token.is_path_start() {
            let lo = self.token.span;
            let (qself, path) = if self.eat_lt() {
                // Parse a qualified path
                let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, self.parse_path(PathStyle::Expr)?)
            };
            let hi = self.prev_span;
            Ok(self.mk_expr(lo.to(hi), ExprKind::Path(qself, path), ThinVec::new()))
        } else {
            self.parse_literal_maybe_minus()
        }
    }

    // helper function to decide whether to parse as ident binding or to try to do
    // something more complex like range patterns
    fn parse_as_ident(&mut self) -> bool {
        self.look_ahead(1, |t| match t.kind {
            token::OpenDelim(token::Paren) | token::OpenDelim(token::Brace) |
            token::DotDotDot | token::DotDotEq | token::ModSep | token::Not => Some(false),
            // ensure slice patterns [a, b.., c] and [a, b, c..] don't go into the
            // range pattern branch
            token::DotDot => None,
            _ => Some(true),
        }).unwrap_or_else(|| self.look_ahead(2, |t| match t.kind {
            token::Comma | token::CloseDelim(token::Bracket) => true,
            _ => false,
        }))
    }

    /// A wrapper around `parse_pat` with some special error handling for the
    /// "top-level" patterns in a match arm, `for` loop, `let`, &c. (in contrast
    /// to subpatterns within such).
    fn parse_top_level_pat(&mut self) -> PResult<'a, P<Pat>> {
        let pat = self.parse_pat(None)?;
        if self.token == token::Comma {
            // An unexpected comma after a top-level pattern is a clue that the
            // user (perhaps more accustomed to some other language) forgot the
            // parentheses in what should have been a tuple pattern; return a
            // suggestion-enhanced error here rather than choking on the comma
            // later.
            let comma_span = self.token.span;
            self.bump();
            if let Err(mut err) = self.parse_pat_list() {
                // We didn't expect this to work anyway; we just wanted
                // to advance to the end of the comma-sequence so we know
                // the span to suggest parenthesizing
                err.cancel();
            }
            let seq_span = pat.span.to(self.prev_span);
            let mut err = self.struct_span_err(comma_span,
                                               "unexpected `,` in pattern");
            if let Ok(seq_snippet) = self.sess.source_map().span_to_snippet(seq_span) {
                err.span_suggestion(
                    seq_span,
                    "try adding parentheses to match on a tuple..",
                    format!("({})", seq_snippet),
                    Applicability::MachineApplicable
                ).span_suggestion(
                    seq_span,
                    "..or a vertical bar to match on multiple alternatives",
                    format!("{}", seq_snippet.replace(",", " |")),
                    Applicability::MachineApplicable
                );
            }
            return Err(err);
        }
        Ok(pat)
    }

    /// Parses a pattern.
    pub fn parse_pat(&mut self, expected: Option<&'static str>) -> PResult<'a, P<Pat>> {
        self.parse_pat_with_range_pat(true, expected)
    }

    /// Parses a pattern, with a setting whether modern range patterns (e.g., `a..=b`, `a..b` are
    /// allowed).
    fn parse_pat_with_range_pat(
        &mut self,
        allow_range_pat: bool,
        expected: Option<&'static str>,
    ) -> PResult<'a, P<Pat>> {
        maybe_recover_from_interpolated_ty_qpath!(self, true);
        maybe_whole!(self, NtPat, |x| x);

        let lo = self.token.span;
        let pat;
        match self.token.kind {
            token::BinOp(token::And) | token::AndAnd => {
                // Parse &pat / &mut pat
                self.expect_and()?;
                let mutbl = self.parse_mutability();
                if let token::Lifetime(name) = self.token.kind {
                    let mut err = self.fatal(&format!("unexpected lifetime `{}` in pattern", name));
                    err.span_label(self.token.span, "unexpected lifetime");
                    return Err(err);
                }
                let subpat = self.parse_pat_with_range_pat(false, expected)?;
                pat = PatKind::Ref(subpat, mutbl);
            }
            token::OpenDelim(token::Paren) => {
                // Parse (pat,pat,pat,...) as tuple pattern
                let (fields, ddpos, trailing_comma) = self.parse_parenthesized_pat_list()?;
                pat = if fields.len() == 1 && ddpos.is_none() && !trailing_comma {
                    PatKind::Paren(fields.into_iter().nth(0).unwrap())
                } else {
                    PatKind::Tuple(fields, ddpos)
                };
            }
            token::OpenDelim(token::Bracket) => {
                // Parse [pat,pat,...] as slice pattern
                self.bump();
                let (before, slice, after) = self.parse_pat_vec_elements()?;
                self.expect(&token::CloseDelim(token::Bracket))?;
                pat = PatKind::Slice(before, slice, after);
            }
            // At this point, token != &, &&, (, [
            _ => if self.eat_keyword(kw::Underscore) {
                // Parse _
                pat = PatKind::Wild;
            } else if self.eat_keyword(kw::Mut) {
                // Parse mut ident @ pat / mut ref ident @ pat
                let mutref_span = self.prev_span.to(self.token.span);
                let binding_mode = if self.eat_keyword(kw::Ref) {
                    self.diagnostic()
                        .struct_span_err(mutref_span, "the order of `mut` and `ref` is incorrect")
                        .span_suggestion(
                            mutref_span,
                            "try switching the order",
                            "ref mut".into(),
                            Applicability::MachineApplicable
                        ).emit();
                    BindingMode::ByRef(Mutability::Mutable)
                } else {
                    BindingMode::ByValue(Mutability::Mutable)
                };
                pat = self.parse_pat_ident(binding_mode)?;
            } else if self.eat_keyword(kw::Ref) {
                // Parse ref ident @ pat / ref mut ident @ pat
                let mutbl = self.parse_mutability();
                pat = self.parse_pat_ident(BindingMode::ByRef(mutbl))?;
            } else if self.eat_keyword(kw::Box) {
                // Parse box pat
                let subpat = self.parse_pat_with_range_pat(false, None)?;
                pat = PatKind::Box(subpat);
            } else if self.token.is_ident() && !self.token.is_reserved_ident() &&
                      self.parse_as_ident() {
                // Parse ident @ pat
                // This can give false positives and parse nullary enums,
                // they are dealt with later in resolve
                let binding_mode = BindingMode::ByValue(Mutability::Immutable);
                pat = self.parse_pat_ident(binding_mode)?;
            } else if self.token.is_path_start() {
                // Parse pattern starting with a path
                let (qself, path) = if self.eat_lt() {
                    // Parse a qualified path
                    let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
                    (Some(qself), path)
                } else {
                    // Parse an unqualified path
                    (None, self.parse_path(PathStyle::Expr)?)
                };
                match self.token.kind {
                    token::Not if qself.is_none() => {
                        // Parse macro invocation
                        self.bump();
                        let (delim, tts) = self.expect_delimited_token_tree()?;
                        let mac = respan(lo.to(self.prev_span), Mac_ { path, tts, delim });
                        pat = PatKind::Mac(mac);
                    }
                    token::DotDotDot | token::DotDotEq | token::DotDot => {
                        let end_kind = match self.token.kind {
                            token::DotDot => RangeEnd::Excluded,
                            token::DotDotDot => RangeEnd::Included(RangeSyntax::DotDotDot),
                            token::DotDotEq => RangeEnd::Included(RangeSyntax::DotDotEq),
                            _ => panic!("can only parse `..`/`...`/`..=` for ranges \
                                         (checked above)"),
                        };
                        let op_span = self.token.span;
                        // Parse range
                        let span = lo.to(self.prev_span);
                        let begin = self.mk_expr(span, ExprKind::Path(qself, path), ThinVec::new());
                        self.bump();
                        let end = self.parse_pat_range_end()?;
                        let op = Spanned { span: op_span, node: end_kind };
                        pat = PatKind::Range(begin, end, op);
                    }
                    token::OpenDelim(token::Brace) => {
                        if qself.is_some() {
                            let msg = "unexpected `{` after qualified path";
                            let mut err = self.fatal(msg);
                            err.span_label(self.token.span, msg);
                            return Err(err);
                        }
                        // Parse struct pattern
                        self.bump();
                        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
                            e.emit();
                            self.recover_stmt();
                            (vec![], true)
                        });
                        self.bump();
                        pat = PatKind::Struct(path, fields, etc);
                    }
                    token::OpenDelim(token::Paren) => {
                        if qself.is_some() {
                            let msg = "unexpected `(` after qualified path";
                            let mut err = self.fatal(msg);
                            err.span_label(self.token.span, msg);
                            return Err(err);
                        }
                        // Parse tuple struct or enum pattern
                        let (fields, ddpos, _) = self.parse_parenthesized_pat_list()?;
                        pat = PatKind::TupleStruct(path, fields, ddpos)
                    }
                    _ => pat = PatKind::Path(qself, path),
                }
            } else {
                // Try to parse everything else as literal with optional minus
                match self.parse_literal_maybe_minus() {
                    Ok(begin) => {
                        let op_span = self.token.span;
                        if self.check(&token::DotDot) || self.check(&token::DotDotEq) ||
                                self.check(&token::DotDotDot) {
                            let end_kind = if self.eat(&token::DotDotDot) {
                                RangeEnd::Included(RangeSyntax::DotDotDot)
                            } else if self.eat(&token::DotDotEq) {
                                RangeEnd::Included(RangeSyntax::DotDotEq)
                            } else if self.eat(&token::DotDot) {
                                RangeEnd::Excluded
                            } else {
                                panic!("impossible case: we already matched \
                                        on a range-operator token")
                            };
                            let end = self.parse_pat_range_end()?;
                            let op = Spanned { span: op_span, node: end_kind };
                            pat = PatKind::Range(begin, end, op);
                        } else {
                            pat = PatKind::Lit(begin);
                        }
                    }
                    Err(mut err) => {
                        self.cancel(&mut err);
                        let expected = expected.unwrap_or("pattern");
                        let msg = format!(
                            "expected {}, found {}",
                            expected,
                            self.this_token_descr(),
                        );
                        let mut err = self.fatal(&msg);
                        err.span_label(self.token.span, format!("expected {}", expected));
                        let sp = self.sess.source_map().start_point(self.token.span);
                        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
                            self.sess.expr_parentheses_needed(&mut err, *sp, None);
                        }
                        return Err(err);
                    }
                }
            }
        }

        let pat = P(Pat { node: pat, span: lo.to(self.prev_span), id: ast::DUMMY_NODE_ID });
        let pat = self.maybe_recover_from_bad_qpath(pat, true)?;

        if !allow_range_pat {
            match pat.node {
                PatKind::Range(
                    _, _, Spanned { node: RangeEnd::Included(RangeSyntax::DotDotDot), .. }
                ) => {},
                PatKind::Range(..) => {
                    let mut err = self.struct_span_err(
                        pat.span,
                        "the range pattern here has ambiguous interpretation",
                    );
                    err.span_suggestion(
                        pat.span,
                        "add parentheses to clarify the precedence",
                        format!("({})", pprust::pat_to_string(&pat)),
                        // "ambiguous interpretation" implies that we have to be guessing
                        Applicability::MaybeIncorrect
                    );
                    return Err(err);
                }
                _ => {}
            }
        }

        Ok(pat)
    }

    /// Parses `ident` or `ident @ pat`.
    /// used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like `ref foo(a, b)`.
    fn parse_pat_ident(&mut self,
                       binding_mode: ast::BindingMode)
                       -> PResult<'a, PatKind> {
        let ident = self.parse_ident()?;
        let sub = if self.eat(&token::At) {
            Some(self.parse_pat(Some("binding pattern"))?)
        } else {
            None
        };

        // just to be friendly, if they write something like
        //   ref Some(i)
        // we end up here with ( as the current token.  This shortly
        // leads to a parse error.  Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to parse_enum_variant()
        if self.token == token::OpenDelim(token::Paren) {
            return Err(self.span_fatal(
                self.prev_span,
                "expected identifier, found enum pattern"))
        }

        Ok(PatKind::Ident(binding_mode, ident, sub))
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

                    let snippet = self.sess.source_map().span_to_snippet(pat.span).unwrap();
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

    /// Parses a structure field.
    fn parse_name_and_ty(&mut self,
                         lo: Span,
                         vis: Visibility,
                         attrs: Vec<Attribute>)
                         -> PResult<'a, StructField> {
        let name = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        Ok(StructField {
            span: lo.to(self.prev_span),
            ident: Some(name),
            vis,
            id: ast::DUMMY_NODE_ID,
            ty,
            attrs,
        })
    }

    /// Emits an expected-item-after-attributes error.
    fn expected_item_err(&mut self, attrs: &[Attribute]) -> PResult<'a,  ()> {
        let message = match attrs.last() {
            Some(&Attribute { is_sugared_doc: true, .. }) => "expected item after doc comment",
            _ => "expected item after attributes",
        };

        let mut err = self.diagnostic().struct_span_err(self.prev_span, message);
        if attrs.last().unwrap().is_sugared_doc {
            err.span_label(self.prev_span, "this doc comment doesn't document anything");
        }
        Err(err)
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

    fn is_async_block(&self) -> bool {
        self.token.is_keyword(kw::Async) &&
        (
            ( // `async move {`
                self.is_keyword_ahead(1, &[kw::Move]) &&
                self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace))
            ) || ( // `async {`
                self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace))
            )
        )
    }

    fn is_async_fn(&self) -> bool {
        self.token.is_keyword(kw::Async) &&
            self.is_keyword_ahead(1, &[kw::Fn])
    }

    fn is_do_catch_block(&self) -> bool {
        self.token.is_keyword(kw::Do) &&
        self.is_keyword_ahead(1, &[kw::Catch]) &&
        self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace)) &&
        !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    fn is_try_block(&self) -> bool {
        self.token.is_keyword(kw::Try) &&
        self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace)) &&
        self.token.span.rust_2018() &&
        // prevent `while try {} {}`, `if try {} {} else {}`, etc.
        !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    fn is_union_item(&self) -> bool {
        self.token.is_keyword(kw::Union) &&
        self.look_ahead(1, |t| t.is_ident() && !t.is_reserved_ident())
    }

    fn is_crate_vis(&self) -> bool {
        self.token.is_keyword(kw::Crate) && self.look_ahead(1, |t| t != &token::ModSep)
    }

    fn is_existential_type_decl(&self) -> bool {
        self.token.is_keyword(kw::Existential) &&
        self.is_keyword_ahead(1, &[kw::Type])
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

    fn eat_macro_def(&mut self, attrs: &[Attribute], vis: &Visibility, lo: Span)
                     -> PResult<'a, Option<P<Item>>> {
        let token_lo = self.token.span;
        let (ident, def) = if self.eat_keyword(kw::Macro) {
            let ident = self.parse_ident()?;
            let tokens = if self.check(&token::OpenDelim(token::Brace)) {
                match self.parse_token_tree() {
                    TokenTree::Delimited(_, _, tts) => tts,
                    _ => unreachable!(),
                }
            } else if self.check(&token::OpenDelim(token::Paren)) {
                let args = self.parse_token_tree();
                let body = if self.check(&token::OpenDelim(token::Brace)) {
                    self.parse_token_tree()
                } else {
                    self.unexpected()?;
                    unreachable!()
                };
                TokenStream::new(vec![
                    args.into(),
                    TokenTree::token(token::FatArrow, token_lo.to(self.prev_span)).into(),
                    body.into(),
                ])
            } else {
                self.unexpected()?;
                unreachable!()
            };

            (ident, ast::MacroDef { tokens: tokens.into(), legacy: false })
        } else if self.check_keyword(sym::macro_rules) &&
                  self.look_ahead(1, |t| *t == token::Not) &&
                  self.look_ahead(2, |t| t.is_ident()) {
            let prev_span = self.prev_span;
            self.complain_if_pub_macro(&vis.node, prev_span);
            self.bump();
            self.bump();

            let ident = self.parse_ident()?;
            let (delim, tokens) = self.expect_delimited_token_tree()?;
            if delim != MacDelimiter::Brace && !self.eat(&token::Semi) {
                self.report_invalid_macro_expansion_item();
            }

            (ident, ast::MacroDef { tokens, legacy: true })
        } else {
            return Ok(None);
        };

        let span = lo.to(self.prev_span);
        Ok(Some(self.mk_item(span, ident, ItemKind::MacroDef(def), vis.clone(), attrs.to_vec())))
    }

    fn parse_stmt_without_recovery(&mut self,
                                   macro_legacy_warnings: bool)
                                   -> PResult<'a, Option<Stmt>> {
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
                  !self.is_existential_type_decl() &&
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

            let mac = respan(lo.to(hi), Mac_ { path, tts, delim });
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

    /// Checks if this expression is a successfully parsed statement.
    fn expr_is_complete(&self, e: &Expr) -> bool {
        self.restrictions.contains(Restrictions::STMT_EXPR) &&
            !classify::expr_requires_semi_to_be_stmt(e)
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
                    let sugg = pprust::to_string(|s| {
                        use crate::print::pprust::{PrintState, INDENT_UNIT};
                        s.ibox(INDENT_UNIT);
                        s.bopen();
                        s.print_stmt(&stmt);
                        s.bclose_maybe_open(stmt.span, INDENT_UNIT, false)
                    });
                    e.span_suggestion(
                        stmt_span,
                        "try placing this code inside a block",
                        sugg,
                        // speculative, has been misleading in the past (closed Issue #46836)
                        Applicability::MaybeIncorrect
                    );
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
    fn parse_inner_attrs_and_block(&mut self) -> PResult<'a, (Vec<Attribute>, P<Block>)> {
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
            } else if self.token == token::Eof {
                break;
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

        stmt.span = stmt.span.with_hi(self.prev_span.hi());
        Ok(Some(stmt))
    }

    fn warn_missing_semicolon(&self) {
        self.diagnostic().struct_span_warn(self.token.span, {
            &format!("expected `;`, found {}", self.this_token_descr())
        }).note({
            "This was erroneously allowed and will become a hard error in a future release"
        }).emit();
    }

    fn err_dotdotdot_syntax(&self, span: Span) {
        self.diagnostic().struct_span_err(span, {
            "unexpected token: `...`"
        }).span_suggestion(
            span, "use `..` for an exclusive range", "..".to_owned(),
            Applicability::MaybeIncorrect
        ).span_suggestion(
            span, "or `..=` for an inclusive range", "..=".to_owned(),
            Applicability::MaybeIncorrect
        ).emit();
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
                        if let Ok(snippet) = self.sess.source_map().span_to_snippet(inner_span) {
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
                        .map(|span| self.sess.source_map().span_to_snippet(span));
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

    /// Parses the following grammar:
    ///
    ///     TraitItemAssocTy = Ident ["<"...">"] [":" [GenericBounds]] ["where" ...] ["=" Ty]
    fn parse_trait_item_assoc_ty(&mut self)
        -> PResult<'a, (Ident, TraitItemKind, ast::Generics)> {
        let ident = self.parse_ident()?;
        let mut generics = self.parse_generics()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_generic_bounds(None)?
        } else {
            Vec::new()
        };
        generics.where_clause = self.parse_where_clause()?;

        let default = if self.eat(&token::Eq) {
            Some(self.parse_ty()?)
        } else {
            None
        };
        self.expect(&token::Semi)?;

        Ok((ident, TraitItemKind::Type(bounds, default), generics))
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
        self.expect(&token::OpenDelim(token::Paren))?;

        let sp = self.token.span;
        let mut c_variadic = false;
        let (args, recovered): (Vec<Option<Arg>>, bool) =
            self.parse_seq_to_before_end(
                &token::CloseDelim(token::Paren),
                SeqSep::trailing_allowed(token::Comma),
                |p| {
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
                }
            )?;

        if !recovered {
            self.eat(&token::CloseDelim(token::Paren));
        }

        let args: Vec<_> = args.into_iter().filter_map(|x| x).collect();

        if c_variadic && args.is_empty() {
            self.span_err(sp,
                          "C-variadic function must be declared with at least one named argument");
        }

        Ok((args, c_variadic))
    }

    /// Parses the argument list and result type of a function declaration.
    fn parse_fn_decl(&mut self, allow_c_variadic: bool) -> PResult<'a, P<FnDecl>> {
        let (args, c_variadic) = self.parse_fn_args(true, allow_c_variadic)?;
        let ret_ty = self.parse_ret_ty(true)?;

        Ok(P(FnDecl {
            inputs: args,
            output: ret_ty,
            c_variadic,
        }))
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
                let (mut input, recovered) = self.parse_seq_to_before_end(
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
            self.parse_seq_to_before_end(&token::CloseDelim(token::Paren), sep, parse_arg_fn)?
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

    /// Parses the name and optional generic types of a function header.
    fn parse_fn_header(&mut self) -> PResult<'a, (Ident, ast::Generics)> {
        let id = self.parse_ident()?;
        let generics = self.parse_generics()?;
        Ok((id, generics))
    }

    fn mk_item(&self, span: Span, ident: Ident, node: ItemKind, vis: Visibility,
               attrs: Vec<Attribute>) -> P<Item> {
        P(Item {
            ident,
            attrs,
            id: ast::DUMMY_NODE_ID,
            node,
            vis,
            span,
            tokens: None,
        })
    }

    /// Parses an item-position function declaration.
    fn parse_item_fn(&mut self,
                     unsafety: Unsafety,
                     asyncness: Spanned<IsAsync>,
                     constness: Spanned<Constness>,
                     abi: Abi)
                     -> PResult<'a, ItemInfo> {
        let (ident, mut generics) = self.parse_fn_header()?;
        let allow_c_variadic = abi == Abi::C && unsafety == Unsafety::Unsafe;
        let decl = self.parse_fn_decl(allow_c_variadic)?;
        generics.where_clause = self.parse_where_clause()?;
        let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
        let header = FnHeader { unsafety, asyncness, constness, abi };
        Ok((ident, ItemKind::Fn(decl, header, generics, body), Some(inner_attrs)))
    }

    /// Returns `true` if we are looking at `const ID`
    /// (returns `false` for things like `const fn`, etc.).
    fn is_const_item(&self) -> bool {
        self.token.is_keyword(kw::Const) &&
            !self.is_keyword_ahead(1, &[kw::Fn, kw::Unsafe])
    }

    /// Parses all the "front matter" for a `fn` declaration, up to
    /// and including the `fn` keyword:
    ///
    /// - `const fn`
    /// - `unsafe fn`
    /// - `const unsafe fn`
    /// - `extern fn`
    /// - etc.
    fn parse_fn_front_matter(&mut self)
        -> PResult<'a, (
            Spanned<Constness>,
            Unsafety,
            Spanned<IsAsync>,
            Abi
        )>
    {
        let is_const_fn = self.eat_keyword(kw::Const);
        let const_span = self.prev_span;
        let asyncness = self.parse_asyncness();
        if let IsAsync::Async { .. } = asyncness {
            self.ban_async_in_2015(self.prev_span);
        }
        let asyncness = respan(self.prev_span, asyncness);
        let unsafety = self.parse_unsafety();
        let (constness, unsafety, abi) = if is_const_fn {
            (respan(const_span, Constness::Const), unsafety, Abi::Rust)
        } else {
            let abi = if self.eat_keyword(kw::Extern) {
                self.parse_opt_abi()?.unwrap_or(Abi::C)
            } else {
                Abi::Rust
            };
            (respan(self.prev_span, Constness::NotConst), unsafety, abi)
        };
        if !self.eat_keyword(kw::Fn) {
            // It is possible for `expect_one_of` to recover given the contents of
            // `self.expected_tokens`, therefore, do not use `self.unexpected()` which doesn't
            // account for this.
            if !self.expect_one_of(&[], &[])? { unreachable!() }
        }
        Ok((constness, unsafety, asyncness, abi))
    }

    /// Parses an impl item.
    pub fn parse_impl_item(&mut self, at_end: &mut bool) -> PResult<'a, ImplItem> {
        maybe_whole!(self, NtImplItem, |x| x);
        let attrs = self.parse_outer_attributes()?;
        let mut unclosed_delims = vec![];
        let (mut item, tokens) = self.collect_tokens(|this| {
            let item = this.parse_impl_item_(at_end, attrs);
            unclosed_delims.append(&mut this.unclosed_delims);
            item
        })?;
        self.unclosed_delims.append(&mut unclosed_delims);

        // See `parse_item` for why this clause is here.
        if !item.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
            item.tokens = Some(tokens);
        }
        Ok(item)
    }

    fn parse_impl_item_(&mut self,
                        at_end: &mut bool,
                        mut attrs: Vec<Attribute>) -> PResult<'a, ImplItem> {
        let lo = self.token.span;
        let vis = self.parse_visibility(false)?;
        let defaultness = self.parse_defaultness();
        let (name, node, generics) = if let Some(type_) = self.eat_type() {
            let (name, alias, generics) = type_?;
            let kind = match alias {
                AliasKind::Weak(typ) => ast::ImplItemKind::Type(typ),
                AliasKind::Existential(bounds) => ast::ImplItemKind::Existential(bounds),
            };
            (name, kind, generics)
        } else if self.is_const_item() {
            // This parses the grammar:
            //     ImplItemConst = "const" Ident ":" Ty "=" Expr ";"
            self.expect_keyword(kw::Const)?;
            let name = self.parse_ident()?;
            self.expect(&token::Colon)?;
            let typ = self.parse_ty()?;
            self.expect(&token::Eq)?;
            let expr = self.parse_expr()?;
            self.expect(&token::Semi)?;
            (name, ast::ImplItemKind::Const(typ, expr), ast::Generics::default())
        } else {
            let (name, inner_attrs, generics, node) = self.parse_impl_method(&vis, at_end)?;
            attrs.extend(inner_attrs);
            (name, node, generics)
        };

        Ok(ImplItem {
            id: ast::DUMMY_NODE_ID,
            span: lo.to(self.prev_span),
            ident: name,
            vis,
            defaultness,
            attrs,
            generics,
            node,
            tokens: None,
        })
    }

    fn complain_if_pub_macro(&self, vis: &VisibilityKind, sp: Span) {
        match *vis {
            VisibilityKind::Inherited => {}
            _ => {
                let mut err = if self.token.is_keyword(sym::macro_rules) {
                    let mut err = self.diagnostic()
                        .struct_span_err(sp, "can't qualify macro_rules invocation with `pub`");
                    err.span_suggestion(
                        sp,
                        "try exporting the macro",
                        "#[macro_export]".to_owned(),
                        Applicability::MaybeIncorrect // speculative
                    );
                    err
                } else {
                    let mut err = self.diagnostic()
                        .struct_span_err(sp, "can't qualify macro invocation with `pub`");
                    err.help("try adjusting the macro to put `pub` inside the invocation");
                    err
                };
                err.emit();
            }
        }
    }

    fn missing_assoc_item_kind_err(&self, item_type: &str, prev_span: Span)
                                   -> DiagnosticBuilder<'a>
    {
        let expected_kinds = if item_type == "extern" {
            "missing `fn`, `type`, or `static`"
        } else {
            "missing `fn`, `type`, or `const`"
        };

        // Given this code `path(`, it seems like this is not
        // setting the visibility of a macro invocation, but rather
        // a mistyped method declaration.
        // Create a diagnostic pointing out that `fn` is missing.
        //
        // x |     pub path(&self) {
        //   |        ^ missing `fn`, `type`, or `const`
        //     pub  path(
        //        ^^ `sp` below will point to this
        let sp = prev_span.between(self.prev_span);
        let mut err = self.diagnostic().struct_span_err(
            sp,
            &format!("{} for {}-item declaration",
                     expected_kinds, item_type));
        err.span_label(sp, expected_kinds);
        err
    }

    /// Parse a method or a macro invocation in a trait impl.
    fn parse_impl_method(&mut self, vis: &Visibility, at_end: &mut bool)
                         -> PResult<'a, (Ident, Vec<Attribute>, ast::Generics,
                             ast::ImplItemKind)> {
        // code copied from parse_macro_use_or_failure... abstraction!
        if let Some(mac) = self.parse_assoc_macro_invoc("impl", Some(vis), at_end)? {
            // method macro
            Ok((Ident::invalid(), vec![], ast::Generics::default(),
                ast::ImplItemKind::Macro(mac)))
        } else {
            let (constness, unsafety, asyncness, abi) = self.parse_fn_front_matter()?;
            let ident = self.parse_ident()?;
            let mut generics = self.parse_generics()?;
            let decl = self.parse_fn_decl_with_self(|p| {
                p.parse_arg_general(true, false, |_| true)
            })?;
            generics.where_clause = self.parse_where_clause()?;
            *at_end = true;
            let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
            let header = ast::FnHeader { abi, unsafety, constness, asyncness };
            Ok((ident, inner_attrs, generics, ast::ImplItemKind::Method(
                ast::MethodSig { header, decl },
                body
            )))
        }
    }

    /// Parses `trait Foo { ... }` or `trait Foo = Bar;`.
    fn parse_item_trait(&mut self, is_auto: IsAuto, unsafety: Unsafety) -> PResult<'a, ItemInfo> {
        let ident = self.parse_ident()?;
        let mut tps = self.parse_generics()?;

        // Parse optional colon and supertrait bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_generic_bounds(Some(self.prev_span))?
        } else {
            Vec::new()
        };

        if self.eat(&token::Eq) {
            // it's a trait alias
            let bounds = self.parse_generic_bounds(None)?;
            tps.where_clause = self.parse_where_clause()?;
            self.expect(&token::Semi)?;
            if is_auto == IsAuto::Yes {
                let msg = "trait aliases cannot be `auto`";
                self.struct_span_err(self.prev_span, msg)
                    .span_label(self.prev_span, msg)
                    .emit();
            }
            if unsafety != Unsafety::Normal {
                let msg = "trait aliases cannot be `unsafe`";
                self.struct_span_err(self.prev_span, msg)
                    .span_label(self.prev_span, msg)
                    .emit();
            }
            Ok((ident, ItemKind::TraitAlias(tps, bounds), None))
        } else {
            // it's a normal trait
            tps.where_clause = self.parse_where_clause()?;
            self.expect(&token::OpenDelim(token::Brace))?;
            let mut trait_items = vec![];
            while !self.eat(&token::CloseDelim(token::Brace)) {
                if let token::DocComment(_) = self.token.kind {
                    if self.look_ahead(1,
                    |tok| tok == &token::CloseDelim(token::Brace)) {
                        let mut err = self.diagnostic().struct_span_err_with_code(
                            self.token.span,
                            "found a documentation comment that doesn't document anything",
                            DiagnosticId::Error("E0584".into()),
                        );
                        err.help("doc comments must come before what they document, maybe a \
                            comment was intended with `//`?",
                        );
                        err.emit();
                        self.bump();
                        continue;
                    }
                }
                let mut at_end = false;
                match self.parse_trait_item(&mut at_end) {
                    Ok(item) => trait_items.push(item),
                    Err(mut e) => {
                        e.emit();
                        if !at_end {
                            self.recover_stmt_(SemiColonMode::Break, BlockMode::Break);
                        }
                    }
                }
            }
            Ok((ident, ItemKind::Trait(is_auto, unsafety, tps, bounds, trait_items), None))
        }
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

    fn parse_impl_body(&mut self) -> PResult<'a, (Vec<ImplItem>, Vec<Attribute>)> {
        self.expect(&token::OpenDelim(token::Brace))?;
        let attrs = self.parse_inner_attributes()?;

        let mut impl_items = Vec::new();
        while !self.eat(&token::CloseDelim(token::Brace)) {
            let mut at_end = false;
            match self.parse_impl_item(&mut at_end) {
                Ok(impl_item) => impl_items.push(impl_item),
                Err(mut err) => {
                    err.emit();
                    if !at_end {
                        self.recover_stmt_(SemiColonMode::Break, BlockMode::Break);
                    }
                }
            }
        }
        Ok((impl_items, attrs))
    }

    /// Parses an implementation item, `impl` keyword is already parsed.
    ///
    ///    impl<'a, T> TYPE { /* impl items */ }
    ///    impl<'a, T> TRAIT for TYPE { /* impl items */ }
    ///    impl<'a, T> !TRAIT for TYPE { /* impl items */ }
    ///
    /// We actually parse slightly more relaxed grammar for better error reporting and recovery.
    ///     `impl` GENERICS `!`? TYPE `for`? (TYPE | `..`) (`where` PREDICATES)? `{` BODY `}`
    ///     `impl` GENERICS `!`? TYPE (`where` PREDICATES)? `{` BODY `}`
    fn parse_item_impl(&mut self, unsafety: Unsafety, defaultness: Defaultness)
                       -> PResult<'a, ItemInfo> {
        // First, parse generic parameters if necessary.
        let mut generics = if self.choose_generics_over_qpath() {
            self.parse_generics()?
        } else {
            ast::Generics::default()
        };

        // Disambiguate `impl !Trait for Type { ... }` and `impl ! { ... }` for the never type.
        let polarity = if self.check(&token::Not) && self.look_ahead(1, |t| t.can_begin_type()) {
            self.bump(); // `!`
            ast::ImplPolarity::Negative
        } else {
            ast::ImplPolarity::Positive
        };

        // Parse both types and traits as a type, then reinterpret if necessary.
        let err_path = |span| ast::Path::from_ident(Ident::new(kw::Invalid, span));
        let ty_first = if self.token.is_keyword(kw::For) &&
                          self.look_ahead(1, |t| t != &token::Lt) {
            let span = self.prev_span.between(self.token.span);
            self.struct_span_err(span, "missing trait in a trait impl").emit();
            P(Ty { node: TyKind::Path(None, err_path(span)), span, id: ast::DUMMY_NODE_ID })
        } else {
            self.parse_ty()?
        };

        // If `for` is missing we try to recover.
        let has_for = self.eat_keyword(kw::For);
        let missing_for_span = self.prev_span.between(self.token.span);

        let ty_second = if self.token == token::DotDot {
            // We need to report this error after `cfg` expansion for compatibility reasons
            self.bump(); // `..`, do not add it to expected tokens
            Some(DummyResult::raw_ty(self.prev_span, true))
        } else if has_for || self.token.can_begin_type() {
            Some(self.parse_ty()?)
        } else {
            None
        };

        generics.where_clause = self.parse_where_clause()?;

        let (impl_items, attrs) = self.parse_impl_body()?;

        let item_kind = match ty_second {
            Some(ty_second) => {
                // impl Trait for Type
                if !has_for {
                    self.struct_span_err(missing_for_span, "missing `for` in a trait impl")
                        .span_suggestion_short(
                            missing_for_span,
                            "add `for` here",
                            " for ".to_string(),
                            Applicability::MachineApplicable,
                        ).emit();
                }

                let ty_first = ty_first.into_inner();
                let path = match ty_first.node {
                    // This notably includes paths passed through `ty` macro fragments (#46438).
                    TyKind::Path(None, path) => path,
                    _ => {
                        self.span_err(ty_first.span, "expected a trait, found type");
                        err_path(ty_first.span)
                    }
                };
                let trait_ref = TraitRef { path, ref_id: ty_first.id };

                ItemKind::Impl(unsafety, polarity, defaultness,
                               generics, Some(trait_ref), ty_second, impl_items)
            }
            None => {
                // impl Type
                ItemKind::Impl(unsafety, polarity, defaultness,
                               generics, None, ty_first, impl_items)
            }
        };

        Ok((Ident::invalid(), item_kind, Some(attrs)))
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

    /// Parses `struct Foo { ... }`.
    fn parse_item_struct(&mut self) -> PResult<'a, ItemInfo> {
        let class_name = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        // There is a special case worth noting here, as reported in issue #17904.
        // If we are parsing a tuple struct it is the case that the where clause
        // should follow the field list. Like so:
        //
        // struct Foo<T>(T) where T: Copy;
        //
        // If we are parsing a normal record-style struct it is the case
        // that the where clause comes before the body, and after the generics.
        // So if we look ahead and see a brace or a where-clause we begin
        // parsing a record style struct.
        //
        // Otherwise if we look ahead and see a paren we parse a tuple-style
        // struct.

        let vdata = if self.token.is_keyword(kw::Where) {
            generics.where_clause = self.parse_where_clause()?;
            if self.eat(&token::Semi) {
                // If we see a: `struct Foo<T> where T: Copy;` style decl.
                VariantData::Unit(ast::DUMMY_NODE_ID)
            } else {
                // If we see: `struct Foo<T> where T: Copy { ... }`
                let (fields, recovered) = self.parse_record_struct_body()?;
                VariantData::Struct(fields, recovered)
            }
        // No `where` so: `struct Foo<T>;`
        } else if self.eat(&token::Semi) {
            VariantData::Unit(ast::DUMMY_NODE_ID)
        // Record-style struct definition
        } else if self.token == token::OpenDelim(token::Brace) {
            let (fields, recovered) = self.parse_record_struct_body()?;
            VariantData::Struct(fields, recovered)
        // Tuple-style struct definition with optional where-clause.
        } else if self.token == token::OpenDelim(token::Paren) {
            let body = VariantData::Tuple(self.parse_tuple_struct_body()?, ast::DUMMY_NODE_ID);
            generics.where_clause = self.parse_where_clause()?;
            self.expect(&token::Semi)?;
            body
        } else {
            let token_str = self.this_token_descr();
            let mut err = self.fatal(&format!(
                "expected `where`, `{{`, `(`, or `;` after struct name, found {}",
                token_str
            ));
            err.span_label(self.token.span, "expected `where`, `{`, `(`, or `;` after struct name");
            return Err(err);
        };

        Ok((class_name, ItemKind::Struct(vdata, generics), None))
    }

    /// Parses `union Foo { ... }`.
    fn parse_item_union(&mut self) -> PResult<'a, ItemInfo> {
        let class_name = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        let vdata = if self.token.is_keyword(kw::Where) {
            generics.where_clause = self.parse_where_clause()?;
            let (fields, recovered) = self.parse_record_struct_body()?;
            VariantData::Struct(fields, recovered)
        } else if self.token == token::OpenDelim(token::Brace) {
            let (fields, recovered) = self.parse_record_struct_body()?;
            VariantData::Struct(fields, recovered)
        } else {
            let token_str = self.this_token_descr();
            let mut err = self.fatal(&format!(
                "expected `where` or `{{` after union name, found {}", token_str));
            err.span_label(self.token.span, "expected `where` or `{` after union name");
            return Err(err);
        };

        Ok((class_name, ItemKind::Union(vdata, generics), None))
    }

    fn parse_record_struct_body(
        &mut self,
    ) -> PResult<'a, (Vec<StructField>, /* recovered */ bool)> {
        let mut fields = Vec::new();
        let mut recovered = false;
        if self.eat(&token::OpenDelim(token::Brace)) {
            while self.token != token::CloseDelim(token::Brace) {
                let field = self.parse_struct_decl_field().map_err(|e| {
                    self.recover_stmt();
                    recovered = true;
                    e
                });
                match field {
                    Ok(field) => fields.push(field),
                    Err(mut err) => {
                        err.emit();
                    }
                }
            }
            self.eat(&token::CloseDelim(token::Brace));
        } else {
            let token_str = self.this_token_descr();
            let mut err = self.fatal(&format!(
                    "expected `where`, or `{{` after struct name, found {}", token_str));
            err.span_label(self.token.span, "expected `where`, or `{` after struct name");
            return Err(err);
        }

        Ok((fields, recovered))
    }

    fn parse_tuple_struct_body(&mut self) -> PResult<'a, Vec<StructField>> {
        // This is the case where we find `struct Foo<T>(T) where T: Copy;`
        // Unit like structs are handled in parse_item_struct function
        let fields = self.parse_unspanned_seq(
            &token::OpenDelim(token::Paren),
            &token::CloseDelim(token::Paren),
            SeqSep::trailing_allowed(token::Comma),
            |p| {
                let attrs = p.parse_outer_attributes()?;
                let lo = p.token.span;
                let vis = p.parse_visibility(true)?;
                let ty = p.parse_ty()?;
                Ok(StructField {
                    span: lo.to(ty.span),
                    vis,
                    ident: None,
                    id: ast::DUMMY_NODE_ID,
                    ty,
                    attrs,
                })
            })?;

        Ok(fields)
    }

    /// Parses a structure field declaration.
    fn parse_single_struct_field(&mut self,
                                     lo: Span,
                                     vis: Visibility,
                                     attrs: Vec<Attribute> )
                                     -> PResult<'a, StructField> {
        let mut seen_comma: bool = false;
        let a_var = self.parse_name_and_ty(lo, vis, attrs)?;
        if self.token == token::Comma {
            seen_comma = true;
        }
        match self.token.kind {
            token::Comma => {
                self.bump();
            }
            token::CloseDelim(token::Brace) => {}
            token::DocComment(_) => {
                let previous_span = self.prev_span;
                let mut err = self.span_fatal_err(self.token.span, Error::UselessDocComment);
                self.bump(); // consume the doc comment
                let comma_after_doc_seen = self.eat(&token::Comma);
                // `seen_comma` is always false, because we are inside doc block
                // condition is here to make code more readable
                if seen_comma == false && comma_after_doc_seen == true {
                    seen_comma = true;
                }
                if comma_after_doc_seen || self.token == token::CloseDelim(token::Brace) {
                    err.emit();
                } else {
                    if seen_comma == false {
                        let sp = self.sess.source_map().next_point(previous_span);
                        err.span_suggestion(
                            sp,
                            "missing comma here",
                            ",".into(),
                            Applicability::MachineApplicable
                        );
                    }
                    return Err(err);
                }
            }
            _ => {
                let sp = self.sess.source_map().next_point(self.prev_span);
                let mut err = self.struct_span_err(sp, &format!("expected `,`, or `}}`, found {}",
                                                                self.this_token_descr()));
                if self.token.is_ident() {
                    // This is likely another field; emit the diagnostic and keep going
                    err.span_suggestion(
                        sp,
                        "try adding a comma",
                        ",".into(),
                        Applicability::MachineApplicable,
                    );
                    err.emit();
                } else {
                    return Err(err)
                }
            }
        }
        Ok(a_var)
    }

    /// Parses an element of a struct declaration.
    fn parse_struct_decl_field(&mut self) -> PResult<'a, StructField> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;
        let vis = self.parse_visibility(false)?;
        self.parse_single_struct_field(lo, vis, attrs)
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
                let mut err = struct_span_err!(self.sess.span_diagnostic, sp, E0704, "{}", msg);
                err.help(suggestion);
                err.span_suggestion(
                    sp, &help_msg, format!("in {}", path), Applicability::MachineApplicable
                );
                err.emit();  // emit diagnostic, but continue with public visibility
            }
        }

        Ok(respan(lo, VisibilityKind::Public))
    }

    /// Parses defaultness (i.e., `default` or nothing).
    fn parse_defaultness(&mut self) -> Defaultness {
        // `pub` is included for better error messages
        if self.check_keyword(kw::Default) &&
            self.is_keyword_ahead(1, &[
                kw::Impl,
                kw::Const,
                kw::Fn,
                kw::Unsafe,
                kw::Extern,
                kw::Type,
                kw::Pub,
            ])
        {
            self.bump(); // `default`
            Defaultness::Default
        } else {
            Defaultness::Final
        }
    }

    /// Given a termination token, parses all of the items in a module.
    fn parse_mod_items(&mut self, term: &TokenKind, inner_lo: Span) -> PResult<'a, Mod> {
        let mut items = vec![];
        while let Some(item) = self.parse_item()? {
            items.push(item);
            self.maybe_consume_incorrect_semicolon(&items);
        }

        if !self.eat(term) {
            let token_str = self.this_token_descr();
            if !self.maybe_consume_incorrect_semicolon(&items) {
                let mut err = self.fatal(&format!("expected item, found {}", token_str));
                err.span_label(self.token.span, "expected item");
                return Err(err);
            }
        }

        let hi = if self.token.span.is_dummy() {
            inner_lo
        } else {
            self.prev_span
        };

        Ok(ast::Mod {
            inner: inner_lo.to(hi),
            items,
            inline: true
        })
    }

    fn parse_item_const(&mut self, m: Option<Mutability>) -> PResult<'a, ItemInfo> {
        let id = if m.is_none() { self.parse_ident_or_underscore() } else { self.parse_ident() }?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        self.expect(&token::Eq)?;
        let e = self.parse_expr()?;
        self.expect(&token::Semi)?;
        let item = match m {
            Some(m) => ItemKind::Static(ty, m, e),
            None => ItemKind::Const(ty, e),
        };
        Ok((id, item, None))
    }

    /// Parse a `mod <foo> { ... }` or `mod <foo>;` item
    fn parse_item_mod(&mut self, outer_attrs: &[Attribute]) -> PResult<'a, ItemInfo> {
        let (in_cfg, outer_attrs) = {
            let mut strip_unconfigured = crate::config::StripUnconfigured {
                sess: self.sess,
                features: None, // don't perform gated feature checking
            };
            let mut outer_attrs = outer_attrs.to_owned();
            strip_unconfigured.process_cfg_attrs(&mut outer_attrs);
            (!self.cfg_mods || strip_unconfigured.in_cfg(&outer_attrs), outer_attrs)
        };

        let id_span = self.token.span;
        let id = self.parse_ident()?;
        if self.eat(&token::Semi) {
            if in_cfg && self.recurse_into_file_modules {
                // This mod is in an external file. Let's go get it!
                let ModulePathSuccess { path, directory_ownership, warn } =
                    self.submod_path(id, &outer_attrs, id_span)?;
                let (module, mut attrs) =
                    self.eval_src_mod(path, directory_ownership, id.to_string(), id_span)?;
                // Record that we fetched the mod from an external file
                if warn {
                    let attr = Attribute {
                        id: attr::mk_attr_id(),
                        style: ast::AttrStyle::Outer,
                        path: ast::Path::from_ident(
                            Ident::with_empty_ctxt(sym::warn_directory_ownership)),
                        tokens: TokenStream::empty(),
                        is_sugared_doc: false,
                        span: DUMMY_SP,
                    };
                    attr::mark_known(&attr);
                    attrs.push(attr);
                }
                Ok((id, ItemKind::Mod(module), Some(attrs)))
            } else {
                let placeholder = ast::Mod {
                    inner: DUMMY_SP,
                    items: Vec::new(),
                    inline: false
                };
                Ok((id, ItemKind::Mod(placeholder), None))
            }
        } else {
            let old_directory = self.directory.clone();
            self.push_directory(id, &outer_attrs);

            self.expect(&token::OpenDelim(token::Brace))?;
            let mod_inner_lo = self.token.span;
            let attrs = self.parse_inner_attributes()?;
            let module = self.parse_mod_items(&token::CloseDelim(token::Brace), mod_inner_lo)?;

            self.directory = old_directory;
            Ok((id, ItemKind::Mod(module), Some(attrs)))
        }
    }

    fn push_directory(&mut self, id: Ident, attrs: &[Attribute]) {
        if let Some(path) = attr::first_attr_value_str_by_name(attrs, sym::path) {
            self.directory.path.to_mut().push(&path.as_str());
            self.directory.ownership = DirectoryOwnership::Owned { relative: None };
        } else {
            // We have to push on the current module name in the case of relative
            // paths in order to ensure that any additional module paths from inline
            // `mod x { ... }` come after the relative extension.
            //
            // For example, a `mod z { ... }` inside `x/y.rs` should set the current
            // directory path to `/x/y/z`, not `/x/z` with a relative offset of `y`.
            if let DirectoryOwnership::Owned { relative } = &mut self.directory.ownership {
                if let Some(ident) = relative.take() { // remove the relative offset
                    self.directory.path.to_mut().push(ident.as_str());
                }
            }
            self.directory.path.to_mut().push(&id.as_str());
        }
    }

    pub fn submod_path_from_attr(attrs: &[Attribute], dir_path: &Path) -> Option<PathBuf> {
        if let Some(s) = attr::first_attr_value_str_by_name(attrs, sym::path) {
            let s = s.as_str();

            // On windows, the base path might have the form
            // `\\?\foo\bar` in which case it does not tolerate
            // mixed `/` and `\` separators, so canonicalize
            // `/` to `\`.
            #[cfg(windows)]
            let s = s.replace("/", "\\");
            Some(dir_path.join(s))
        } else {
            None
        }
    }

    /// Returns a path to a module.
    pub fn default_submod_path(
        id: ast::Ident,
        relative: Option<ast::Ident>,
        dir_path: &Path,
        source_map: &SourceMap) -> ModulePath
    {
        // If we're in a foo.rs file instead of a mod.rs file,
        // we need to look for submodules in
        // `./foo/<id>.rs` and `./foo/<id>/mod.rs` rather than
        // `./<id>.rs` and `./<id>/mod.rs`.
        let relative_prefix_string;
        let relative_prefix = if let Some(ident) = relative {
            relative_prefix_string = format!("{}{}", ident.as_str(), path::MAIN_SEPARATOR);
            &relative_prefix_string
        } else {
            ""
        };

        let mod_name = id.to_string();
        let default_path_str = format!("{}{}.rs", relative_prefix, mod_name);
        let secondary_path_str = format!("{}{}{}mod.rs",
                                         relative_prefix, mod_name, path::MAIN_SEPARATOR);
        let default_path = dir_path.join(&default_path_str);
        let secondary_path = dir_path.join(&secondary_path_str);
        let default_exists = source_map.file_exists(&default_path);
        let secondary_exists = source_map.file_exists(&secondary_path);

        let result = match (default_exists, secondary_exists) {
            (true, false) => Ok(ModulePathSuccess {
                path: default_path,
                directory_ownership: DirectoryOwnership::Owned {
                    relative: Some(id),
                },
                warn: false,
            }),
            (false, true) => Ok(ModulePathSuccess {
                path: secondary_path,
                directory_ownership: DirectoryOwnership::Owned {
                    relative: None,
                },
                warn: false,
            }),
            (false, false) => Err(Error::FileNotFoundForModule {
                mod_name: mod_name.clone(),
                default_path: default_path_str,
                secondary_path: secondary_path_str,
                dir_path: dir_path.display().to_string(),
            }),
            (true, true) => Err(Error::DuplicatePaths {
                mod_name: mod_name.clone(),
                default_path: default_path_str,
                secondary_path: secondary_path_str,
            }),
        };

        ModulePath {
            name: mod_name,
            path_exists: default_exists || secondary_exists,
            result,
        }
    }

    fn submod_path(&mut self,
                   id: ast::Ident,
                   outer_attrs: &[Attribute],
                   id_sp: Span)
                   -> PResult<'a, ModulePathSuccess> {
        if let Some(path) = Parser::submod_path_from_attr(outer_attrs, &self.directory.path) {
            return Ok(ModulePathSuccess {
                directory_ownership: match path.file_name().and_then(|s| s.to_str()) {
                    // All `#[path]` files are treated as though they are a `mod.rs` file.
                    // This means that `mod foo;` declarations inside `#[path]`-included
                    // files are siblings,
                    //
                    // Note that this will produce weirdness when a file named `foo.rs` is
                    // `#[path]` included and contains a `mod foo;` declaration.
                    // If you encounter this, it's your own darn fault :P
                    Some(_) => DirectoryOwnership::Owned { relative: None },
                    _ => DirectoryOwnership::UnownedViaMod(true),
                },
                path,
                warn: false,
            });
        }

        let relative = match self.directory.ownership {
            DirectoryOwnership::Owned { relative } => relative,
            DirectoryOwnership::UnownedViaBlock |
            DirectoryOwnership::UnownedViaMod(_) => None,
        };
        let paths = Parser::default_submod_path(
                        id, relative, &self.directory.path, self.sess.source_map());

        match self.directory.ownership {
            DirectoryOwnership::Owned { .. } => {
                paths.result.map_err(|err| self.span_fatal_err(id_sp, err))
            },
            DirectoryOwnership::UnownedViaBlock => {
                let msg =
                    "Cannot declare a non-inline module inside a block \
                    unless it has a path attribute";
                let mut err = self.diagnostic().struct_span_err(id_sp, msg);
                if paths.path_exists {
                    let msg = format!("Maybe `use` the module `{}` instead of redeclaring it",
                                      paths.name);
                    err.span_note(id_sp, &msg);
                }
                Err(err)
            }
            DirectoryOwnership::UnownedViaMod(warn) => {
                if warn {
                    if let Ok(result) = paths.result {
                        return Ok(ModulePathSuccess { warn: true, ..result });
                    }
                }
                let mut err = self.diagnostic().struct_span_err(id_sp,
                    "cannot declare a new module at this location");
                if !id_sp.is_dummy() {
                    let src_path = self.sess.source_map().span_to_filename(id_sp);
                    if let FileName::Real(src_path) = src_path {
                        if let Some(stem) = src_path.file_stem() {
                            let mut dest_path = src_path.clone();
                            dest_path.set_file_name(stem);
                            dest_path.push("mod.rs");
                            err.span_note(id_sp,
                                    &format!("maybe move this module `{}` to its own \
                                                directory via `{}`", src_path.display(),
                                            dest_path.display()));
                        }
                    }
                }
                if paths.path_exists {
                    err.span_note(id_sp,
                                  &format!("... or maybe `use` the module `{}` instead \
                                            of possibly redeclaring it",
                                           paths.name));
                }
                Err(err)
            }
        }
    }

    /// Reads a module from a source file.
    fn eval_src_mod(&mut self,
                    path: PathBuf,
                    directory_ownership: DirectoryOwnership,
                    name: String,
                    id_sp: Span)
                    -> PResult<'a, (ast::Mod, Vec<Attribute> )> {
        let mut included_mod_stack = self.sess.included_mod_stack.borrow_mut();
        if let Some(i) = included_mod_stack.iter().position(|p| *p == path) {
            let mut err = String::from("circular modules: ");
            let len = included_mod_stack.len();
            for p in &included_mod_stack[i.. len] {
                err.push_str(&p.to_string_lossy());
                err.push_str(" -> ");
            }
            err.push_str(&path.to_string_lossy());
            return Err(self.span_fatal(id_sp, &err[..]));
        }
        included_mod_stack.push(path.clone());
        drop(included_mod_stack);

        let mut p0 =
            new_sub_parser_from_file(self.sess, &path, directory_ownership, Some(name), id_sp);
        p0.cfg_mods = self.cfg_mods;
        let mod_inner_lo = p0.token.span;
        let mod_attrs = p0.parse_inner_attributes()?;
        let mut m0 = p0.parse_mod_items(&token::Eof, mod_inner_lo)?;
        m0.inline = false;
        self.sess.included_mod_stack.borrow_mut().pop();
        Ok((m0, mod_attrs))
    }

    /// Parses a function declaration from a foreign module.
    fn parse_item_foreign_fn(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                             -> PResult<'a, ForeignItem> {
        self.expect_keyword(kw::Fn)?;

        let (ident, mut generics) = self.parse_fn_header()?;
        let decl = self.parse_fn_decl(true)?;
        generics.where_clause = self.parse_where_clause()?;
        let hi = self.token.span;
        self.expect(&token::Semi)?;
        Ok(ast::ForeignItem {
            ident,
            attrs,
            node: ForeignItemKind::Fn(decl, generics),
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis,
        })
    }

    /// Parses a static item from a foreign module.
    /// Assumes that the `static` keyword is already parsed.
    fn parse_item_foreign_static(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                                 -> PResult<'a, ForeignItem> {
        let mutbl = self.parse_mutability();
        let ident = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        let hi = self.token.span;
        self.expect(&token::Semi)?;
        Ok(ForeignItem {
            ident,
            attrs,
            node: ForeignItemKind::Static(ty, mutbl),
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis,
        })
    }

    /// Parses a type from a foreign module.
    fn parse_item_foreign_type(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                             -> PResult<'a, ForeignItem> {
        self.expect_keyword(kw::Type)?;

        let ident = self.parse_ident()?;
        let hi = self.token.span;
        self.expect(&token::Semi)?;
        Ok(ast::ForeignItem {
            ident,
            attrs,
            node: ForeignItemKind::Ty,
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis
        })
    }

    fn parse_crate_name_with_dashes(&mut self) -> PResult<'a, ast::Ident> {
        let error_msg = "crate name using dashes are not valid in `extern crate` statements";
        let suggestion_msg = "if the original crate name uses dashes you need to use underscores \
                              in the code";
        let mut ident = if self.token.is_keyword(kw::SelfLower) {
            self.parse_path_segment_ident()
        } else {
            self.parse_ident()
        }?;
        let mut idents = vec![];
        let mut replacement = vec![];
        let mut fixed_crate_name = false;
        // Accept `extern crate name-like-this` for better diagnostics
        let dash = token::BinOp(token::BinOpToken::Minus);
        if self.token == dash {  // Do not include `-` as part of the expected tokens list
            while self.eat(&dash) {
                fixed_crate_name = true;
                replacement.push((self.prev_span, "_".to_string()));
                idents.push(self.parse_ident()?);
            }
        }
        if fixed_crate_name {
            let fixed_name_sp = ident.span.to(idents.last().unwrap().span);
            let mut fixed_name = format!("{}", ident.name);
            for part in idents {
                fixed_name.push_str(&format!("_{}", part.name));
            }
            ident = Ident::from_str(&fixed_name).with_span_pos(fixed_name_sp);

            let mut err = self.struct_span_err(fixed_name_sp, error_msg);
            err.span_label(fixed_name_sp, "dash-separated idents are not valid");
            err.multipart_suggestion(
                suggestion_msg,
                replacement,
                Applicability::MachineApplicable,
            );
            err.emit();
        }
        Ok(ident)
    }

    /// Parses `extern crate` links.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate foo;
    /// extern crate bar as foo;
    /// ```
    fn parse_item_extern_crate(&mut self,
                               lo: Span,
                               visibility: Visibility,
                               attrs: Vec<Attribute>)
                               -> PResult<'a, P<Item>> {
        // Accept `extern crate name-like-this` for better diagnostics
        let orig_name = self.parse_crate_name_with_dashes()?;
        let (item_name, orig_name) = if let Some(rename) = self.parse_rename()? {
            (rename, Some(orig_name.name))
        } else {
            (orig_name, None)
        };
        self.expect(&token::Semi)?;

        let span = lo.to(self.prev_span);
        Ok(self.mk_item(span, item_name, ItemKind::ExternCrate(orig_name), visibility, attrs))
    }

    /// Parses `extern` for foreign ABIs modules.
    ///
    /// `extern` is expected to have been
    /// consumed before calling this method.
    ///
    /// # Examples
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// extern "C" {}
    /// extern {}
    /// ```
    fn parse_item_foreign_mod(&mut self,
                              lo: Span,
                              opt_abi: Option<Abi>,
                              visibility: Visibility,
                              mut attrs: Vec<Attribute>)
                              -> PResult<'a, P<Item>> {
        self.expect(&token::OpenDelim(token::Brace))?;

        let abi = opt_abi.unwrap_or(Abi::C);

        attrs.extend(self.parse_inner_attributes()?);

        let mut foreign_items = vec![];
        while !self.eat(&token::CloseDelim(token::Brace)) {
            foreign_items.push(self.parse_foreign_item()?);
        }

        let prev_span = self.prev_span;
        let m = ast::ForeignMod {
            abi,
            items: foreign_items
        };
        let invalid = Ident::invalid();
        Ok(self.mk_item(lo.to(prev_span), invalid, ItemKind::ForeignMod(m), visibility, attrs))
    }

    /// Parses `type Foo = Bar;`
    /// or
    /// `existential type Foo: Bar;`
    /// or
    /// `return `None``
    /// without modifying the parser state.
    fn eat_type(&mut self) -> Option<PResult<'a, (Ident, AliasKind, ast::Generics)>> {
        // This parses the grammar:
        //     Ident ["<"...">"] ["where" ...] ("=" | ":") Ty ";"
        if self.check_keyword(kw::Type) ||
           self.check_keyword(kw::Existential) &&
                self.is_keyword_ahead(1, &[kw::Type]) {
            let existential = self.eat_keyword(kw::Existential);
            assert!(self.eat_keyword(kw::Type));
            Some(self.parse_existential_or_alias(existential))
        } else {
            None
        }
    }

    /// Parses a type alias or existential type.
    fn parse_existential_or_alias(
        &mut self,
        existential: bool,
    ) -> PResult<'a, (Ident, AliasKind, ast::Generics)> {
        let ident = self.parse_ident()?;
        let mut tps = self.parse_generics()?;
        tps.where_clause = self.parse_where_clause()?;
        let alias = if existential {
            self.expect(&token::Colon)?;
            let bounds = self.parse_generic_bounds(Some(self.prev_span))?;
            AliasKind::Existential(bounds)
        } else {
            self.expect(&token::Eq)?;
            let ty = self.parse_ty()?;
            AliasKind::Weak(ty)
        };
        self.expect(&token::Semi)?;
        Ok((ident, alias, tps))
    }

    /// Parses the part of an enum declaration following the `{`.
    fn parse_enum_def(&mut self, _generics: &ast::Generics) -> PResult<'a, EnumDef> {
        let mut variants = Vec::new();
        while self.token != token::CloseDelim(token::Brace) {
            let variant_attrs = self.parse_outer_attributes()?;
            let vlo = self.token.span;

            self.eat_bad_pub();
            let ident = self.parse_ident()?;

            let struct_def = if self.check(&token::OpenDelim(token::Brace)) {
                // Parse a struct variant.
                let (fields, recovered) = self.parse_record_struct_body()?;
                VariantData::Struct(fields, recovered)
            } else if self.check(&token::OpenDelim(token::Paren)) {
                VariantData::Tuple(
                    self.parse_tuple_struct_body()?,
                    ast::DUMMY_NODE_ID,
                )
            } else {
                VariantData::Unit(ast::DUMMY_NODE_ID)
            };

            let disr_expr = if self.eat(&token::Eq) {
                Some(AnonConst {
                    id: ast::DUMMY_NODE_ID,
                    value: self.parse_expr()?,
                })
            } else {
                None
            };

            let vr = ast::Variant_ {
                ident,
                id: ast::DUMMY_NODE_ID,
                attrs: variant_attrs,
                data: struct_def,
                disr_expr,
            };
            variants.push(respan(vlo.to(self.prev_span), vr));

            if !self.eat(&token::Comma) {
                if self.token.is_ident() && !self.token.is_reserved_ident() {
                    let sp = self.sess.source_map().next_point(self.prev_span);
                    let mut err = self.struct_span_err(sp, "missing comma");
                    err.span_suggestion_short(
                        sp,
                        "missing comma",
                        ",".to_owned(),
                        Applicability::MaybeIncorrect,
                    );
                    err.emit();
                } else {
                    break;
                }
            }
        }
        self.expect(&token::CloseDelim(token::Brace))?;

        Ok(ast::EnumDef { variants })
    }

    /// Parses an enum declaration.
    fn parse_item_enum(&mut self) -> PResult<'a, ItemInfo> {
        let id = self.parse_ident()?;
        let mut generics = self.parse_generics()?;
        generics.where_clause = self.parse_where_clause()?;
        self.expect(&token::OpenDelim(token::Brace))?;

        let enum_definition = self.parse_enum_def(&generics).map_err(|e| {
            self.recover_stmt();
            self.eat(&token::CloseDelim(token::Brace));
            e
        })?;
        Ok((id, ItemKind::Enum(enum_definition, generics), None))
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
                        let mut err = struct_span_err!(
                            self.sess.span_diagnostic,
                            prev_span,
                            E0703,
                            "invalid ABI: found `{}`",
                            symbol);
                        err.span_label(prev_span, "invalid ABI");
                        err.help(&format!("valid ABIs: {}", abi::all_names().join(", ")));
                        err.emit();
                        Ok(None)
                    }
                }
            }

            _ => Ok(None),
        }
    }

    fn is_static_global(&mut self) -> bool {
        if self.check_keyword(kw::Static) {
            // Check if this could be a closure
            !self.look_ahead(1, |token| {
                if token.is_keyword(kw::Move) {
                    return true;
                }
                match token.kind {
                    token::BinOp(token::Or) | token::OrOr => true,
                    _ => false,
                }
            })
        } else {
            false
        }
    }

    fn parse_item_(
        &mut self,
        attrs: Vec<Attribute>,
        macros_allowed: bool,
        attributes_allowed: bool,
    ) -> PResult<'a, Option<P<Item>>> {
        let mut unclosed_delims = vec![];
        let (ret, tokens) = self.collect_tokens(|this| {
            let item = this.parse_item_implementation(attrs, macros_allowed, attributes_allowed);
            unclosed_delims.append(&mut this.unclosed_delims);
            item
        })?;
        self.unclosed_delims.append(&mut unclosed_delims);

        // Once we've parsed an item and recorded the tokens we got while
        // parsing we may want to store `tokens` into the item we're about to
        // return. Note, though, that we specifically didn't capture tokens
        // related to outer attributes. The `tokens` field here may later be
        // used with procedural macros to convert this item back into a token
        // stream, but during expansion we may be removing attributes as we go
        // along.
        //
        // If we've got inner attributes then the `tokens` we've got above holds
        // these inner attributes. If an inner attribute is expanded we won't
        // actually remove it from the token stream, so we'll just keep yielding
        // it (bad!). To work around this case for now we just avoid recording
        // `tokens` if we detect any inner attributes. This should help keep
        // expansion correct, but we should fix this bug one day!
        Ok(ret.map(|item| {
            item.map(|mut i| {
                if !i.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
                    i.tokens = Some(tokens);
                }
                i
            })
        }))
    }

    /// Parses one of the items allowed by the flags.
    fn parse_item_implementation(
        &mut self,
        attrs: Vec<Attribute>,
        macros_allowed: bool,
        attributes_allowed: bool,
    ) -> PResult<'a, Option<P<Item>>> {
        maybe_whole!(self, NtItem, |item| {
            let mut item = item.into_inner();
            let mut attrs = attrs;
            mem::swap(&mut item.attrs, &mut attrs);
            item.attrs.extend(attrs);
            Some(P(item))
        });

        let lo = self.token.span;

        let visibility = self.parse_visibility(false)?;

        if self.eat_keyword(kw::Use) {
            // USE ITEM
            let item_ = ItemKind::Use(P(self.parse_use_tree()?));
            self.expect(&token::Semi)?;

            let span = lo.to(self.prev_span);
            let item =
                self.mk_item(span, Ident::invalid(), item_, visibility, attrs);
            return Ok(Some(item));
        }

        if self.eat_keyword(kw::Extern) {
            if self.eat_keyword(kw::Crate) {
                return Ok(Some(self.parse_item_extern_crate(lo, visibility, attrs)?));
            }

            let opt_abi = self.parse_opt_abi()?;

            if self.eat_keyword(kw::Fn) {
                // EXTERN FUNCTION ITEM
                let fn_span = self.prev_span;
                let abi = opt_abi.unwrap_or(Abi::C);
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(Unsafety::Normal,
                                       respan(fn_span, IsAsync::NotAsync),
                                       respan(fn_span, Constness::NotConst),
                                       abi)?;
                let prev_span = self.prev_span;
                let item = self.mk_item(lo.to(prev_span),
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return Ok(Some(item));
            } else if self.check(&token::OpenDelim(token::Brace)) {
                return Ok(Some(self.parse_item_foreign_mod(lo, opt_abi, visibility, attrs)?));
            }

            self.unexpected()?;
        }

        if self.is_static_global() {
            self.bump();
            // STATIC ITEM
            let m = if self.eat_keyword(kw::Mut) {
                Mutability::Mutable
            } else {
                Mutability::Immutable
            };
            let (ident, item_, extra_attrs) = self.parse_item_const(Some(m))?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Const) {
            let const_span = self.prev_span;
            if self.check_keyword(kw::Fn)
                || (self.check_keyword(kw::Unsafe)
                    && self.is_keyword_ahead(1, &[kw::Fn])) {
                // CONST FUNCTION ITEM
                let unsafety = self.parse_unsafety();
                self.bump();
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(unsafety,
                                       respan(const_span, IsAsync::NotAsync),
                                       respan(const_span, Constness::Const),
                                       Abi::Rust)?;
                let prev_span = self.prev_span;
                let item = self.mk_item(lo.to(prev_span),
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return Ok(Some(item));
            }

            // CONST ITEM
            if self.eat_keyword(kw::Mut) {
                let prev_span = self.prev_span;
                let mut err = self.diagnostic()
                    .struct_span_err(prev_span, "const globals cannot be mutable");
                err.span_label(prev_span, "cannot be mutable");
                err.span_suggestion(
                    const_span,
                    "you might want to declare a static instead",
                    "static".to_owned(),
                    Applicability::MaybeIncorrect,
                );
                err.emit();
            }
            let (ident, item_, extra_attrs) = self.parse_item_const(None)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }

        // Parse `async unsafe? fn`.
        if self.check_keyword(kw::Async) {
            let async_span = self.token.span;
            if self.is_keyword_ahead(1, &[kw::Fn])
                || self.is_keyword_ahead(2, &[kw::Fn])
            {
                // ASYNC FUNCTION ITEM
                self.bump(); // `async`
                let unsafety = self.parse_unsafety(); // `unsafe`?
                self.expect_keyword(kw::Fn)?; // `fn`
                let fn_span = self.prev_span;
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(unsafety,
                                    respan(async_span, IsAsync::Async {
                                        closure_id: ast::DUMMY_NODE_ID,
                                        return_impl_trait_id: ast::DUMMY_NODE_ID,
                                    }),
                                    respan(fn_span, Constness::NotConst),
                                    Abi::Rust)?;
                let prev_span = self.prev_span;
                let item = self.mk_item(lo.to(prev_span),
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                self.ban_async_in_2015(async_span);
                return Ok(Some(item));
            }
        }
        if self.check_keyword(kw::Unsafe) &&
            self.is_keyword_ahead(1, &[kw::Trait, kw::Auto])
        {
            // UNSAFE TRAIT ITEM
            self.bump(); // `unsafe`
            let is_auto = if self.eat_keyword(kw::Trait) {
                IsAuto::No
            } else {
                self.expect_keyword(kw::Auto)?;
                self.expect_keyword(kw::Trait)?;
                IsAuto::Yes
            };
            let (ident, item_, extra_attrs) =
                self.parse_item_trait(is_auto, Unsafety::Unsafe)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(kw::Impl) ||
           self.check_keyword(kw::Unsafe) &&
                self.is_keyword_ahead(1, &[kw::Impl]) ||
           self.check_keyword(kw::Default) &&
                self.is_keyword_ahead(1, &[kw::Impl, kw::Unsafe]) {
            // IMPL ITEM
            let defaultness = self.parse_defaultness();
            let unsafety = self.parse_unsafety();
            self.expect_keyword(kw::Impl)?;
            let (ident, item, extra_attrs) = self.parse_item_impl(unsafety, defaultness)?;
            let span = lo.to(self.prev_span);
            return Ok(Some(self.mk_item(span, ident, item, visibility,
                                        maybe_append(attrs, extra_attrs))));
        }
        if self.check_keyword(kw::Fn) {
            // FUNCTION ITEM
            self.bump();
            let fn_span = self.prev_span;
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(Unsafety::Normal,
                                   respan(fn_span, IsAsync::NotAsync),
                                   respan(fn_span, Constness::NotConst),
                                   Abi::Rust)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(kw::Unsafe)
            && self.look_ahead(1, |t| *t != token::OpenDelim(token::Brace)) {
            // UNSAFE FUNCTION ITEM
            self.bump(); // `unsafe`
            // `{` is also expected after `unsafe`, in case of error, include it in the diagnostic
            self.check(&token::OpenDelim(token::Brace));
            let abi = if self.eat_keyword(kw::Extern) {
                self.parse_opt_abi()?.unwrap_or(Abi::C)
            } else {
                Abi::Rust
            };
            self.expect_keyword(kw::Fn)?;
            let fn_span = self.prev_span;
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(Unsafety::Unsafe,
                                   respan(fn_span, IsAsync::NotAsync),
                                   respan(fn_span, Constness::NotConst),
                                   abi)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Mod) {
            // MODULE ITEM
            let (ident, item_, extra_attrs) =
                self.parse_item_mod(&attrs[..])?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if let Some(type_) = self.eat_type() {
            let (ident, alias, generics) = type_?;
            // TYPE ITEM
            let item_ = match alias {
                AliasKind::Weak(ty) => ItemKind::Ty(ty, generics),
                AliasKind::Existential(bounds) => ItemKind::Existential(bounds, generics),
            };
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    attrs);
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Enum) {
            // ENUM ITEM
            let (ident, item_, extra_attrs) = self.parse_item_enum()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(kw::Trait)
            || (self.check_keyword(kw::Auto)
                && self.is_keyword_ahead(1, &[kw::Trait]))
        {
            let is_auto = if self.eat_keyword(kw::Trait) {
                IsAuto::No
            } else {
                self.expect_keyword(kw::Auto)?;
                self.expect_keyword(kw::Trait)?;
                IsAuto::Yes
            };
            // TRAIT ITEM
            let (ident, item_, extra_attrs) =
                self.parse_item_trait(is_auto, Unsafety::Normal)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Struct) {
            // STRUCT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_struct()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.is_union_item() {
            // UNION ITEM
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_union()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if let Some(macro_def) = self.eat_macro_def(&attrs, &visibility, lo)? {
            return Ok(Some(macro_def));
        }

        // Verify whether we have encountered a struct or method definition where the user forgot to
        // add the `struct` or `fn` keyword after writing `pub`: `pub S {}`
        if visibility.node.is_pub() &&
            self.check_ident() &&
            self.look_ahead(1, |t| *t != token::Not)
        {
            // Space between `pub` keyword and the identifier
            //
            //     pub   S {}
            //        ^^^ `sp` points here
            let sp = self.prev_span.between(self.token.span);
            let full_sp = self.prev_span.to(self.token.span);
            let ident_sp = self.token.span;
            if self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace)) {
                // possible public struct definition where `struct` was forgotten
                let ident = self.parse_ident().unwrap();
                let msg = format!("add `struct` here to parse `{}` as a public struct",
                                  ident);
                let mut err = self.diagnostic()
                    .struct_span_err(sp, "missing `struct` for struct definition");
                err.span_suggestion_short(
                    sp, &msg, " struct ".into(), Applicability::MaybeIncorrect // speculative
                );
                return Err(err);
            } else if self.look_ahead(1, |t| *t == token::OpenDelim(token::Paren)) {
                let ident = self.parse_ident().unwrap();
                self.bump();  // `(`
                let kw_name = if let Ok(Some(_)) = self.parse_self_arg_with_attrs() {
                    "method"
                } else {
                    "function"
                };
                self.consume_block(token::Paren);
                let (kw, kw_name, ambiguous) = if self.check(&token::RArrow) {
                    self.eat_to_tokens(&[&token::OpenDelim(token::Brace)]);
                    self.bump();  // `{`
                    ("fn", kw_name, false)
                } else if self.check(&token::OpenDelim(token::Brace)) {
                    self.bump();  // `{`
                    ("fn", kw_name, false)
                } else if self.check(&token::Colon) {
                    let kw = "struct";
                    (kw, kw, false)
                } else {
                    ("fn` or `struct", "function or struct", true)
                };

                let msg = format!("missing `{}` for {} definition", kw, kw_name);
                let mut err = self.diagnostic().struct_span_err(sp, &msg);
                if !ambiguous {
                    self.consume_block(token::Brace);
                    let suggestion = format!("add `{}` here to parse `{}` as a public {}",
                                             kw,
                                             ident,
                                             kw_name);
                    err.span_suggestion_short(
                        sp, &suggestion, format!(" {} ", kw), Applicability::MachineApplicable
                    );
                } else {
                    if let Ok(snippet) = self.sess.source_map().span_to_snippet(ident_sp) {
                        err.span_suggestion(
                            full_sp,
                            "if you meant to call a macro, try",
                            format!("{}!", snippet),
                            // this is the `ambiguous` conditional branch
                            Applicability::MaybeIncorrect
                        );
                    } else {
                        err.help("if you meant to call a macro, remove the `pub` \
                                  and add a trailing `!` after the identifier");
                    }
                }
                return Err(err);
            } else if self.look_ahead(1, |t| *t == token::Lt) {
                let ident = self.parse_ident().unwrap();
                self.eat_to_tokens(&[&token::Gt]);
                self.bump();  // `>`
                let (kw, kw_name, ambiguous) = if self.eat(&token::OpenDelim(token::Paren)) {
                    if let Ok(Some(_)) = self.parse_self_arg_with_attrs() {
                        ("fn", "method", false)
                    } else {
                        ("fn", "function", false)
                    }
                } else if self.check(&token::OpenDelim(token::Brace)) {
                    ("struct", "struct", false)
                } else {
                    ("fn` or `struct", "function or struct", true)
                };
                let msg = format!("missing `{}` for {} definition", kw, kw_name);
                let mut err = self.diagnostic().struct_span_err(sp, &msg);
                if !ambiguous {
                    err.span_suggestion_short(
                        sp,
                        &format!("add `{}` here to parse `{}` as a public {}", kw, ident, kw_name),
                        format!(" {} ", kw),
                        Applicability::MachineApplicable,
                    );
                }
                return Err(err);
            }
        }
        self.parse_macro_use_or_failure(attrs, macros_allowed, attributes_allowed, lo, visibility)
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

    /// Parses a foreign item.
    crate fn parse_foreign_item(&mut self) -> PResult<'a, ForeignItem> {
        maybe_whole!(self, NtForeignItem, |ni| ni);

        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;
        let visibility = self.parse_visibility(false)?;

        // FOREIGN STATIC ITEM
        // Treat `const` as `static` for error recovery, but don't add it to expected tokens.
        if self.check_keyword(kw::Static) || self.token.is_keyword(kw::Const) {
            if self.token.is_keyword(kw::Const) {
                self.diagnostic()
                    .struct_span_err(self.token.span, "extern items cannot be `const`")
                    .span_suggestion(
                        self.token.span,
                        "try using a static value",
                        "static".to_owned(),
                        Applicability::MachineApplicable
                    ).emit();
            }
            self.bump(); // `static` or `const`
            return Ok(self.parse_item_foreign_static(visibility, lo, attrs)?);
        }
        // FOREIGN FUNCTION ITEM
        if self.check_keyword(kw::Fn) {
            return Ok(self.parse_item_foreign_fn(visibility, lo, attrs)?);
        }
        // FOREIGN TYPE ITEM
        if self.check_keyword(kw::Type) {
            return Ok(self.parse_item_foreign_type(visibility, lo, attrs)?);
        }

        match self.parse_assoc_macro_invoc("extern", Some(&visibility), &mut false)? {
            Some(mac) => {
                Ok(
                    ForeignItem {
                        ident: Ident::invalid(),
                        span: lo.to(self.prev_span),
                        id: ast::DUMMY_NODE_ID,
                        attrs,
                        vis: visibility,
                        node: ForeignItemKind::Macro(mac),
                    }
                )
            }
            None => {
                if !attrs.is_empty()  {
                    self.expected_item_err(&attrs)?;
                }

                self.unexpected()
            }
        }
    }

    /// This is the fall-through for parsing items.
    fn parse_macro_use_or_failure(
        &mut self,
        attrs: Vec<Attribute> ,
        macros_allowed: bool,
        attributes_allowed: bool,
        lo: Span,
        visibility: Visibility
    ) -> PResult<'a, Option<P<Item>>> {
        if macros_allowed && self.token.is_path_start() &&
                !(self.is_async_fn() && self.token.span.rust_2015()) {
            // MACRO INVOCATION ITEM

            let prev_span = self.prev_span;
            self.complain_if_pub_macro(&visibility.node, prev_span);

            let mac_lo = self.token.span;

            // item macro.
            let path = self.parse_path(PathStyle::Mod)?;
            self.expect(&token::Not)?;
            let (delim, tts) = self.expect_delimited_token_tree()?;
            if delim != MacDelimiter::Brace && !self.eat(&token::Semi) {
                self.report_invalid_macro_expansion_item();
            }

            let hi = self.prev_span;
            let mac = respan(mac_lo.to(hi), Mac_ { path, tts, delim });
            let item =
                self.mk_item(lo.to(hi), Ident::invalid(), ItemKind::Mac(mac), visibility, attrs);
            return Ok(Some(item));
        }

        // FAILURE TO PARSE ITEM
        match visibility.node {
            VisibilityKind::Inherited => {}
            _ => {
                return Err(self.span_fatal(self.prev_span, "unmatched visibility `pub`"));
            }
        }

        if !attributes_allowed && !attrs.is_empty() {
            self.expected_item_err(&attrs)?;
        }
        Ok(None)
    }

    /// Parses a macro invocation inside a `trait`, `impl` or `extern` block.
    fn parse_assoc_macro_invoc(&mut self, item_kind: &str, vis: Option<&Visibility>,
                               at_end: &mut bool) -> PResult<'a, Option<Mac>>
    {
        if self.token.is_path_start() &&
                !(self.is_async_fn() && self.token.span.rust_2015()) {
            let prev_span = self.prev_span;
            let lo = self.token.span;
            let path = self.parse_path(PathStyle::Mod)?;

            if path.segments.len() == 1 {
                if !self.eat(&token::Not) {
                    return Err(self.missing_assoc_item_kind_err(item_kind, prev_span));
                }
            } else {
                self.expect(&token::Not)?;
            }

            if let Some(vis) = vis {
                self.complain_if_pub_macro(&vis.node, prev_span);
            }

            *at_end = true;

            // eat a matched-delimiter token tree:
            let (delim, tts) = self.expect_delimited_token_tree()?;
            if delim != MacDelimiter::Brace {
                self.expect(&token::Semi)?;
            }

            Ok(Some(respan(lo.to(self.prev_span), Mac_ { path, tts, delim })))
        } else {
            Ok(None)
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
        } else {
            &mut self.token_cursor.stack[prev].last_token
        };

        // Pull out the tokens that we've collected from the call to `f` above.
        let mut collected_tokens = match *last_token {
            LastToken::Collecting(ref mut v) => mem::take(v),
            LastToken::Was(_) => panic!("our vector went away?"),
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

    pub fn parse_item(&mut self) -> PResult<'a, Option<P<Item>>> {
        let attrs = self.parse_outer_attributes()?;
        self.parse_item_(attrs, true, false)
    }

    /// `::{` or `::*`
    fn is_import_coupler(&mut self) -> bool {
        self.check(&token::ModSep) &&
            self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace) ||
                                   *t == token::BinOp(token::Star))
    }

    /// Parses a `UseTree`.
    ///
    /// ```
    /// USE_TREE = [`::`] `*` |
    ///            [`::`] `{` USE_TREE_LIST `}` |
    ///            PATH `::` `*` |
    ///            PATH `::` `{` USE_TREE_LIST `}` |
    ///            PATH [`as` IDENT]
    /// ```
    fn parse_use_tree(&mut self) -> PResult<'a, UseTree> {
        let lo = self.token.span;

        let mut prefix = ast::Path { segments: Vec::new(), span: lo.shrink_to_lo() };
        let kind = if self.check(&token::OpenDelim(token::Brace)) ||
                      self.check(&token::BinOp(token::Star)) ||
                      self.is_import_coupler() {
            // `use *;` or `use ::*;` or `use {...};` or `use ::{...};`
            let mod_sep_ctxt = self.token.span.ctxt();
            if self.eat(&token::ModSep) {
                prefix.segments.push(
                    PathSegment::path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt))
                );
            }

            if self.eat(&token::BinOp(token::Star)) {
                UseTreeKind::Glob
            } else {
                UseTreeKind::Nested(self.parse_use_tree_list()?)
            }
        } else {
            // `use path::*;` or `use path::{...};` or `use path;` or `use path as bar;`
            prefix = self.parse_path(PathStyle::Mod)?;

            if self.eat(&token::ModSep) {
                if self.eat(&token::BinOp(token::Star)) {
                    UseTreeKind::Glob
                } else {
                    UseTreeKind::Nested(self.parse_use_tree_list()?)
                }
            } else {
                UseTreeKind::Simple(self.parse_rename()?, ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID)
            }
        };

        Ok(UseTree { prefix, kind, span: lo.to(self.prev_span) })
    }

    /// Parses a `UseTreeKind::Nested(list)`.
    ///
    /// ```
    /// USE_TREE_LIST = Ø | (USE_TREE `,`)* USE_TREE [`,`]
    /// ```
    fn parse_use_tree_list(&mut self) -> PResult<'a, Vec<(UseTree, ast::NodeId)>> {
        self.parse_unspanned_seq(&token::OpenDelim(token::Brace),
                                 &token::CloseDelim(token::Brace),
                                 SeqSep::trailing_allowed(token::Comma), |this| {
            Ok((this.parse_use_tree()?, ast::DUMMY_NODE_ID))
        })
    }

    fn parse_rename(&mut self) -> PResult<'a, Option<Ident>> {
        if self.eat_keyword(kw::As) {
            self.parse_ident_or_underscore().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Parses a source module as a crate. This is the main entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<'a, Crate> {
        let lo = self.token.span;
        let krate = Ok(ast::Crate {
            attrs: self.parse_inner_attributes()?,
            module: self.parse_mod_items(&token::Eof, lo)?,
            span: lo.to(self.token.span),
        });
        krate
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
