// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_target::spec::abi::{self, Abi};
use ast::{AngleBracketedParameterData, ParenthesizedParameterData, AttrStyle, BareFnTy};
use ast::{RegionTyParamBound, TraitTyParamBound, TraitBoundModifier};
use ast::Unsafety;
use ast::{Mod, Arg, Arm, Attribute, BindingMode, TraitItemKind};
use ast::Block;
use ast::{BlockCheckMode, CaptureBy, Movability};
use ast::{Constness, Crate};
use ast::Defaultness;
use ast::EnumDef;
use ast::{Expr, ExprKind, RangeLimits};
use ast::{Field, FnDecl};
use ast::{ForeignItem, ForeignItemKind, FunctionRetTy};
use ast::GenericParam;
use ast::{Ident, ImplItem, IsAuto, Item, ItemKind};
use ast::{Label, Lifetime, LifetimeDef, Lit, LitKind, UintTy};
use ast::Local;
use ast::MacStmtStyle;
use ast::{Mac, Mac_};
use ast::{MutTy, Mutability};
use ast::{Pat, PatKind, PathSegment};
use ast::{PolyTraitRef, QSelf};
use ast::{Stmt, StmtKind};
use ast::{VariantData, StructField};
use ast::StrStyle;
use ast::SelfKind;
use ast::{TraitItem, TraitRef, TraitObjectSyntax};
use ast::{Ty, TyKind, TypeBinding, TyParam, TyParamBounds};
use ast::{Visibility, VisibilityKind, WhereClause, CrateSugar};
use ast::{UseTree, UseTreeKind};
use ast::{BinOpKind, UnOp};
use ast::{RangeEnd, RangeSyntax};
use {ast, attr};
use codemap::{self, CodeMap, Spanned, respan};
use syntax_pos::{self, Span, MultiSpan, BytePos, FileName, DUMMY_SP};
use errors::{self, DiagnosticBuilder};
use parse::{self, classify, token};
use parse::common::SeqSep;
use parse::lexer::TokenAndSpan;
use parse::lexer::comments::{doc_comment_style, strip_doc_comment_decoration};
use parse::{new_sub_parser_from_file, ParseSess, Directory, DirectoryOwnership};
use util::parser::{AssocOp, Fixity};
use print::pprust;
use ptr::P;
use parse::PResult;
use tokenstream::{self, Delimited, ThinTokenStream, TokenTree, TokenStream};
use symbol::{Symbol, keywords};
use util::ThinVec;

use std::cmp;
use std::mem;
use std::path::{self, Path, PathBuf};
use std::slice;

bitflags! {
    pub struct Restrictions: u8 {
        const STMT_EXPR         = 1 << 0;
        const NO_STRUCT_LITERAL = 1 << 1;
    }
}

type ItemInfo = (Ident, ItemKind, Option<Vec<Attribute>>);

/// How to parse a path.
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
    /// without the disambiguator, e.g. `x<y>` - unambiguously a path.
    /// Paths with disambiguators are still accepted, `x::<Y>` - unambiguously a path too.
    Type,
    /// A path with generic arguments disallowed, e.g. `foo::bar::Baz`, used in imports,
    /// visibilities or attributes.
    /// Technically, this variant is unnecessary and e.g. `Expr` can be used instead
    /// (paths in "mod" contexts have to be checked later for absence of generic arguments
    /// anyway, due to macros), but it is used to avoid weird suggestions about expected
    /// tokens when something goes wrong.
    Mod,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SemiColonMode {
    Break,
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockMode {
    Break,
    Ignore,
}

/// Possibly accept an `token::Interpolated` expression (a pre-parsed expression
/// dropped into the token stream, which happens while parsing the result of
/// macro expansion). Placement of these is not as complex as I feared it would
/// be. The important thing is to make sure that lookahead doesn't balk at
/// `token::Interpolated` tokens.
macro_rules! maybe_whole_expr {
    ($p:expr) => {
        if let token::Interpolated(nt) = $p.token.clone() {
            match nt.0 {
                token::NtExpr(ref e) => {
                    $p.bump();
                    return Ok((*e).clone());
                }
                token::NtPath(ref path) => {
                    $p.bump();
                    let span = $p.span;
                    let kind = ExprKind::Path(None, (*path).clone());
                    return Ok($p.mk_expr(span, kind, ThinVec::new()));
                }
                token::NtBlock(ref block) => {
                    $p.bump();
                    let span = $p.span;
                    let kind = ExprKind::Block((*block).clone());
                    return Ok($p.mk_expr(span, kind, ThinVec::new()));
                }
                _ => {},
            };
        }
    }
}

/// As maybe_whole_expr, but for things other than expressions
macro_rules! maybe_whole {
    ($p:expr, $constructor:ident, |$x:ident| $e:expr) => {
        if let token::Interpolated(nt) = $p.token.clone() {
            if let token::$constructor($x) = nt.0.clone() {
                $p.bump();
                return Ok($e);
            }
        }
    };
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
    Other,
}

trait RecoverQPath: Sized {
    const PATH_STYLE: PathStyle = PathStyle::Expr;
    fn to_ty(&self) -> Option<P<Ty>>;
    fn to_recovered(&self, qself: Option<QSelf>, path: ast::Path) -> Self;
    fn to_string(&self) -> String;
}

impl RecoverQPath for Ty {
    const PATH_STYLE: PathStyle = PathStyle::Type;
    fn to_ty(&self) -> Option<P<Ty>> {
        Some(P(self.clone()))
    }
    fn to_recovered(&self, qself: Option<QSelf>, path: ast::Path) -> Self {
        Self { span: path.span, node: TyKind::Path(qself, path), id: self.id }
    }
    fn to_string(&self) -> String {
        pprust::ty_to_string(self)
    }
}

impl RecoverQPath for Pat {
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn to_recovered(&self, qself: Option<QSelf>, path: ast::Path) -> Self {
        Self { span: path.span, node: PatKind::Path(qself, path), id: self.id }
    }
    fn to_string(&self) -> String {
        pprust::pat_to_string(self)
    }
}

impl RecoverQPath for Expr {
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn to_recovered(&self, qself: Option<QSelf>, path: ast::Path) -> Self {
        Self { span: path.span, node: ExprKind::Path(qself, path),
               id: self.id, attrs: self.attrs.clone() }
    }
    fn to_string(&self) -> String {
        pprust::expr_to_string(self)
    }
}

/* ident is handled by common.rs */

#[derive(Clone)]
pub struct Parser<'a> {
    pub sess: &'a ParseSess,
    /// the current token:
    pub token: token::Token,
    /// the span of the current token:
    pub span: Span,
    /// the span of the previous token:
    pub meta_var_span: Option<Span>,
    pub prev_span: Span,
    /// the previous token kind
    prev_token_kind: PrevTokenKind,
    pub restrictions: Restrictions,
    /// Used to determine the path to externally loaded source files
    pub directory: Directory,
    /// Whether to parse sub-modules in other files.
    pub recurse_into_file_modules: bool,
    /// Name of the root module this parser originated from. If `None`, then the
    /// name is not known. This does not change while the parser is descending
    /// into modules, and sub-parsers have new values for this name.
    pub root_module_name: Option<String>,
    pub expected_tokens: Vec<TokenType>,
    token_cursor: TokenCursor,
    pub desugar_doc_comments: bool,
    /// Whether we should configure out of line modules as we parse.
    pub cfg_mods: bool,
}


#[derive(Clone)]
struct TokenCursor {
    frame: TokenCursorFrame,
    stack: Vec<TokenCursorFrame>,
}

#[derive(Clone)]
struct TokenCursorFrame {
    delim: token::DelimToken,
    span: Span,
    open_delim: bool,
    tree_cursor: tokenstream::Cursor,
    close_delim: bool,
    last_token: LastToken,
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
enum LastToken {
    Collecting(Vec<TokenTree>),
    Was(Option<TokenTree>),
}

impl TokenCursorFrame {
    fn new(sp: Span, delimited: &Delimited) -> Self {
        TokenCursorFrame {
            delim: delimited.delim,
            span: sp,
            open_delim: delimited.delim == token::NoDelim,
            tree_cursor: delimited.stream().into_trees(),
            close_delim: delimited.delim == token::NoDelim,
            last_token: LastToken::Was(None),
        }
    }
}

impl TokenCursor {
    fn next(&mut self) -> TokenAndSpan {
        loop {
            let tree = if !self.frame.open_delim {
                self.frame.open_delim = true;
                Delimited { delim: self.frame.delim, tts: TokenStream::empty().into() }
                    .open_tt(self.frame.span)
            } else if let Some(tree) = self.frame.tree_cursor.next() {
                tree
            } else if !self.frame.close_delim {
                self.frame.close_delim = true;
                Delimited { delim: self.frame.delim, tts: TokenStream::empty().into() }
                    .close_tt(self.frame.span)
            } else if let Some(frame) = self.stack.pop() {
                self.frame = frame;
                continue
            } else {
                return TokenAndSpan { tok: token::Eof, sp: syntax_pos::DUMMY_SP }
            };

            match self.frame.last_token {
                LastToken::Collecting(ref mut v) => v.push(tree.clone()),
                LastToken::Was(ref mut t) => *t = Some(tree.clone()),
            }

            match tree {
                TokenTree::Token(sp, tok) => return TokenAndSpan { tok: tok, sp: sp },
                TokenTree::Delimited(sp, ref delimited) => {
                    let frame = TokenCursorFrame::new(sp, delimited);
                    self.stack.push(mem::replace(&mut self.frame, frame));
                }
            }
        }
    }

    fn next_desugared(&mut self) -> TokenAndSpan {
        let (sp, name) = match self.next() {
            TokenAndSpan { sp, tok: token::DocComment(name) } => (sp, name),
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

        let body = TokenTree::Delimited(sp, Delimited {
            delim: token::Bracket,
            tts: [TokenTree::Token(sp, token::Ident(ast::Ident::from_str("doc"), false)),
                  TokenTree::Token(sp, token::Eq),
                  TokenTree::Token(sp, token::Literal(
                      token::StrRaw(Symbol::intern(&stripped), num_of_hashes), None))]
                .iter().cloned().collect::<TokenStream>().into(),
        });

        self.stack.push(mem::replace(&mut self.frame, TokenCursorFrame::new(sp, &Delimited {
            delim: token::NoDelim,
            tts: if doc_comment_style(&name.as_str()) == AttrStyle::Inner {
                [TokenTree::Token(sp, token::Pound), TokenTree::Token(sp, token::Not), body]
                    .iter().cloned().collect::<TokenStream>().into()
            } else {
                [TokenTree::Token(sp, token::Pound), body]
                    .iter().cloned().collect::<TokenStream>().into()
            },
        })));

        self.next()
    }
}

#[derive(PartialEq, Eq, Clone)]
pub enum TokenType {
    Token(token::Token),
    Keyword(keywords::Keyword),
    Operator,
    Lifetime,
    Ident,
    Path,
    Type,
}

impl TokenType {
    fn to_string(&self) -> String {
        match *self {
            TokenType::Token(ref t) => format!("`{}`", Parser::token_to_string(t)),
            TokenType::Keyword(kw) => format!("`{}`", kw.name()),
            TokenType::Operator => "an operator".to_string(),
            TokenType::Lifetime => "lifetime".to_string(),
            TokenType::Ident => "identifier".to_string(),
            TokenType::Path => "path".to_string(),
            TokenType::Type => "type".to_string(),
        }
    }
}

/// Returns true if `IDENT t` can start a type - `IDENT::a::b`, `IDENT<u8, u8>`,
/// `IDENT<<u8 as Trait>::AssocTy>`.
///
/// Types can also be of the form `IDENT(u8, u8) -> u8`, however this assumes
/// that IDENT is not the ident of a fn trait
fn can_continue_type_after_non_fn_ident(t: &token::Token) -> bool {
    t == &token::ModSep || t == &token::Lt ||
    t == &token::BinOp(token::Shl)
}

/// Information about the path to a module.
pub struct ModulePath {
    pub name: String,
    pub path_exists: bool,
    pub result: Result<ModulePathSuccess, Error>,
}

pub struct ModulePathSuccess {
    pub path: PathBuf,
    pub directory_ownership: DirectoryOwnership,
    warn: bool,
}

pub struct ModulePathError {
    pub err_msg: String,
    pub help_msg: String,
}

pub enum Error {
    FileNotFoundForModule {
        mod_name: String,
        default_path: String,
        secondary_path: String,
        dir_path: String,
    },
    DuplicatePaths {
        mod_name: String,
        default_path: String,
        secondary_path: String,
    },
    UselessDocComment,
    InclusiveRangeWithNoEnd,
}

impl Error {
    pub fn span_err<S: Into<MultiSpan>>(self,
                                        sp: S,
                                        handler: &errors::Handler) -> DiagnosticBuilder {
        match self {
            Error::FileNotFoundForModule { ref mod_name,
                                           ref default_path,
                                           ref secondary_path,
                                           ref dir_path } => {
                let mut err = struct_span_err!(handler, sp, E0583,
                                               "file not found for module `{}`", mod_name);
                err.help(&format!("name the file either {} or {} inside the directory \"{}\"",
                                  default_path,
                                  secondary_path,
                                  dir_path));
                err
            }
            Error::DuplicatePaths { ref mod_name, ref default_path, ref secondary_path } => {
                let mut err = struct_span_err!(handler, sp, E0584,
                                               "file for module `{}` found at both {} and {}",
                                               mod_name,
                                               default_path,
                                               secondary_path);
                err.help("delete or rename one of them to remove the ambiguity");
                err
            }
            Error::UselessDocComment => {
                let mut err = struct_span_err!(handler, sp, E0585,
                                  "found a documentation comment that doesn't document anything");
                err.help("doc comments must come before what they document, maybe a comment was \
                          intended with `//`?");
                err
            }
            Error::InclusiveRangeWithNoEnd => {
                let mut err = struct_span_err!(handler, sp, E0586,
                                               "inclusive range with no end");
                err.help("inclusive ranges must be bounded at the end (`..=b` or `a..=b`)");
                err
            }
        }
    }
}

#[derive(Debug)]
pub enum LhsExpr {
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

/// Create a placeholder argument.
fn dummy_arg(span: Span) -> Arg {
    let ident = Ident::new(keywords::Invalid.name(), span);
    let pat = P(Pat {
        id: ast::DUMMY_NODE_ID,
        node: PatKind::Ident(BindingMode::ByValue(Mutability::Immutable), ident, None),
        span,
    });
    let ty = Ty {
        node: TyKind::Err,
        span,
        id: ast::DUMMY_NODE_ID
    };
    Arg { ty: P(ty), pat: pat, id: ast::DUMMY_NODE_ID }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TokenExpectType {
    Expect,
    NoExpect,
}

impl<'a> Parser<'a> {
    pub fn new(sess: &'a ParseSess,
               tokens: TokenStream,
               directory: Option<Directory>,
               recurse_into_file_modules: bool,
               desugar_doc_comments: bool)
               -> Self {
        let mut parser = Parser {
            sess,
            token: token::Whitespace,
            span: syntax_pos::DUMMY_SP,
            prev_span: syntax_pos::DUMMY_SP,
            meta_var_span: None,
            prev_token_kind: PrevTokenKind::Other,
            restrictions: Restrictions::empty(),
            recurse_into_file_modules,
            directory: Directory {
                path: PathBuf::new(),
                ownership: DirectoryOwnership::Owned { relative: None }
            },
            root_module_name: None,
            expected_tokens: Vec::new(),
            token_cursor: TokenCursor {
                frame: TokenCursorFrame::new(syntax_pos::DUMMY_SP, &Delimited {
                    delim: token::NoDelim,
                    tts: tokens.into(),
                }),
                stack: Vec::new(),
            },
            desugar_doc_comments,
            cfg_mods: true,
        };

        let tok = parser.next_tok();
        parser.token = tok.tok;
        parser.span = tok.sp;

        if let Some(directory) = directory {
            parser.directory = directory;
        } else if !parser.span.source_equal(&DUMMY_SP) {
            if let FileName::Real(path) = sess.codemap().span_to_unmapped_path(parser.span) {
                parser.directory.path = path;
                parser.directory.path.pop();
            }
        }

        parser.process_potential_macro_variable();
        parser
    }

    fn next_tok(&mut self) -> TokenAndSpan {
        let mut next = if self.desugar_doc_comments {
            self.token_cursor.next_desugared()
        } else {
            self.token_cursor.next()
        };
        if next.sp == syntax_pos::DUMMY_SP {
            // Tweak the location for better diagnostics, but keep syntactic context intact.
            next.sp = self.prev_span.with_ctxt(next.sp.ctxt());
        }
        next
    }

    /// Convert a token to a string using self's reader
    pub fn token_to_string(token: &token::Token) -> String {
        pprust::token_to_string(token)
    }

    /// Convert the current token to a string using self's reader
    pub fn this_token_to_string(&self) -> String {
        Parser::token_to_string(&self.token)
    }

    pub fn token_descr(&self) -> Option<&'static str> {
        Some(match &self.token {
            t if t.is_special_ident() => "reserved identifier",
            t if t.is_used_keyword() => "keyword",
            t if t.is_unused_keyword() => "reserved keyword",
            _ => return None,
        })
    }

    pub fn this_token_descr(&self) -> String {
        if let Some(prefix) = self.token_descr() {
            format!("{} `{}`", prefix, self.this_token_to_string())
        } else {
            format!("`{}`", self.this_token_to_string())
        }
    }

    pub fn unexpected_last<T>(&self, t: &token::Token) -> PResult<'a, T> {
        let token_str = Parser::token_to_string(t);
        Err(self.span_fatal(self.prev_span, &format!("unexpected token: `{}`", token_str)))
    }

    pub fn unexpected<T>(&mut self) -> PResult<'a, T> {
        match self.expect_one_of(&[], &[]) {
            Err(e) => Err(e),
            Ok(_) => unreachable!(),
        }
    }

    /// Expect and consume the token t. Signal an error if
    /// the next token is not t.
    pub fn expect(&mut self, t: &token::Token) -> PResult<'a,  ()> {
        if self.expected_tokens.is_empty() {
            if self.token == *t {
                self.bump();
                Ok(())
            } else {
                let token_str = Parser::token_to_string(t);
                let this_token_str = self.this_token_to_string();
                let mut err = self.fatal(&format!("expected `{}`, found `{}`",
                                                  token_str,
                                                  this_token_str));
                err.span_label(self.span, format!("expected `{}`", token_str));
                Err(err)
            }
        } else {
            self.expect_one_of(unsafe { slice::from_raw_parts(t, 1) }, &[])
        }
    }

    /// Expect next token to be edible or inedible token.  If edible,
    /// then consume it; if inedible, then return without consuming
    /// anything.  Signal a fatal error if next token is unexpected.
    pub fn expect_one_of(&mut self,
                         edible: &[token::Token],
                         inedible: &[token::Token]) -> PResult<'a,  ()>{
        fn tokens_to_string(tokens: &[TokenType]) -> String {
            let mut i = tokens.iter();
            // This might be a sign we need a connect method on Iterator.
            let b = i.next()
                     .map_or("".to_string(), |t| t.to_string());
            i.enumerate().fold(b, |mut b, (i, a)| {
                if tokens.len() > 2 && i == tokens.len() - 2 {
                    b.push_str(", or ");
                } else if tokens.len() == 2 && i == tokens.len() - 2 {
                    b.push_str(" or ");
                } else {
                    b.push_str(", ");
                }
                b.push_str(&a.to_string());
                b
            })
        }
        if edible.contains(&self.token) {
            self.bump();
            Ok(())
        } else if inedible.contains(&self.token) {
            // leave it in the input
            Ok(())
        } else {
            let mut expected = edible.iter()
                .map(|x| TokenType::Token(x.clone()))
                .chain(inedible.iter().map(|x| TokenType::Token(x.clone())))
                .chain(self.expected_tokens.iter().cloned())
                .collect::<Vec<_>>();
            expected.sort_by_cached_key(|x| x.to_string());
            expected.dedup();
            let expect = tokens_to_string(&expected[..]);
            let actual = self.this_token_to_string();
            let (msg_exp, (label_sp, label_exp)) = if expected.len() > 1 {
                let short_expect = if expected.len() > 6 {
                    format!("{} possible tokens", expected.len())
                } else {
                    expect.clone()
                };
                (format!("expected one of {}, found `{}`", expect, actual),
                 (self.sess.codemap().next_point(self.prev_span),
                  format!("expected one of {} here", short_expect)))
            } else if expected.is_empty() {
                (format!("unexpected token: `{}`", actual),
                 (self.prev_span, "unexpected token after this".to_string()))
            } else {
                (format!("expected {}, found `{}`", expect, actual),
                 (self.sess.codemap().next_point(self.prev_span),
                  format!("expected {} here", expect)))
            };
            let mut err = self.fatal(&msg_exp);
            let sp = if self.token == token::Token::Eof {
                // This is EOF, don't want to point at the following char, but rather the last token
                self.prev_span
            } else {
                label_sp
            };

            let cm = self.sess.codemap();
            match (cm.lookup_line(self.span.lo()), cm.lookup_line(sp.lo())) {
                (Ok(ref a), Ok(ref b)) if a.line == b.line => {
                    // When the spans are in the same line, it means that the only content between
                    // them is whitespace, point at the found token in that case:
                    //
                    // X |     () => { syntax error };
                    //   |                    ^^^^^ expected one of 8 possible tokens here
                    //
                    // instead of having:
                    //
                    // X |     () => { syntax error };
                    //   |                   -^^^^^ unexpected token
                    //   |                   |
                    //   |                   expected one of 8 possible tokens here
                    err.span_label(self.span, label_exp);
                }
                _ => {
                    err.span_label(sp, label_exp);
                    err.span_label(self.span, "unexpected token");
                }
            }
            Err(err)
        }
    }

    /// returns the span of expr, if it was not interpolated or the span of the interpolated token
    fn interpolated_or_expr_span(&self,
                                 expr: PResult<'a, P<Expr>>)
                                 -> PResult<'a, (Span, P<Expr>)> {
        expr.map(|e| {
            if self.prev_token_kind == PrevTokenKind::Interpolated {
                (self.prev_span, e)
            } else {
                (e.span, e)
            }
        })
    }

    fn expected_ident_found(&self) -> DiagnosticBuilder<'a> {
        let mut err = self.struct_span_err(self.span,
                                           &format!("expected identifier, found {}",
                                                    self.this_token_descr()));
        if let Some(token_descr) = self.token_descr() {
            err.span_label(self.span, format!("expected identifier, found {}", token_descr));
        } else {
            err.span_label(self.span, "expected identifier");
        }
        err
    }

    pub fn parse_ident(&mut self) -> PResult<'a, ast::Ident> {
        self.parse_ident_common(true)
    }

    fn parse_ident_common(&mut self, recover: bool) -> PResult<'a, ast::Ident> {
        match self.token {
            token::Ident(ident, _) => {
                if self.token.is_reserved_ident() {
                    let mut err = self.expected_ident_found();
                    if recover {
                        err.emit();
                    } else {
                        return Err(err);
                    }
                }
                let span = self.span;
                self.bump();
                Ok(Ident::new(ident.name, span))
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

    /// Check if the next token is `tok`, and return `true` if so.
    ///
    /// This method will automatically add `tok` to `expected_tokens` if `tok` is not
    /// encountered.
    pub fn check(&mut self, tok: &token::Token) -> bool {
        let is_present = self.token == *tok;
        if !is_present { self.expected_tokens.push(TokenType::Token(tok.clone())); }
        is_present
    }

    /// Consume token 'tok' if it exists. Returns true if the given
    /// token was present, false otherwise.
    pub fn eat(&mut self, tok: &token::Token) -> bool {
        let is_present = self.check(tok);
        if is_present { self.bump() }
        is_present
    }

    pub fn check_keyword(&mut self, kw: keywords::Keyword) -> bool {
        self.expected_tokens.push(TokenType::Keyword(kw));
        self.token.is_keyword(kw)
    }

    /// If the next token is the given keyword, eat it and return
    /// true. Otherwise, return false.
    pub fn eat_keyword(&mut self, kw: keywords::Keyword) -> bool {
        if self.check_keyword(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub fn eat_keyword_noexpect(&mut self, kw: keywords::Keyword) -> bool {
        if self.token.is_keyword(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// If the given word is not a keyword, signal an error.
    /// If the next token is not the given word, signal an error.
    /// Otherwise, eat it.
    pub fn expect_keyword(&mut self, kw: keywords::Keyword) -> PResult<'a, ()> {
        if !self.eat_keyword(kw) {
            self.unexpected()
        } else {
            Ok(())
        }
    }

    fn check_ident(&mut self) -> bool {
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

    /// Expect and consume an `&`. If `&&` is seen, replace it with a single
    /// `&` and continue. If an `&` is not seen, signal an error.
    fn expect_and(&mut self) -> PResult<'a, ()> {
        self.expected_tokens.push(TokenType::Token(token::BinOp(token::And)));
        match self.token {
            token::BinOp(token::And) => {
                self.bump();
                Ok(())
            }
            token::AndAnd => {
                let span = self.span.with_lo(self.span.lo() + BytePos(1));
                Ok(self.bump_with(token::BinOp(token::And), span))
            }
            _ => self.unexpected()
        }
    }

    /// Expect and consume an `|`. If `||` is seen, replace it with a single
    /// `|` and continue. If an `|` is not seen, signal an error.
    fn expect_or(&mut self) -> PResult<'a, ()> {
        self.expected_tokens.push(TokenType::Token(token::BinOp(token::Or)));
        match self.token {
            token::BinOp(token::Or) => {
                self.bump();
                Ok(())
            }
            token::OrOr => {
                let span = self.span.with_lo(self.span.lo() + BytePos(1));
                Ok(self.bump_with(token::BinOp(token::Or), span))
            }
            _ => self.unexpected()
        }
    }

    pub fn expect_no_suffix(&self, sp: Span, kind: &str, suffix: Option<ast::Name>) {
        match suffix {
            None => {/* everything ok */}
            Some(suf) => {
                let text = suf.as_str();
                if text.is_empty() {
                    self.span_bug(sp, "found empty literal suffix in Some")
                }
                self.span_err(sp, &format!("{} with a suffix is invalid", kind));
            }
        }
    }

    /// Attempt to consume a `<`. If `<<` is seen, replace it with a single
    /// `<` and continue. If a `<` is not seen, return false.
    ///
    /// This is meant to be used when parsing generics on a path to get the
    /// starting token.
    fn eat_lt(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Token(token::Lt));
        match self.token {
            token::Lt => {
                self.bump();
                true
            }
            token::BinOp(token::Shl) => {
                let span = self.span.with_lo(self.span.lo() + BytePos(1));
                self.bump_with(token::Lt, span);
                true
            }
            _ => false,
        }
    }

    fn expect_lt(&mut self) -> PResult<'a, ()> {
        if !self.eat_lt() {
            self.unexpected()
        } else {
            Ok(())
        }
    }

    /// Expect and consume a GT. if a >> is seen, replace it
    /// with a single > and continue. If a GT is not seen,
    /// signal an error.
    pub fn expect_gt(&mut self) -> PResult<'a, ()> {
        self.expected_tokens.push(TokenType::Token(token::Gt));
        match self.token {
            token::Gt => {
                self.bump();
                Ok(())
            }
            token::BinOp(token::Shr) => {
                let span = self.span.with_lo(self.span.lo() + BytePos(1));
                Ok(self.bump_with(token::Gt, span))
            }
            token::BinOpEq(token::Shr) => {
                let span = self.span.with_lo(self.span.lo() + BytePos(1));
                Ok(self.bump_with(token::Ge, span))
            }
            token::Ge => {
                let span = self.span.with_lo(self.span.lo() + BytePos(1));
                Ok(self.bump_with(token::Eq, span))
            }
            _ => self.unexpected()
        }
    }

    pub fn parse_seq_to_before_gt_or_return<T, F>(&mut self,
                                                  sep: Option<token::Token>,
                                                  mut f: F)
                                                  -> PResult<'a, (Vec<T>, bool)>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, Option<T>>,
    {
        let mut v = Vec::new();
        // This loop works by alternating back and forth between parsing types
        // and commas.  For example, given a string `A, B,>`, the parser would
        // first parse `A`, then a comma, then `B`, then a comma. After that it
        // would encounter a `>` and stop. This lets the parser handle trailing
        // commas in generic parameters, because it can stop either after
        // parsing a type or after parsing a comma.
        for i in 0.. {
            if self.check(&token::Gt)
                || self.token == token::BinOp(token::Shr)
                || self.token == token::Ge
                || self.token == token::BinOpEq(token::Shr) {
                break;
            }

            if i % 2 == 0 {
                match f(self)? {
                    Some(result) => v.push(result),
                    None => return Ok((v, true))
                }
            } else {
                if let Some(t) = sep.as_ref() {
                    self.expect(t)?;
                }

            }
        }
        return Ok((v, false));
    }

    /// Parse a sequence bracketed by '<' and '>', stopping
    /// before the '>'.
    pub fn parse_seq_to_before_gt<T, F>(&mut self,
                                        sep: Option<token::Token>,
                                        mut f: F)
                                        -> PResult<'a, Vec<T>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        let (result, returned) = self.parse_seq_to_before_gt_or_return(sep,
                                                                       |p| Ok(Some(f(p)?)))?;
        assert!(!returned);
        return Ok(result);
    }

    pub fn parse_seq_to_gt<T, F>(&mut self,
                                 sep: Option<token::Token>,
                                 f: F)
                                 -> PResult<'a, Vec<T>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        let v = self.parse_seq_to_before_gt(sep, f)?;
        self.expect_gt()?;
        return Ok(v);
    }

    pub fn parse_seq_to_gt_or_return<T, F>(&mut self,
                                           sep: Option<token::Token>,
                                           f: F)
                                           -> PResult<'a, (Vec<T>, bool)> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a, Option<T>>,
    {
        let (v, returned) = self.parse_seq_to_before_gt_or_return(sep, f)?;
        if !returned {
            self.expect_gt()?;
        }
        return Ok((v, returned));
    }

    /// Eat and discard tokens until one of `kets` is encountered. Respects token trees,
    /// passes through any errors encountered. Used for error recovery.
    pub fn eat_to_tokens(&mut self, kets: &[&token::Token]) {
        let handler = self.diagnostic();

        if let Err(ref mut err) = self.parse_seq_to_before_tokens(kets,
                                                                  SeqSep::none(),
                                                                  TokenExpectType::Expect,
                                                                  |p| Ok(p.parse_token_tree())) {
            handler.cancel(err);
        }
    }

    /// Parse a sequence, including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_end<T, F>(&mut self,
                                  ket: &token::Token,
                                  sep: SeqSep,
                                  f: F)
                                  -> PResult<'a, Vec<T>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a,  T>,
    {
        let val = self.parse_seq_to_before_end(ket, sep, f)?;
        self.bump();
        Ok(val)
    }

    /// Parse a sequence, not including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_before_end<T, F>(&mut self,
                                         ket: &token::Token,
                                         sep: SeqSep,
                                         f: F)
                                         -> PResult<'a, Vec<T>>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, T>
    {
        self.parse_seq_to_before_tokens(&[ket], sep, TokenExpectType::Expect, f)
    }

    fn parse_seq_to_before_tokens<T, F>(&mut self,
                                            kets: &[&token::Token],
                                            sep: SeqSep,
                                            expect: TokenExpectType,
                                            mut f: F)
                                            -> PResult<'a, Vec<T>>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a, T>
    {
        let mut first: bool = true;
        let mut v = vec![];
        while !kets.contains(&&self.token) {
            match self.token {
                token::CloseDelim(..) | token::Eof => break,
                _ => {}
            };
            if let Some(ref t) = sep.sep {
                if first {
                    first = false;
                } else {
                    if let Err(mut e) = self.expect(t) {
                        // Attempt to keep parsing if it was a similar separator
                        if let Some(ref tokens) = t.similar_tokens() {
                            if tokens.contains(&self.token) {
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

        Ok(v)
    }

    /// Parse a sequence, including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_unspanned_seq<T, F>(&mut self,
                                     bra: &token::Token,
                                     ket: &token::Token,
                                     sep: SeqSep,
                                     f: F)
                                     -> PResult<'a, Vec<T>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        self.expect(bra)?;
        let result = self.parse_seq_to_before_end(ket, sep, f)?;
        if self.token == *ket {
            self.bump();
        }
        Ok(result)
    }

    // NB: Do not use this function unless you actually plan to place the
    // spanned list in the AST.
    pub fn parse_seq<T, F>(&mut self,
                           bra: &token::Token,
                           ket: &token::Token,
                           sep: SeqSep,
                           f: F)
                           -> PResult<'a, Spanned<Vec<T>>> where
        F: FnMut(&mut Parser<'a>) -> PResult<'a, T>,
    {
        let lo = self.span;
        self.expect(bra)?;
        let result = self.parse_seq_to_before_end(ket, sep, f)?;
        let hi = self.span;
        self.bump();
        Ok(respan(lo.to(hi), result))
    }

    /// Advance the parser by one token
    pub fn bump(&mut self) {
        if self.prev_token_kind == PrevTokenKind::Eof {
            // Bumping after EOF is a bad sign, usually an infinite loop.
            self.bug("attempted to bump the parser past EOF (may be stuck in a loop)");
        }

        self.prev_span = self.meta_var_span.take().unwrap_or(self.span);

        // Record last token kind for possible error recovery.
        self.prev_token_kind = match self.token {
            token::DocComment(..) => PrevTokenKind::DocComment,
            token::Comma => PrevTokenKind::Comma,
            token::BinOp(token::Plus) => PrevTokenKind::Plus,
            token::Interpolated(..) => PrevTokenKind::Interpolated,
            token::Eof => PrevTokenKind::Eof,
            token::Ident(..) => PrevTokenKind::Ident,
            _ => PrevTokenKind::Other,
        };

        let next = self.next_tok();
        self.span = next.sp;
        self.token = next.tok;
        self.expected_tokens.clear();
        // check after each token
        self.process_potential_macro_variable();
    }

    /// Advance the parser using provided token as a next one. Use this when
    /// consuming a part of a token. For example a single `<` from `<<`.
    pub fn bump_with(&mut self, next: token::Token, span: Span) {
        self.prev_span = self.span.with_hi(span.lo());
        // It would be incorrect to record the kind of the current token, but
        // fortunately for tokens currently using `bump_with`, the
        // prev_token_kind will be of no use anyway.
        self.prev_token_kind = PrevTokenKind::Other;
        self.span = span;
        self.token = next;
        self.expected_tokens.clear();
    }

    pub fn look_ahead<R, F>(&self, dist: usize, f: F) -> R where
        F: FnOnce(&token::Token) -> R,
    {
        if dist == 0 {
            return f(&self.token)
        }

        f(&match self.token_cursor.frame.tree_cursor.look_ahead(dist - 1) {
            Some(tree) => match tree {
                TokenTree::Token(_, tok) => tok,
                TokenTree::Delimited(_, delimited) => token::OpenDelim(delimited.delim),
            },
            None => token::CloseDelim(self.token_cursor.frame.delim),
        })
    }

    fn look_ahead_span(&self, dist: usize) -> Span {
        if dist == 0 {
            return self.span
        }

        match self.token_cursor.frame.tree_cursor.look_ahead(dist - 1) {
            Some(TokenTree::Token(span, _)) | Some(TokenTree::Delimited(span, _)) => span,
            None => self.look_ahead_span(dist - 1),
        }
    }
    pub fn fatal(&self, m: &str) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_fatal(self.span, m)
    }
    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_fatal(sp, m)
    }
    pub fn span_fatal_err<S: Into<MultiSpan>>(&self, sp: S, err: Error) -> DiagnosticBuilder<'a> {
        err.span_err(sp, self.diagnostic())
    }
    pub fn span_fatal_help<S: Into<MultiSpan>>(&self,
                                            sp: S,
                                            m: &str,
                                            help: &str) -> DiagnosticBuilder<'a> {
        let mut err = self.sess.span_diagnostic.struct_span_fatal(sp, m);
        err.help(help);
        err
    }
    pub fn bug(&self, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(self.span, m)
    }
    pub fn warn(&self, m: &str) {
        self.sess.span_diagnostic.span_warn(self.span, m)
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, m: &str) {
        self.sess.span_diagnostic.span_warn(sp, m)
    }
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, m: &str) {
        self.sess.span_diagnostic.span_err(sp, m)
    }
    pub fn struct_span_err<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_err(sp, m)
    }
    pub fn span_err_help<S: Into<MultiSpan>>(&self, sp: S, m: &str, h: &str) {
        let mut err = self.sess.span_diagnostic.mut_span_err(sp, m);
        err.help(h);
        err.emit();
    }
    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(sp, m)
    }
    pub fn abort_if_errors(&self) {
        self.sess.span_diagnostic.abort_if_errors();
    }

    fn cancel(&self, err: &mut DiagnosticBuilder) {
        self.sess.span_diagnostic.cancel(err)
    }

    pub fn diagnostic(&self) -> &'a errors::Handler {
        &self.sess.span_diagnostic
    }

    /// Is the current token one of the keywords that signals a bare function
    /// type?
    pub fn token_is_bare_fn_keyword(&mut self) -> bool {
        self.check_keyword(keywords::Fn) ||
            self.check_keyword(keywords::Unsafe) ||
            self.check_keyword(keywords::Extern) && self.is_extern_non_path()
    }

    /// parse a TyKind::BareFn type:
    pub fn parse_ty_bare_fn(&mut self, generic_params: Vec<GenericParam>)
                            -> PResult<'a, TyKind> {
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
        let abi = if self.eat_keyword(keywords::Extern) {
            self.parse_opt_abi()?.unwrap_or(Abi::C)
        } else {
            Abi::Rust
        };

        self.expect_keyword(keywords::Fn)?;
        let (inputs, variadic) = self.parse_fn_args(false, true)?;
        let ret_ty = self.parse_ret_ty(false)?;
        let decl = P(FnDecl {
            inputs,
            output: ret_ty,
            variadic,
        });
        Ok(TyKind::BareFn(P(BareFnTy {
            abi,
            unsafety,
            generic_params,
            decl,
        })))
    }

    /// Parse unsafety: `unsafe` or nothing.
    fn parse_unsafety(&mut self) -> Unsafety {
        if self.eat_keyword(keywords::Unsafe) {
            Unsafety::Unsafe
        } else {
            Unsafety::Normal
        }
    }

    /// Parse the items in a trait declaration
    pub fn parse_trait_item(&mut self, at_end: &mut bool) -> PResult<'a, TraitItem> {
        maybe_whole!(self, NtTraitItem, |x| x);
        let attrs = self.parse_outer_attributes()?;
        let (mut item, tokens) = self.collect_tokens(|this| {
            this.parse_trait_item_(at_end, attrs)
        })?;
        // See `parse_item` for why this clause is here.
        if !item.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
            item.tokens = Some(tokens);
        }
        Ok(item)
    }

    fn parse_trait_item_(&mut self,
                         at_end: &mut bool,
                         mut attrs: Vec<Attribute>) -> PResult<'a, TraitItem> {
        let lo = self.span;

        let (name, node, generics) = if self.eat_keyword(keywords::Type) {
            let (generics, TyParam {ident, bounds, default, ..}) =
                self.parse_trait_item_assoc_ty(vec![])?;
            (ident, TraitItemKind::Type(bounds, default), generics)
        } else if self.is_const_item() {
            self.expect_keyword(keywords::Const)?;
            let ident = self.parse_ident()?;
            self.expect(&token::Colon)?;
            let ty = self.parse_ty()?;
            let default = if self.check(&token::Eq) {
                self.bump();
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
            (keywords::Invalid.ident(), ast::TraitItemKind::Macro(mac), ast::Generics::default())
        } else {
            let (constness, unsafety, abi) = self.parse_fn_front_matter()?;

            let ident = self.parse_ident()?;
            let mut generics = self.parse_generics()?;

            let d = self.parse_fn_decl_with_self(|p: &mut Parser<'a>|{
                // This is somewhat dubious; We don't want to allow
                // argument names to be left off if there is a
                // definition...
                p.parse_arg_general(false)
            })?;
            generics.where_clause = self.parse_where_clause()?;

            let sig = ast::MethodSig {
                unsafety,
                constness,
                decl: d,
                abi,
            };

            let body = match self.token {
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
                _ => {
                    let token_str = self.this_token_to_string();
                    let mut err = self.fatal(&format!("expected `;` or `{{`, found `{}`",
                                                      token_str));
                    err.span_label(self.span, "expected `;` or `{`");
                    return Err(err);
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

    /// Parse optional return type [ -> TY ] in function decl
    fn parse_ret_ty(&mut self, allow_plus: bool) -> PResult<'a, FunctionRetTy> {
        if self.eat(&token::RArrow) {
            Ok(FunctionRetTy::Ty(self.parse_ty_common(allow_plus, true)?))
        } else {
            Ok(FunctionRetTy::Default(self.span.shrink_to_lo()))
        }
    }

    // Parse a type
    pub fn parse_ty(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(true, true)
    }

    /// Parse a type in restricted contexts where `+` is not permitted.
    /// Example 1: `&'a TYPE`
    ///     `+` is prohibited to maintain operator priority (P(+) < P(&)).
    /// Example 2: `value1 as TYPE + value2`
    ///     `+` is prohibited to avoid interactions with expression grammar.
    fn parse_ty_no_plus(&mut self) -> PResult<'a, P<Ty>> {
        self.parse_ty_common(false, true)
    }

    fn parse_ty_common(&mut self, allow_plus: bool, allow_qpath_recovery: bool)
                       -> PResult<'a, P<Ty>> {
        maybe_whole!(self, NtTy, |x| x);

        let lo = self.span;
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
                let maybe_bounds = allow_plus && self.token == token::BinOp(token::Plus);
                match ty.node {
                    // `(TY_BOUND_NOPAREN) + BOUND + ...`.
                    TyKind::Path(None, ref path) if maybe_bounds => {
                        self.parse_remaining_bounds(Vec::new(), path.clone(), lo, true)?
                    }
                    TyKind::TraitObject(ref bounds, TraitObjectSyntax::None)
                            if maybe_bounds && bounds.len() == 1 && !trailing_plus => {
                        let path = match bounds[0] {
                            TraitTyParamBound(ref pt, ..) => pt.trait_ref.path.clone(),
                            _ => self.bug("unexpected lifetime bound"),
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
                Some(suffix) => TyKind::Array(t, suffix),
            };
            self.expect(&token::CloseDelim(token::Bracket))?;
            t
        } else if self.check(&token::BinOp(token::And)) || self.check(&token::AndAnd) {
            // Reference
            self.expect_and()?;
            self.parse_borrowed_pointee()?
        } else if self.eat_keyword_noexpect(keywords::Typeof) {
            // `typeof(EXPR)`
            // In order to not be ambiguous, the type must be surrounded by parens.
            self.expect(&token::OpenDelim(token::Paren))?;
            let e = self.parse_expr()?;
            self.expect(&token::CloseDelim(token::Paren))?;
            TyKind::Typeof(e)
        } else if self.eat_keyword(keywords::Underscore) {
            // A type to be inferred `_`
            TyKind::Infer
        } else if self.token_is_bare_fn_keyword() {
            // Function pointer type
            self.parse_ty_bare_fn(Vec::new())?
        } else if self.check_keyword(keywords::For) {
            // Function pointer type or bound list (trait object type) starting with a poly-trait.
            //   `for<'lt> [unsafe] [extern "ABI"] fn (&'lt S) -> T`
            //   `for<'lt> Trait1<'lt> + Trait2 + 'a`
            let lo = self.span;
            let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
            if self.token_is_bare_fn_keyword() {
                self.parse_ty_bare_fn(lifetime_defs)?
            } else {
                let path = self.parse_path(PathStyle::Type)?;
                let parse_plus = allow_plus && self.check(&token::BinOp(token::Plus));
                self.parse_remaining_bounds(lifetime_defs, path, lo, parse_plus)?
            }
        } else if self.eat_keyword(keywords::Impl) {
            // Always parse bounds greedily for better error recovery.
            let bounds = self.parse_ty_param_bounds()?;
            impl_dyn_multi = bounds.len() > 1 || self.prev_token_kind == PrevTokenKind::Plus;
            TyKind::ImplTrait(bounds)
        } else if self.check_keyword(keywords::Dyn) &&
                  self.look_ahead(1, |t| t.can_begin_bound() &&
                                         !can_continue_type_after_non_fn_ident(t)) {
            self.bump(); // `dyn`
            // Always parse bounds greedily for better error recovery.
            let bounds = self.parse_ty_param_bounds()?;
            impl_dyn_multi = bounds.len() > 1 || self.prev_token_kind == PrevTokenKind::Plus;
            TyKind::TraitObject(bounds, TraitObjectSyntax::Dyn)
        } else if self.check(&token::Question) ||
                  self.check_lifetime() && self.look_ahead(1, |t| t == &token::BinOp(token::Plus)) {
            // Bound list (trait object type)
            TyKind::TraitObject(self.parse_ty_param_bounds_common(allow_plus)?,
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
                let (_, tts) = self.expect_delimited_token_tree()?;
                TyKind::Mac(respan(lo.to(self.prev_span), Mac_ { path: path, tts: tts }))
            } else {
                // Just a type path or bound list (trait object type) starting with a trait.
                //   `Type`
                //   `Trait1 + Trait2 + 'a`
                if allow_plus && self.check(&token::BinOp(token::Plus)) {
                    self.parse_remaining_bounds(Vec::new(), path, lo, true)?
                } else {
                    TyKind::Path(None, path)
                }
            }
        } else {
            let msg = format!("expected type, found {}", self.this_token_descr());
            return Err(self.fatal(&msg));
        };

        let span = lo.to(self.prev_span);
        let ty = Ty { node, span, id: ast::DUMMY_NODE_ID };

        // Try to recover from use of `+` with incorrect priority.
        self.maybe_report_ambiguous_plus(allow_plus, impl_dyn_multi, &ty);
        self.maybe_recover_from_bad_type_plus(allow_plus, &ty)?;
        let ty = self.maybe_recover_from_bad_qpath(ty, allow_qpath_recovery)?;

        Ok(P(ty))
    }

    fn parse_remaining_bounds(&mut self, generic_params: Vec<GenericParam>, path: ast::Path,
                              lo: Span, parse_plus: bool) -> PResult<'a, TyKind> {
        let poly_trait_ref = PolyTraitRef::new(generic_params, path, lo.to(self.prev_span));
        let mut bounds = vec![TraitTyParamBound(poly_trait_ref, TraitBoundModifier::None)];
        if parse_plus {
            self.bump(); // `+`
            bounds.append(&mut self.parse_ty_param_bounds()?);
        }
        Ok(TyKind::TraitObject(bounds, TraitObjectSyntax::None))
    }

    fn maybe_report_ambiguous_plus(&mut self, allow_plus: bool, impl_dyn_multi: bool, ty: &Ty) {
        if !allow_plus && impl_dyn_multi {
            let sum_with_parens = format!("({})", pprust::ty_to_string(&ty));
            self.struct_span_err(ty.span, "ambiguous `+` in a type")
                .span_suggestion(ty.span, "use parentheses to disambiguate", sum_with_parens)
                .emit();
        }
    }

    fn maybe_recover_from_bad_type_plus(&mut self, allow_plus: bool, ty: &Ty) -> PResult<'a, ()> {
        // Do not add `+` to expected tokens.
        if !allow_plus || self.token != token::BinOp(token::Plus) {
            return Ok(())
        }

        self.bump(); // `+`
        let bounds = self.parse_ty_param_bounds()?;
        let sum_span = ty.span.to(self.prev_span);

        let mut err = struct_span_err!(self.sess.span_diagnostic, sum_span, E0178,
            "expected a path on the left-hand side of `+`, not `{}`", pprust::ty_to_string(ty));

        match ty.node {
            TyKind::Rptr(ref lifetime, ref mut_ty) => {
                let sum_with_parens = pprust::to_string(|s| {
                    use print::pprust::PrintState;

                    s.s.word("&")?;
                    s.print_opt_lifetime(lifetime)?;
                    s.print_mutability(mut_ty.mutbl)?;
                    s.popen()?;
                    s.print_type(&mut_ty.ty)?;
                    s.print_bounds(" +", &bounds)?;
                    s.pclose()
                });
                err.span_suggestion(sum_span, "try adding parentheses", sum_with_parens);
            }
            TyKind::Ptr(..) | TyKind::BareFn(..) => {
                err.span_label(sum_span, "perhaps you forgot parentheses?");
            }
            _ => {
                err.span_label(sum_span, "expected a path");
            },
        }
        err.emit();
        Ok(())
    }

    // Try to recover from associated item paths like `[T]::AssocItem`/`(T, U)::AssocItem`.
    fn maybe_recover_from_bad_qpath<T: RecoverQPath>(&mut self, base: T, allow_recovery: bool)
                                                     -> PResult<'a, T> {
        // Do not add `::` to expected tokens.
        if !allow_recovery || self.token != token::ModSep {
            return Ok(base);
        }
        let ty = match base.to_ty() {
            Some(ty) => ty,
            None => return Ok(base),
        };

        self.bump(); // `::`
        let mut segments = Vec::new();
        self.parse_path_segments(&mut segments, T::PATH_STYLE, true)?;

        let span = ty.span.to(self.prev_span);
        let recovered =
            base.to_recovered(Some(QSelf { ty, position: 0 }), ast::Path { segments, span });

        self.diagnostic()
            .struct_span_err(span, "missing angle brackets in associated item path")
            .span_suggestion(span, "try", recovered.to_string()).emit();

        Ok(recovered)
    }

    fn parse_borrowed_pointee(&mut self) -> PResult<'a, TyKind> {
        let opt_lifetime = if self.check_lifetime() { Some(self.expect_lifetime()) } else { None };
        let mutbl = self.parse_mutability();
        let ty = self.parse_ty_no_plus()?;
        return Ok(TyKind::Rptr(opt_lifetime, MutTy { ty: ty, mutbl: mutbl }));
    }

    pub fn parse_ptr(&mut self) -> PResult<'a, MutTy> {
        let mutbl = if self.eat_keyword(keywords::Mut) {
            Mutability::Mutable
        } else if self.eat_keyword(keywords::Const) {
            Mutability::Immutable
        } else {
            let span = self.prev_span;
            self.span_err(span,
                          "expected mut or const in raw pointer type (use \
                           `*mut T` or `*const T` as appropriate)");
            Mutability::Immutable
        };
        let t = self.parse_ty_no_plus()?;
        Ok(MutTy { ty: t, mutbl: mutbl })
    }

    fn is_named_argument(&mut self) -> bool {
        let offset = match self.token {
            token::Interpolated(ref nt) => match nt.0 {
                token::NtPat(..) => return self.look_ahead(1, |t| t == &token::Colon),
                _ => 0,
            }
            token::BinOp(token::And) | token::AndAnd => 1,
            _ if self.token.is_keyword(keywords::Mut) => 1,
            _ => 0,
        };

        self.look_ahead(offset, |t| t.is_ident()) &&
        self.look_ahead(offset + 1, |t| t == &token::Colon)
    }

    /// This version of parse arg doesn't necessarily require
    /// identifier names.
    pub fn parse_arg_general(&mut self, require_name: bool) -> PResult<'a, Arg> {
        maybe_whole!(self, NtArg, |x| x);

        let pat = if require_name || self.is_named_argument() {
            debug!("parse_arg_general parse_pat (require_name:{})",
                   require_name);
            let pat = self.parse_pat()?;

            self.expect(&token::Colon)?;
            pat
        } else {
            debug!("parse_arg_general ident_to_pat");
            let ident = Ident::new(keywords::Invalid.name(), self.prev_span);
            P(Pat {
                id: ast::DUMMY_NODE_ID,
                node: PatKind::Ident(BindingMode::ByValue(Mutability::Immutable), ident, None),
                span: ident.span,
            })
        };

        let t = self.parse_ty()?;

        Ok(Arg {
            ty: t,
            pat,
            id: ast::DUMMY_NODE_ID,
        })
    }

    /// Parse a single function argument
    pub fn parse_arg(&mut self) -> PResult<'a, Arg> {
        self.parse_arg_general(true)
    }

    /// Parse an argument in a lambda header e.g. |arg, arg|
    pub fn parse_fn_block_arg(&mut self) -> PResult<'a, Arg> {
        let pat = self.parse_pat()?;
        let t = if self.eat(&token::Colon) {
            self.parse_ty()?
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyKind::Infer,
                span: self.span,
            })
        };
        Ok(Arg {
            ty: t,
            pat,
            id: ast::DUMMY_NODE_ID
        })
    }

    pub fn maybe_parse_fixed_length_of_vec(&mut self) -> PResult<'a, Option<P<ast::Expr>>> {
        if self.eat(&token::Semi) {
            Ok(Some(self.parse_expr()?))
        } else {
            Ok(None)
        }
    }

    /// Matches token_lit = LIT_INTEGER | ...
    pub fn parse_lit_token(&mut self) -> PResult<'a, LitKind> {
        let out = match self.token {
            token::Interpolated(ref nt) => match nt.0 {
                token::NtExpr(ref v) => match v.node {
                    ExprKind::Lit(ref lit) => { lit.node.clone() }
                    _ => { return self.unexpected_last(&self.token); }
                },
                _ => { return self.unexpected_last(&self.token); }
            },
            token::Literal(lit, suf) => {
                let diag = Some((self.span, &self.sess.span_diagnostic));
                let (suffix_illegal, result) = parse::lit_token(lit, suf, diag);

                if suffix_illegal {
                    let sp = self.span;
                    self.expect_no_suffix(sp, &format!("{} literal", lit.short_name()), suf)
                }

                result.unwrap()
            }
            _ => { return self.unexpected_last(&self.token); }
        };

        self.bump();
        Ok(out)
    }

    /// Matches lit = true | false | token_lit
    pub fn parse_lit(&mut self) -> PResult<'a, Lit> {
        let lo = self.span;
        let lit = if self.eat_keyword(keywords::True) {
            LitKind::Bool(true)
        } else if self.eat_keyword(keywords::False) {
            LitKind::Bool(false)
        } else {
            let lit = self.parse_lit_token()?;
            lit
        };
        Ok(codemap::Spanned { node: lit, span: lo.to(self.prev_span) })
    }

    /// matches '-' lit | lit (cf. ast_validation::AstValidator::check_expr_within_pat)
    pub fn parse_pat_literal_maybe_minus(&mut self) -> PResult<'a, P<Expr>> {
        maybe_whole_expr!(self);

        let minus_lo = self.span;
        let minus_present = self.eat(&token::BinOp(token::Minus));
        let lo = self.span;
        let literal = P(self.parse_lit()?);
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

    pub fn parse_path_segment_ident(&mut self) -> PResult<'a, ast::Ident> {
        match self.token {
            token::Ident(ident, _) if self.token.is_path_segment_keyword() => {
                let span = self.span;
                self.bump();
                Ok(Ident::new(ident.name, span))
            }
            _ => self.parse_ident(),
        }
    }

    /// Parses qualified path.
    /// Assumes that the leading `<` has been parsed already.
    ///
    /// `qualified_path = <type [as trait_ref]>::path`
    ///
    /// # Examples
    /// `<T as U>::a`
    /// `<T as U>::F::a<S>` (without disambiguator)
    /// `<T as U>::F::a::<S>` (with disambiguator)
    fn parse_qpath(&mut self, style: PathStyle) -> PResult<'a, (QSelf, ast::Path)> {
        let lo = self.prev_span;
        let ty = self.parse_ty()?;
        let mut path = if self.eat_keyword(keywords::As) {
            self.parse_path(PathStyle::Type)?
        } else {
            ast::Path { segments: Vec::new(), span: syntax_pos::DUMMY_SP }
        };
        self.expect(&token::Gt)?;
        self.expect(&token::ModSep)?;

        let qself = QSelf { ty, position: path.segments.len() };
        self.parse_path_segments(&mut path.segments, style, true)?;

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
        self.parse_path_common(style, true)
    }

    pub fn parse_path_common(&mut self, style: PathStyle, enable_warning: bool)
                             -> PResult<'a, ast::Path> {
        maybe_whole!(self, NtPath, |path| {
            if style == PathStyle::Mod &&
               path.segments.iter().any(|segment| segment.parameters.is_some()) {
                self.diagnostic().span_err(path.span, "unexpected generic arguments in path");
            }
            path
        });

        let lo = self.meta_var_span.unwrap_or(self.span);
        let mut segments = Vec::new();
        if self.eat(&token::ModSep) {
            segments.push(PathSegment::crate_root(lo.shrink_to_lo()));
        }
        self.parse_path_segments(&mut segments, style, enable_warning)?;

        Ok(ast::Path { segments, span: lo.to(self.prev_span) })
    }

    /// Like `parse_path`, but also supports parsing `Word` meta items into paths for back-compat.
    /// This is used when parsing derive macro paths in `#[derive]` attributes.
    pub fn parse_path_allowing_meta(&mut self, style: PathStyle) -> PResult<'a, ast::Path> {
        let meta_ident = match self.token {
            token::Interpolated(ref nt) => match nt.0 {
                token::NtMeta(ref meta) => match meta.node {
                    ast::MetaItemKind::Word => Some(meta.ident.clone()),
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

    fn parse_path_segments(&mut self,
                           segments: &mut Vec<PathSegment>,
                           style: PathStyle,
                           enable_warning: bool)
                           -> PResult<'a, ()> {
        loop {
            segments.push(self.parse_path_segment(style, enable_warning)?);

            if self.is_import_coupler() || !self.eat(&token::ModSep) {
                return Ok(());
            }
        }
    }

    fn parse_path_segment(&mut self, style: PathStyle, enable_warning: bool)
                          -> PResult<'a, PathSegment> {
        let ident = self.parse_path_segment_ident()?;

        let is_args_start = |token: &token::Token| match *token {
            token::Lt | token::BinOp(token::Shl) | token::OpenDelim(token::Paren) => true,
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
            // Generic arguments are found - `<`, `(`, `::<` or `::(`.
            let lo = self.span;
            if self.eat(&token::ModSep) && style == PathStyle::Type && enable_warning {
                self.diagnostic().struct_span_warn(self.prev_span, "unnecessary path disambiguator")
                                 .span_label(self.prev_span, "try removing `::`").emit();
            }

            let parameters = if self.eat_lt() {
                // `<'a, T, A = U>`
                let (lifetimes, types, bindings) = self.parse_generic_args()?;
                self.expect_gt()?;
                let span = lo.to(self.prev_span);
                AngleBracketedParameterData { lifetimes, types, bindings, span }.into()
            } else {
                // `(T, U) -> R`
                self.bump(); // `(`
                let inputs = self.parse_seq_to_before_tokens(
                    &[&token::CloseDelim(token::Paren)],
                    SeqSep::trailing_allowed(token::Comma),
                    TokenExpectType::Expect,
                    |p| p.parse_ty())?;
                self.bump(); // `)`
                let output = if self.eat(&token::RArrow) {
                    Some(self.parse_ty_common(false, false)?)
                } else {
                    None
                };
                let span = lo.to(self.prev_span);
                ParenthesizedParameterData { inputs, output, span }.into()
            };

            PathSegment { ident, parameters }
        } else {
            // Generic arguments are not found.
            PathSegment::from_ident(ident)
        })
    }

    pub fn check_lifetime(&mut self) -> bool {
        self.expected_tokens.push(TokenType::Lifetime);
        self.token.is_lifetime()
    }

    /// Parse single lifetime 'a or panic.
    pub fn expect_lifetime(&mut self) -> Lifetime {
        if let Some(ident) = self.token.lifetime() {
            let span = self.span;
            self.bump();
            Lifetime { ident: Ident::new(ident.name, span), id: ast::DUMMY_NODE_ID }
        } else {
            self.span_bug(self.span, "not a lifetime")
        }
    }

    fn eat_label(&mut self) -> Option<Label> {
        if let Some(ident) = self.token.lifetime() {
            let span = self.span;
            self.bump();
            Some(Label { ident: Ident::new(ident.name, span) })
        } else {
            None
        }
    }

    /// Parse mutability (`mut` or nothing).
    fn parse_mutability(&mut self) -> Mutability {
        if self.eat_keyword(keywords::Mut) {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        }
    }

    pub fn parse_field_name(&mut self) -> PResult<'a, Ident> {
        if let token::Literal(token::Integer(name), None) = self.token {
            self.bump();
            Ok(Ident::new(name, self.prev_span))
        } else {
            self.parse_ident_common(false)
        }
    }

    /// Parse ident (COLON expr)?
    pub fn parse_field(&mut self) -> PResult<'a, Field> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.span;

        // Check if a colon exists one ahead. This means we're parsing a fieldname.
        let (fieldname, expr, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
            let fieldname = self.parse_field_name()?;
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

    pub fn mk_expr(&mut self, span: Span, node: ExprKind, attrs: ThinVec<Attribute>) -> P<Expr> {
        P(Expr { node, span, attrs, id: ast::DUMMY_NODE_ID })
    }

    pub fn mk_unary(&mut self, unop: ast::UnOp, expr: P<Expr>) -> ast::ExprKind {
        ExprKind::Unary(unop, expr)
    }

    pub fn mk_binary(&mut self, binop: ast::BinOp, lhs: P<Expr>, rhs: P<Expr>) -> ast::ExprKind {
        ExprKind::Binary(binop, lhs, rhs)
    }

    pub fn mk_call(&mut self, f: P<Expr>, args: Vec<P<Expr>>) -> ast::ExprKind {
        ExprKind::Call(f, args)
    }

    pub fn mk_index(&mut self, expr: P<Expr>, idx: P<Expr>) -> ast::ExprKind {
        ExprKind::Index(expr, idx)
    }

    pub fn mk_range(&mut self,
                    start: Option<P<Expr>>,
                    end: Option<P<Expr>>,
                    limits: RangeLimits)
                    -> PResult<'a, ast::ExprKind> {
        if end.is_none() && limits == RangeLimits::Closed {
            Err(self.span_fatal_err(self.span, Error::InclusiveRangeWithNoEnd))
        } else {
            Ok(ExprKind::Range(start, end, limits))
        }
    }

    pub fn mk_assign_op(&mut self, binop: ast::BinOp,
                        lhs: P<Expr>, rhs: P<Expr>) -> ast::ExprKind {
        ExprKind::AssignOp(binop, lhs, rhs)
    }

    pub fn mk_mac_expr(&mut self, span: Span, m: Mac_, attrs: ThinVec<Attribute>) -> P<Expr> {
        P(Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprKind::Mac(codemap::Spanned {node: m, span: span}),
            span,
            attrs,
        })
    }

    pub fn mk_lit_u32(&mut self, i: u32, attrs: ThinVec<Attribute>) -> P<Expr> {
        let span = &self.span;
        let lv_lit = P(codemap::Spanned {
            node: LitKind::Int(i as u128, ast::LitIntType::Unsigned(UintTy::U32)),
            span: *span
        });

        P(Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprKind::Lit(lv_lit),
            span: *span,
            attrs,
        })
    }

    fn expect_delimited_token_tree(&mut self) -> PResult<'a, (token::DelimToken, ThinTokenStream)> {
        match self.token {
            token::OpenDelim(delim) => match self.parse_token_tree() {
                TokenTree::Delimited(_, delimited) => Ok((delim, delimited.stream().into())),
                _ => unreachable!(),
            },
            _ => {
                let msg = "expected open delimiter";
                let mut err = self.fatal(msg);
                err.span_label(self.span, msg);
                Err(err)
            }
        }
    }

    /// At the bottom (top?) of the precedence hierarchy,
    /// parse things like parenthesized exprs,
    /// macros, return, etc.
    ///
    /// NB: This does not parse outer attributes,
    ///     and is private because it only works
    ///     correctly if called from parse_dot_or_call_expr().
    fn parse_bottom_expr(&mut self) -> PResult<'a, P<Expr>> {
        maybe_whole_expr!(self);

        // Outer attributes are already parsed and will be
        // added to the return value after the fact.
        //
        // Therefore, prevent sub-parser from parsing
        // attributes by giving them a empty "already parsed" list.
        let mut attrs = ThinVec::new();

        let lo = self.span;
        let mut hi = self.span;

        let ex: ExprKind;

        // Note: when adding new syntax here, don't forget to adjust Token::can_begin_expr().
        match self.token {
            token::OpenDelim(token::Paren) => {
                self.bump();

                attrs.extend(self.parse_inner_attributes()?);

                // (e) is parenthesized e
                // (e,) is a tuple with only one field, e
                let mut es = vec![];
                let mut trailing_comma = false;
                while self.token != token::CloseDelim(token::Paren) {
                    es.push(self.parse_expr()?);
                    self.expect_one_of(&[], &[token::Comma, token::CloseDelim(token::Paren)])?;
                    if self.check(&token::Comma) {
                        trailing_comma = true;

                        self.bump();
                    } else {
                        trailing_comma = false;
                        break;
                    }
                }
                self.bump();

                hi = self.prev_span;
                ex = if es.len() == 1 && !trailing_comma {
                    ExprKind::Paren(es.into_iter().nth(0).unwrap())
                } else {
                    ExprKind::Tup(es)
                };
            }
            token::OpenDelim(token::Brace) => {
                return self.parse_block_expr(lo, BlockCheckMode::Default, attrs);
            }
            token::BinOp(token::Or) | token::OrOr => {
                return self.parse_lambda_expr(attrs);
            }
            token::OpenDelim(token::Bracket) => {
                self.bump();

                attrs.extend(self.parse_inner_attributes()?);

                if self.check(&token::CloseDelim(token::Bracket)) {
                    // Empty vector.
                    self.bump();
                    ex = ExprKind::Array(Vec::new());
                } else {
                    // Nonempty vector.
                    let first_expr = self.parse_expr()?;
                    if self.check(&token::Semi) {
                        // Repeating array syntax: [ 0; 512 ]
                        self.bump();
                        let count = self.parse_expr()?;
                        self.expect(&token::CloseDelim(token::Bracket))?;
                        ex = ExprKind::Repeat(first_expr, count);
                    } else if self.check(&token::Comma) {
                        // Vector with two or more elements.
                        self.bump();
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
                if self.check_keyword(keywords::Move) || self.check_keyword(keywords::Static) {
                    return self.parse_lambda_expr(attrs);
                }
                if self.eat_keyword(keywords::If) {
                    return self.parse_if_expr(attrs);
                }
                if self.eat_keyword(keywords::For) {
                    let lo = self.prev_span;
                    return self.parse_for_expr(None, lo, attrs);
                }
                if self.eat_keyword(keywords::While) {
                    let lo = self.prev_span;
                    return self.parse_while_expr(None, lo, attrs);
                }
                if let Some(label) = self.eat_label() {
                    let lo = label.ident.span;
                    self.expect(&token::Colon)?;
                    if self.eat_keyword(keywords::While) {
                        return self.parse_while_expr(Some(label), lo, attrs)
                    }
                    if self.eat_keyword(keywords::For) {
                        return self.parse_for_expr(Some(label), lo, attrs)
                    }
                    if self.eat_keyword(keywords::Loop) {
                        return self.parse_loop_expr(Some(label), lo, attrs)
                    }
                    let msg = "expected `while`, `for`, or `loop` after a label";
                    let mut err = self.fatal(msg);
                    err.span_label(self.span, msg);
                    return Err(err);
                }
                if self.eat_keyword(keywords::Loop) {
                    let lo = self.prev_span;
                    return self.parse_loop_expr(None, lo, attrs);
                }
                if self.eat_keyword(keywords::Continue) {
                    let label = self.eat_label();
                    let ex = ExprKind::Continue(label);
                    let hi = self.prev_span;
                    return Ok(self.mk_expr(lo.to(hi), ex, attrs));
                }
                if self.eat_keyword(keywords::Match) {
                    return self.parse_match_expr(attrs);
                }
                if self.eat_keyword(keywords::Unsafe) {
                    return self.parse_block_expr(
                        lo,
                        BlockCheckMode::Unsafe(ast::UserProvided),
                        attrs);
                }
                if self.is_catch_expr() {
                    let lo = self.span;
                    assert!(self.eat_keyword(keywords::Do));
                    assert!(self.eat_keyword(keywords::Catch));
                    return self.parse_catch_expr(lo, attrs);
                }
                if self.eat_keyword(keywords::Return) {
                    if self.token.can_begin_expr() {
                        let e = self.parse_expr()?;
                        hi = e.span;
                        ex = ExprKind::Ret(Some(e));
                    } else {
                        ex = ExprKind::Ret(None);
                    }
                } else if self.eat_keyword(keywords::Break) {
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
                } else if self.eat_keyword(keywords::Yield) {
                    if self.token.can_begin_expr() {
                        let e = self.parse_expr()?;
                        hi = e.span;
                        ex = ExprKind::Yield(Some(e));
                    } else {
                        ex = ExprKind::Yield(None);
                    }
                } else if self.token.is_keyword(keywords::Let) {
                    // Catch this syntax error here, instead of in `parse_ident`, so
                    // that we can explicitly mention that let is not to be used as an expression
                    let mut db = self.fatal("expected expression, found statement (`let`)");
                    db.span_label(self.span, "expected expression");
                    db.note("variable declaration using `let` is a statement");
                    return Err(db);
                } else if self.token.is_path_start() {
                    let pth = self.parse_path(PathStyle::Expr)?;

                    // `!`, as an operator, is prefix, so we know this isn't that
                    if self.eat(&token::Not) {
                        // MACRO INVOCATION expression
                        let (_, tts) = self.expect_delimited_token_tree()?;
                        let hi = self.prev_span;
                        return Ok(self.mk_mac_expr(lo.to(hi), Mac_ { path: pth, tts: tts }, attrs));
                    }
                    if self.check(&token::OpenDelim(token::Brace)) {
                        // This is a struct literal, unless we're prohibited
                        // from parsing struct literals here.
                        let prohibited = self.restrictions.contains(
                            Restrictions::NO_STRUCT_LITERAL
                        );
                        if !prohibited {
                            return self.parse_struct_expr(lo, pth, attrs);
                        }
                    }

                    hi = pth.span;
                    ex = ExprKind::Path(None, pth);
                } else {
                    match self.parse_lit() {
                        Ok(lit) => {
                            hi = lit.span;
                            ex = ExprKind::Lit(P(lit));
                        }
                        Err(mut err) => {
                            self.cancel(&mut err);
                            let msg = format!("expected expression, found {}",
                                              self.this_token_descr());
                            let mut err = self.fatal(&msg);
                            err.span_label(self.span, "expected expression");
                            return Err(err);
                        }
                    }
                }
            }
        }

        let expr = Expr { node: ex, span: lo.to(hi), id: ast::DUMMY_NODE_ID, attrs };
        let expr = self.maybe_recover_from_bad_qpath(expr, true)?;

        return Ok(P(expr));
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
                    err.span_suggestion_short(self.span, "remove this comma", "".to_owned());
                    err.note("the base struct must always be the last field");
                    err.emit();
                    self.recover_stmt();
                }
                break;
            }

            match self.parse_field() {
                Ok(f) => fields.push(f),
                Err(mut e) => {
                    e.span_label(struct_sp, "while parsing this struct");
                    e.emit();
                    self.recover_stmt();
                    break;
                }
            }

            match self.expect_one_of(&[token::Comma],
                                     &[token::CloseDelim(token::Brace)]) {
                Ok(()) => {}
                Err(mut e) => {
                    e.emit();
                    self.recover_stmt();
                    break;
                }
            }
        }

        let span = lo.to(self.span);
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

    /// Parse a block or unsafe block
    pub fn parse_block_expr(&mut self, lo: Span, blk_mode: BlockCheckMode,
                            outer_attrs: ThinVec<Attribute>)
                            -> PResult<'a, P<Expr>> {
        self.expect(&token::OpenDelim(token::Brace))?;

        let mut attrs = outer_attrs;
        attrs.extend(self.parse_inner_attributes()?);

        let blk = self.parse_block_tail(lo, blk_mode)?;
        return Ok(self.mk_expr(blk.span, ExprKind::Block(blk), attrs));
    }

    /// parse a.b or a(13) or a[4] or just a
    pub fn parse_dot_or_call_expr(&mut self,
                                  already_parsed_attrs: Option<ThinVec<Attribute>>)
                                  -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;

        let b = self.parse_bottom_expr();
        let (span, b) = self.interpolated_or_expr_span(b)?;
        self.parse_dot_or_call_expr_with(b, span, attrs)
    }

    pub fn parse_dot_or_call_expr_with(&mut self,
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
                    ExprKind::If(..) | ExprKind::IfLet(..) => {
                        if !expr.attrs.is_empty() {
                            // Just point to the first attribute in there...
                            let span = expr.attrs[0].span;

                            self.span_err(span,
                                "attributes are not yet allowed on `if` \
                                expressions");
                        }
                    }
                    _ => {}
                }
                expr
            })
        )
    }

    // Assuming we have just parsed `.`, continue parsing into an expression.
    fn parse_dot_suffix(&mut self, self_arg: P<Expr>, lo: Span) -> PResult<'a, P<Expr>> {
        let segment = self.parse_path_segment(PathStyle::Expr, true)?;
        Ok(match self.token {
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
                if let Some(parameters) = segment.parameters {
                    self.span_err(parameters.span(),
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
                match self.token {
                  token::Ident(..) => {
                    e = self.parse_dot_suffix(e, lo)?;
                  }
                  token::Literal(token::Integer(name), _) => {
                    let span = self.span;
                    self.bump();
                    let field = ExprKind::Field(e, Ident::new(name, span));
                    e = self.mk_expr(lo.to(span), field, ThinVec::new());
                  }
                  token::Literal(token::Float(n), _suf) => {
                    self.bump();
                    let fstr = n.as_str();
                    let mut err = self.diagnostic().struct_span_err(self.prev_span,
                        &format!("unexpected token: `{}`", n));
                    err.span_label(self.prev_span, "unexpected token");
                    if fstr.chars().all(|x| "0123456789.".contains(x)) {
                        let float = match fstr.parse::<f64>().ok() {
                            Some(f) => f,
                            None => continue,
                        };
                        let sugg = pprust::to_string(|s| {
                            use print::pprust::PrintState;
                            s.popen()?;
                            s.print_expr(&e)?;
                            s.s.word( ".")?;
                            s.print_usize(float.trunc() as usize)?;
                            s.pclose()?;
                            s.s.word(".")?;
                            s.s.word(fstr.splitn(2, ".").last().unwrap())
                        });
                        err.span_suggestion(
                            lo.to(self.prev_span),
                            "try parenthesizing the first index",
                            sugg);
                    }
                    return Err(err);

                  }
                  _ => {
                    // FIXME Could factor this out into non_fatal_unexpected or something.
                    let actual = self.this_token_to_string();
                    self.span_err(self.span, &format!("unexpected token: `{}`", actual));
                  }
                }
                continue;
            }
            if self.expr_is_complete(&e) { break; }
            match self.token {
              // expr(...)
              token::OpenDelim(token::Paren) => {
                let es = self.parse_unspanned_seq(
                    &token::OpenDelim(token::Paren),
                    &token::CloseDelim(token::Paren),
                    SeqSep::trailing_allowed(token::Comma),
                    |p| Ok(p.parse_expr()?)
                )?;
                hi = self.prev_span;

                let nd = self.mk_call(e, es);
                e = self.mk_expr(lo.to(hi), nd, ThinVec::new());
              }

              // expr[...]
              // Could be either an index expression or a slicing expression.
              token::OpenDelim(token::Bracket) => {
                self.bump();
                let ix = self.parse_expr()?;
                hi = self.span;
                self.expect(&token::CloseDelim(token::Bracket))?;
                let index = self.mk_index(e, ix);
                e = self.mk_expr(lo.to(hi), index, ThinVec::new())
              }
              _ => return Ok(e)
            }
        }
        return Ok(e);
    }

    pub fn process_potential_macro_variable(&mut self) {
        let (token, span) = match self.token {
            token::Dollar if self.span.ctxt() != syntax_pos::hygiene::SyntaxContext::empty() &&
                             self.look_ahead(1, |t| t.is_ident()) => {
                self.bump();
                let name = match self.token {
                    token::Ident(ident, _) => ident,
                    _ => unreachable!()
                };
                let mut err = self.fatal(&format!("unknown macro variable `{}`", name));
                err.span_label(self.span, "unknown macro variable");
                err.emit();
                return
            }
            token::Interpolated(ref nt) => {
                self.meta_var_span = Some(self.span);
                // Interpolated identifier and lifetime tokens are replaced with usual identifier
                // and lifetime tokens, so the former are never encountered during normal parsing.
                match nt.0 {
                    token::NtIdent(ident, is_raw) => (token::Ident(ident, is_raw), ident.span),
                    token::NtLifetime(ident) => (token::Lifetime(ident), ident.span),
                    _ => return,
                }
            }
            _ => return,
        };
        self.token = token;
        self.span = span;
    }

    /// parse a single token tree from the input.
    pub fn parse_token_tree(&mut self) -> TokenTree {
        match self.token {
            token::OpenDelim(..) => {
                let frame = mem::replace(&mut self.token_cursor.frame,
                                         self.token_cursor.stack.pop().unwrap());
                self.span = frame.span;
                self.bump();
                TokenTree::Delimited(frame.span, Delimited {
                    delim: frame.delim,
                    tts: frame.tree_cursor.original_stream().into(),
                })
            },
            token::CloseDelim(_) | token::Eof => unreachable!(),
            _ => {
                let (token, span) = (mem::replace(&mut self.token, token::Whitespace), self.span);
                self.bump();
                TokenTree::Token(span, token)
            }
        }
    }

    // parse a stream of tokens into a list of TokenTree's,
    // up to EOF.
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
            match self.token {
                token::Eof | token::CloseDelim(..) => break,
                _ => result.push(self.parse_token_tree().into()),
            }
        }
        TokenStream::concat(result)
    }

    /// Parse a prefix-unary-operator expr
    pub fn parse_prefix_expr(&mut self,
                             already_parsed_attrs: Option<ThinVec<Attribute>>)
                             -> PResult<'a, P<Expr>> {
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;
        let lo = self.span;
        // Note: when adding new unary operators, don't forget to adjust Token::can_begin_expr()
        let (hi, ex) = match self.token {
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
                let mut err = self.diagnostic().struct_span_err(span_of_tilde,
                        "`~` cannot be used as a unary operator");
                err.span_suggestion_short(span_of_tilde,
                                          "use `!` to perform bitwise negation",
                                          "!".to_owned());
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
            token::Ident(..) if self.token.is_keyword(keywords::In) => {
                self.bump();
                let place = self.parse_expr_res(
                    Restrictions::NO_STRUCT_LITERAL,
                    None,
                )?;
                let blk = self.parse_block()?;
                let span = blk.span;
                let blk_expr = self.mk_expr(span, ExprKind::Block(blk), ThinVec::new());
                (lo.to(span), ExprKind::ObsoleteInPlace(place, blk_expr))
            }
            token::Ident(..) if self.token.is_keyword(keywords::Box) => {
                self.bump();
                let e = self.parse_prefix_expr(None);
                let (span, e) = self.interpolated_or_expr_span(e)?;
                (lo.to(span), ExprKind::Box(e))
            }
            token::Ident(..) if self.token.is_ident_named("not") => {
                // `not` is just an ordinary identifier in Rust-the-language,
                // but as `rustc`-the-compiler, we can issue clever diagnostics
                // for confused users who really want to say `!`
                let token_cannot_continue_expr = |t: &token::Token| match *t {
                    // These tokens can start an expression after `!`, but
                    // can't continue an expression after an ident
                    token::Ident(ident, is_raw) => token::ident_can_begin_expr(ident, is_raw),
                    token::Literal(..) | token::Pound => true,
                    token::Interpolated(ref nt) => match nt.0 {
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
                        .struct_span_err(self.span,
                                         &format!("unexpected {} after identifier",
                                                  self.this_token_descr()));
                    // span the `not` plus trailing whitespace to avoid
                    // trailing whitespace after the `!` in our suggestion
                    let to_replace = self.sess.codemap()
                        .span_until_non_whitespace(lo.to(self.span));
                    err.span_suggestion_short(to_replace,
                                              "use `!` to perform logical negation",
                                              "!".to_owned());
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

    /// Parse an associative expression
    ///
    /// This parses an expression accounting for associativity and precedence of the operators in
    /// the expression.
    pub fn parse_assoc_expr(&mut self,
                            already_parsed_attrs: Option<ThinVec<Attribute>>)
                            -> PResult<'a, P<Expr>> {
        self.parse_assoc_expr_with(0, already_parsed_attrs.into())
    }

    /// Parse an associative expression with operators of at least `min_prec` precedence
    pub fn parse_assoc_expr_with(&mut self,
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
            if [token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token) {
                return self.parse_prefix_range_expr(attrs);
            } else {
                self.parse_prefix_expr(attrs)?
            }
        };

        if self.expr_is_complete(&lhs) {
            // Semi-statement forms are odd. See https://github.com/rust-lang/rust/issues/29071
            return Ok(lhs);
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

            let cur_op_span = self.span;
            let restrictions = if op.is_assign_like() {
                self.restrictions & Restrictions::NO_STRUCT_LITERAL
            } else {
                self.restrictions
            };
            if op.precedence() < min_prec {
                break;
            }
            // Check for deprecated `...` syntax
            if self.token == token::DotDotDot && op == AssocOp::DotDotEq {
                self.err_dotdotdot_syntax(self.span);
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
                lhs = match self.parse_assoc_op_cast(lhs, lhs_span, ExprKind::Type) {
                    Ok(lhs) => lhs,
                    Err(mut err) => {
                        err.span_label(self.span,
                                       "expecting a type here because of type ascription");
                        let cm = self.sess.codemap();
                        let cur_pos = cm.lookup_char_pos(self.span.lo());
                        let op_pos = cm.lookup_char_pos(cur_op_span.hi());
                        if cur_pos.line != op_pos.line {
                            err.span_suggestion_short(cur_op_span,
                                                      "did you mean to use `;` here?",
                                                      ";".to_string());
                        }
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
                    Some(self.parse_assoc_expr_with(op.precedence() + 1,
                                                    LhsExpr::NotYetParsed)?)
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

                let r = try!(self.mk_range(Some(lhs), rhs, limits));
                lhs = self.mk_expr(lhs_span.to(rhs_span), r, ThinVec::new());
                break
            }

            let rhs = match op.fixity() {
                Fixity::Right => self.with_res(
                    restrictions - Restrictions::STMT_EXPR,
                    |this| {
                        this.parse_assoc_expr_with(op.precedence(),
                            LhsExpr::NotYetParsed)
                }),
                Fixity::Left => self.with_res(
                    restrictions - Restrictions::STMT_EXPR,
                    |this| {
                        this.parse_assoc_expr_with(op.precedence() + 1,
                            LhsExpr::NotYetParsed)
                }),
                // We currently have no non-associative operators that are not handled above by
                // the special cases. The code is here only for future convenience.
                Fixity::None => self.with_res(
                    restrictions - Restrictions::STMT_EXPR,
                    |this| {
                        this.parse_assoc_expr_with(op.precedence() + 1,
                            LhsExpr::NotYetParsed)
                }),
            }?;

            let span = lhs_span.to(rhs.span);
            lhs = match op {
                AssocOp::Add | AssocOp::Subtract | AssocOp::Multiply | AssocOp::Divide |
                AssocOp::Modulus | AssocOp::LAnd | AssocOp::LOr | AssocOp::BitXor |
                AssocOp::BitAnd | AssocOp::BitOr | AssocOp::ShiftLeft | AssocOp::ShiftRight |
                AssocOp::Equal | AssocOp::Less | AssocOp::LessEqual | AssocOp::NotEqual |
                AssocOp::Greater | AssocOp::GreaterEqual => {
                    let ast_op = op.to_ast_binop().unwrap();
                    let binary = self.mk_binary(codemap::respan(cur_op_span, ast_op), lhs, rhs);
                    self.mk_expr(span, binary, ThinVec::new())
                }
                AssocOp::Assign =>
                    self.mk_expr(span, ExprKind::Assign(lhs, rhs), ThinVec::new()),
                AssocOp::ObsoleteInPlace =>
                    self.mk_expr(span, ExprKind::ObsoleteInPlace(lhs, rhs), ThinVec::new()),
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
                    let aopexpr = self.mk_assign_op(codemap::respan(cur_op_span, aop), lhs, rhs);
                    self.mk_expr(span, aopexpr, ThinVec::new())
                }
                AssocOp::As | AssocOp::Colon | AssocOp::DotDot | AssocOp::DotDotEq => {
                    self.bug("AssocOp should have been handled by special case")
                }
            };

            if op.fixity() == Fixity::None { break }
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
                        let (op_noun, op_verb) = match self.token {
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
                        let mut err = self.sess.span_diagnostic.struct_span_err(self.span, &msg);
                        err.span_label(self.look_ahead_span(1).to(parser_snapshot_after_type.span),
                                       "interpreted as generic arguments");
                        err.span_label(self.span, format!("not interpreted as {}", op_noun));

                        let expr = mk_expr(self, P(Ty {
                            span: path.span,
                            node: TyKind::Path(None, path),
                            id: ast::DUMMY_NODE_ID
                        }));

                        let expr_str = self.sess.codemap().span_to_snippet(expr.span)
                                                .unwrap_or(pprust::expr_to_string(&expr));
                        err.span_suggestion(expr.span,
                                            &format!("try {} the cast value", op_verb),
                                            format!("({})", expr_str));
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

    /// Produce an error if comparison operators are chained (RFC #558).
    /// We only need to check lhs, not rhs, because all comparison ops
    /// have same precedence and are left-associative
    fn check_no_chained_comparison(&mut self, lhs: &Expr, outer_op: &AssocOp) {
        debug_assert!(outer_op.is_comparison(),
                      "check_no_chained_comparison: {:?} is not comparison",
                      outer_op);
        match lhs.node {
            ExprKind::Binary(op, _, _) if op.node.is_comparison() => {
                // respan to include both operators
                let op_span = op.span.to(self.span);
                let mut err = self.diagnostic().struct_span_err(op_span,
                    "chained comparison operators require parentheses");
                if op.node == BinOpKind::Lt &&
                    *outer_op == AssocOp::Less ||  // Include `<` to provide this recommendation
                    *outer_op == AssocOp::Greater  // even in a case like the following:
                {                                  //     Foo<Bar<Baz<Qux, ()>>>
                    err.help(
                        "use `::<...>` instead of `<...>` if you meant to specify type arguments");
                    err.help("or use `(...)` if you meant to specify fn arguments");
                }
                err.emit();
            }
            _ => {}
        }
    }

    /// Parse prefix-forms of range notation: `..expr`, `..`, `..=expr`
    fn parse_prefix_range_expr(&mut self,
                               already_parsed_attrs: Option<ThinVec<Attribute>>)
                               -> PResult<'a, P<Expr>> {
        // Check for deprecated `...` syntax
        if self.token == token::DotDotDot {
            self.err_dotdotdot_syntax(self.span);
        }

        debug_assert!([token::DotDot, token::DotDotDot, token::DotDotEq].contains(&self.token),
                      "parse_prefix_range_expr: token {:?} is not DotDot/DotDotEq",
                      self.token);
        let tok = self.token.clone();
        let attrs = self.parse_or_use_outer_attributes(already_parsed_attrs)?;
        let lo = self.span;
        let mut hi = self.span;
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

        let r = try!(self.mk_range(None,
                                   opt_end,
                                   limits));
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

    /// Parse an 'if' or 'if let' expression ('if' token already eaten)
    pub fn parse_if_expr(&mut self, attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        if self.check_keyword(keywords::Let) {
            return self.parse_if_let_expr(attrs);
        }
        let lo = self.prev_span;
        let cond = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;

        // Verify that the parsed `if` condition makes sense as a condition. If it is a block, then
        // verify that the last statement is either an implicit return (no `;`) or an explicit
        // return. This won't catch blocks with an explicit `return`, but that would be caught by
        // the dead code lint.
        if self.eat_keyword(keywords::Else) || !cond.returns() {
            let sp = self.sess.codemap().next_point(lo);
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
        if self.eat_keyword(keywords::Else) {
            let elexpr = self.parse_else_expr()?;
            hi = elexpr.span;
            els = Some(elexpr);
        }
        Ok(self.mk_expr(lo.to(hi), ExprKind::If(cond, thn, els), attrs))
    }

    /// Parse an 'if let' expression ('if' token already eaten)
    pub fn parse_if_let_expr(&mut self, attrs: ThinVec<Attribute>)
                             -> PResult<'a, P<Expr>> {
        let lo = self.prev_span;
        self.expect_keyword(keywords::Let)?;
        let pats = self.parse_pats()?;
        self.expect(&token::Eq)?;
        let expr = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        let thn = self.parse_block()?;
        let (hi, els) = if self.eat_keyword(keywords::Else) {
            let expr = self.parse_else_expr()?;
            (expr.span, Some(expr))
        } else {
            (thn.span, None)
        };
        Ok(self.mk_expr(lo.to(hi), ExprKind::IfLet(pats, expr, thn, els), attrs))
    }

    // `move |args| expr`
    pub fn parse_lambda_expr(&mut self,
                             attrs: ThinVec<Attribute>)
                             -> PResult<'a, P<Expr>>
    {
        let lo = self.span;
        let movability = if self.eat_keyword(keywords::Static) {
            Movability::Static
        } else {
            Movability::Movable
        };
        let capture_clause = if self.eat_keyword(keywords::Move) {
            CaptureBy::Value
        } else {
            CaptureBy::Ref
        };
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
                let body_lo = self.span;
                self.parse_block_expr(body_lo, BlockCheckMode::Default, ThinVec::new())?
            }
        };

        Ok(self.mk_expr(
            lo.to(body.span),
            ExprKind::Closure(capture_clause, movability, decl, body, lo.to(decl_hi)),
            attrs))
    }

    // `else` token already eaten
    pub fn parse_else_expr(&mut self) -> PResult<'a, P<Expr>> {
        if self.eat_keyword(keywords::If) {
            return self.parse_if_expr(ThinVec::new());
        } else {
            let blk = self.parse_block()?;
            return Ok(self.mk_expr(blk.span, ExprKind::Block(blk), ThinVec::new()));
        }
    }

    /// Parse a 'for' .. 'in' expression ('for' token already eaten)
    pub fn parse_for_expr(&mut self, opt_label: Option<Label>,
                          span_lo: Span,
                          mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        // Parse: `for <src_pat> in <src_expr> <src_loop_block>`

        let pat = self.parse_top_level_pat()?;
        if !self.eat_keyword(keywords::In) {
            let in_span = self.prev_span.between(self.span);
            let mut err = self.sess.span_diagnostic
                .struct_span_err(in_span, "missing `in` in `for` loop");
            err.span_suggestion_short(in_span, "try adding `in` here", " in ".into());
            err.emit();
        }
        let expr = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        let (iattrs, loop_block) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);

        let hi = self.prev_span;
        Ok(self.mk_expr(span_lo.to(hi), ExprKind::ForLoop(pat, expr, loop_block, opt_label), attrs))
    }

    /// Parse a 'while' or 'while let' expression ('while' token already eaten)
    pub fn parse_while_expr(&mut self, opt_label: Option<Label>,
                            span_lo: Span,
                            mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        if self.token.is_keyword(keywords::Let) {
            return self.parse_while_let_expr(opt_label, span_lo, attrs);
        }
        let cond = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        return Ok(self.mk_expr(span, ExprKind::While(cond, body, opt_label), attrs));
    }

    /// Parse a 'while let' expression ('while' token already eaten)
    pub fn parse_while_let_expr(&mut self, opt_label: Option<Label>,
                                span_lo: Span,
                                mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        self.expect_keyword(keywords::Let)?;
        let pats = self.parse_pats()?;
        self.expect(&token::Eq)?;
        let expr = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL, None)?;
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        return Ok(self.mk_expr(span, ExprKind::WhileLet(pats, expr, body, opt_label), attrs));
    }

    // parse `loop {...}`, `loop` token already eaten
    pub fn parse_loop_expr(&mut self, opt_label: Option<Label>,
                           span_lo: Span,
                           mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        let span = span_lo.to(body.span);
        Ok(self.mk_expr(span, ExprKind::Loop(body, opt_label), attrs))
    }

    /// Parse a `do catch {...}` expression (`do catch` token already eaten)
    pub fn parse_catch_expr(&mut self, span_lo: Span, mut attrs: ThinVec<Attribute>)
        -> PResult<'a, P<Expr>>
    {
        let (iattrs, body) = self.parse_inner_attrs_and_block()?;
        attrs.extend(iattrs);
        Ok(self.mk_expr(span_lo.to(body.span), ExprKind::Catch(body), attrs))
    }

    // `match` token already eaten
    fn parse_match_expr(&mut self, mut attrs: ThinVec<Attribute>) -> PResult<'a, P<Expr>> {
        let match_span = self.prev_span;
        let lo = self.prev_span;
        let discriminant = self.parse_expr_res(Restrictions::NO_STRUCT_LITERAL,
                                               None)?;
        if let Err(mut e) = self.expect(&token::OpenDelim(token::Brace)) {
            if self.token == token::Token::Semi {
                e.span_suggestion_short(match_span, "try removing this `match`", "".to_owned());
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
                    let span = lo.to(self.span);
                    if self.token == token::CloseDelim(token::Brace) {
                        self.bump();
                    }
                    return Ok(self.mk_expr(span, ExprKind::Match(discriminant, arms), attrs));
                }
            }
        }
        let hi = self.span;
        self.bump();
        return Ok(self.mk_expr(lo.to(hi), ExprKind::Match(discriminant, arms), attrs));
    }

    pub fn parse_arm(&mut self) -> PResult<'a, Arm> {
        maybe_whole!(self, NtArm, |x| x);

        let attrs = self.parse_outer_attributes()?;
        // Allow a '|' before the pats (RFC 1925)
        self.eat(&token::BinOp(token::Or));
        let pats = self.parse_pats()?;
        let guard = if self.eat_keyword(keywords::If) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let arrow_span = self.span;
        self.expect(&token::FatArrow)?;
        let arm_start_span = self.span;

        let expr = self.parse_expr_res(Restrictions::STMT_EXPR, None)
            .map_err(|mut err| {
                err.span_label(arrow_span, "while parsing the `match` arm starting here");
                err
            })?;

        let require_comma = classify::expr_requires_semi_to_be_stmt(&expr)
            && self.token != token::CloseDelim(token::Brace);

        if require_comma {
            let cm = self.sess.codemap();
            self.expect_one_of(&[token::Comma], &[token::CloseDelim(token::Brace)])
                .map_err(|mut err| {
                    match (cm.span_to_lines(expr.span), cm.span_to_lines(arm_start_span)) {
                        (Ok(ref expr_lines), Ok(ref arm_start_lines))
                        if arm_start_lines.lines[0].end_col == expr_lines.lines[0].end_col
                            && expr_lines.lines.len() == 2
                            && self.token == token::FatArrow => {
                            // We check wether there's any trailing code in the parse span, if there
                            // isn't, we very likely have the following:
                            //
                            // X |     &Y => "y"
                            //   |        --    - missing comma
                            //   |        |
                            //   |        arrow_span
                            // X |     &X => "x"
                            //   |      - ^^ self.span
                            //   |      |
                            //   |      parsed until here as `"y" & X`
                            err.span_suggestion_short(
                                cm.next_point(arm_start_span),
                                "missing a comma here to end this `match` arm",
                                ",".to_owned()
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
        })
    }

    /// Parse an expression
    pub fn parse_expr(&mut self) -> PResult<'a, P<Expr>> {
        self.parse_expr_res(Restrictions::empty(), None)
    }

    /// Evaluate the closure with restrictions in place.
    ///
    /// After the closure is evaluated, restrictions are reset.
    pub fn with_res<F, T>(&mut self, r: Restrictions, f: F) -> T
        where F: FnOnce(&mut Self) -> T
    {
        let old = self.restrictions;
        self.restrictions = r;
        let r = f(self);
        self.restrictions = old;
        return r;

    }

    /// Parse an expression, subject to the given restrictions
    pub fn parse_expr_res(&mut self, r: Restrictions,
                          already_parsed_attrs: Option<ThinVec<Attribute>>)
                          -> PResult<'a, P<Expr>> {
        self.with_res(r, |this| this.parse_assoc_expr(already_parsed_attrs))
    }

    /// Parse the RHS of a local variable declaration (e.g. '= 14;')
    fn parse_initializer(&mut self, skip_eq: bool) -> PResult<'a, Option<P<Expr>>> {
        if self.check(&token::Eq) {
            self.bump();
            Ok(Some(self.parse_expr()?))
        } else if skip_eq {
            Ok(Some(self.parse_expr()?))
        } else {
            Ok(None)
        }
    }

    /// Parse patterns, separated by '|' s
    fn parse_pats(&mut self) -> PResult<'a, Vec<P<Pat>>> {
        let mut pats = Vec::new();
        loop {
            pats.push(self.parse_top_level_pat()?);

            if self.token == token::OrOr {
                let mut err = self.struct_span_err(self.span,
                                                   "unexpected token `||` after pattern");
                err.span_suggestion(self.span,
                                    "use a single `|` to specify multiple patterns",
                                    "|".to_owned());
                err.emit();
                self.bump();
            } else if self.check(&token::BinOp(token::Or)) {
                self.bump();
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
        let result = self.parse_pat_list()?;
        self.expect(&token::CloseDelim(token::Paren))?;
        Ok(result)
    }

    fn parse_pat_list(&mut self) -> PResult<'a, (Vec<P<Pat>>, Option<usize>, bool)> {
        let mut fields = Vec::new();
        let mut ddpos = None;
        let mut trailing_comma = false;
        loop {
            if self.eat(&token::DotDot) {
                if ddpos.is_none() {
                    ddpos = Some(fields.len());
                } else {
                    // Emit a friendly error, ignore `..` and continue parsing
                    self.span_err(self.prev_span,
                                  "`..` can only be used once per tuple or tuple struct pattern");
                }
            } else if !self.check(&token::CloseDelim(token::Paren)) {
                fields.push(self.parse_pat()?);
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
            self.span_err(self.prev_span, "trailing comma is not permitted after `..`");
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

            let subpat = self.parse_pat()?;
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

    /// Parse the fields of a struct-like pattern
    fn parse_pat_fields(&mut self) -> PResult<'a, (Vec<codemap::Spanned<ast::FieldPat>>, bool)> {
        let mut fields = Vec::new();
        let mut etc = false;
        let mut first = true;
        while self.token != token::CloseDelim(token::Brace) {
            if first {
                first = false;
            } else {
                self.expect(&token::Comma)?;
                // accept trailing commas
                if self.check(&token::CloseDelim(token::Brace)) { break }
            }

            let attrs = self.parse_outer_attributes()?;
            let lo = self.span;
            let hi;

            if self.check(&token::DotDot) || self.token == token::DotDotDot {
                if self.token == token::DotDotDot { // Issue #46718
                    let mut err = self.struct_span_err(self.span,
                                                       "expected field pattern, found `...`");
                    err.span_suggestion(self.span,
                                        "to omit remaining fields, use one fewer `.`",
                                        "..".to_owned());
                    err.emit();
                }

                self.bump();
                if self.token != token::CloseDelim(token::Brace) {
                    let token_str = self.this_token_to_string();
                    let mut err = self.fatal(&format!("expected `{}`, found `{}`", "}", token_str));
                    if self.token == token::Comma { // Issue #49257
                        err.span_label(self.span,
                                       "`..` must be in the last position, \
                                        and cannot have a trailing comma");
                    } else {
                        err.span_label(self.span, "expected `}`");
                    }
                    return Err(err);
                }
                etc = true;
                break;
            }

            // Check if a colon exists one ahead. This means we're parsing a fieldname.
            let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
                // Parsing a pattern of the form "fieldname: pat"
                let fieldname = self.parse_field_name()?;
                self.bump();
                let pat = self.parse_pat()?;
                hi = pat.span;
                (pat, fieldname, false)
            } else {
                // Parsing a pattern of the form "(box) (ref) (mut) fieldname"
                let is_box = self.eat_keyword(keywords::Box);
                let boxed_span = self.span;
                let is_ref = self.eat_keyword(keywords::Ref);
                let is_mut = self.eat_keyword(keywords::Mut);
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

            fields.push(codemap::Spanned { span: lo.to(hi),
                                           node: ast::FieldPat {
                                               ident: fieldname,
                                               pat: subpat,
                                               is_shorthand,
                                               attrs: attrs.into(),
                                           }
            });
        }
        return Ok((fields, etc));
    }

    fn parse_pat_range_end(&mut self) -> PResult<'a, P<Expr>> {
        if self.token.is_path_start() {
            let lo = self.span;
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
            self.parse_pat_literal_maybe_minus()
        }
    }

    // helper function to decide whether to parse as ident binding or to try to do
    // something more complex like range patterns
    fn parse_as_ident(&mut self) -> bool {
        self.look_ahead(1, |t| match *t {
            token::OpenDelim(token::Paren) | token::OpenDelim(token::Brace) |
            token::DotDotDot | token::DotDotEq | token::ModSep | token::Not => Some(false),
            // ensure slice patterns [a, b.., c] and [a, b, c..] don't go into the
            // range pattern branch
            token::DotDot => None,
            _ => Some(true),
        }).unwrap_or_else(|| self.look_ahead(2, |t| match *t {
            token::Comma | token::CloseDelim(token::Bracket) => true,
            _ => false,
        }))
    }

    /// A wrapper around `parse_pat` with some special error handling for the
    /// "top-level" patterns in a match arm, `for` loop, `let`, &c. (in contast
    /// to subpatterns within such).
    pub fn parse_top_level_pat(&mut self) -> PResult<'a, P<Pat>> {
        let pat = self.parse_pat()?;
        if self.token == token::Comma {
            // An unexpected comma after a top-level pattern is a clue that the
            // user (perhaps more accustomed to some other language) forgot the
            // parentheses in what should have been a tuple pattern; return a
            // suggestion-enhanced error here rather than choking on the comma
            // later.
            let comma_span = self.span;
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
            if let Ok(seq_snippet) = self.sess.codemap().span_to_snippet(seq_span) {
                err.span_suggestion(seq_span, "try adding parentheses",
                                    format!("({})", seq_snippet));
            }
            return Err(err);
        }
        Ok(pat)
    }

    /// Parse a pattern.
    pub fn parse_pat(&mut self) -> PResult<'a, P<Pat>> {
        self.parse_pat_with_range_pat(true)
    }

    /// Parse a pattern, with a setting whether modern range patterns e.g. `a..=b`, `a..b` are
    /// allowed.
    fn parse_pat_with_range_pat(&mut self, allow_range_pat: bool) -> PResult<'a, P<Pat>> {
        maybe_whole!(self, NtPat, |x| x);

        let lo = self.span;
        let pat;
        match self.token {
            token::BinOp(token::And) | token::AndAnd => {
                // Parse &pat / &mut pat
                self.expect_and()?;
                let mutbl = self.parse_mutability();
                if let token::Lifetime(ident) = self.token {
                    let mut err = self.fatal(&format!("unexpected lifetime `{}` in pattern",
                                                      ident));
                    err.span_label(self.span, "unexpected lifetime");
                    return Err(err);
                }
                let subpat = self.parse_pat_with_range_pat(false)?;
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
            _ => if self.eat_keyword(keywords::Underscore) {
                // Parse _
                pat = PatKind::Wild;
            } else if self.eat_keyword(keywords::Mut) {
                // Parse mut ident @ pat / mut ref ident @ pat
                let mutref_span = self.prev_span.to(self.span);
                let binding_mode = if self.eat_keyword(keywords::Ref) {
                    self.diagnostic()
                        .struct_span_err(mutref_span, "the order of `mut` and `ref` is incorrect")
                        .span_suggestion(mutref_span, "try switching the order", "ref mut".into())
                        .emit();
                    BindingMode::ByRef(Mutability::Mutable)
                } else {
                    BindingMode::ByValue(Mutability::Mutable)
                };
                pat = self.parse_pat_ident(binding_mode)?;
            } else if self.eat_keyword(keywords::Ref) {
                // Parse ref ident @ pat / ref mut ident @ pat
                let mutbl = self.parse_mutability();
                pat = self.parse_pat_ident(BindingMode::ByRef(mutbl))?;
            } else if self.eat_keyword(keywords::Box) {
                // Parse box pat
                let subpat = self.parse_pat_with_range_pat(false)?;
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
                match self.token {
                    token::Not if qself.is_none() => {
                        // Parse macro invocation
                        self.bump();
                        let (_, tts) = self.expect_delimited_token_tree()?;
                        let mac = respan(lo.to(self.prev_span), Mac_ { path: path, tts: tts });
                        pat = PatKind::Mac(mac);
                    }
                    token::DotDotDot | token::DotDotEq | token::DotDot => {
                        let end_kind = match self.token {
                            token::DotDot => RangeEnd::Excluded,
                            token::DotDotDot => RangeEnd::Included(RangeSyntax::DotDotDot),
                            token::DotDotEq => RangeEnd::Included(RangeSyntax::DotDotEq),
                            _ => panic!("can only parse `..`/`...`/`..=` for ranges \
                                         (checked above)"),
                        };
                        // Parse range
                        let span = lo.to(self.prev_span);
                        let begin = self.mk_expr(span, ExprKind::Path(qself, path), ThinVec::new());
                        self.bump();
                        let end = self.parse_pat_range_end()?;
                        pat = PatKind::Range(begin, end, end_kind);
                    }
                    token::OpenDelim(token::Brace) => {
                        if qself.is_some() {
                            let msg = "unexpected `{` after qualified path";
                            let mut err = self.fatal(msg);
                            err.span_label(self.span, msg);
                            return Err(err);
                        }
                        // Parse struct pattern
                        self.bump();
                        let (fields, etc) = self.parse_pat_fields().unwrap_or_else(|mut e| {
                            e.emit();
                            self.recover_stmt();
                            (vec![], false)
                        });
                        self.bump();
                        pat = PatKind::Struct(path, fields, etc);
                    }
                    token::OpenDelim(token::Paren) => {
                        if qself.is_some() {
                            let msg = "unexpected `(` after qualified path";
                            let mut err = self.fatal(msg);
                            err.span_label(self.span, msg);
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
                match self.parse_pat_literal_maybe_minus() {
                    Ok(begin) => {
                        if self.eat(&token::DotDotDot) {
                            let end = self.parse_pat_range_end()?;
                            pat = PatKind::Range(begin, end,
                                    RangeEnd::Included(RangeSyntax::DotDotDot));
                        } else if self.eat(&token::DotDotEq) {
                            let end = self.parse_pat_range_end()?;
                            pat = PatKind::Range(begin, end,
                                    RangeEnd::Included(RangeSyntax::DotDotEq));
                        } else if self.eat(&token::DotDot) {
                            let end = self.parse_pat_range_end()?;
                            pat = PatKind::Range(begin, end, RangeEnd::Excluded);
                        } else {
                            pat = PatKind::Lit(begin);
                        }
                    }
                    Err(mut err) => {
                        self.cancel(&mut err);
                        let msg = format!("expected pattern, found {}", self.this_token_descr());
                        let mut err = self.fatal(&msg);
                        err.span_label(self.span, "expected pattern");
                        return Err(err);
                    }
                }
            }
        }

        let pat = Pat { node: pat, span: lo.to(self.prev_span), id: ast::DUMMY_NODE_ID };
        let pat = self.maybe_recover_from_bad_qpath(pat, true)?;

        if !allow_range_pat {
            match pat.node {
                PatKind::Range(_, _, RangeEnd::Included(RangeSyntax::DotDotDot)) => {}
                PatKind::Range(..) => {
                    let mut err = self.struct_span_err(
                        pat.span,
                        "the range pattern here has ambiguous interpretation",
                    );
                    err.span_suggestion(
                        pat.span,
                        "add parentheses to clarify the precedence",
                        format!("({})", pprust::pat_to_string(&pat)),
                    );
                    return Err(err);
                }
                _ => {}
            }
        }

        Ok(P(pat))
    }

    /// Parse ident or ident @ pat
    /// used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like ref foo(a,b)
    fn parse_pat_ident(&mut self,
                       binding_mode: ast::BindingMode)
                       -> PResult<'a, PatKind> {
        let ident = self.parse_ident()?;
        let sub = if self.eat(&token::At) {
            Some(self.parse_pat()?)
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

    /// Parse a local variable declaration
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

                    let snippet = self.sess.codemap().span_to_snippet(pat.span).unwrap();
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
                err.span_suggestion_short(colon_sp,
                                          "use `=` if you meant to assign",
                                          "=".to_string());
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
            self.span
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

    /// Parse a structure field
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

    /// Emit an expected item after attributes error.
    fn expected_item_err(&self, attrs: &[Attribute]) {
        let message = match attrs.last() {
            Some(&Attribute { is_sugared_doc: true, .. }) => "expected item after doc comment",
            _ => "expected item after attributes",
        };

        self.span_err(self.prev_span, message);
    }

    /// Parse a statement. This stops just before trailing semicolons on everything but items.
    /// e.g. a `StmtKind::Semi` parses to a `StmtKind::Expr`, leaving the trailing `;` unconsumed.
    pub fn parse_stmt(&mut self) -> PResult<'a, Option<Stmt>> {
        Ok(self.parse_stmt_(true))
    }

    // Eat tokens until we can be relatively sure we reached the end of the
    // statement. This is something of a best-effort heuristic.
    //
    // We terminate when we find an unmatched `}` (without consuming it).
    fn recover_stmt(&mut self) {
        self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore)
    }

    // If `break_on_semi` is `Break`, then we will stop consuming tokens after
    // finding (and consuming) a `;` outside of `{}` or `[]` (note that this is
    // approximate - it can mean we break too early due to macros, but that
    // shoud only lead to sub-optimal recovery, not inaccurate parsing).
    //
    // If `break_on_block` is `Break`, then we will stop consuming tokens
    // after finding (and consuming) a brace-delimited block.
    fn recover_stmt_(&mut self, break_on_semi: SemiColonMode, break_on_block: BlockMode) {
        let mut brace_depth = 0;
        let mut bracket_depth = 0;
        let mut in_block = false;
        debug!("recover_stmt_ enter loop (semi={:?}, block={:?})",
               break_on_semi, break_on_block);
        loop {
            debug!("recover_stmt_ loop {:?}", self.token);
            match self.token {
                token::OpenDelim(token::DelimToken::Brace) => {
                    brace_depth += 1;
                    self.bump();
                    if break_on_block == BlockMode::Break &&
                       brace_depth == 1 &&
                       bracket_depth == 0 {
                        in_block = true;
                    }
                }
                token::OpenDelim(token::DelimToken::Bracket) => {
                    bracket_depth += 1;
                    self.bump();
                }
                token::CloseDelim(token::DelimToken::Brace) => {
                    if brace_depth == 0 {
                        debug!("recover_stmt_ return - close delim {:?}", self.token);
                        return;
                    }
                    brace_depth -= 1;
                    self.bump();
                    if in_block && bracket_depth == 0 && brace_depth == 0 {
                        debug!("recover_stmt_ return - block end {:?}", self.token);
                        return;
                    }
                }
                token::CloseDelim(token::DelimToken::Bracket) => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        bracket_depth = 0;
                    }
                    self.bump();
                }
                token::Eof => {
                    debug!("recover_stmt_ return - Eof");
                    return;
                }
                token::Semi => {
                    self.bump();
                    if break_on_semi == SemiColonMode::Break &&
                       brace_depth == 0 &&
                       bracket_depth == 0 {
                        debug!("recover_stmt_ return - Semi");
                        return;
                    }
                }
                _ => {
                    self.bump()
                }
            }
        }
    }

    fn parse_stmt_(&mut self, macro_legacy_warnings: bool) -> Option<Stmt> {
        self.parse_stmt_without_recovery(macro_legacy_warnings).unwrap_or_else(|mut e| {
            e.emit();
            self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
            None
        })
    }

    fn is_catch_expr(&mut self) -> bool {
        self.token.is_keyword(keywords::Do) &&
        self.look_ahead(1, |t| t.is_keyword(keywords::Catch)) &&
        self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace)) &&

        // prevent `while catch {} {}`, `if catch {} {} else {}`, etc.
        !self.restrictions.contains(Restrictions::NO_STRUCT_LITERAL)
    }

    fn is_union_item(&self) -> bool {
        self.token.is_keyword(keywords::Union) &&
        self.look_ahead(1, |t| t.is_ident() && !t.is_reserved_ident())
    }

    fn is_crate_vis(&self) -> bool {
        self.token.is_keyword(keywords::Crate) && self.look_ahead(1, |t| t != &token::ModSep)
    }

    fn is_extern_non_path(&self) -> bool {
        self.token.is_keyword(keywords::Extern) && self.look_ahead(1, |t| t != &token::ModSep)
    }

    fn is_auto_trait_item(&mut self) -> bool {
        // auto trait
        (self.token.is_keyword(keywords::Auto)
            && self.look_ahead(1, |t| t.is_keyword(keywords::Trait)))
        || // unsafe auto trait
        (self.token.is_keyword(keywords::Unsafe) &&
         self.look_ahead(1, |t| t.is_keyword(keywords::Auto)) &&
         self.look_ahead(2, |t| t.is_keyword(keywords::Trait)))
    }

    fn eat_macro_def(&mut self, attrs: &[Attribute], vis: &Visibility, lo: Span)
                     -> PResult<'a, Option<P<Item>>> {
        let token_lo = self.span;
        let (ident, def) = match self.token {
            token::Ident(ident, false) if ident.name == keywords::Macro.name() => {
                self.bump();
                let ident = self.parse_ident()?;
                let tokens = if self.check(&token::OpenDelim(token::Brace)) {
                    match self.parse_token_tree() {
                        TokenTree::Delimited(_, ref delimited) => delimited.stream(),
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
                    TokenStream::concat(vec![
                        args.into(),
                        TokenTree::Token(token_lo.to(self.prev_span), token::FatArrow).into(),
                        body.into(),
                    ])
                } else {
                    self.unexpected()?;
                    unreachable!()
                };

                (ident, ast::MacroDef { tokens: tokens.into(), legacy: false })
            }
            token::Ident(ident, _) if ident.name == "macro_rules" &&
                                   self.look_ahead(1, |t| *t == token::Not) => {
                let prev_span = self.prev_span;
                self.complain_if_pub_macro(&vis.node, prev_span);
                self.bump();
                self.bump();

                let ident = self.parse_ident()?;
                let (delim, tokens) = self.expect_delimited_token_tree()?;
                if delim != token::Brace {
                    if !self.eat(&token::Semi) {
                        let msg = "macros that expand to items must either \
                                   be surrounded with braces or followed by a semicolon";
                        self.span_err(self.prev_span, msg);
                    }
                }

                (ident, ast::MacroDef { tokens: tokens, legacy: true })
            }
            _ => return Ok(None),
        };

        let span = lo.to(self.prev_span);
        Ok(Some(self.mk_item(span, ident, ItemKind::MacroDef(def), vis.clone(), attrs.to_vec())))
    }

    fn parse_stmt_without_recovery(&mut self,
                                   macro_legacy_warnings: bool)
                                   -> PResult<'a, Option<Stmt>> {
        maybe_whole!(self, NtStmt, |x| Some(x));

        let attrs = self.parse_outer_attributes()?;
        let lo = self.span;

        Ok(Some(if self.eat_keyword(keywords::Let) {
            Stmt {
                id: ast::DUMMY_NODE_ID,
                node: StmtKind::Local(self.parse_local(attrs.into())?),
                span: lo.to(self.prev_span),
            }
        } else if let Some(macro_def) = self.eat_macro_def(
            &attrs,
            &codemap::respan(lo, VisibilityKind::Inherited),
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
        // `extern::b::c` - path, `extern crate c;` - not a path.
        } else if self.token.is_path_start() &&
                  !self.token.is_qpath_start() &&
                  !self.is_union_item() &&
                  !self.is_crate_vis() &&
                  !self.is_extern_non_path() &&
                  !self.is_auto_trait_item() {
            let pth = self.parse_path(PathStyle::Expr)?;

            if !self.eat(&token::Not) {
                let expr = if self.check(&token::OpenDelim(token::Brace)) {
                    self.parse_struct_expr(lo, pth, ThinVec::new())?
                } else {
                    let hi = self.prev_span;
                    self.mk_expr(lo.to(hi), ExprKind::Path(None, pth), ThinVec::new())
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

            // it's a macro invocation
            let id = match self.token {
                token::OpenDelim(_) => keywords::Invalid.ident(), // no special identifier
                _ => self.parse_ident()?,
            };

            // check that we're pointing at delimiters (need to check
            // again after the `if`, because of `parse_ident`
            // consuming more tokens).
            let delim = match self.token {
                token::OpenDelim(delim) => delim,
                _ => {
                    // we only expect an ident if we didn't parse one
                    // above.
                    let ident_str = if id.name == keywords::Invalid.name() {
                        "identifier, "
                    } else {
                        ""
                    };
                    let tok_str = self.this_token_to_string();
                    let mut err = self.fatal(&format!("expected {}`(` or `{{`, found `{}`",
                                                      ident_str,
                                                      tok_str));
                    err.span_label(self.span, format!("expected {}`(` or `{{`", ident_str));
                    return Err(err)
                },
            };

            let (_, tts) = self.expect_delimited_token_tree()?;
            let hi = self.prev_span;

            let style = if delim == token::Brace {
                MacStmtStyle::Braces
            } else {
                MacStmtStyle::NoBraces
            };

            if id.name == keywords::Invalid.name() {
                let mac = respan(lo.to(hi), Mac_ { path: pth, tts: tts });
                let node = if delim == token::Brace ||
                              self.token == token::Semi || self.token == token::Eof {
                    StmtKind::Mac(P((mac, style, attrs.into())))
                }
                // We used to incorrectly stop parsing macro-expanded statements here.
                // If the next token will be an error anyway but could have parsed with the
                // earlier behavior, stop parsing here and emit a warning to avoid breakage.
                else if macro_legacy_warnings && self.token.can_begin_expr() && match self.token {
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
                    let e = self.mk_mac_expr(lo.to(hi), mac.node, ThinVec::new());
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
                // if it has a special ident, it's definitely an item
                //
                // Require a semicolon or braces.
                if style != MacStmtStyle::Braces {
                    if !self.eat(&token::Semi) {
                        self.span_err(self.prev_span,
                                      "macros that expand to items must \
                                       either be surrounded with braces or \
                                       followed by a semicolon");
                    }
                }
                let span = lo.to(hi);
                Stmt {
                    id: ast::DUMMY_NODE_ID,
                    span,
                    node: StmtKind::Item({
                        self.mk_item(
                            span, id /*id is good here*/,
                            ItemKind::Mac(respan(span, Mac_ { path: pth, tts: tts })),
                            respan(lo, VisibilityKind::Inherited),
                            attrs)
                    }),
                }
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
                                s.span_err(s.span, "expected statement after outer attribute");
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

    /// Is this expression a successfully-parsed statement?
    fn expr_is_complete(&mut self, e: &Expr) -> bool {
        self.restrictions.contains(Restrictions::STMT_EXPR) &&
            !classify::expr_requires_semi_to_be_stmt(e)
    }

    /// Parse a block. No inner attrs are allowed.
    pub fn parse_block(&mut self) -> PResult<'a, P<Block>> {
        maybe_whole!(self, NtBlock, |x| x);

        let lo = self.span;

        if !self.eat(&token::OpenDelim(token::Brace)) {
            let sp = self.span;
            let tok = self.this_token_to_string();
            let mut e = self.span_fatal(sp, &format!("expected `{{`, found `{}`", tok));

            // Check to see if the user has written something like
            //
            //    if (cond)
            //      bar;
            //
            // Which is valid in other languages, but not Rust.
            match self.parse_stmt_without_recovery(false) {
                Ok(Some(stmt)) => {
                    if self.look_ahead(1, |t| t == &token::OpenDelim(token::Brace)) {
                        // if the next token is an open brace (e.g., `if a b {`), the place-
                        // inside-a-block suggestion would be more likely wrong than right
                        return Err(e);
                    }
                    let mut stmt_span = stmt.span;
                    // expand the span to include the semicolon, if it exists
                    if self.eat(&token::Semi) {
                        stmt_span = stmt_span.with_hi(self.prev_span.hi());
                    }
                    let sugg = pprust::to_string(|s| {
                        use print::pprust::{PrintState, INDENT_UNIT};
                        s.ibox(INDENT_UNIT)?;
                        s.bopen()?;
                        s.print_stmt(&stmt)?;
                        s.bclose_maybe_open(stmt.span, INDENT_UNIT, false)
                    });
                    e.span_suggestion(stmt_span, "try placing this code inside a block", sugg);
                }
                Err(mut e) => {
                    self.recover_stmt_(SemiColonMode::Break, BlockMode::Ignore);
                    self.cancel(&mut e);
                }
                _ => ()
            }
            return Err(e);
        }

        self.parse_block_tail(lo, BlockCheckMode::Default)
    }

    /// Parse a block. Inner attrs are allowed.
    fn parse_inner_attrs_and_block(&mut self) -> PResult<'a, (Vec<Attribute>, P<Block>)> {
        maybe_whole!(self, NtBlock, |x| (Vec::new(), x));

        let lo = self.span;
        self.expect(&token::OpenDelim(token::Brace))?;
        Ok((self.parse_inner_attributes()?,
            self.parse_block_tail(lo, BlockCheckMode::Default)?))
    }

    /// Parse the rest of a block expression or function body
    /// Precondition: already parsed the '{'.
    fn parse_block_tail(&mut self, lo: Span, s: BlockCheckMode) -> PResult<'a, P<Block>> {
        let mut stmts = vec![];
        let mut recovered = false;

        while !self.eat(&token::CloseDelim(token::Brace)) {
            let stmt = match self.parse_full_stmt(false) {
                Err(mut err) => {
                    err.emit();
                    self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore);
                    self.eat(&token::CloseDelim(token::Brace));
                    recovered = true;
                    break;
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
            recovered,
        }))
    }

    /// Parse a statement, including the trailing semicolon.
    pub fn parse_full_stmt(&mut self, macro_legacy_warnings: bool) -> PResult<'a, Option<Stmt>> {
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
                    self.expect_one_of(&[token::Semi], &[])?;
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
        self.diagnostic().struct_span_warn(self.span, {
            &format!("expected `;`, found `{}`", self.this_token_to_string())
        }).note({
            "This was erroneously allowed and will become a hard error in a future release"
        }).emit();
    }

    fn err_dotdotdot_syntax(&self, span: Span) {
        self.diagnostic().struct_span_err(span, {
            "`...` syntax cannot be used in expressions"
        }).help({
            "Use `..` if you need an exclusive range (a < b)"
        }).help({
            "or `..=` if you need an inclusive range (a <= b)"
        }).emit();
    }

    // Parse bounds of a type parameter `BOUND + BOUND + BOUND`, possibly with trailing `+`.
    // BOUND = TY_BOUND | LT_BOUND
    // LT_BOUND = LIFETIME (e.g. `'a`)
    // TY_BOUND = TY_BOUND_NOPAREN | (TY_BOUND_NOPAREN)
    // TY_BOUND_NOPAREN = [?] [for<LT_PARAM_DEFS>] SIMPLE_PATH (e.g. `?for<'a: 'b> m::Trait<'a>`)
    fn parse_ty_param_bounds_common(&mut self, allow_plus: bool) -> PResult<'a, TyParamBounds> {
        let mut bounds = Vec::new();
        loop {
            // This needs to be syncronized with `Token::can_begin_bound`.
            let is_bound_start = self.check_path() || self.check_lifetime() ||
                                 self.check(&token::Question) ||
                                 self.check_keyword(keywords::For) ||
                                 self.check(&token::OpenDelim(token::Paren));
            if is_bound_start {
                let has_parens = self.eat(&token::OpenDelim(token::Paren));
                let question = if self.eat(&token::Question) { Some(self.prev_span) } else { None };
                if self.token.is_lifetime() {
                    if let Some(question_span) = question {
                        self.span_err(question_span,
                                      "`?` may only modify trait bounds, not lifetime bounds");
                    }
                    bounds.push(RegionTyParamBound(self.expect_lifetime()));
                } else {
                    let lo = self.span;
                    let lifetime_defs = self.parse_late_bound_lifetime_defs()?;
                    let path = self.parse_path(PathStyle::Type)?;
                    let poly_trait = PolyTraitRef::new(lifetime_defs, path, lo.to(self.prev_span));
                    let modifier = if question.is_some() {
                        TraitBoundModifier::Maybe
                    } else {
                        TraitBoundModifier::None
                    };
                    bounds.push(TraitTyParamBound(poly_trait, modifier));
                }
                if has_parens {
                    self.expect(&token::CloseDelim(token::Paren))?;
                    if let Some(&RegionTyParamBound(..)) = bounds.last() {
                        self.span_err(self.prev_span,
                                      "parenthesized lifetime bounds are not supported");
                    }
                }
            } else {
                break
            }

            if !allow_plus || !self.eat(&token::BinOp(token::Plus)) {
                break
            }
        }

        return Ok(bounds);
    }

    fn parse_ty_param_bounds(&mut self) -> PResult<'a, TyParamBounds> {
        self.parse_ty_param_bounds_common(true)
    }

    // Parse bounds of a lifetime parameter `BOUND + BOUND + BOUND`, possibly with trailing `+`.
    // BOUND = LT_BOUND (e.g. `'a`)
    fn parse_lt_param_bounds(&mut self) -> Vec<Lifetime> {
        let mut lifetimes = Vec::new();
        while self.check_lifetime() {
            lifetimes.push(self.expect_lifetime());

            if !self.eat(&token::BinOp(token::Plus)) {
                break
            }
        }
        lifetimes
    }

    /// Matches typaram = IDENT (`?` unbound)? optbounds ( EQ ty )?
    fn parse_ty_param(&mut self, preceding_attrs: Vec<Attribute>) -> PResult<'a, TyParam> {
        let ident = self.parse_ident()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_ty_param_bounds()?
        } else {
            Vec::new()
        };

        let default = if self.eat(&token::Eq) {
            Some(self.parse_ty()?)
        } else {
            None
        };

        Ok(TyParam {
            attrs: preceding_attrs.into(),
            ident,
            id: ast::DUMMY_NODE_ID,
            bounds,
            default,
        })
    }

    /// Parses the following grammar:
    ///     TraitItemAssocTy = Ident ["<"...">"] [":" [TyParamBounds]] ["where" ...] ["=" Ty]
    fn parse_trait_item_assoc_ty(&mut self, preceding_attrs: Vec<Attribute>)
        -> PResult<'a, (ast::Generics, TyParam)> {
        let ident = self.parse_ident()?;
        let mut generics = self.parse_generics()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_ty_param_bounds()?
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

        Ok((generics, TyParam {
            attrs: preceding_attrs.into(),
            ident,
            id: ast::DUMMY_NODE_ID,
            bounds,
            default,
        }))
    }

    /// Parses (possibly empty) list of lifetime and type parameters, possibly including
    /// trailing comma and erroneous trailing attributes.
    pub fn parse_generic_params(&mut self) -> PResult<'a, Vec<ast::GenericParam>> {
        let mut params = Vec::new();
        let mut seen_ty_param = false;
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
                params.push(ast::GenericParam::Lifetime(LifetimeDef {
                    attrs: attrs.into(),
                    lifetime,
                    bounds,
                }));
                if seen_ty_param {
                    self.span_err(self.prev_span,
                        "lifetime parameters must be declared prior to type parameters");
                }
            } else if self.check_ident() {
                // Parse type parameter.
                params.push(ast::GenericParam::Type(self.parse_ty_param(attrs)?));
                seen_ty_param = true;
            } else {
                // Check for trailing attributes and stop parsing.
                if !attrs.is_empty() {
                    let param_kind = if seen_ty_param { "type" } else { "lifetime" };
                    self.span_err(attrs[0].span,
                        &format!("trailing attribute after {} parameters", param_kind));
                }
                break
            }

            if !self.eat(&token::Comma) {
                break
            }
        }
        Ok(params)
    }

    /// Parse a set of optional generic type parameter declarations. Where
    /// clauses are not parsed here, and must be added later via
    /// `parse_where_clause()`.
    ///
    /// matches generics = ( ) | ( < > ) | ( < typaramseq ( , )? > ) | ( < lifetimes ( , )? > )
    ///                  | ( < lifetimes , typaramseq ( , )? > )
    /// where   typaramseq = ( typaram ) | ( typaram , typaramseq )
    pub fn parse_generics(&mut self) -> PResult<'a, ast::Generics> {
        maybe_whole!(self, NtGenerics, |x| x);

        let span_lo = self.span;
        if self.eat_lt() {
            let params = self.parse_generic_params()?;
            self.expect_gt()?;
            Ok(ast::Generics {
                params,
                where_clause: WhereClause {
                    id: ast::DUMMY_NODE_ID,
                    predicates: Vec::new(),
                    span: syntax_pos::DUMMY_SP,
                },
                span: span_lo.to(self.prev_span),
            })
        } else {
            Ok(ast::Generics::default())
        }
    }

    /// Parses (possibly empty) list of lifetime and type arguments and associated type bindings,
    /// possibly including trailing comma.
    fn parse_generic_args(&mut self) -> PResult<'a, (Vec<Lifetime>, Vec<P<Ty>>, Vec<TypeBinding>)> {
        let mut lifetimes = Vec::new();
        let mut types = Vec::new();
        let mut bindings = Vec::new();
        let mut seen_type = false;
        let mut seen_binding = false;
        loop {
            if self.check_lifetime() && self.look_ahead(1, |t| t != &token::BinOp(token::Plus)) {
                // Parse lifetime argument.
                lifetimes.push(self.expect_lifetime());
                if seen_type || seen_binding {
                    self.span_err(self.prev_span,
                        "lifetime parameters must be declared prior to type parameters");
                }
            } else if self.check_ident() && self.look_ahead(1, |t| t == &token::Eq) {
                // Parse associated type binding.
                let lo = self.span;
                let ident = self.parse_ident()?;
                self.bump();
                let ty = self.parse_ty()?;
                bindings.push(TypeBinding {
                    id: ast::DUMMY_NODE_ID,
                    ident,
                    ty,
                    span: lo.to(self.prev_span),
                });
                seen_binding = true;
            } else if self.check_type() {
                // Parse type argument.
                types.push(self.parse_ty()?);
                if seen_binding {
                    self.span_err(types[types.len() - 1].span,
                        "type parameters must be declared prior to associated type bindings");
                }
                seen_type = true;
            } else {
                break
            }

            if !self.eat(&token::Comma) {
                break
            }
        }
        Ok((lifetimes, types, bindings))
    }

    /// Parses an optional `where` clause and places it in `generics`.
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// where T : Trait<U, V> + 'b, 'a : 'b
    /// ```
    pub fn parse_where_clause(&mut self) -> PResult<'a, WhereClause> {
        maybe_whole!(self, NtWhereClause, |x| x);

        let mut where_clause = WhereClause {
            id: ast::DUMMY_NODE_ID,
            predicates: Vec::new(),
            span: syntax_pos::DUMMY_SP,
        };

        if !self.eat_keyword(keywords::Where) {
            return Ok(where_clause);
        }
        let lo = self.prev_span;

        // We are considering adding generics to the `where` keyword as an alternative higher-rank
        // parameter syntax (as in `where<'a>` or `where<T>`. To avoid that being a breaking
        // change we parse those generics now, but report an error.
        if self.choose_generics_over_qpath() {
            let generics = self.parse_generics()?;
            self.span_err(generics.span,
                          "generic parameters on `where` clauses are reserved for future use");
        }

        loop {
            let lo = self.span;
            if self.check_lifetime() && self.look_ahead(1, |t| t != &token::BinOp(token::Plus)) {
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
                // Example 1: for<'a> Trait1<'a>: Trait2<'a /*ok*/>
                // Example 2: (for<'a> Trait1<'a>): Trait2<'a /*not ok*/>
                // Example 3: for<'a> for<'b> Trait1<'a, 'b>: Trait2<'a /*ok*/, 'b /*not ok*/>
                let lifetime_defs = self.parse_late_bound_lifetime_defs()?;

                // Parse type with mandatory colon and (possibly empty) bounds,
                // or with mandatory equality sign and the second type.
                let ty = self.parse_ty()?;
                if self.eat(&token::Colon) {
                    let bounds = self.parse_ty_param_bounds()?;
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

    fn parse_fn_args(&mut self, named_args: bool, allow_variadic: bool)
                     -> PResult<'a, (Vec<Arg> , bool)> {
        let sp = self.span;
        let mut variadic = false;
        let args: Vec<Option<Arg>> =
            self.parse_unspanned_seq(
                &token::OpenDelim(token::Paren),
                &token::CloseDelim(token::Paren),
                SeqSep::trailing_allowed(token::Comma),
                |p| {
                    if p.token == token::DotDotDot {
                        p.bump();
                        variadic = true;
                        if allow_variadic {
                            if p.token != token::CloseDelim(token::Paren) {
                                let span = p.span;
                                p.span_err(span,
                                    "`...` must be last in argument list for variadic function");
                            }
                            Ok(None)
                        } else {
                            let span = p.prev_span;
                            if p.token == token::CloseDelim(token::Paren) {
                                // continue parsing to present any further errors
                                p.struct_span_err(
                                    span,
                                    "only foreign functions are allowed to be variadic"
                                ).emit();
                                Ok(Some(dummy_arg(span)))
                           } else {
                               // this function definition looks beyond recovery, stop parsing
                                p.span_err(span,
                                           "only foreign functions are allowed to be variadic");
                                Ok(None)
                            }
                        }
                    } else {
                        match p.parse_arg_general(named_args) {
                            Ok(arg) => Ok(Some(arg)),
                            Err(mut e) => {
                                e.emit();
                                let lo = p.prev_span;
                                // Skip every token until next possible arg or end.
                                p.eat_to_tokens(&[&token::Comma, &token::CloseDelim(token::Paren)]);
                                // Create a placeholder argument for proper arg count (#34264).
                                let span = lo.to(p.prev_span);
                                Ok(Some(dummy_arg(span)))
                            }
                        }
                    }
                }
            )?;

        let args: Vec<_> = args.into_iter().filter_map(|x| x).collect();

        if variadic && args.is_empty() {
            self.span_err(sp,
                          "variadic function must be declared with at least one named argument");
        }

        Ok((args, variadic))
    }

    /// Parse the argument list and result type of a function declaration
    pub fn parse_fn_decl(&mut self, allow_variadic: bool) -> PResult<'a, P<FnDecl>> {

        let (args, variadic) = self.parse_fn_args(true, allow_variadic)?;
        let ret_ty = self.parse_ret_ty(true)?;

        Ok(P(FnDecl {
            inputs: args,
            output: ret_ty,
            variadic,
        }))
    }

    /// Returns the parsed optional self argument and whether a self shortcut was used.
    fn parse_self_arg(&mut self) -> PResult<'a, Option<Arg>> {
        let expect_ident = |this: &mut Self| match this.token {
            // Preserve hygienic context.
            token::Ident(ident, _) =>
                { let span = this.span; this.bump(); Ident::new(ident.name, span) }
            _ => unreachable!()
        };
        let isolated_self = |this: &mut Self, n| {
            this.look_ahead(n, |t| t.is_keyword(keywords::SelfValue)) &&
            this.look_ahead(n + 1, |t| t != &token::ModSep)
        };

        // Parse optional self parameter of a method.
        // Only a limited set of initial token sequences is considered self parameters, anything
        // else is parsed as a normal function parameter list, so some lookahead is required.
        let eself_lo = self.span;
        let (eself, eself_ident) = match self.token {
            token::BinOp(token::And) => {
                // &self
                // &mut self
                // &'lt self
                // &'lt mut self
                // &not_self
                if isolated_self(self, 1) {
                    self.bump();
                    (SelfKind::Region(None, Mutability::Immutable), expect_ident(self))
                } else if self.look_ahead(1, |t| t.is_keyword(keywords::Mut)) &&
                          isolated_self(self, 2) {
                    self.bump();
                    self.bump();
                    (SelfKind::Region(None, Mutability::Mutable), expect_ident(self))
                } else if self.look_ahead(1, |t| t.is_lifetime()) &&
                          isolated_self(self, 2) {
                    self.bump();
                    let lt = self.expect_lifetime();
                    (SelfKind::Region(Some(lt), Mutability::Immutable), expect_ident(self))
                } else if self.look_ahead(1, |t| t.is_lifetime()) &&
                          self.look_ahead(2, |t| t.is_keyword(keywords::Mut)) &&
                          isolated_self(self, 3) {
                    self.bump();
                    let lt = self.expect_lifetime();
                    self.bump();
                    (SelfKind::Region(Some(lt), Mutability::Mutable), expect_ident(self))
                } else {
                    return Ok(None);
                }
            }
            token::BinOp(token::Star) => {
                // *self
                // *const self
                // *mut self
                // *not_self
                // Emit special error for `self` cases.
                if isolated_self(self, 1) {
                    self.bump();
                    self.span_err(self.span, "cannot pass `self` by raw pointer");
                    (SelfKind::Value(Mutability::Immutable), expect_ident(self))
                } else if self.look_ahead(1, |t| t.is_mutability()) &&
                          isolated_self(self, 2) {
                    self.bump();
                    self.bump();
                    self.span_err(self.span, "cannot pass `self` by raw pointer");
                    (SelfKind::Value(Mutability::Immutable), expect_ident(self))
                } else {
                    return Ok(None);
                }
            }
            token::Ident(..) => {
                if isolated_self(self, 0) {
                    // self
                    // self: TYPE
                    let eself_ident = expect_ident(self);
                    if self.eat(&token::Colon) {
                        let ty = self.parse_ty()?;
                        (SelfKind::Explicit(ty, Mutability::Immutable), eself_ident)
                    } else {
                        (SelfKind::Value(Mutability::Immutable), eself_ident)
                    }
                } else if self.token.is_keyword(keywords::Mut) &&
                          isolated_self(self, 1) {
                    // mut self
                    // mut self: TYPE
                    self.bump();
                    let eself_ident = expect_ident(self);
                    if self.eat(&token::Colon) {
                        let ty = self.parse_ty()?;
                        (SelfKind::Explicit(ty, Mutability::Mutable), eself_ident)
                    } else {
                        (SelfKind::Value(Mutability::Mutable), eself_ident)
                    }
                } else {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        };

        let eself = codemap::respan(eself_lo.to(self.prev_span), eself);
        Ok(Some(Arg::from_self(eself, eself_ident)))
    }

    /// Parse the parameter list and result type of a function that may have a `self` parameter.
    fn parse_fn_decl_with_self<F>(&mut self, parse_arg_fn: F) -> PResult<'a, P<FnDecl>>
        where F: FnMut(&mut Parser<'a>) -> PResult<'a,  Arg>,
    {
        self.expect(&token::OpenDelim(token::Paren))?;

        // Parse optional self argument
        let self_arg = self.parse_self_arg()?;

        // Parse the rest of the function parameter list.
        let sep = SeqSep::trailing_allowed(token::Comma);
        let fn_inputs = if let Some(self_arg) = self_arg {
            if self.check(&token::CloseDelim(token::Paren)) {
                vec![self_arg]
            } else if self.eat(&token::Comma) {
                let mut fn_inputs = vec![self_arg];
                fn_inputs.append(&mut self.parse_seq_to_before_end(
                    &token::CloseDelim(token::Paren), sep, parse_arg_fn)?
                );
                fn_inputs
            } else {
                return self.unexpected();
            }
        } else {
            self.parse_seq_to_before_end(&token::CloseDelim(token::Paren), sep, parse_arg_fn)?
        };

        // Parse closing paren and return type.
        self.expect(&token::CloseDelim(token::Paren))?;
        Ok(P(FnDecl {
            inputs: fn_inputs,
            output: self.parse_ret_ty(true)?,
            variadic: false
        }))
    }

    // parse the |arg, arg| header on a lambda
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
                )?;
                self.expect_or()?;
                args
            }
        };
        let output = self.parse_ret_ty(true)?;

        Ok(P(FnDecl {
            inputs: inputs_captures,
            output,
            variadic: false
        }))
    }

    /// Parse the name and optional generic types of a function header.
    fn parse_fn_header(&mut self) -> PResult<'a, (Ident, ast::Generics)> {
        let id = self.parse_ident()?;
        let generics = self.parse_generics()?;
        Ok((id, generics))
    }

    fn mk_item(&mut self, span: Span, ident: Ident, node: ItemKind, vis: Visibility,
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

    /// Parse an item-position function declaration.
    fn parse_item_fn(&mut self,
                     unsafety: Unsafety,
                     constness: Spanned<Constness>,
                     abi: Abi)
                     -> PResult<'a, ItemInfo> {
        let (ident, mut generics) = self.parse_fn_header()?;
        let decl = self.parse_fn_decl(false)?;
        generics.where_clause = self.parse_where_clause()?;
        let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
        Ok((ident, ItemKind::Fn(decl, unsafety, constness, abi, generics, body), Some(inner_attrs)))
    }

    /// true if we are looking at `const ID`, false for things like `const fn` etc
    pub fn is_const_item(&mut self) -> bool {
        self.token.is_keyword(keywords::Const) &&
            !self.look_ahead(1, |t| t.is_keyword(keywords::Fn)) &&
            !self.look_ahead(1, |t| t.is_keyword(keywords::Unsafe))
    }

    /// parses all the "front matter" for a `fn` declaration, up to
    /// and including the `fn` keyword:
    ///
    /// - `const fn`
    /// - `unsafe fn`
    /// - `const unsafe fn`
    /// - `extern fn`
    /// - etc
    pub fn parse_fn_front_matter(&mut self) -> PResult<'a, (Spanned<Constness>, Unsafety, Abi)> {
        let is_const_fn = self.eat_keyword(keywords::Const);
        let const_span = self.prev_span;
        let unsafety = self.parse_unsafety();
        let (constness, unsafety, abi) = if is_const_fn {
            (respan(const_span, Constness::Const), unsafety, Abi::Rust)
        } else {
            let abi = if self.eat_keyword(keywords::Extern) {
                self.parse_opt_abi()?.unwrap_or(Abi::C)
            } else {
                Abi::Rust
            };
            (respan(self.prev_span, Constness::NotConst), unsafety, abi)
        };
        self.expect_keyword(keywords::Fn)?;
        Ok((constness, unsafety, abi))
    }

    /// Parse an impl item.
    pub fn parse_impl_item(&mut self, at_end: &mut bool) -> PResult<'a, ImplItem> {
        maybe_whole!(self, NtImplItem, |x| x);
        let attrs = self.parse_outer_attributes()?;
        let (mut item, tokens) = self.collect_tokens(|this| {
            this.parse_impl_item_(at_end, attrs)
        })?;

        // See `parse_item` for why this clause is here.
        if !item.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
            item.tokens = Some(tokens);
        }
        Ok(item)
    }

    fn parse_impl_item_(&mut self,
                        at_end: &mut bool,
                        mut attrs: Vec<Attribute>) -> PResult<'a, ImplItem> {
        let lo = self.span;
        let vis = self.parse_visibility(false)?;
        let defaultness = self.parse_defaultness();
        let (name, node, generics) = if self.eat_keyword(keywords::Type) {
            // This parses the grammar:
            //     ImplItemAssocTy = Ident ["<"...">"] ["where" ...] "=" Ty ";"
            let name = self.parse_ident()?;
            let mut generics = self.parse_generics()?;
            generics.where_clause = self.parse_where_clause()?;
            self.expect(&token::Eq)?;
            let typ = self.parse_ty()?;
            self.expect(&token::Semi)?;
            (name, ast::ImplItemKind::Type(typ), generics)
        } else if self.is_const_item() {
            // This parses the grammar:
            //     ImplItemConst = "const" Ident ":" Ty "=" Expr ";"
            self.expect_keyword(keywords::Const)?;
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

    fn complain_if_pub_macro(&mut self, vis: &VisibilityKind, sp: Span) {
        if let Err(mut err) = self.complain_if_pub_macro_diag(vis, sp) {
            err.emit();
        }
    }

    fn complain_if_pub_macro_diag(&mut self, vis: &VisibilityKind, sp: Span) -> PResult<'a, ()> {
        match *vis {
            VisibilityKind::Inherited => Ok(()),
            _ => {
                let is_macro_rules: bool = match self.token {
                    token::Ident(sid, _) => sid.name == Symbol::intern("macro_rules"),
                    _ => false,
                };
                if is_macro_rules {
                    let mut err = self.diagnostic()
                        .struct_span_err(sp, "can't qualify macro_rules invocation with `pub`");
                    err.span_suggestion(sp,
                                        "try exporting the macro",
                                        "#[macro_export]".to_owned());
                    Err(err)
                } else {
                    let mut err = self.diagnostic()
                        .struct_span_err(sp, "can't qualify macro invocation with `pub`");
                    err.help("try adjusting the macro to put `pub` inside the invocation");
                    Err(err)
                }
            }
        }
    }

    fn missing_assoc_item_kind_err(&mut self, item_type: &str, prev_span: Span)
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
            // Method macro.
            Ok((keywords::Invalid.ident(), vec![], ast::Generics::default(),
                ast::ImplItemKind::Macro(mac)))
        } else {
            let (constness, unsafety, abi) = self.parse_fn_front_matter()?;
            let ident = self.parse_ident()?;
            let mut generics = self.parse_generics()?;
            let decl = self.parse_fn_decl_with_self(|p| p.parse_arg())?;
            generics.where_clause = self.parse_where_clause()?;
            *at_end = true;
            let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
            Ok((ident, inner_attrs, generics, ast::ImplItemKind::Method(ast::MethodSig {
                abi,
                unsafety,
                constness,
                decl,
             }, body)))
        }
    }

    /// Parse `trait Foo { ... }` or `trait Foo = Bar;`
    fn parse_item_trait(&mut self, is_auto: IsAuto, unsafety: Unsafety) -> PResult<'a, ItemInfo> {
        let ident = self.parse_ident()?;
        let mut tps = self.parse_generics()?;

        // Parse optional colon and supertrait bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_ty_param_bounds()?
        } else {
            Vec::new()
        };

        if self.eat(&token::Eq) {
            // it's a trait alias
            let bounds = self.parse_ty_param_bounds()?;
            tps.where_clause = self.parse_where_clause()?;
            self.expect(&token::Semi)?;
            if unsafety != Unsafety::Normal {
                self.span_err(self.prev_span, "trait aliases cannot be unsafe");
            }
            Ok((ident, ItemKind::TraitAlias(tps, bounds), None))
        } else {
            // it's a normal trait
            tps.where_clause = self.parse_where_clause()?;
            self.expect(&token::OpenDelim(token::Brace))?;
            let mut trait_items = vec![];
            while !self.eat(&token::CloseDelim(token::Brace)) {
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
        // The only truly ambiguous case is
        //     `<` IDENT `>` `::` IDENT ...
        // we disambiguate it in favor of generics (`impl<T> ::absolute::Path<T> { ... }`)
        // because this is what almost always expected in practice, qualified paths in impls
        // (`impl <Type>::AssocTy { ... }`) aren't even allowed by type checker at the moment.
        self.token == token::Lt &&
            (self.look_ahead(1, |t| t == &token::Pound || t == &token::Gt) ||
             self.look_ahead(1, |t| t.is_lifetime() || t.is_ident()) &&
                self.look_ahead(2, |t| t == &token::Gt || t == &token::Comma ||
                                       t == &token::Colon || t == &token::Eq))
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
    ///    impl<'a, T> TYPE { /* impl items */ }
    ///    impl<'a, T> TRAIT for TYPE { /* impl items */ }
    ///    impl<'a, T> !TRAIT for TYPE { /* impl items */ }
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
        let ty_first = self.parse_ty()?;

        // If `for` is missing we try to recover.
        let has_for = self.eat_keyword(keywords::For);
        let missing_for_span = self.prev_span.between(self.span);

        let ty_second = if self.token == token::DotDot {
            // We need to report this error after `cfg` expansion for compatibility reasons
            self.bump(); // `..`, do not add it to expected tokens
            Some(P(Ty { node: TyKind::Err, span: self.prev_span, id: ast::DUMMY_NODE_ID }))
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
                    self.span_err(missing_for_span, "missing `for` in a trait impl");
                }

                let ty_first = ty_first.into_inner();
                let path = match ty_first.node {
                    // This notably includes paths passed through `ty` macro fragments (#46438).
                    TyKind::Path(None, path) => path,
                    _ => {
                        self.span_err(ty_first.span, "expected a trait, found type");
                        ast::Path::from_ident(Ident::new(keywords::Invalid.name(), ty_first.span))
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

        Ok((keywords::Invalid.ident(), item_kind, Some(attrs)))
    }

    fn parse_late_bound_lifetime_defs(&mut self) -> PResult<'a, Vec<GenericParam>> {
        if self.eat_keyword(keywords::For) {
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

    /// Parse struct Foo { ... }
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

        let vdata = if self.token.is_keyword(keywords::Where) {
            generics.where_clause = self.parse_where_clause()?;
            if self.eat(&token::Semi) {
                // If we see a: `struct Foo<T> where T: Copy;` style decl.
                VariantData::Unit(ast::DUMMY_NODE_ID)
            } else {
                // If we see: `struct Foo<T> where T: Copy { ... }`
                VariantData::Struct(self.parse_record_struct_body()?, ast::DUMMY_NODE_ID)
            }
        // No `where` so: `struct Foo<T>;`
        } else if self.eat(&token::Semi) {
            VariantData::Unit(ast::DUMMY_NODE_ID)
        // Record-style struct definition
        } else if self.token == token::OpenDelim(token::Brace) {
            VariantData::Struct(self.parse_record_struct_body()?, ast::DUMMY_NODE_ID)
        // Tuple-style struct definition with optional where-clause.
        } else if self.token == token::OpenDelim(token::Paren) {
            let body = VariantData::Tuple(self.parse_tuple_struct_body()?, ast::DUMMY_NODE_ID);
            generics.where_clause = self.parse_where_clause()?;
            self.expect(&token::Semi)?;
            body
        } else {
            let token_str = self.this_token_to_string();
            let mut err = self.fatal(&format!(
                "expected `where`, `{{`, `(`, or `;` after struct name, found `{}`",
                token_str
            ));
            err.span_label(self.span, "expected `where`, `{`, `(`, or `;` after struct name");
            return Err(err);
        };

        Ok((class_name, ItemKind::Struct(vdata, generics), None))
    }

    /// Parse union Foo { ... }
    fn parse_item_union(&mut self) -> PResult<'a, ItemInfo> {
        let class_name = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        let vdata = if self.token.is_keyword(keywords::Where) {
            generics.where_clause = self.parse_where_clause()?;
            VariantData::Struct(self.parse_record_struct_body()?, ast::DUMMY_NODE_ID)
        } else if self.token == token::OpenDelim(token::Brace) {
            VariantData::Struct(self.parse_record_struct_body()?, ast::DUMMY_NODE_ID)
        } else {
            let token_str = self.this_token_to_string();
            let mut err = self.fatal(&format!(
                "expected `where` or `{{` after union name, found `{}`", token_str));
            err.span_label(self.span, "expected `where` or `{` after union name");
            return Err(err);
        };

        Ok((class_name, ItemKind::Union(vdata, generics), None))
    }

    fn consume_block(&mut self, delim: token::DelimToken) {
        let mut brace_depth = 0;
        if !self.eat(&token::OpenDelim(delim)) {
            return;
        }
        loop {
            if self.eat(&token::OpenDelim(delim)) {
                brace_depth += 1;
            } else if self.eat(&token::CloseDelim(delim)) {
                if brace_depth == 0 {
                    return;
                } else {
                    brace_depth -= 1;
                    continue;
                }
            } else if self.eat(&token::Eof) || self.eat(&token::CloseDelim(token::NoDelim)) {
                return;
            } else {
                self.bump();
            }
        }
    }

    pub fn parse_record_struct_body(&mut self) -> PResult<'a, Vec<StructField>> {
        let mut fields = Vec::new();
        if self.eat(&token::OpenDelim(token::Brace)) {
            while self.token != token::CloseDelim(token::Brace) {
                let field = self.parse_struct_decl_field().map_err(|e| {
                    self.recover_stmt();
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
            let token_str = self.this_token_to_string();
            let mut err = self.fatal(&format!(
                    "expected `where`, or `{{` after struct name, found `{}`", token_str));
            err.span_label(self.span, "expected `where`, or `{` after struct name");
            return Err(err);
        }

        Ok(fields)
    }

    pub fn parse_tuple_struct_body(&mut self) -> PResult<'a, Vec<StructField>> {
        // This is the case where we find `struct Foo<T>(T) where T: Copy;`
        // Unit like structs are handled in parse_item_struct function
        let fields = self.parse_unspanned_seq(
            &token::OpenDelim(token::Paren),
            &token::CloseDelim(token::Paren),
            SeqSep::trailing_allowed(token::Comma),
            |p| {
                let attrs = p.parse_outer_attributes()?;
                let lo = p.span;
                let vis = p.parse_visibility(true)?;
                let ty = p.parse_ty()?;
                Ok(StructField {
                    span: lo.to(p.span),
                    vis,
                    ident: None,
                    id: ast::DUMMY_NODE_ID,
                    ty,
                    attrs,
                })
            })?;

        Ok(fields)
    }

    /// Parse a structure field declaration
    pub fn parse_single_struct_field(&mut self,
                                     lo: Span,
                                     vis: Visibility,
                                     attrs: Vec<Attribute> )
                                     -> PResult<'a, StructField> {
        let mut seen_comma: bool = false;
        let a_var = self.parse_name_and_ty(lo, vis, attrs)?;
        if self.token == token::Comma {
            seen_comma = true;
        }
        match self.token {
            token::Comma => {
                self.bump();
            }
            token::CloseDelim(token::Brace) => {}
            token::DocComment(_) => {
                let previous_span = self.prev_span;
                let mut err = self.span_fatal_err(self.span, Error::UselessDocComment);
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
                        let sp = self.sess.codemap().next_point(previous_span);
                        err.span_suggestion(sp, "missing comma here", ",".into());
                    }
                    return Err(err);
                }
            }
            _ => return Err(self.span_fatal_help(self.span,
                    &format!("expected `,`, or `}}`, found `{}`", self.this_token_to_string()),
                    "struct fields should be separated by commas")),
        }
        Ok(a_var)
    }

    /// Parse an element of a struct definition
    fn parse_struct_decl_field(&mut self) -> PResult<'a, StructField> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.span;
        let vis = self.parse_visibility(false)?;
        self.parse_single_struct_field(lo, vis, attrs)
    }

    /// Parse `pub`, `pub(crate)` and `pub(in path)` plus shortcuts `pub(self)` for `pub(in self)`
    /// and `pub(super)` for `pub(in super)`.  If the following element can't be a tuple (i.e. it's
    /// a function definition, it's not a tuple struct field) and the contents within the parens
    /// isn't valid, emit a proper diagnostic.
    pub fn parse_visibility(&mut self, can_take_tuple: bool) -> PResult<'a, Visibility> {
        maybe_whole!(self, NtVis, |x| x);

        self.expected_tokens.push(TokenType::Keyword(keywords::Crate));
        if self.is_crate_vis() {
            self.bump(); // `crate`
            return Ok(respan(self.prev_span, VisibilityKind::Crate(CrateSugar::JustCrate)));
        }

        if !self.eat_keyword(keywords::Pub) {
            return Ok(respan(self.prev_span, VisibilityKind::Inherited))
        }
        let lo = self.prev_span;

        if self.check(&token::OpenDelim(token::Paren)) {
            // We don't `self.bump()` the `(` yet because this might be a struct definition where
            // `()` or a tuple might be allowed. For example, `struct Struct(pub (), pub (usize));`.
            // Because of this, we only `bump` the `(` if we're assured it is appropriate to do so
            // by the following tokens.
            if self.look_ahead(1, |t| t.is_keyword(keywords::Crate)) {
                // `pub(crate)`
                self.bump(); // `(`
                self.bump(); // `crate`
                self.expect(&token::CloseDelim(token::Paren))?; // `)`
                let vis = respan(
                    lo.to(self.prev_span),
                    VisibilityKind::Crate(CrateSugar::PubCrate),
                );
                return Ok(vis)
            } else if self.look_ahead(1, |t| t.is_keyword(keywords::In)) {
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
                      self.look_ahead(1, |t| t.is_keyword(keywords::Super) ||
                                             t.is_keyword(keywords::SelfValue))
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
                let path_span = self.prev_span;
                let help_msg = format!("make this visible only to module `{}` with `in`", path);
                self.expect(&token::CloseDelim(token::Paren))?;  // `)`
                let mut err = self.span_fatal_help(path_span, msg, suggestion);
                err.span_suggestion(path_span, &help_msg, format!("in {}", path));
                err.emit();  // emit diagnostic, but continue with public visibility
            }
        }

        Ok(respan(lo, VisibilityKind::Public))
    }

    /// Parse defaultness: `default` or nothing.
    fn parse_defaultness(&mut self) -> Defaultness {
        // `pub` is included for better error messages
        if self.check_keyword(keywords::Default) &&
           self.look_ahead(1, |t| t.is_keyword(keywords::Impl) ||
                                  t.is_keyword(keywords::Const) ||
                                  t.is_keyword(keywords::Fn) ||
                                  t.is_keyword(keywords::Unsafe) ||
                                  t.is_keyword(keywords::Extern) ||
                                  t.is_keyword(keywords::Type) ||
                                  t.is_keyword(keywords::Pub)) {
            self.bump(); // `default`
            Defaultness::Default
        } else {
            Defaultness::Final
        }
    }

    /// Given a termination token, parse all of the items in a module
    fn parse_mod_items(&mut self, term: &token::Token, inner_lo: Span) -> PResult<'a, Mod> {
        let mut items = vec![];
        while let Some(item) = self.parse_item()? {
            items.push(item);
        }

        if !self.eat(term) {
            let token_str = self.this_token_to_string();
            let mut err = self.fatal(&format!("expected item, found `{}`", token_str));
            if token_str == ";" {
                let msg = "consider removing this semicolon";
                err.span_suggestion_short(self.span, msg, "".to_string());
            } else {
                err.span_label(self.span, "expected item");
            }
            return Err(err);
        }

        let hi = if self.span == syntax_pos::DUMMY_SP {
            inner_lo
        } else {
            self.prev_span
        };

        Ok(ast::Mod {
            inner: inner_lo.to(hi),
            items,
        })
    }

    fn parse_item_const(&mut self, m: Option<Mutability>) -> PResult<'a, ItemInfo> {
        let id = self.parse_ident()?;
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
            let mut strip_unconfigured = ::config::StripUnconfigured {
                sess: self.sess,
                should_test: false, // irrelevant
                features: None, // don't perform gated feature checking
            };
            let outer_attrs = strip_unconfigured.process_cfg_attrs(outer_attrs.to_owned());
            (!self.cfg_mods || strip_unconfigured.in_cfg(&outer_attrs), outer_attrs)
        };

        let id_span = self.span;
        let id = self.parse_ident()?;
        if self.check(&token::Semi) {
            self.bump();
            if in_cfg && self.recurse_into_file_modules {
                // This mod is in an external file. Let's go get it!
                let ModulePathSuccess { path, directory_ownership, warn } =
                    self.submod_path(id, &outer_attrs, id_span)?;
                let (module, mut attrs) =
                    self.eval_src_mod(path, directory_ownership, id.to_string(), id_span)?;
                if warn {
                    let attr = Attribute {
                        id: attr::mk_attr_id(),
                        style: ast::AttrStyle::Outer,
                        path: ast::Path::from_ident(Ident::from_str("warn_directory_ownership")),
                        tokens: TokenStream::empty(),
                        is_sugared_doc: false,
                        span: syntax_pos::DUMMY_SP,
                    };
                    attr::mark_known(&attr);
                    attrs.push(attr);
                }
                Ok((id, module, Some(attrs)))
            } else {
                let placeholder = ast::Mod { inner: syntax_pos::DUMMY_SP, items: Vec::new() };
                Ok((id, ItemKind::Mod(placeholder), None))
            }
        } else {
            let old_directory = self.directory.clone();
            self.push_directory(id, &outer_attrs);

            self.expect(&token::OpenDelim(token::Brace))?;
            let mod_inner_lo = self.span;
            let attrs = self.parse_inner_attributes()?;
            let module = self.parse_mod_items(&token::CloseDelim(token::Brace), mod_inner_lo)?;

            self.directory = old_directory;
            Ok((id, ItemKind::Mod(module), Some(attrs)))
        }
    }

    fn push_directory(&mut self, id: Ident, attrs: &[Attribute]) {
        if let Some(path) = attr::first_attr_value_str_by_name(attrs, "path") {
            self.directory.path.push(&path.as_str());
            self.directory.ownership = DirectoryOwnership::Owned { relative: None };
        } else {
            self.directory.path.push(&id.name.as_str());
        }
    }

    pub fn submod_path_from_attr(attrs: &[Attribute], dir_path: &Path) -> Option<PathBuf> {
        attr::first_attr_value_str_by_name(attrs, "path").map(|d| dir_path.join(&d.as_str()))
    }

    /// Returns either a path to a module, or .
    pub fn default_submod_path(
        id: ast::Ident,
        relative: Option<ast::Ident>,
        dir_path: &Path,
        codemap: &CodeMap) -> ModulePath
    {
        // If we're in a foo.rs file instead of a mod.rs file,
        // we need to look for submodules in
        // `./foo/<id>.rs` and `./foo/<id>/mod.rs` rather than
        // `./<id>.rs` and `./<id>/mod.rs`.
        let relative_prefix_string;
        let relative_prefix = if let Some(ident) = relative {
            relative_prefix_string = format!("{}{}", ident.name.as_str(), path::MAIN_SEPARATOR);
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
        let default_exists = codemap.file_exists(&default_path);
        let secondary_exists = codemap.file_exists(&secondary_path);

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
                dir_path: format!("{}", dir_path.display()),
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
            DirectoryOwnership::Owned { relative } => {
                // Push the usage onto the list of non-mod.rs mod uses.
                // This is used later for feature-gate error reporting.
                if let Some(cur_file_ident) = relative {
                    self.sess
                        .non_modrs_mods.borrow_mut()
                        .push((cur_file_ident, id_sp));
                }
                relative
            },
            DirectoryOwnership::UnownedViaBlock |
            DirectoryOwnership::UnownedViaMod(_) => None,
        };
        let paths = Parser::default_submod_path(
                        id, relative, &self.directory.path, self.sess.codemap());

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
                if id_sp != syntax_pos::DUMMY_SP {
                    let src_path = self.sess.codemap().span_to_filename(id_sp);
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

    /// Read a module from a source file.
    fn eval_src_mod(&mut self,
                    path: PathBuf,
                    directory_ownership: DirectoryOwnership,
                    name: String,
                    id_sp: Span)
                    -> PResult<'a, (ast::ItemKind, Vec<Attribute> )> {
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
        let mod_inner_lo = p0.span;
        let mod_attrs = p0.parse_inner_attributes()?;
        let m0 = p0.parse_mod_items(&token::Eof, mod_inner_lo)?;
        self.sess.included_mod_stack.borrow_mut().pop();
        Ok((ast::ItemKind::Mod(m0), mod_attrs))
    }

    /// Parse a function declaration from a foreign module
    fn parse_item_foreign_fn(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                             -> PResult<'a, ForeignItem> {
        self.expect_keyword(keywords::Fn)?;

        let (ident, mut generics) = self.parse_fn_header()?;
        let decl = self.parse_fn_decl(true)?;
        generics.where_clause = self.parse_where_clause()?;
        let hi = self.span;
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

    /// Parse a static item from a foreign module.
    /// Assumes that the `static` keyword is already parsed.
    fn parse_item_foreign_static(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                                 -> PResult<'a, ForeignItem> {
        let mutbl = self.eat_keyword(keywords::Mut);
        let ident = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        let hi = self.span;
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

    /// Parse a type from a foreign module
    fn parse_item_foreign_type(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                             -> PResult<'a, ForeignItem> {
        self.expect_keyword(keywords::Type)?;

        let ident = self.parse_ident()?;
        let hi = self.span;
        self.expect(&token::Semi)?;
        Ok(ast::ForeignItem {
            ident: ident,
            attrs: attrs,
            node: ForeignItemKind::Ty,
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis: vis
        })
    }

    /// Parse extern crate links
    ///
    /// # Examples
    ///
    /// extern crate foo;
    /// extern crate bar as foo;
    fn parse_item_extern_crate(&mut self,
                               lo: Span,
                               visibility: Visibility,
                               attrs: Vec<Attribute>)
                               -> PResult<'a, P<Item>> {
        let orig_name = self.parse_ident()?;
        let (item_name, orig_name) = if let Some(rename) = self.parse_rename()? {
            (rename, Some(orig_name.name))
        } else {
            (orig_name, None)
        };
        self.expect(&token::Semi)?;

        let span = lo.to(self.prev_span);
        Ok(self.mk_item(span, item_name, ItemKind::ExternCrate(orig_name), visibility, attrs))
    }

    /// Parse `extern` for foreign ABIs
    /// modules.
    ///
    /// `extern` is expected to have been
    /// consumed before calling this method
    ///
    /// # Examples:
    ///
    /// extern "C" {}
    /// extern {}
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
        while let Some(item) = self.parse_foreign_item()? {
            foreign_items.push(item);
        }
        self.expect(&token::CloseDelim(token::Brace))?;

        let prev_span = self.prev_span;
        let m = ast::ForeignMod {
            abi,
            items: foreign_items
        };
        let invalid = keywords::Invalid.ident();
        Ok(self.mk_item(lo.to(prev_span), invalid, ItemKind::ForeignMod(m), visibility, attrs))
    }

    /// Parse type Foo = Bar;
    fn parse_item_type(&mut self) -> PResult<'a, ItemInfo> {
        let ident = self.parse_ident()?;
        let mut tps = self.parse_generics()?;
        tps.where_clause = self.parse_where_clause()?;
        self.expect(&token::Eq)?;
        let ty = self.parse_ty()?;
        self.expect(&token::Semi)?;
        Ok((ident, ItemKind::Ty(ty, tps), None))
    }

    /// Parse the part of an "enum" decl following the '{'
    fn parse_enum_def(&mut self, _generics: &ast::Generics) -> PResult<'a, EnumDef> {
        let mut variants = Vec::new();
        let mut all_nullary = true;
        let mut any_disr = None;
        while self.token != token::CloseDelim(token::Brace) {
            let variant_attrs = self.parse_outer_attributes()?;
            let vlo = self.span;

            let struct_def;
            let mut disr_expr = None;
            let ident = self.parse_ident()?;
            if self.check(&token::OpenDelim(token::Brace)) {
                // Parse a struct variant.
                all_nullary = false;
                struct_def = VariantData::Struct(self.parse_record_struct_body()?,
                                                 ast::DUMMY_NODE_ID);
            } else if self.check(&token::OpenDelim(token::Paren)) {
                all_nullary = false;
                struct_def = VariantData::Tuple(self.parse_tuple_struct_body()?,
                                                ast::DUMMY_NODE_ID);
            } else if self.eat(&token::Eq) {
                disr_expr = Some(self.parse_expr()?);
                any_disr = disr_expr.as_ref().map(|expr| expr.span);
                struct_def = VariantData::Unit(ast::DUMMY_NODE_ID);
            } else {
                struct_def = VariantData::Unit(ast::DUMMY_NODE_ID);
            }

            let vr = ast::Variant_ {
                ident,
                attrs: variant_attrs,
                data: struct_def,
                disr_expr,
            };
            variants.push(respan(vlo.to(self.prev_span), vr));

            if !self.eat(&token::Comma) { break; }
        }
        self.expect(&token::CloseDelim(token::Brace))?;
        match any_disr {
            Some(disr_span) if !all_nullary =>
                self.span_err(disr_span,
                    "discriminator values can only be used with a field-less enum"),
            _ => ()
        }

        Ok(ast::EnumDef { variants: variants })
    }

    /// Parse an "enum" declaration
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
        match self.token {
            token::Literal(token::Str_(s), suf) | token::Literal(token::StrRaw(s, _), suf) => {
                let sp = self.span;
                self.expect_no_suffix(sp, "ABI spec", suf);
                self.bump();
                match abi::lookup(&s.as_str()) {
                    Some(abi) => Ok(Some(abi)),
                    None => {
                        let prev_span = self.prev_span;
                        self.span_err(
                            prev_span,
                            &format!("invalid ABI: expected one of [{}], \
                                     found `{}`",
                                    abi::all_names().join(", "),
                                    s));
                        Ok(None)
                    }
                }
            }

            _ => Ok(None),
        }
    }

    fn is_static_global(&mut self) -> bool {
        if self.check_keyword(keywords::Static) {
            // Check if this could be a closure
            !self.look_ahead(1, |token| {
                if token.is_keyword(keywords::Move) {
                    return true;
                }
                match *token {
                    token::BinOp(token::Or) | token::OrOr => true,
                    _ => false,
                }
            })
        } else {
            false
        }
    }

    /// Parse one of the items allowed by the flags.
    /// NB: this function no longer parses the items inside an
    /// extern crate.
    fn parse_item_(&mut self, attrs: Vec<Attribute>,
                   macros_allowed: bool, attributes_allowed: bool) -> PResult<'a, Option<P<Item>>> {
        maybe_whole!(self, NtItem, |item| {
            let mut item = item.into_inner();
            let mut attrs = attrs;
            mem::swap(&mut item.attrs, &mut attrs);
            item.attrs.extend(attrs);
            Some(P(item))
        });

        let lo = self.span;

        let visibility = self.parse_visibility(false)?;

        if self.eat_keyword(keywords::Use) {
            // USE ITEM
            let item_ = ItemKind::Use(P(self.parse_use_tree()?));
            self.expect(&token::Semi)?;

            let span = lo.to(self.prev_span);
            let item = self.mk_item(span, keywords::Invalid.ident(), item_, visibility, attrs);
            return Ok(Some(item));
        }

        if self.check_keyword(keywords::Extern) && self.is_extern_non_path() {
            self.bump(); // `extern`
            if self.eat_keyword(keywords::Crate) {
                return Ok(Some(self.parse_item_extern_crate(lo, visibility, attrs)?));
            }

            let opt_abi = self.parse_opt_abi()?;

            if self.eat_keyword(keywords::Fn) {
                // EXTERN FUNCTION ITEM
                let fn_span = self.prev_span;
                let abi = opt_abi.unwrap_or(Abi::C);
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(Unsafety::Normal,
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
            let m = if self.eat_keyword(keywords::Mut) {
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
        if self.eat_keyword(keywords::Const) {
            let const_span = self.prev_span;
            if self.check_keyword(keywords::Fn)
                || (self.check_keyword(keywords::Unsafe)
                    && self.look_ahead(1, |t| t.is_keyword(keywords::Fn))) {
                // CONST FUNCTION ITEM
                let unsafety = self.parse_unsafety();
                self.bump();
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(unsafety,
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
            if self.eat_keyword(keywords::Mut) {
                let prev_span = self.prev_span;
                self.diagnostic().struct_span_err(prev_span, "const globals cannot be mutable")
                                 .help("did you mean to declare a static?")
                                 .emit();
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
        if self.check_keyword(keywords::Unsafe) &&
            (self.look_ahead(1, |t| t.is_keyword(keywords::Trait)) ||
            self.look_ahead(1, |t| t.is_keyword(keywords::Auto)))
        {
            // UNSAFE TRAIT ITEM
            self.bump(); // `unsafe`
            let is_auto = if self.eat_keyword(keywords::Trait) {
                IsAuto::No
            } else {
                self.expect_keyword(keywords::Auto)?;
                self.expect_keyword(keywords::Trait)?;
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
        if self.check_keyword(keywords::Impl) ||
           self.check_keyword(keywords::Unsafe) &&
                self.look_ahead(1, |t| t.is_keyword(keywords::Impl)) ||
           self.check_keyword(keywords::Default) &&
                self.look_ahead(1, |t| t.is_keyword(keywords::Impl)) ||
           self.check_keyword(keywords::Default) &&
                self.look_ahead(1, |t| t.is_keyword(keywords::Unsafe)) {
            // IMPL ITEM
            let defaultness = self.parse_defaultness();
            let unsafety = self.parse_unsafety();
            self.expect_keyword(keywords::Impl)?;
            let (ident, item, extra_attrs) = self.parse_item_impl(unsafety, defaultness)?;
            let span = lo.to(self.prev_span);
            return Ok(Some(self.mk_item(span, ident, item, visibility,
                                        maybe_append(attrs, extra_attrs))));
        }
        if self.check_keyword(keywords::Fn) {
            // FUNCTION ITEM
            self.bump();
            let fn_span = self.prev_span;
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(Unsafety::Normal,
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
        if self.check_keyword(keywords::Unsafe)
            && self.look_ahead(1, |t| *t != token::OpenDelim(token::Brace)) {
            // UNSAFE FUNCTION ITEM
            self.bump(); // `unsafe`
            // `{` is also expected after `unsafe`, in case of error, include it in the diagnostic
            self.check(&token::OpenDelim(token::Brace));
            let abi = if self.eat_keyword(keywords::Extern) {
                self.parse_opt_abi()?.unwrap_or(Abi::C)
            } else {
                Abi::Rust
            };
            self.expect_keyword(keywords::Fn)?;
            let fn_span = self.prev_span;
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(Unsafety::Unsafe,
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
        if self.eat_keyword(keywords::Mod) {
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
        if self.eat_keyword(keywords::Type) {
            // TYPE ITEM
            let (ident, item_, extra_attrs) = self.parse_item_type()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(keywords::Enum) {
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
        if self.check_keyword(keywords::Trait)
            || (self.check_keyword(keywords::Auto)
                && self.look_ahead(1, |t| t.is_keyword(keywords::Trait)))
        {
            let is_auto = if self.eat_keyword(keywords::Trait) {
                IsAuto::No
            } else {
                self.expect_keyword(keywords::Auto)?;
                self.expect_keyword(keywords::Trait)?;
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
        if self.eat_keyword(keywords::Struct) {
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
        if visibility.node == VisibilityKind::Public &&
            self.check_ident() &&
            self.look_ahead(1, |t| *t != token::Not)
        {
            // Space between `pub` keyword and the identifier
            //
            //     pub   S {}
            //        ^^^ `sp` points here
            let sp = self.prev_span.between(self.span);
            let full_sp = self.prev_span.to(self.span);
            let ident_sp = self.span;
            if self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace)) {
                // possible public struct definition where `struct` was forgotten
                let ident = self.parse_ident().unwrap();
                let msg = format!("add `struct` here to parse `{}` as a public struct",
                                  ident);
                let mut err = self.diagnostic()
                    .struct_span_err(sp, "missing `struct` for struct definition");
                err.span_suggestion_short(sp, &msg, " struct ".into());
                return Err(err);
            } else if self.look_ahead(1, |t| *t == token::OpenDelim(token::Paren)) {
                let ident = self.parse_ident().unwrap();
                self.consume_block(token::Paren);
                let (kw, kw_name, ambiguous) = if self.check(&token::RArrow) ||
                    self.check(&token::OpenDelim(token::Brace))
                {
                    ("fn", "method", false)
                } else if self.check(&token::Colon) {
                    let kw = "struct";
                    (kw, kw, false)
                } else {
                    ("fn` or `struct", "method or struct", true)
                };

                let msg = format!("missing `{}` for {} definition", kw, kw_name);
                let mut err = self.diagnostic().struct_span_err(sp, &msg);
                if !ambiguous {
                    let suggestion = format!("add `{}` here to parse `{}` as a public {}",
                                             kw,
                                             ident,
                                             kw_name);
                    err.span_suggestion_short(sp, &suggestion, format!(" {} ", kw));
                } else {
                    if let Ok(snippet) = self.sess.codemap().span_to_snippet(ident_sp) {
                        err.span_suggestion(
                            full_sp,
                            "if you meant to call a macro, write instead",
                            format!("{}!", snippet));
                    } else {
                        err.help("if you meant to call a macro, remove the `pub` \
                                  and add a trailing `!` after the identifier");
                    }
                }
                return Err(err);
            }
        }
        self.parse_macro_use_or_failure(attrs, macros_allowed, attributes_allowed, lo, visibility)
    }

    /// Parse a foreign item.
    pub fn parse_foreign_item(&mut self) -> PResult<'a, Option<ForeignItem>> {
        maybe_whole!(self, NtForeignItem, |ni| Some(ni));

        let attrs = self.parse_outer_attributes()?;
        let lo = self.span;
        let visibility = self.parse_visibility(false)?;

        // FOREIGN STATIC ITEM
        // Treat `const` as `static` for error recovery, but don't add it to expected tokens.
        if self.check_keyword(keywords::Static) || self.token.is_keyword(keywords::Const) {
            if self.token.is_keyword(keywords::Const) {
                self.diagnostic()
                    .struct_span_err(self.span, "extern items cannot be `const`")
                    .span_suggestion(self.span, "instead try using", "static".to_owned())
                    .emit();
            }
            self.bump(); // `static` or `const`
            return Ok(Some(self.parse_item_foreign_static(visibility, lo, attrs)?));
        }
        // FOREIGN FUNCTION ITEM
        if self.check_keyword(keywords::Fn) {
            return Ok(Some(self.parse_item_foreign_fn(visibility, lo, attrs)?));
        }
        // FOREIGN TYPE ITEM
        if self.check_keyword(keywords::Type) {
            return Ok(Some(self.parse_item_foreign_type(visibility, lo, attrs)?));
        }

        match self.parse_assoc_macro_invoc("extern", Some(&visibility), &mut false)? {
            Some(mac) => {
                Ok(Some(
                    ForeignItem {
                        ident: keywords::Invalid.ident(),
                        span: lo.to(self.prev_span),
                        id: ast::DUMMY_NODE_ID,
                        attrs,
                        vis: visibility,
                        node: ForeignItemKind::Macro(mac),
                    }
                ))
            }
            None => {
                if !attrs.is_empty() {
                    self.expected_item_err(&attrs);
                }

                Ok(None)
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
        if macros_allowed && self.token.is_path_start() {
            // MACRO INVOCATION ITEM

            let prev_span = self.prev_span;
            self.complain_if_pub_macro(&visibility.node, prev_span);

            let mac_lo = self.span;

            // item macro.
            let pth = self.parse_path(PathStyle::Mod)?;
            self.expect(&token::Not)?;

            // a 'special' identifier (like what `macro_rules!` uses)
            // is optional. We should eventually unify invoc syntax
            // and remove this.
            let id = if self.token.is_ident() {
                self.parse_ident()?
            } else {
                keywords::Invalid.ident() // no special identifier
            };
            // eat a matched-delimiter token tree:
            let (delim, tts) = self.expect_delimited_token_tree()?;
            if delim != token::Brace {
                if !self.eat(&token::Semi) {
                    self.span_err(self.prev_span,
                                  "macros that expand to items must either \
                                   be surrounded with braces or followed by \
                                   a semicolon");
                }
            }

            let hi = self.prev_span;
            let mac = respan(mac_lo.to(hi), Mac_ { path: pth, tts: tts });
            let item = self.mk_item(lo.to(hi), id, ItemKind::Mac(mac), visibility, attrs);
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
            self.expected_item_err(&attrs);
        }
        Ok(None)
    }

    /// Parse a macro invocation inside a `trait`, `impl` or `extern` block
    fn parse_assoc_macro_invoc(&mut self, item_kind: &str, vis: Option<&Visibility>,
                               at_end: &mut bool) -> PResult<'a, Option<Mac>>
    {
        if self.token.is_path_start() && !self.is_extern_non_path() {
            let prev_span = self.prev_span;
            let lo = self.span;
            let pth = self.parse_path(PathStyle::Mod)?;

            if pth.segments.len() == 1 {
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
            if delim != token::Brace {
                self.expect(&token::Semi)?
            }

            Ok(Some(respan(lo.to(self.prev_span), Mac_ { path: pth, tts: tts })))
        } else {
            Ok(None)
        }
    }

    fn collect_tokens<F, R>(&mut self, f: F) -> PResult<'a, (R, TokenStream)>
        where F: FnOnce(&mut Self) -> PResult<'a, R>
    {
        // Record all tokens we parse when parsing this item.
        let mut tokens = Vec::new();
        match self.token_cursor.frame.last_token {
            LastToken::Collecting(_) => {
                panic!("cannot collect tokens recursively yet")
            }
            LastToken::Was(ref mut last) => tokens.extend(last.take()),
        }
        self.token_cursor.frame.last_token = LastToken::Collecting(tokens);
        let prev = self.token_cursor.stack.len();
        let ret = f(self);
        let last_token = if self.token_cursor.stack.len() == prev {
            &mut self.token_cursor.frame.last_token
        } else {
            &mut self.token_cursor.stack[prev].last_token
        };
        let mut tokens = match *last_token {
            LastToken::Collecting(ref mut v) => mem::replace(v, Vec::new()),
            LastToken::Was(_) => panic!("our vector went away?"),
        };

        // If we're not at EOF our current token wasn't actually consumed by
        // `f`, but it'll still be in our list that we pulled out. In that case
        // put it back.
        if self.token == token::Eof {
            *last_token = LastToken::Was(None);
        } else {
            *last_token = LastToken::Was(tokens.pop());
        }

        Ok((ret?, tokens.into_iter().collect()))
    }

    pub fn parse_item(&mut self) -> PResult<'a, Option<P<Item>>> {
        let attrs = self.parse_outer_attributes()?;

        let (ret, tokens) = self.collect_tokens(|this| {
            this.parse_item_(attrs, true, false)
        })?;

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

    /// `::{` or `::*`
    fn is_import_coupler(&mut self) -> bool {
        self.check(&token::ModSep) &&
            self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace) ||
                                   *t == token::BinOp(token::Star))
    }

    /// Parse UseTree
    ///
    /// USE_TREE = [`::`] `*` |
    ///            [`::`] `{` USE_TREE_LIST `}` |
    ///            PATH `::` `*` |
    ///            PATH `::` `{` USE_TREE_LIST `}` |
    ///            PATH [`as` IDENT]
    fn parse_use_tree(&mut self) -> PResult<'a, UseTree> {
        let lo = self.span;

        let mut prefix = ast::Path { segments: Vec::new(), span: lo.shrink_to_lo() };
        let kind = if self.check(&token::OpenDelim(token::Brace)) ||
                      self.check(&token::BinOp(token::Star)) ||
                      self.is_import_coupler() {
            // `use *;` or `use ::*;` or `use {...};` or `use ::{...};`
            if self.eat(&token::ModSep) {
                prefix.segments.push(PathSegment::crate_root(lo.shrink_to_lo()));
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
                UseTreeKind::Simple(self.parse_rename()?)
            }
        };

        Ok(UseTree { prefix, kind, span: lo.to(self.prev_span) })
    }

    /// Parse UseTreeKind::Nested(list)
    ///
    /// USE_TREE_LIST = Ø | (USE_TREE `,`)* USE_TREE [`,`]
    fn parse_use_tree_list(&mut self) -> PResult<'a, Vec<(UseTree, ast::NodeId)>> {
        self.parse_unspanned_seq(&token::OpenDelim(token::Brace),
                                 &token::CloseDelim(token::Brace),
                                 SeqSep::trailing_allowed(token::Comma), |this| {
            Ok((this.parse_use_tree()?, ast::DUMMY_NODE_ID))
        })
    }

    fn parse_rename(&mut self) -> PResult<'a, Option<Ident>> {
        if self.eat_keyword(keywords::As) {
            match self.token {
                token::Ident(ident, false) if ident.name == keywords::Underscore.name() => {
                    self.bump(); // `_`
                    Ok(Some(Ident::new(ident.name.gensymed(), ident.span)))
                }
                _ => self.parse_ident().map(Some),
            }
        } else {
            Ok(None)
        }
    }

    /// Parses a source module as a crate. This is the main
    /// entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<'a, Crate> {
        let lo = self.span;
        Ok(ast::Crate {
            attrs: self.parse_inner_attributes()?,
            module: self.parse_mod_items(&token::Eof, lo)?,
            span: lo.to(self.span),
        })
    }

    pub fn parse_optional_str(&mut self) -> Option<(Symbol, ast::StrStyle, Option<ast::Name>)> {
        let ret = match self.token {
            token::Literal(token::Str_(s), suf) => (s, ast::StrStyle::Cooked, suf),
            token::Literal(token::StrRaw(s, n), suf) => (s, ast::StrStyle::Raw(n), suf),
            _ => return None
        };
        self.bump();
        Some(ret)
    }

    pub fn parse_str(&mut self) -> PResult<'a, (Symbol, StrStyle)> {
        match self.parse_optional_str() {
            Some((s, style, suf)) => {
                let sp = self.prev_span;
                self.expect_no_suffix(sp, "string literal", suf);
                Ok((s, style))
            }
            _ => {
                let msg = "expected string literal";
                let mut err = self.fatal(msg);
                err.span_label(self.span, msg);
                Err(err)
            }
        }
    }
}
