// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]

use abi;
use ast::{BareFnTy, ClosureTy};
use ast::{StaticRegionTyParamBound, OtherRegionTyParamBound, TraitTyParamBound};
use ast::{ProvidedMethod, Public, FnStyle};
use ast::{Mod, BiAdd, Arg, Arm, Attribute, BindByRef, BindByValue};
use ast::{BiBitAnd, BiBitOr, BiBitXor, Block};
use ast::{BlockCheckMode, UnBox};
use ast::{CaptureByRef, CaptureByValue, CaptureClause};
use ast::{Crate, CrateConfig, Decl, DeclItem};
use ast::{DeclLocal, DefaultBlock, UnDeref, BiDiv, EMPTY_CTXT, EnumDef, ExplicitSelf};
use ast::{Expr, Expr_, ExprAddrOf, ExprMatch, ExprAgain};
use ast::{ExprAssign, ExprAssignOp, ExprBinary, ExprBlock, ExprBox};
use ast::{ExprBreak, ExprCall, ExprCast};
use ast::{ExprField, ExprFnBlock, ExprIf, ExprIndex};
use ast::{ExprLit, ExprLoop, ExprMac};
use ast::{ExprMethodCall, ExprParen, ExprPath, ExprProc};
use ast::{ExprRepeat, ExprRet, ExprStruct, ExprTup, ExprUnary, ExprUnboxedFn};
use ast::{ExprVec, ExprWhile, ExprForLoop, Field, FnDecl};
use ast::{Once, Many};
use ast::{FnUnboxedClosureKind, FnMutUnboxedClosureKind};
use ast::{FnOnceUnboxedClosureKind};
use ast::{ForeignItem, ForeignItemStatic, ForeignItemFn, ForeignMod};
use ast::{Ident, NormalFn, Inherited, ImplItem, Item, Item_, ItemStatic};
use ast::{ItemEnum, ItemFn, ItemForeignMod, ItemImpl};
use ast::{ItemMac, ItemMod, ItemStruct, ItemTrait, ItemTy, Lit, Lit_};
use ast::{LitBool, LitChar, LitByte, LitBinary};
use ast::{LitNil, LitStr, LitInt, Local, LocalLet};
use ast::{MutImmutable, MutMutable, Mac_, MacInvocTT, Matcher, MatchNonterminal};
use ast::{MatchSeq, MatchTok, Method, MutTy, BiMul, Mutability};
use ast::{MethodImplItem};
use ast::{NamedField, UnNeg, NoReturn, UnNot, P, Pat, PatEnum};
use ast::{PatIdent, PatLit, PatRange, PatRegion, PatStruct};
use ast::{PatTup, PatBox, PatWild, PatWildMulti, PatWildSingle};
use ast::{BiRem, RequiredMethod};
use ast::{RetStyle, Return, BiShl, BiShr, Stmt, StmtDecl};
use ast::{StmtExpr, StmtSemi, StmtMac, StructDef, StructField};
use ast::{StructVariantKind, BiSub};
use ast::StrStyle;
use ast::{SelfExplicit, SelfRegion, SelfStatic, SelfValue};
use ast::{TokenTree, TraitItem, TraitRef, TTDelim, TTSeq, TTTok};
use ast::{TTNonterminal, TupleVariantKind, Ty, Ty_, TyBot, TyBox};
use ast::{TypeField, TyFixedLengthVec, TyClosure, TyProc, TyBareFn};
use ast::{TyTypeof, TyInfer, TypeMethod};
use ast::{TyNil, TyParam, TyParamBound, TyParen, TyPath, TyPtr, TyRptr};
use ast::{TyTup, TyU32, TyUnboxedFn, TyUniq, TyVec, UnUniq};
use ast::{UnboxedClosureKind, UnboxedFnTy, UnboxedFnTyParamBound};
use ast::{UnnamedField, UnsafeBlock};
use ast::{UnsafeFn, ViewItem, ViewItem_, ViewItemExternCrate, ViewItemUse};
use ast::{ViewPath, ViewPathGlob, ViewPathList, ViewPathSimple};
use ast::{Visibility, WhereClause, WherePredicate};
use ast;
use ast_util::{as_prec, ident_to_path, operator_prec};
use ast_util;
use attr;
use codemap::{Span, BytePos, Spanned, spanned, mk_sp};
use codemap;
use parse;
use parse::attr::ParserAttr;
use parse::classify;
use parse::common::{SeqSep, seq_sep_none};
use parse::common::{seq_sep_trailing_disallowed, seq_sep_trailing_allowed};
use parse::lexer::Reader;
use parse::lexer::TokenAndSpan;
use parse::obsolete::*;
use parse::token::{INTERPOLATED, InternedString, can_begin_expr};
use parse::token::{is_ident, is_ident_or_path, is_plain_ident};
use parse::token::{keywords, special_idents, token_to_binop};
use parse::token;
use parse::{new_sub_parser_from_file, ParseSess};
use owned_slice::OwnedSlice;

use std::collections::HashSet;
use std::mem::replace;
use std::rc::Rc;
use std::gc::{Gc, GC};
use std::iter;

#[allow(non_camel_case_types)]
#[deriving(PartialEq)]
pub enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_BAR_OP,
    RESTRICT_NO_BAR_OR_DOUBLEBAR_OP,
    RESTRICT_NO_STRUCT_LITERAL,
}

type ItemInfo = (Ident, Item_, Option<Vec<Attribute> >);

/// How to parse a path. There are four different kinds of paths, all of which
/// are parsed somewhat differently.
#[deriving(PartialEq)]
pub enum PathParsingMode {
    /// A path with no type parameters; e.g. `foo::bar::Baz`
    NoTypesAllowed,
    /// A path with a lifetime and type parameters, with no double colons
    /// before the type parameters; e.g. `foo::bar<'a>::Baz<T>`
    LifetimeAndTypesWithoutColons,
    /// A path with a lifetime and type parameters with double colons before
    /// the type parameters; e.g. `foo::bar::<'a>::Baz::<T>`
    LifetimeAndTypesWithColons,
    /// A path with a lifetime and type parameters with bounds before the last
    /// set of type parameters only; e.g. `foo::bar<'a>::Baz+X+Y<T>` This
    /// form does not use extra double colons.
    LifetimeAndTypesAndBounds,
}

/// A path paired with optional type bounds.
pub struct PathAndBounds {
    pub path: ast::Path,
    pub bounds: Option<OwnedSlice<TyParamBound>>,
}

enum ItemOrViewItem {
    /// Indicates a failure to parse any kind of item. The attributes are
    /// returned.
    IoviNone(Vec<Attribute>),
    IoviItem(Gc<Item>),
    IoviForeignItem(Gc<ForeignItem>),
    IoviViewItem(ViewItem)
}


/// Possibly accept an `INTERPOLATED` expression (a pre-parsed expression
/// dropped into the token stream, which happens while parsing the
/// result of macro expansion)
/// Placement of these is not as complex as I feared it would be.
/// The important thing is to make sure that lookahead doesn't balk
/// at INTERPOLATED tokens
macro_rules! maybe_whole_expr (
    ($p:expr) => (
        {
            let found = match $p.token {
                INTERPOLATED(token::NtExpr(e)) => {
                    Some(e)
                }
                INTERPOLATED(token::NtPath(_)) => {
                    // FIXME: The following avoids an issue with lexical borrowck scopes,
                    // but the clone is unfortunate.
                    let pt = match $p.token {
                        INTERPOLATED(token::NtPath(ref pt)) => (**pt).clone(),
                        _ => unreachable!()
                    };
                    let span = $p.span;
                    Some($p.mk_expr(span.lo, span.hi, ExprPath(pt)))
                }
                INTERPOLATED(token::NtBlock(b)) => {
                    let span = $p.span;
                    Some($p.mk_expr(span.lo, span.hi, ExprBlock(b)))
                }
                _ => None
            };
            match found {
                Some(e) => {
                    $p.bump();
                    return e;
                }
                None => ()
            }
        }
    )
)

/// As maybe_whole_expr, but for things other than expressions
macro_rules! maybe_whole (
    ($p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match found {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return x.clone()
                }
                _ => {}
            }
        }
    );
    (no_clone $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match found {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return x
                }
                _ => {}
            }
        }
    );
    (deref $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match found {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return (*x).clone()
                }
                _ => {}
            }
        }
    );
    (Some $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match found {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return Some(x.clone()),
                }
                _ => {}
            }
        }
    );
    (iovi $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match found {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return IoviItem(x.clone())
                }
                _ => {}
            }
        }
    );
    (pair_empty $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match found {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return (Vec::new(), x)
                }
                _ => {}
            }
        }
    )
)


fn maybe_append(lhs: Vec<Attribute> , rhs: Option<Vec<Attribute> >)
             -> Vec<Attribute> {
    match rhs {
        None => lhs,
        Some(ref attrs) => lhs.append(attrs.as_slice())
    }
}


struct ParsedItemsAndViewItems {
    attrs_remaining: Vec<Attribute>,
    view_items: Vec<ViewItem>,
    items: Vec<Gc<Item>>,
    foreign_items: Vec<Gc<ForeignItem>>
}

/* ident is handled by common.rs */

pub struct Parser<'a> {
    pub sess: &'a ParseSess,
    /// the current token:
    pub token: token::Token,
    /// the span of the current token:
    pub span: Span,
    /// the span of the prior token:
    pub last_span: Span,
    pub cfg: CrateConfig,
    /// the previous token or None (only stashed sometimes).
    pub last_token: Option<Box<token::Token>>,
    pub buffer: [TokenAndSpan, ..4],
    pub buffer_start: int,
    pub buffer_end: int,
    pub tokens_consumed: uint,
    pub restriction: restriction,
    pub quote_depth: uint, // not (yet) related to the quasiquoter
    pub reader: Box<Reader>,
    pub interner: Rc<token::IdentInterner>,
    /// The set of seen errors about obsolete syntax. Used to suppress
    /// extra detail when the same error is seen twice
    pub obsolete_set: HashSet<ObsoleteSyntax>,
    /// Used to determine the path to externally loaded source files
    pub mod_path_stack: Vec<InternedString>,
    /// Stack of spans of open delimiters. Used for error message.
    pub open_braces: Vec<Span>,
    /// Flag if this parser "owns" the directory that it is currently parsing
    /// in. This will affect how nested files are looked up.
    pub owns_directory: bool,
    /// Name of the root module this parser originated from. If `None`, then the
    /// name is not known. This does not change while the parser is descending
    /// into modules, and sub-parsers have new values for this name.
    pub root_module_name: Option<String>,
}

fn is_plain_ident_or_underscore(t: &token::Token) -> bool {
    is_plain_ident(t) || *t == token::UNDERSCORE
}

/// Get a token the parser cares about
fn real_token(rdr: &mut Reader) -> TokenAndSpan {
    let mut t = rdr.next_token();
    loop {
        match t.tok {
            token::WS | token::COMMENT | token::SHEBANG(_) => {
                t = rdr.next_token();
            },
            _ => break
        }
    }
    t
}

impl<'a> Parser<'a> {
    pub fn new(sess: &'a ParseSess, cfg: ast::CrateConfig,
               mut rdr: Box<Reader>) -> Parser<'a> {
        let tok0 = real_token(rdr);
        let span = tok0.sp;
        let placeholder = TokenAndSpan {
            tok: token::UNDERSCORE,
            sp: span,
        };

        Parser {
            reader: rdr,
            interner: token::get_ident_interner(),
            sess: sess,
            cfg: cfg,
            token: tok0.tok,
            span: span,
            last_span: span,
            last_token: None,
            buffer: [
                placeholder.clone(),
                placeholder.clone(),
                placeholder.clone(),
                placeholder.clone(),
            ],
            buffer_start: 0,
            buffer_end: 0,
            tokens_consumed: 0,
            restriction: UNRESTRICTED,
            quote_depth: 0,
            obsolete_set: HashSet::new(),
            mod_path_stack: Vec::new(),
            open_braces: Vec::new(),
            owns_directory: true,
            root_module_name: None,
        }
    }

    /// Convert a token to a string using self's reader
    pub fn token_to_string(token: &token::Token) -> String {
        token::to_string(token)
    }

    /// Convert the current token to a string using self's reader
    pub fn this_token_to_string(&mut self) -> String {
        Parser::token_to_string(&self.token)
    }

    pub fn unexpected_last(&mut self, t: &token::Token) -> ! {
        let token_str = Parser::token_to_string(t);
        let last_span = self.last_span;
        self.span_fatal(last_span, format!("unexpected token: `{}`",
                                                token_str).as_slice());
    }

    pub fn unexpected(&mut self) -> ! {
        let this_token = self.this_token_to_string();
        self.fatal(format!("unexpected token: `{}`", this_token).as_slice());
    }

    /// Expect and consume the token t. Signal an error if
    /// the next token is not t.
    pub fn expect(&mut self, t: &token::Token) {
        if self.token == *t {
            self.bump();
        } else {
            let token_str = Parser::token_to_string(t);
            let this_token_str = self.this_token_to_string();
            self.fatal(format!("expected `{}`, found `{}`",
                               token_str,
                               this_token_str).as_slice())
        }
    }

    /// Expect next token to be edible or inedible token.  If edible,
    /// then consume it; if inedible, then return without consuming
    /// anything.  Signal a fatal error if next token is unexpected.
    pub fn expect_one_of(&mut self,
                         edible: &[token::Token],
                         inedible: &[token::Token]) {
        fn tokens_to_string(tokens: &[token::Token]) -> String {
            let mut i = tokens.iter();
            // This might be a sign we need a connect method on Iterator.
            let b = i.next()
                     .map_or("".to_string(), |t| Parser::token_to_string(t));
            i.fold(b, |b,a| {
                let mut b = b;
                b.push_str("`, `");
                b.push_str(Parser::token_to_string(a).as_slice());
                b
            })
        }
        if edible.contains(&self.token) {
            self.bump();
        } else if inedible.contains(&self.token) {
            // leave it in the input
        } else {
            let expected = edible.iter().map(|x| (*x).clone()).collect::<Vec<_>>().append(inedible);
            let expect = tokens_to_string(expected.as_slice());
            let actual = self.this_token_to_string();
            self.fatal(
                (if expected.len() != 1 {
                    (format!("expected one of `{}`, found `{}`",
                             expect,
                             actual))
                } else {
                    (format!("expected `{}`, found `{}`",
                             expect,
                             actual))
                }).as_slice()
            )
        }
    }

    /// Check for erroneous `ident { }`; if matches, signal error and
    /// recover (without consuming any expected input token).  Returns
    /// true if and only if input was consumed for recovery.
    pub fn check_for_erroneous_unit_struct_expecting(&mut self, expected: &[token::Token]) -> bool {
        if self.token == token::LBRACE
            && expected.iter().all(|t| *t != token::LBRACE)
            && self.look_ahead(1, |t| *t == token::RBRACE) {
            // matched; signal non-fatal error and recover.
            let span = self.span;
            self.span_err(span,
                          "unit-like struct construction is written with no trailing `{ }`");
            self.eat(&token::LBRACE);
            self.eat(&token::RBRACE);
            true
        } else {
            false
        }
    }

    /// Commit to parsing a complete expression `e` expected to be
    /// followed by some token from the set edible + inedible.  Recover
    /// from anticipated input errors, discarding erroneous characters.
    pub fn commit_expr(&mut self, e: Gc<Expr>, edible: &[token::Token],
                       inedible: &[token::Token]) {
        debug!("commit_expr {:?}", e);
        match e.node {
            ExprPath(..) => {
                // might be unit-struct construction; check for recoverableinput error.
                let expected = edible.iter().map(|x| (*x).clone()).collect::<Vec<_>>()
                              .append(inedible);
                self.check_for_erroneous_unit_struct_expecting(
                    expected.as_slice());
            }
            _ => {}
        }
        self.expect_one_of(edible, inedible)
    }

    pub fn commit_expr_expecting(&mut self, e: Gc<Expr>, edible: token::Token) {
        self.commit_expr(e, &[edible], &[])
    }

    /// Commit to parsing a complete statement `s`, which expects to be
    /// followed by some token from the set edible + inedible.  Check
    /// for recoverable input errors, discarding erroneous characters.
    pub fn commit_stmt(&mut self, s: Gc<Stmt>, edible: &[token::Token],
                       inedible: &[token::Token]) {
        debug!("commit_stmt {:?}", s);
        let _s = s; // unused, but future checks might want to inspect `s`.
        if self.last_token
               .as_ref()
               .map_or(false, |t| is_ident_or_path(&**t)) {
            let expected = edible.iter().map(|x| (*x).clone()).collect::<Vec<_>>()
                           .append(inedible.as_slice());
            self.check_for_erroneous_unit_struct_expecting(
                expected.as_slice());
        }
        self.expect_one_of(edible, inedible)
    }

    pub fn commit_stmt_expecting(&mut self, s: Gc<Stmt>, edible: token::Token) {
        self.commit_stmt(s, &[edible], &[])
    }

    pub fn parse_ident(&mut self) -> ast::Ident {
        self.check_strict_keywords();
        self.check_reserved_keywords();
        match self.token {
            token::IDENT(i, _) => {
                self.bump();
                i
            }
            token::INTERPOLATED(token::NtIdent(..)) => {
                self.bug("ident interpolation not converted to real token");
            }
            _ => {
                let token_str = self.this_token_to_string();
                self.fatal((format!("expected ident, found `{}`",
                                    token_str)).as_slice())
            }
        }
    }

    pub fn parse_path_list_item(&mut self) -> ast::PathListItem {
        let lo = self.span.lo;
        let node = if self.eat_keyword(keywords::Mod) {
            ast::PathListMod { id: ast::DUMMY_NODE_ID }
        } else {
            let ident = self.parse_ident();
            ast::PathListIdent { name: ident, id: ast::DUMMY_NODE_ID }
        };
        let hi = self.last_span.hi;
        spanned(lo, hi, node)
    }

    /// Consume token 'tok' if it exists. Returns true if the given
    /// token was present, false otherwise.
    pub fn eat(&mut self, tok: &token::Token) -> bool {
        let is_present = self.token == *tok;
        if is_present { self.bump() }
        is_present
    }

    pub fn is_keyword(&mut self, kw: keywords::Keyword) -> bool {
        token::is_keyword(kw, &self.token)
    }

    /// If the next token is the given keyword, eat it and return
    /// true. Otherwise, return false.
    pub fn eat_keyword(&mut self, kw: keywords::Keyword) -> bool {
        match self.token {
            token::IDENT(sid, false) if kw.to_name() == sid.name => {
                self.bump();
                true
            }
            _ => false
        }
    }

    /// If the given word is not a keyword, signal an error.
    /// If the next token is not the given word, signal an error.
    /// Otherwise, eat it.
    pub fn expect_keyword(&mut self, kw: keywords::Keyword) {
        if !self.eat_keyword(kw) {
            let id_interned_str = token::get_name(kw.to_name());
            let token_str = self.this_token_to_string();
            self.fatal(format!("expected `{}`, found `{}`",
                               id_interned_str, token_str).as_slice())
        }
    }

    /// Signal an error if the given string is a strict keyword
    pub fn check_strict_keywords(&mut self) {
        if token::is_strict_keyword(&self.token) {
            let token_str = self.this_token_to_string();
            let span = self.span;
            self.span_err(span,
                          format!("found `{}` in ident position",
                                  token_str).as_slice());
        }
    }

    /// Signal an error if the current token is a reserved keyword
    pub fn check_reserved_keywords(&mut self) {
        if token::is_reserved_keyword(&self.token) {
            let token_str = self.this_token_to_string();
            self.fatal(format!("`{}` is a reserved keyword",
                               token_str).as_slice())
        }
    }

    /// Expect and consume an `&`. If `&&` is seen, replace it with a single
    /// `&` and continue. If an `&` is not seen, signal an error.
    fn expect_and(&mut self) {
        match self.token {
            token::BINOP(token::AND) => self.bump(),
            token::ANDAND => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                self.replace_token(token::BINOP(token::AND), lo, span.hi)
            }
            _ => {
                let token_str = self.this_token_to_string();
                let found_token =
                    Parser::token_to_string(&token::BINOP(token::AND));
                self.fatal(format!("expected `{}`, found `{}`",
                                   found_token,
                                   token_str).as_slice())
            }
        }
    }

    /// Expect and consume a `|`. If `||` is seen, replace it with a single
    /// `|` and continue. If a `|` is not seen, signal an error.
    fn expect_or(&mut self) {
        match self.token {
            token::BINOP(token::OR) => self.bump(),
            token::OROR => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                self.replace_token(token::BINOP(token::OR), lo, span.hi)
            }
            _ => {
                let found_token = self.this_token_to_string();
                let token_str =
                    Parser::token_to_string(&token::BINOP(token::OR));
                self.fatal(format!("expected `{}`, found `{}`",
                                   token_str,
                                   found_token).as_slice())
            }
        }
    }

    /// Attempt to consume a `<`. If `<<` is seen, replace it with a single
    /// `<` and continue. If a `<` is not seen, return false.
    ///
    /// This is meant to be used when parsing generics on a path to get the
    /// starting token. The `force` parameter is used to forcefully break up a
    /// `<<` token. If `force` is false, then `<<` is only broken when a lifetime
    /// shows up next. For example, consider the expression:
    ///
    ///      foo as bar << test
    ///
    /// The parser needs to know if `bar <<` is the start of a generic path or if
    /// it's a left-shift token. If `test` were a lifetime, then it's impossible
    /// for the token to be a left-shift, but if it's not a lifetime, then it's
    /// considered a left-shift.
    ///
    /// The reason for this is that the only current ambiguity with `<<` is when
    /// parsing closure types:
    ///
    ///      foo::<<'a> ||>();
    ///      impl Foo<<'a> ||>() { ... }
    fn eat_lt(&mut self, force: bool) -> bool {
        match self.token {
            token::LT => { self.bump(); true }
            token::BINOP(token::SHL) => {
                let next_lifetime = self.look_ahead(1, |t| match *t {
                    token::LIFETIME(..) => true,
                    _ => false,
                });
                if force || next_lifetime {
                    let span = self.span;
                    let lo = span.lo + BytePos(1);
                    self.replace_token(token::LT, lo, span.hi);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn expect_lt(&mut self) {
        if !self.eat_lt(true) {
            let found_token = self.this_token_to_string();
            let token_str = Parser::token_to_string(&token::LT);
            self.fatal(format!("expected `{}`, found `{}`",
                               token_str,
                               found_token).as_slice())
        }
    }

    /// Parse a sequence bracketed by `|` and `|`, stopping before the `|`.
    fn parse_seq_to_before_or<T>(
                              &mut self,
                              sep: &token::Token,
                              f: |&mut Parser| -> T)
                              -> Vec<T> {
        let mut first = true;
        let mut vector = Vec::new();
        while self.token != token::BINOP(token::OR) &&
                self.token != token::OROR {
            if first {
                first = false
            } else {
                self.expect(sep)
            }

            vector.push(f(self))
        }
        vector
    }

    /// Expect and consume a GT. if a >> is seen, replace it
    /// with a single > and continue. If a GT is not seen,
    /// signal an error.
    pub fn expect_gt(&mut self) {
        match self.token {
            token::GT => self.bump(),
            token::BINOP(token::SHR) => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                self.replace_token(token::GT, lo, span.hi)
            }
            token::BINOPEQ(token::SHR) => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                self.replace_token(token::GE, lo, span.hi)
            }
            token::GE => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                self.replace_token(token::EQ, lo, span.hi)
            }
            _ => {
                let gt_str = Parser::token_to_string(&token::GT);
                let this_token_str = self.this_token_to_string();
                self.fatal(format!("expected `{}`, found `{}`",
                                   gt_str,
                                   this_token_str).as_slice())
            }
        }
    }

    /// Parse a sequence bracketed by '<' and '>', stopping
    /// before the '>'.
    pub fn parse_seq_to_before_gt<T>(
                                  &mut self,
                                  sep: Option<token::Token>,
                                  f: |&mut Parser| -> T)
                                  -> OwnedSlice<T> {
        let mut v = Vec::new();
        // This loop works by alternating back and forth between parsing types
        // and commas.  For example, given a string `A, B,>`, the parser would
        // first parse `A`, then a comma, then `B`, then a comma. After that it
        // would encounter a `>` and stop. This lets the parser handle trailing
        // commas in generic parameters, because it can stop either after
        // parsing a type or after parsing a comma.
        for i in iter::count(0u, 1) {
            if self.token == token::GT
                || self.token == token::BINOP(token::SHR)
                || self.token == token::GE
                || self.token == token::BINOPEQ(token::SHR) {
                break;
            }

            if i % 2 == 0 {
                v.push(f(self));
            } else {
                sep.as_ref().map(|t| self.expect(t));
            }
        }
        return OwnedSlice::from_vec(v);
    }

    pub fn parse_seq_to_gt<T>(
                           &mut self,
                           sep: Option<token::Token>,
                           f: |&mut Parser| -> T)
                           -> OwnedSlice<T> {
        let v = self.parse_seq_to_before_gt(sep, f);
        self.expect_gt();
        return v;
    }

    /// Parse a sequence, including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_end<T>(
                            &mut self,
                            ket: &token::Token,
                            sep: SeqSep,
                            f: |&mut Parser| -> T)
                            -> Vec<T> {
        let val = self.parse_seq_to_before_end(ket, sep, f);
        self.bump();
        val
    }

    /// Parse a sequence, not including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_before_end<T>(
                                   &mut self,
                                   ket: &token::Token,
                                   sep: SeqSep,
                                   f: |&mut Parser| -> T)
                                   -> Vec<T> {
        let mut first: bool = true;
        let mut v = vec!();
        while self.token != *ket {
            match sep.sep {
              Some(ref t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            if sep.trailing_sep_allowed && self.token == *ket { break; }
            v.push(f(self));
        }
        return v;
    }

    /// Parse a sequence, including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_unspanned_seq<T>(
                               &mut self,
                               bra: &token::Token,
                               ket: &token::Token,
                               sep: SeqSep,
                               f: |&mut Parser| -> T)
                               -> Vec<T> {
        self.expect(bra);
        let result = self.parse_seq_to_before_end(ket, sep, f);
        self.bump();
        result
    }

    /// Parse a sequence parameter of enum variant. For consistency purposes,
    /// these should not be empty.
    pub fn parse_enum_variant_seq<T>(
                               &mut self,
                               bra: &token::Token,
                               ket: &token::Token,
                               sep: SeqSep,
                               f: |&mut Parser| -> T)
                               -> Vec<T> {
        let result = self.parse_unspanned_seq(bra, ket, sep, f);
        if result.is_empty() {
            let last_span = self.last_span;
            self.span_err(last_span,
            "nullary enum variants are written with no trailing `( )`");
        }
        result
    }

    // NB: Do not use this function unless you actually plan to place the
    // spanned list in the AST.
    pub fn parse_seq<T>(
                     &mut self,
                     bra: &token::Token,
                     ket: &token::Token,
                     sep: SeqSep,
                     f: |&mut Parser| -> T)
                     -> Spanned<Vec<T> > {
        let lo = self.span.lo;
        self.expect(bra);
        let result = self.parse_seq_to_before_end(ket, sep, f);
        let hi = self.span.hi;
        self.bump();
        spanned(lo, hi, result)
    }

    /// Advance the parser by one token
    pub fn bump(&mut self) {
        self.last_span = self.span;
        // Stash token for error recovery (sometimes; clone is not necessarily cheap).
        self.last_token = if is_ident_or_path(&self.token) {
            Some(box self.token.clone())
        } else {
            None
        };
        let next = if self.buffer_start == self.buffer_end {
            real_token(self.reader)
        } else {
            // Avoid token copies with `replace`.
            let buffer_start = self.buffer_start as uint;
            let next_index = (buffer_start + 1) & 3 as uint;
            self.buffer_start = next_index as int;

            let placeholder = TokenAndSpan {
                tok: token::UNDERSCORE,
                sp: self.span,
            };
            replace(&mut self.buffer[buffer_start], placeholder)
        };
        self.span = next.sp;
        self.token = next.tok;
        self.tokens_consumed += 1u;
    }

    /// Advance the parser by one token and return the bumped token.
    pub fn bump_and_get(&mut self) -> token::Token {
        let old_token = replace(&mut self.token, token::UNDERSCORE);
        self.bump();
        old_token
    }

    /// EFFECT: replace the current token and span with the given one
    pub fn replace_token(&mut self,
                         next: token::Token,
                         lo: BytePos,
                         hi: BytePos) {
        self.last_span = mk_sp(self.span.lo, lo);
        self.token = next;
        self.span = mk_sp(lo, hi);
    }
    pub fn buffer_length(&mut self) -> int {
        if self.buffer_start <= self.buffer_end {
            return self.buffer_end - self.buffer_start;
        }
        return (4 - self.buffer_start) + self.buffer_end;
    }
    pub fn look_ahead<R>(&mut self, distance: uint, f: |&token::Token| -> R)
                      -> R {
        let dist = distance as int;
        while self.buffer_length() < dist {
            self.buffer[self.buffer_end as uint] = real_token(self.reader);
            self.buffer_end = (self.buffer_end + 1) & 3;
        }
        f(&self.buffer[((self.buffer_start + dist - 1) & 3) as uint].tok)
    }
    pub fn fatal(&mut self, m: &str) -> ! {
        self.sess.span_diagnostic.span_fatal(self.span, m)
    }
    pub fn span_fatal(&mut self, sp: Span, m: &str) -> ! {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }
    pub fn span_note(&mut self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_note(sp, m)
    }
    pub fn bug(&mut self, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(self.span, m)
    }
    pub fn warn(&mut self, m: &str) {
        self.sess.span_diagnostic.span_warn(self.span, m)
    }
    pub fn span_warn(&mut self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_warn(sp, m)
    }
    pub fn span_err(&mut self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_err(sp, m)
    }
    pub fn abort_if_errors(&mut self) {
        self.sess.span_diagnostic.handler().abort_if_errors();
    }

    pub fn id_to_interned_str(&mut self, id: Ident) -> InternedString {
        token::get_ident(id)
    }

    /// Is the current token one of the keywords that signals a bare function
    /// type?
    pub fn token_is_bare_fn_keyword(&mut self) -> bool {
        if token::is_keyword(keywords::Fn, &self.token) {
            return true
        }

        if token::is_keyword(keywords::Unsafe, &self.token) ||
            token::is_keyword(keywords::Once, &self.token) {
            return self.look_ahead(1, |t| token::is_keyword(keywords::Fn, t))
        }

        false
    }

    /// Is the current token one of the keywords that signals a closure type?
    pub fn token_is_closure_keyword(&mut self) -> bool {
        token::is_keyword(keywords::Unsafe, &self.token) ||
            token::is_keyword(keywords::Once, &self.token)
    }

    /// Is the current token one of the keywords that signals an old-style
    /// closure type (with explicit sigil)?
    pub fn token_is_old_style_closure_keyword(&mut self) -> bool {
        token::is_keyword(keywords::Unsafe, &self.token) ||
            token::is_keyword(keywords::Once, &self.token) ||
            token::is_keyword(keywords::Fn, &self.token)
    }

    pub fn token_is_lifetime(tok: &token::Token) -> bool {
        match *tok {
            token::LIFETIME(..) => true,
            _ => false,
        }
    }

    pub fn get_lifetime(&mut self) -> ast::Ident {
        match self.token {
            token::LIFETIME(ref ident) => *ident,
            _ => self.bug("not a lifetime"),
        }
    }

    /// parse a TyBareFn type:
    pub fn parse_ty_bare_fn(&mut self) -> Ty_ {
        /*

        [unsafe] [extern "ABI"] fn <'lt> (S) -> T
         ^~~~^           ^~~~^     ^~~~^ ^~^    ^
           |               |         |    |     |
           |               |         |    |   Return type
           |               |         |  Argument types
           |               |     Lifetimes
           |              ABI
        Function Style
        */

        let fn_style = self.parse_unsafety();
        let abi = if self.eat_keyword(keywords::Extern) {
            self.parse_opt_abi().unwrap_or(abi::C)
        } else {
            abi::Rust
        };

        self.expect_keyword(keywords::Fn);
        let (decl, lifetimes) = self.parse_ty_fn_decl(true);
        return TyBareFn(box(GC) BareFnTy {
            abi: abi,
            fn_style: fn_style,
            lifetimes: lifetimes,
            decl: decl
        });
    }

    /// Parses a procedure type (`proc`). The initial `proc` keyword must
    /// already have been parsed.
    pub fn parse_proc_type(&mut self) -> Ty_ {
        /*

        proc <'lt> (S) [:Bounds] -> T
        ^~~^ ^~~~^  ^  ^~~~~~~~^    ^
         |     |    |      |        |
         |     |    |      |      Return type
         |     |    |    Bounds
         |     |  Argument types
         |   Lifetimes
        the `proc` keyword

        */

        let lifetime_defs = if self.eat(&token::LT) {
            let lifetime_defs = self.parse_lifetime_defs();
            self.expect_gt();
            lifetime_defs
        } else {
            Vec::new()
        };

        let (inputs, variadic) = self.parse_fn_args(false, false);
        let bounds = {
            if self.eat(&token::COLON) {
                let (_, bounds) = self.parse_ty_param_bounds(false);
                Some(bounds)
            } else {
                None
            }
        };
        let (ret_style, ret_ty) = self.parse_ret_ty();
        let decl = P(FnDecl {
            inputs: inputs,
            output: ret_ty,
            cf: ret_style,
            variadic: variadic
        });
        TyProc(box(GC) ClosureTy {
            fn_style: NormalFn,
            onceness: Once,
            bounds: bounds,
            decl: decl,
            lifetimes: lifetime_defs,
        })
    }

    /// Parses an optional unboxed closure kind (`&:`, `&mut:`, or `:`).
    pub fn parse_optional_unboxed_closure_kind(&mut self)
                                               -> Option<UnboxedClosureKind> {
        if self.token == token::BINOP(token::AND) &&
                    self.look_ahead(1, |t| {
                        token::is_keyword(keywords::Mut, t)
                    }) &&
                    self.look_ahead(2, |t| *t == token::COLON) {
            self.bump();
            self.bump();
            self.bump();
            return Some(FnMutUnboxedClosureKind)
        }

        if self.token == token::BINOP(token::AND) &&
                    self.look_ahead(1, |t| *t == token::COLON) {
            self.bump();
            self.bump();
            return Some(FnUnboxedClosureKind)
        }

        if self.eat(&token::COLON) {
            return Some(FnOnceUnboxedClosureKind)
        }

        return None
    }

    /// Parse a TyClosure type
    pub fn parse_ty_closure(&mut self) -> Ty_ {
        /*

        [unsafe] [once] <'lt> |S| [:Bounds] -> T
        ^~~~~~~^ ^~~~~^ ^~~~^  ^  ^~~~~~~~^    ^
          |        |      |    |      |        |
          |        |      |    |      |      Return type
          |        |      |    |  Closure bounds
          |        |      |  Argument types
          |        |    Lifetime defs
          |     Once-ness (a.k.a., affine)
        Function Style

        */

        let fn_style = self.parse_unsafety();
        let onceness = if self.eat_keyword(keywords::Once) {Once} else {Many};

        let lifetime_defs = if self.eat(&token::LT) {
            let lifetime_defs = self.parse_lifetime_defs();
            self.expect_gt();

            lifetime_defs
        } else {
            Vec::new()
        };

        let (optional_unboxed_closure_kind, inputs) = if self.eat(&token::OROR) {
            (None, Vec::new())
        } else {
            self.expect_or();

            let optional_unboxed_closure_kind =
                self.parse_optional_unboxed_closure_kind();

            let inputs = self.parse_seq_to_before_or(
                &token::COMMA,
                |p| p.parse_arg_general(false));
            self.expect_or();
            (optional_unboxed_closure_kind, inputs)
        };

        let (region, bounds) = {
            if self.eat(&token::COLON) {
                let (region, bounds) = self.parse_ty_param_bounds(true);
                (region, Some(bounds))
            } else {
                (None, None)
            }
        };

        let (return_style, output) = self.parse_ret_ty();
        let decl = P(FnDecl {
            inputs: inputs,
            output: output,
            cf: return_style,
            variadic: false
        });

        match optional_unboxed_closure_kind {
            Some(unboxed_closure_kind) => {
                TyUnboxedFn(box(GC) UnboxedFnTy {
                    kind: unboxed_closure_kind,
                    decl: decl,
                })
            }
            None => {
                TyClosure(box(GC) ClosureTy {
                    fn_style: fn_style,
                    onceness: onceness,
                    bounds: bounds,
                    decl: decl,
                    lifetimes: lifetime_defs,
                }, region)
            }
        }
    }

    pub fn parse_unsafety(&mut self) -> FnStyle {
        if self.eat_keyword(keywords::Unsafe) {
            return UnsafeFn;
        } else {
            return NormalFn;
        }
    }

    /// Parse a function type (following the 'fn')
    pub fn parse_ty_fn_decl(&mut self, allow_variadic: bool)
                            -> (P<FnDecl>, Vec<ast::LifetimeDef>) {
        /*

        (fn) <'lt> (S) -> T
             ^~~~^ ^~^    ^
               |    |     |
               |    |   Return type
               |  Argument types
           Lifetime_defs

        */
        let lifetime_defs = if self.eat(&token::LT) {
            let lifetime_defs = self.parse_lifetime_defs();
            self.expect_gt();
            lifetime_defs
        } else {
            Vec::new()
        };

        let (inputs, variadic) = self.parse_fn_args(false, allow_variadic);
        let (ret_style, ret_ty) = self.parse_ret_ty();
        let decl = P(FnDecl {
            inputs: inputs,
            output: ret_ty,
            cf: ret_style,
            variadic: variadic
        });
        (decl, lifetime_defs)
    }

    /// Parse the methods in a trait declaration
    pub fn parse_trait_methods(&mut self) -> Vec<TraitItem> {
        self.parse_unspanned_seq(
            &token::LBRACE,
            &token::RBRACE,
            seq_sep_none(),
            |p| {
            let attrs = p.parse_outer_attributes();
            let lo = p.span.lo;

            // NB: at the moment, trait methods are public by default; this
            // could change.
            let vis = p.parse_visibility();
            let abi = if p.eat_keyword(keywords::Extern) {
                p.parse_opt_abi().unwrap_or(abi::C)
            } else if attr::contains_name(attrs.as_slice(),
                                          "rust_call_abi_hack") {
                // FIXME(stage0, pcwalton): Remove this awful hack after a
                // snapshot, and change to `extern "rust-call" fn`.
                abi::RustCall
            } else {
                abi::Rust
            };
            let style = p.parse_fn_style();
            let ident = p.parse_ident();

            let mut generics = p.parse_generics();

            let (explicit_self, d) = p.parse_fn_decl_with_self(|p| {
                // This is somewhat dubious; We don't want to allow argument
                // names to be left off if there is a definition...
                p.parse_arg_general(false)
            });

            p.parse_where_clause(&mut generics);

            let hi = p.last_span.hi;
            match p.token {
              token::SEMI => {
                p.bump();
                debug!("parse_trait_methods(): parsing required method");
                RequiredMethod(TypeMethod {
                    ident: ident,
                    attrs: attrs,
                    fn_style: style,
                    decl: d,
                    generics: generics,
                    abi: abi,
                    explicit_self: explicit_self,
                    id: ast::DUMMY_NODE_ID,
                    span: mk_sp(lo, hi),
                    vis: vis,
                })
              }
              token::LBRACE => {
                debug!("parse_trait_methods(): parsing provided method");
                let (inner_attrs, body) =
                    p.parse_inner_attrs_and_block();
                let attrs = attrs.append(inner_attrs.as_slice());
                ProvidedMethod(box(GC) ast::Method {
                    attrs: attrs,
                    id: ast::DUMMY_NODE_ID,
                    span: mk_sp(lo, hi),
                    node: ast::MethDecl(ident,
                                        generics,
                                        abi,
                                        explicit_self,
                                        style,
                                        d,
                                        body,
                                        vis)
                })
              }

              _ => {
                  let token_str = p.this_token_to_string();
                  p.fatal((format!("expected `;` or `{{`, found `{}`",
                                   token_str)).as_slice())
              }
            }
        })
    }

    /// Parse a possibly mutable type
    pub fn parse_mt(&mut self) -> MutTy {
        let mutbl = self.parse_mutability();
        let t = self.parse_ty(true);
        MutTy { ty: t, mutbl: mutbl }
    }

    /// Parse [mut/const/imm] ID : TY
    /// now used only by obsolete record syntax parser...
    pub fn parse_ty_field(&mut self) -> TypeField {
        let lo = self.span.lo;
        let mutbl = self.parse_mutability();
        let id = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(true);
        let hi = ty.span.hi;
        ast::TypeField {
            ident: id,
            mt: MutTy { ty: ty, mutbl: mutbl },
            span: mk_sp(lo, hi),
        }
    }

    /// Parse optional return type [ -> TY ] in function decl
    pub fn parse_ret_ty(&mut self) -> (RetStyle, P<Ty>) {
        return if self.eat(&token::RARROW) {
            let lo = self.span.lo;
            if self.eat(&token::NOT) {
                (
                    NoReturn,
                    P(Ty {
                        id: ast::DUMMY_NODE_ID,
                        node: TyBot,
                        span: mk_sp(lo, self.last_span.hi)
                    })
                )
            } else {
                (Return, self.parse_ty(true))
            }
        } else {
            let pos = self.span.lo;
            (
                Return,
                P(Ty {
                    id: ast::DUMMY_NODE_ID,
                    node: TyNil,
                    span: mk_sp(pos, pos),
                })
            )
        }
    }

    /// Parse a type.
    ///
    /// The second parameter specifies whether the `+` binary operator is
    /// allowed in the type grammar.
    pub fn parse_ty(&mut self, plus_allowed: bool) -> P<Ty> {
        maybe_whole!(no_clone self, NtTy);

        let lo = self.span.lo;

        let t = if self.token == token::LPAREN {
            self.bump();
            if self.token == token::RPAREN {
                self.bump();
                TyNil
            } else {
                // (t) is a parenthesized ty
                // (t,) is the type of a tuple with only one field,
                // of type t
                let mut ts = vec!(self.parse_ty(true));
                let mut one_tuple = false;
                while self.token == token::COMMA {
                    self.bump();
                    if self.token != token::RPAREN {
                        ts.push(self.parse_ty(true));
                    }
                    else {
                        one_tuple = true;
                    }
                }

                if ts.len() == 1 && !one_tuple {
                    self.expect(&token::RPAREN);
                    TyParen(*ts.get(0))
                } else {
                    let t = TyTup(ts);
                    self.expect(&token::RPAREN);
                    t
                }
            }
        } else if self.token == token::AT {
            // MANAGED POINTER
            self.bump();
            let span = self.last_span;
            self.obsolete(span, ObsoleteManagedType);
            TyBox(self.parse_ty(plus_allowed))
        } else if self.token == token::TILDE {
            // OWNED POINTER
            self.bump();
            let span = self.last_span;
            match self.token {
                token::IDENT(ref ident, _)
                        if "str" == token::get_ident(*ident).get() => {
                    // This is OK (for now).
                }
                token::LBRACKET => {}   // Also OK.
                _ => self.obsolete(span, ObsoleteOwnedType)
            }
            TyUniq(self.parse_ty(false))
        } else if self.token == token::BINOP(token::STAR) {
            // STAR POINTER (bare pointer?)
            self.bump();
            TyPtr(self.parse_ptr())
        } else if self.token == token::LBRACKET {
            // VECTOR
            self.expect(&token::LBRACKET);
            let t = self.parse_ty(true);

            // Parse the `, ..e` in `[ int, ..e ]`
            // where `e` is a const expression
            let t = match self.maybe_parse_fixed_vstore() {
                None => TyVec(t),
                Some(suffix) => TyFixedLengthVec(t, suffix)
            };
            self.expect(&token::RBRACKET);
            t
        } else if self.token == token::BINOP(token::AND) ||
                self.token == token::ANDAND {
            // BORROWED POINTER
            self.expect_and();
            self.parse_borrowed_pointee()
        } else if self.is_keyword(keywords::Extern) ||
                  self.is_keyword(keywords::Unsafe) ||
                self.token_is_bare_fn_keyword() {
            // BARE FUNCTION
            self.parse_ty_bare_fn()
        } else if self.token_is_closure_keyword() ||
                self.token == token::BINOP(token::OR) ||
                self.token == token::OROR ||
                self.token == token::LT {
            // CLOSURE
            //
            // FIXME(pcwalton): Eventually `token::LT` will not unambiguously
            // introduce a closure, once procs can have lifetime bounds. We
            // will need to refactor the grammar a little bit at that point.

            self.parse_ty_closure()
        } else if self.eat_keyword(keywords::Typeof) {
            // TYPEOF
            // In order to not be ambiguous, the type must be surrounded by parens.
            self.expect(&token::LPAREN);
            let e = self.parse_expr();
            self.expect(&token::RPAREN);
            TyTypeof(e)
        } else if self.eat_keyword(keywords::Proc) {
            self.parse_proc_type()
        } else if self.token == token::MOD_SEP
            || is_ident_or_path(&self.token) {
            // NAMED TYPE
            let mode = if plus_allowed {
                LifetimeAndTypesAndBounds
            } else {
                LifetimeAndTypesWithoutColons
            };
            let PathAndBounds {
                path,
                bounds
            } = self.parse_path(mode);
            TyPath(path, bounds, ast::DUMMY_NODE_ID)
        } else if self.eat(&token::UNDERSCORE) {
            // TYPE TO BE INFERRED
            TyInfer
        } else {
            let msg = format!("expected type, found token {:?}", self.token);
            self.fatal(msg.as_slice());
        };

        let sp = mk_sp(lo, self.last_span.hi);
        P(Ty {id: ast::DUMMY_NODE_ID, node: t, span: sp})
    }

    pub fn parse_borrowed_pointee(&mut self) -> Ty_ {
        // look for `&'lt` or `&'foo ` and interpret `foo` as the region name:
        let opt_lifetime = self.parse_opt_lifetime();

        let mt = self.parse_mt();
        return TyRptr(opt_lifetime, mt);
    }

    pub fn parse_ptr(&mut self) -> MutTy {
        let mutbl = if self.eat_keyword(keywords::Mut) {
            MutMutable
        } else if self.eat_keyword(keywords::Const) {
            MutImmutable
        } else {
            let span = self.last_span;
            self.span_err(span,
                          "bare raw pointers are no longer allowed, you should \
                           likely use `*mut T`, but otherwise `*T` is now \
                           known as `*const T`");
            MutImmutable
        };
        let t = self.parse_ty(true);
        MutTy { ty: t, mutbl: mutbl }
    }

    pub fn is_named_argument(&mut self) -> bool {
        let offset = match self.token {
            token::BINOP(token::AND) => 1,
            token::ANDAND => 1,
            _ if token::is_keyword(keywords::Mut, &self.token) => 1,
            _ => 0
        };

        debug!("parser is_named_argument offset:{}", offset);

        if offset == 0 {
            is_plain_ident_or_underscore(&self.token)
                && self.look_ahead(1, |t| *t == token::COLON)
        } else {
            self.look_ahead(offset, |t| is_plain_ident_or_underscore(t))
                && self.look_ahead(offset + 1, |t| *t == token::COLON)
        }
    }

    /// This version of parse arg doesn't necessarily require
    /// identifier names.
    pub fn parse_arg_general(&mut self, require_name: bool) -> Arg {
        let pat = if require_name || self.is_named_argument() {
            debug!("parse_arg_general parse_pat (require_name:{:?})",
                   require_name);
            let pat = self.parse_pat();

            self.expect(&token::COLON);
            pat
        } else {
            debug!("parse_arg_general ident_to_pat");
            ast_util::ident_to_pat(ast::DUMMY_NODE_ID,
                                   self.last_span,
                                   special_idents::invalid)
        };

        let t = self.parse_ty(true);

        Arg {
            ty: t,
            pat: pat,
            id: ast::DUMMY_NODE_ID,
        }
    }

    /// Parse a single function argument
    pub fn parse_arg(&mut self) -> Arg {
        self.parse_arg_general(true)
    }

    /// Parse an argument in a lambda header e.g. |arg, arg|
    pub fn parse_fn_block_arg(&mut self) -> Arg {
        let pat = self.parse_pat();
        let t = if self.eat(&token::COLON) {
            self.parse_ty(true)
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyInfer,
                span: mk_sp(self.span.lo, self.span.hi),
            })
        };
        Arg {
            ty: t,
            pat: pat,
            id: ast::DUMMY_NODE_ID
        }
    }

    pub fn maybe_parse_fixed_vstore(&mut self) -> Option<Gc<ast::Expr>> {
        if self.token == token::COMMA &&
                self.look_ahead(1, |t| *t == token::DOTDOT) {
            self.bump();
            self.bump();
            Some(self.parse_expr())
        } else {
            None
        }
    }

    /// Matches token_lit = LIT_INTEGER | ...
    pub fn lit_from_token(&mut self, tok: &token::Token) -> Lit_ {
        match *tok {
            token::LIT_BYTE(i) => LitByte(parse::byte_lit(i.as_str()).val0()),
            token::LIT_CHAR(i) => LitChar(parse::char_lit(i.as_str()).val0()),
            token::LIT_INTEGER(s) => parse::integer_lit(s.as_str(),
                                                        &self.sess.span_diagnostic, self.span),
            token::LIT_FLOAT(s) => parse::float_lit(s.as_str()),
            token::LIT_STR(s) => {
                LitStr(token::intern_and_get_ident(parse::str_lit(s.as_str()).as_slice()),
                       ast::CookedStr)
            }
            token::LIT_STR_RAW(s, n) => {
                LitStr(token::intern_and_get_ident(parse::raw_str_lit(s.as_str()).as_slice()),
                       ast::RawStr(n))
            }
            token::LIT_BINARY(i) =>
                LitBinary(parse::binary_lit(i.as_str())),
            token::LIT_BINARY_RAW(i, _) =>
                LitBinary(Rc::new(i.as_str().as_bytes().iter().map(|&x| x).collect())),
            token::LPAREN => { self.expect(&token::RPAREN); LitNil },
            _ => { self.unexpected_last(tok); }
        }
    }

    /// Matches lit = true | false | token_lit
    pub fn parse_lit(&mut self) -> Lit {
        let lo = self.span.lo;
        let lit = if self.eat_keyword(keywords::True) {
            LitBool(true)
        } else if self.eat_keyword(keywords::False) {
            LitBool(false)
        } else {
            let token = self.bump_and_get();
            let lit = self.lit_from_token(&token);
            lit
        };
        codemap::Spanned { node: lit, span: mk_sp(lo, self.last_span.hi) }
    }

    /// matches '-' lit | lit
    pub fn parse_literal_maybe_minus(&mut self) -> Gc<Expr> {
        let minus_lo = self.span.lo;
        let minus_present = self.eat(&token::BINOP(token::MINUS));

        let lo = self.span.lo;
        let literal = box(GC) self.parse_lit();
        let hi = self.span.hi;
        let expr = self.mk_expr(lo, hi, ExprLit(literal));

        if minus_present {
            let minus_hi = self.span.hi;
            let unary = self.mk_unary(UnNeg, expr);
            self.mk_expr(minus_lo, minus_hi, unary)
        } else {
            expr
        }
    }

    /// Parses a path and optional type parameter bounds, depending on the
    /// mode. The `mode` parameter determines whether lifetimes, types, and/or
    /// bounds are permitted and whether `::` must precede type parameter
    /// groups.
    pub fn parse_path(&mut self, mode: PathParsingMode) -> PathAndBounds {
        // Check for a whole path...
        let found = match self.token {
            INTERPOLATED(token::NtPath(_)) => Some(self.bump_and_get()),
            _ => None,
        };
        match found {
            Some(INTERPOLATED(token::NtPath(box path))) => {
                return PathAndBounds {
                    path: path,
                    bounds: None,
                }
            }
            _ => {}
        }

        let lo = self.span.lo;
        let is_global = self.eat(&token::MOD_SEP);

        // Parse any number of segments and bound sets. A segment is an
        // identifier followed by an optional lifetime and a set of types.
        // A bound set is a set of type parameter bounds.
        let mut segments = Vec::new();
        loop {
            // First, parse an identifier.
            let identifier = self.parse_ident();

            // Parse the '::' before type parameters if it's required. If
            // it is required and wasn't present, then we're done.
            if mode == LifetimeAndTypesWithColons &&
                    !self.eat(&token::MOD_SEP) {
                segments.push(ast::PathSegment {
                    identifier: identifier,
                    lifetimes: Vec::new(),
                    types: OwnedSlice::empty(),
                });
                break
            }

            // Parse the `<` before the lifetime and types, if applicable.
            let (any_lifetime_or_types, lifetimes, types) = {
                if mode != NoTypesAllowed && self.eat_lt(false) {
                    let (lifetimes, types) =
                        self.parse_generic_values_after_lt();
                    (true, lifetimes, OwnedSlice::from_vec(types))
                } else {
                    (false, Vec::new(), OwnedSlice::empty())
                }
            };

            // Assemble and push the result.
            segments.push(ast::PathSegment {
                identifier: identifier,
                lifetimes: lifetimes,
                types: types,
            });

            // We're done if we don't see a '::', unless the mode required
            // a double colon to get here in the first place.
            if !(mode == LifetimeAndTypesWithColons &&
                    !any_lifetime_or_types) {
                if !self.eat(&token::MOD_SEP) {
                    break
                }
            }
        }

        // Next, parse a plus and bounded type parameters, if applicable.
        let bounds = if mode == LifetimeAndTypesAndBounds {
            let bounds = {
                if self.eat(&token::BINOP(token::PLUS)) {
                    let (_, bounds) = self.parse_ty_param_bounds(false);
                    if bounds.len() == 0 {
                        let last_span = self.last_span;
                        self.span_err(last_span,
                                      "at least one type parameter bound \
                                       must be specified after the `+`");
                    }
                    Some(bounds)
                } else {
                    None
                }
            };
            bounds
        } else {
            None
        };

        // Assemble the span.
        let span = mk_sp(lo, self.last_span.hi);

        // Assemble the result.
        PathAndBounds {
            path: ast::Path {
                span: span,
                global: is_global,
                segments: segments,
            },
            bounds: bounds,
        }
    }

    /// parses 0 or 1 lifetime
    pub fn parse_opt_lifetime(&mut self) -> Option<ast::Lifetime> {
        match self.token {
            token::LIFETIME(..) => {
                Some(self.parse_lifetime())
            }
            _ => {
                None
            }
        }
    }

    /// Parses a single lifetime
    /// Matches lifetime = LIFETIME
    pub fn parse_lifetime(&mut self) -> ast::Lifetime {
        match self.token {
            token::LIFETIME(i) => {
                let span = self.span;
                self.bump();
                return ast::Lifetime {
                    id: ast::DUMMY_NODE_ID,
                    span: span,
                    name: i.name
                };
            }
            _ => {
                self.fatal(format!("expected a lifetime name").as_slice());
            }
        }
    }

    pub fn parse_lifetime_defs(&mut self) -> Vec<ast::LifetimeDef> {
        /*!
         * Parses `lifetime_defs = [ lifetime_defs { ',' lifetime_defs } ]`
         * where `lifetime_def  = lifetime [':' lifetimes]`
         */

        let mut res = Vec::new();
        loop {
            match self.token {
                token::LIFETIME(_) => {
                    let lifetime = self.parse_lifetime();
                    let bounds =
                        if self.eat(&token::COLON) {
                            self.parse_lifetimes(token::BINOP(token::PLUS))
                        } else {
                            Vec::new()
                        };
                    res.push(ast::LifetimeDef { lifetime: lifetime,
                                                bounds: bounds });
                }

                _ => {
                    return res;
                }
            }

            match self.token {
                token::COMMA => { self.bump(); }
                token::GT => { return res; }
                token::BINOP(token::SHR) => { return res; }
                _ => {
                    let msg = format!("expected `,` or `>` after lifetime \
                                      name, got: {:?}",
                                      self.token);
                    self.fatal(msg.as_slice());
                }
            }
        }
    }

    // matches lifetimes = ( lifetime ) | ( lifetime , lifetimes )
    // actually, it matches the empty one too, but putting that in there
    // messes up the grammar....
    pub fn parse_lifetimes(&mut self, sep: token::Token) -> Vec<ast::Lifetime> {
        /*!
         * Parses zero or more comma separated lifetimes.
         * Expects each lifetime to be followed by either
         * a comma or `>`.  Used when parsing type parameter
         * lists, where we expect something like `<'a, 'b, T>`.
         */

        let mut res = Vec::new();
        loop {
            match self.token {
                token::LIFETIME(_) => {
                    res.push(self.parse_lifetime());
                }
                _ => {
                    return res;
                }
            }

            if self.token != sep {
                return res;
            }

            self.bump();
        }
    }

    pub fn token_is_mutability(tok: &token::Token) -> bool {
        token::is_keyword(keywords::Mut, tok) ||
        token::is_keyword(keywords::Const, tok)
    }

    /// Parse mutability declaration (mut/const/imm)
    pub fn parse_mutability(&mut self) -> Mutability {
        if self.eat_keyword(keywords::Mut) {
            MutMutable
        } else {
            MutImmutable
        }
    }

    /// Parse ident COLON expr
    pub fn parse_field(&mut self) -> Field {
        let lo = self.span.lo;
        let i = self.parse_ident();
        let hi = self.last_span.hi;
        self.expect(&token::COLON);
        let e = self.parse_expr();
        ast::Field {
            ident: spanned(lo, hi, i),
            expr: e,
            span: mk_sp(lo, e.span.hi),
        }
    }

    pub fn mk_expr(&mut self, lo: BytePos, hi: BytePos, node: Expr_) -> Gc<Expr> {
        box(GC) Expr {
            id: ast::DUMMY_NODE_ID,
            node: node,
            span: mk_sp(lo, hi),
        }
    }

    pub fn mk_unary(&mut self, unop: ast::UnOp, expr: Gc<Expr>) -> ast::Expr_ {
        ExprUnary(unop, expr)
    }

    pub fn mk_binary(&mut self, binop: ast::BinOp,
                     lhs: Gc<Expr>, rhs: Gc<Expr>) -> ast::Expr_ {
        ExprBinary(binop, lhs, rhs)
    }

    pub fn mk_call(&mut self, f: Gc<Expr>, args: Vec<Gc<Expr>>) -> ast::Expr_ {
        ExprCall(f, args)
    }

    fn mk_method_call(&mut self,
                      ident: ast::SpannedIdent,
                      tps: Vec<P<Ty>>,
                      args: Vec<Gc<Expr>>)
                      -> ast::Expr_ {
        ExprMethodCall(ident, tps, args)
    }

    pub fn mk_index(&mut self, expr: Gc<Expr>, idx: Gc<Expr>) -> ast::Expr_ {
        ExprIndex(expr, idx)
    }

    pub fn mk_field(&mut self, expr: Gc<Expr>, ident: ast::SpannedIdent,
                    tys: Vec<P<Ty>>) -> ast::Expr_ {
        ExprField(expr, ident, tys)
    }

    pub fn mk_assign_op(&mut self, binop: ast::BinOp,
                        lhs: Gc<Expr>, rhs: Gc<Expr>) -> ast::Expr_ {
        ExprAssignOp(binop, lhs, rhs)
    }

    pub fn mk_mac_expr(&mut self, lo: BytePos, hi: BytePos, m: Mac_) -> Gc<Expr> {
        box(GC) Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprMac(codemap::Spanned {node: m, span: mk_sp(lo, hi)}),
            span: mk_sp(lo, hi),
        }
    }

    pub fn mk_lit_u32(&mut self, i: u32) -> Gc<Expr> {
        let span = &self.span;
        let lv_lit = box(GC) codemap::Spanned {
            node: LitInt(i as u64, ast::UnsignedIntLit(TyU32)),
            span: *span
        };

        box(GC) Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprLit(lv_lit),
            span: *span,
        }
    }

    /// At the bottom (top?) of the precedence hierarchy,
    /// parse things like parenthesized exprs,
    /// macros, return, etc.
    pub fn parse_bottom_expr(&mut self) -> Gc<Expr> {
        maybe_whole_expr!(self);

        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let ex: Expr_;

        match self.token {
            token::LPAREN => {
                self.bump();
                // (e) is parenthesized e
                // (e,) is a tuple with only one field, e
                let mut trailing_comma = false;
                if self.token == token::RPAREN {
                    hi = self.span.hi;
                    self.bump();
                    let lit = box(GC) spanned(lo, hi, LitNil);
                    return self.mk_expr(lo, hi, ExprLit(lit));
                }
                let mut es = vec!(self.parse_expr());
                self.commit_expr(*es.last().unwrap(), &[], &[token::COMMA, token::RPAREN]);
                while self.token == token::COMMA {
                    self.bump();
                    if self.token != token::RPAREN {
                        es.push(self.parse_expr());
                        self.commit_expr(*es.last().unwrap(), &[], &[token::COMMA, token::RPAREN]);
                    }
                        else {
                        trailing_comma = true;
                    }
                }
                hi = self.span.hi;
                self.commit_expr_expecting(*es.last().unwrap(), token::RPAREN);

                return if es.len() == 1 && !trailing_comma {
                    self.mk_expr(lo, hi, ExprParen(*es.get(0)))
                }
                    else {
                    self.mk_expr(lo, hi, ExprTup(es))
                }
            },
            token::LBRACE => {
                self.bump();
                let blk = self.parse_block_tail(lo, DefaultBlock);
                return self.mk_expr(blk.span.lo, blk.span.hi,
                                    ExprBlock(blk));
            },
            token::BINOP(token::OR) |  token::OROR => {
                return self.parse_lambda_expr(CaptureByValue);
            },
            // FIXME #13626: Should be able to stick in
            // token::SELF_KEYWORD_NAME
            token::IDENT(id @ ast::Ident{
                        name: ast::Name(token::SELF_KEYWORD_NAME_NUM),
                        ctxt: _
                    } ,false) => {
                self.bump();
                let path = ast_util::ident_to_path(mk_sp(lo, hi), id);
                ex = ExprPath(path);
                hi = self.last_span.hi;
            }
            token::LBRACKET => {
                self.bump();

                if self.token == token::RBRACKET {
                    // Empty vector.
                    self.bump();
                    ex = ExprVec(Vec::new());
                } else {
                    // Nonempty vector.
                    let first_expr = self.parse_expr();
                    if self.token == token::COMMA &&
                        self.look_ahead(1, |t| *t == token::DOTDOT) {
                        // Repeating vector syntax: [ 0, ..512 ]
                        self.bump();
                        self.bump();
                        let count = self.parse_expr();
                        self.expect(&token::RBRACKET);
                        ex = ExprRepeat(first_expr, count);
                    } else if self.token == token::COMMA {
                        // Vector with two or more elements.
                        self.bump();
                        let remaining_exprs = self.parse_seq_to_end(
                            &token::RBRACKET,
                            seq_sep_trailing_allowed(token::COMMA),
                            |p| p.parse_expr()
                                );
                        let mut exprs = vec!(first_expr);
                        exprs.push_all_move(remaining_exprs);
                        ex = ExprVec(exprs);
                    } else {
                        // Vector with one element.
                        self.expect(&token::RBRACKET);
                        ex = ExprVec(vec!(first_expr));
                    }
                }
                hi = self.last_span.hi;
            },
            _ => {
                if self.eat_keyword(keywords::Ref) {
                    return self.parse_lambda_expr(CaptureByRef);
                }
                if self.eat_keyword(keywords::Proc) {
                    let decl = self.parse_proc_decl();
                    let body = self.parse_expr();
                    let fakeblock = P(ast::Block {
                            view_items: Vec::new(),
                            stmts: Vec::new(),
                            expr: Some(body),
                            id: ast::DUMMY_NODE_ID,
                            rules: DefaultBlock,
                            span: body.span,
                        });
                    return self.mk_expr(lo, body.span.hi, ExprProc(decl, fakeblock));
                }
                if self.eat_keyword(keywords::If) {
                    return self.parse_if_expr();
                }
                if self.eat_keyword(keywords::For) {
                    return self.parse_for_expr(None);
                }
                if self.eat_keyword(keywords::While) {
                    return self.parse_while_expr();
                }
                if Parser::token_is_lifetime(&self.token) {
                    let lifetime = self.get_lifetime();
                    self.bump();
                    self.expect(&token::COLON);
                    if self.eat_keyword(keywords::For) {
                        return self.parse_for_expr(Some(lifetime))
                    }
                    if self.eat_keyword(keywords::Loop) {
                        return self.parse_loop_expr(Some(lifetime))
                    }
                    self.fatal("expected `for` or `loop` after a label")
                }
                if self.eat_keyword(keywords::Loop) {
                    return self.parse_loop_expr(None);
                }
                if self.eat_keyword(keywords::Continue) {
                    let lo = self.span.lo;
                    let ex = if Parser::token_is_lifetime(&self.token) {
                        let lifetime = self.get_lifetime();
                        self.bump();
                        ExprAgain(Some(lifetime))
                    } else {
                        ExprAgain(None)
                    };
                    let hi = self.span.hi;
                    return self.mk_expr(lo, hi, ex);
                }
                if self.eat_keyword(keywords::Match) {
                    return self.parse_match_expr();
                }
                if self.eat_keyword(keywords::Unsafe) {
                    return self.parse_block_expr(
                        lo,
                        UnsafeBlock(ast::UserProvided));
                }
                if self.eat_keyword(keywords::Return) {
                    // RETURN expression
                    if can_begin_expr(&self.token) {
                        let e = self.parse_expr();
                        hi = e.span.hi;
                        ex = ExprRet(Some(e));
                    } else {
                        ex = ExprRet(None);
                    }
                } else if self.eat_keyword(keywords::Break) {
                    // BREAK expression
                    if Parser::token_is_lifetime(&self.token) {
                        let lifetime = self.get_lifetime();
                        self.bump();
                        ex = ExprBreak(Some(lifetime));
                    } else {
                        ex = ExprBreak(None);
                    }
                    hi = self.span.hi;
                } else if self.token == token::MOD_SEP ||
                        is_ident(&self.token) &&
                        !self.is_keyword(keywords::True) &&
                        !self.is_keyword(keywords::False) {
                    let pth =
                        self.parse_path(LifetimeAndTypesWithColons).path;

                    // `!`, as an operator, is prefix, so we know this isn't that
                    if self.token == token::NOT {
                        // MACRO INVOCATION expression
                        self.bump();

                        let ket = token::close_delimiter_for(&self.token)
                            .unwrap_or_else(|| {
                                self.fatal("expected open delimiter")
                            });
                        self.bump();

                        let tts = self.parse_seq_to_end(
                            &ket,
                            seq_sep_none(),
                            |p| p.parse_token_tree());
                        let hi = self.span.hi;

                        return self.mk_mac_expr(lo,
                                                hi,
                                                MacInvocTT(pth,
                                                           tts,
                                                           EMPTY_CTXT));
                    }
                    if self.token == token::LBRACE {
                        // This is a struct literal, unless we're prohibited
                        // from parsing struct literals here.
                        if self.restriction != RESTRICT_NO_STRUCT_LITERAL {
                            // It's a struct literal.
                            self.bump();
                            let mut fields = Vec::new();
                            let mut base = None;

                            while self.token != token::RBRACE {
                                if self.eat(&token::DOTDOT) {
                                    base = Some(self.parse_expr());
                                    break;
                                }

                                fields.push(self.parse_field());
                                self.commit_expr(fields.last().unwrap().expr,
                                                 &[token::COMMA],
                                                 &[token::RBRACE]);
                            }

                            if fields.len() == 0 && base.is_none() {
                                let last_span = self.last_span;
                                self.span_err(last_span,
                                              "structure literal must either \
                                              have at least one field or use \
                                              functional structure update \
                                              syntax");
                            }

                            hi = self.span.hi;
                            self.expect(&token::RBRACE);
                            ex = ExprStruct(pth, fields, base);
                            return self.mk_expr(lo, hi, ex);
                        }
                    }

                    hi = pth.span.hi;
                    ex = ExprPath(pth);
                } else {
                    // other literal expression
                    let lit = self.parse_lit();
                    hi = lit.span.hi;
                    ex = ExprLit(box(GC) lit);
                }
            }
        }

        return self.mk_expr(lo, hi, ex);
    }

    /// Parse a block or unsafe block
    pub fn parse_block_expr(&mut self, lo: BytePos, blk_mode: BlockCheckMode)
                            -> Gc<Expr> {
        self.expect(&token::LBRACE);
        let blk = self.parse_block_tail(lo, blk_mode);
        return self.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk));
    }

    /// parse a.b or a(13) or a[4] or just a
    pub fn parse_dot_or_call_expr(&mut self) -> Gc<Expr> {
        let b = self.parse_bottom_expr();
        self.parse_dot_or_call_expr_with(b)
    }

    pub fn parse_dot_or_call_expr_with(&mut self, e0: Gc<Expr>) -> Gc<Expr> {
        let mut e = e0;
        let lo = e.span.lo;
        let mut hi;
        loop {
            // expr.f
            if self.eat(&token::DOT) {
                match self.token {
                  token::IDENT(i, _) => {
                    let dot = self.last_span.hi;
                    hi = self.span.hi;
                    self.bump();
                    let (_, tys) = if self.eat(&token::MOD_SEP) {
                        self.expect_lt();
                        self.parse_generic_values_after_lt()
                    } else {
                        (Vec::new(), Vec::new())
                    };

                    // expr.f() method call
                    match self.token {
                        token::LPAREN => {
                            let mut es = self.parse_unspanned_seq(
                                &token::LPAREN,
                                &token::RPAREN,
                                seq_sep_trailing_allowed(token::COMMA),
                                |p| p.parse_expr()
                            );
                            hi = self.last_span.hi;

                            es.unshift(e);
                            let id = spanned(dot, hi, i);
                            let nd = self.mk_method_call(id, tys, es);
                            e = self.mk_expr(lo, hi, nd);
                        }
                        _ => {
                            let id = spanned(dot, hi, i);
                            let field = self.mk_field(e, id, tys);
                            e = self.mk_expr(lo, hi, field)
                        }
                    }
                  }
                  _ => self.unexpected()
                }
                continue;
            }
            if self.expr_is_complete(e) { break; }
            match self.token {
              // expr(...)
              token::LPAREN => {
                let es = self.parse_unspanned_seq(
                    &token::LPAREN,
                    &token::RPAREN,
                    seq_sep_trailing_allowed(token::COMMA),
                    |p| p.parse_expr()
                );
                hi = self.last_span.hi;

                let nd = self.mk_call(e, es);
                e = self.mk_expr(lo, hi, nd);
              }

              // expr[...]
              token::LBRACKET => {
                self.bump();
                let ix = self.parse_expr();
                hi = self.span.hi;
                self.commit_expr_expecting(ix, token::RBRACKET);
                let index = self.mk_index(e, ix);
                e = self.mk_expr(lo, hi, index)
              }

              _ => return e
            }
        }
        return e;
    }

    /// Parse an optional separator followed by a kleene-style
    /// repetition token (+ or *).
    pub fn parse_sep_and_zerok(&mut self) -> (Option<token::Token>, bool) {
        fn parse_zerok(parser: &mut Parser) -> Option<bool> {
            match parser.token {
                token::BINOP(token::STAR) | token::BINOP(token::PLUS) => {
                    let zerok = parser.token == token::BINOP(token::STAR);
                    parser.bump();
                    Some(zerok)
                },
                _ => None
            }
        };

        match parse_zerok(self) {
            Some(zerok) => return (None, zerok),
            None => {}
        }

        let separator = self.bump_and_get();
        match parse_zerok(self) {
            Some(zerok) => (Some(separator), zerok),
            None => self.fatal("expected `*` or `+`")
        }
    }

    /// parse a single token tree from the input.
    pub fn parse_token_tree(&mut self) -> TokenTree {
        // FIXME #6994: currently, this is too eager. It
        // parses token trees but also identifies TTSeq's
        // and TTNonterminal's; it's too early to know yet
        // whether something will be a nonterminal or a seq
        // yet.
        maybe_whole!(deref self, NtTT);

        // this is the fall-through for the 'match' below.
        // invariants: the current token is not a left-delimiter,
        // not an EOF, and not the desired right-delimiter (if
        // it were, parse_seq_to_before_end would have prevented
        // reaching this point.
        fn parse_non_delim_tt_tok(p: &mut Parser) -> TokenTree {
            maybe_whole!(deref p, NtTT);
            match p.token {
              token::RPAREN | token::RBRACE | token::RBRACKET => {
                  // This is a conservative error: only report the last unclosed delimiter. The
                  // previous unclosed delimiters could actually be closed! The parser just hasn't
                  // gotten to them yet.
                  match p.open_braces.last() {
                      None => {}
                      Some(&sp) => p.span_note(sp, "unclosed delimiter"),
                  };
                  let token_str = p.this_token_to_string();
                  p.fatal(format!("incorrect close delimiter: `{}`",
                                  token_str).as_slice())
              },
              /* we ought to allow different depths of unquotation */
              token::DOLLAR if p.quote_depth > 0u => {
                p.bump();
                let sp = p.span;

                if p.token == token::LPAREN {
                    let seq = p.parse_seq(
                        &token::LPAREN,
                        &token::RPAREN,
                        seq_sep_none(),
                        |p| p.parse_token_tree()
                    );
                    let (s, z) = p.parse_sep_and_zerok();
                    let seq = match seq {
                        Spanned { node, .. } => node,
                    };
                    TTSeq(mk_sp(sp.lo, p.span.hi), Rc::new(seq), s, z)
                } else {
                    TTNonterminal(sp, p.parse_ident())
                }
              }
              _ => {
                  parse_any_tt_tok(p)
              }
            }
        }

        // turn the next token into a TTTok:
        fn parse_any_tt_tok(p: &mut Parser) -> TokenTree {
            TTTok(p.span, p.bump_and_get())
        }

        match (&self.token, token::close_delimiter_for(&self.token)) {
            (&token::EOF, _) => {
                let open_braces = self.open_braces.clone();
                for sp in open_braces.iter() {
                    self.span_note(*sp, "Did you mean to close this delimiter?");
                }
                // There shouldn't really be a span, but it's easier for the test runner
                // if we give it one
                self.fatal("this file contains an un-closed delimiter ");
            }
            (_, Some(close_delim)) => {
                // Parse the open delimiter.
                self.open_braces.push(self.span);
                let mut result = vec!(parse_any_tt_tok(self));

                let trees =
                    self.parse_seq_to_before_end(&close_delim,
                                                 seq_sep_none(),
                                                 |p| p.parse_token_tree());
                result.push_all_move(trees);

                // Parse the close delimiter.
                result.push(parse_any_tt_tok(self));
                self.open_braces.pop().unwrap();

                TTDelim(Rc::new(result))
            }
            _ => parse_non_delim_tt_tok(self)
        }
    }

    // parse a stream of tokens into a list of TokenTree's,
    // up to EOF.
    pub fn parse_all_token_trees(&mut self) -> Vec<TokenTree> {
        let mut tts = Vec::new();
        while self.token != token::EOF {
            tts.push(self.parse_token_tree());
        }
        tts
    }

    pub fn parse_matchers(&mut self) -> Vec<Matcher> {
        // unification of Matcher's and TokenTree's would vastly improve
        // the interpolation of Matcher's
        maybe_whole!(self, NtMatchers);
        let mut name_idx = 0u;
        match token::close_delimiter_for(&self.token) {
            Some(other_delimiter) => {
                self.bump();
                self.parse_matcher_subseq_upto(&mut name_idx, &other_delimiter)
            }
            None => self.fatal("expected open delimiter")
        }
    }

    /// This goofy function is necessary to correctly match parens in Matcher's.
    /// Otherwise, `$( ( )` would be a valid Matcher, and `$( () )` would be
    /// invalid. It's similar to common::parse_seq.
    pub fn parse_matcher_subseq_upto(&mut self,
                                     name_idx: &mut uint,
                                     ket: &token::Token)
                                     -> Vec<Matcher> {
        let mut ret_val = Vec::new();
        let mut lparens = 0u;

        while self.token != *ket || lparens > 0u {
            if self.token == token::LPAREN { lparens += 1u; }
            if self.token == token::RPAREN { lparens -= 1u; }
            ret_val.push(self.parse_matcher(name_idx));
        }

        self.bump();

        return ret_val;
    }

    pub fn parse_matcher(&mut self, name_idx: &mut uint) -> Matcher {
        let lo = self.span.lo;

        let m = if self.token == token::DOLLAR {
            self.bump();
            if self.token == token::LPAREN {
                let name_idx_lo = *name_idx;
                self.bump();
                let ms = self.parse_matcher_subseq_upto(name_idx,
                                                        &token::RPAREN);
                if ms.len() == 0u {
                    self.fatal("repetition body must be nonempty");
                }
                let (sep, zerok) = self.parse_sep_and_zerok();
                MatchSeq(ms, sep, zerok, name_idx_lo, *name_idx)
            } else {
                let bound_to = self.parse_ident();
                self.expect(&token::COLON);
                let nt_name = self.parse_ident();
                let m = MatchNonterminal(bound_to, nt_name, *name_idx);
                *name_idx += 1;
                m
            }
        } else {
            MatchTok(self.bump_and_get())
        };

        return spanned(lo, self.span.hi, m);
    }

    /// Parse a prefix-operator expr
    pub fn parse_prefix_expr(&mut self) -> Gc<Expr> {
        let lo = self.span.lo;
        let hi;

        let ex;
        match self.token {
          token::NOT => {
            self.bump();
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            ex = self.mk_unary(UnNot, e);
          }
          token::BINOP(token::MINUS) => {
            self.bump();
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            ex = self.mk_unary(UnNeg, e);
          }
          token::BINOP(token::STAR) => {
            self.bump();
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            ex = self.mk_unary(UnDeref, e);
          }
          token::BINOP(token::AND) | token::ANDAND => {
            self.expect_and();
            let m = self.parse_mutability();
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            ex = ExprAddrOf(m, e);
          }
          token::AT => {
            self.bump();
            let span = self.last_span;
            self.obsolete(span, ObsoleteManagedExpr);
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            ex = self.mk_unary(UnBox, e);
          }
          token::TILDE => {
            self.bump();
            let span = self.last_span;
            match self.token {
                token::LIT_STR(_) => {
                    // This is OK (for now).
                }
                token::LBRACKET => {}   // Also OK.
                _ => self.obsolete(span, ObsoleteOwnedExpr)
            }

            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            ex = self.mk_unary(UnUniq, e);
          }
          token::IDENT(_, _) => {
            if !self.is_keyword(keywords::Box) {
                return self.parse_dot_or_call_expr();
            }

            self.bump();

            // Check for a place: `box(PLACE) EXPR`.
            if self.eat(&token::LPAREN) {
                // Support `box() EXPR` as the default.
                if !self.eat(&token::RPAREN) {
                    let place = self.parse_expr();
                    self.expect(&token::RPAREN);
                    let subexpression = self.parse_prefix_expr();
                    hi = subexpression.span.hi;
                    ex = ExprBox(place, subexpression);
                    return self.mk_expr(lo, hi, ex);
                }
            }

            // Otherwise, we use the unique pointer default.
            let subexpression = self.parse_prefix_expr();
            hi = subexpression.span.hi;
            ex = self.mk_unary(UnUniq, subexpression);
          }
          _ => return self.parse_dot_or_call_expr()
        }
        return self.mk_expr(lo, hi, ex);
    }

    /// Parse an expression of binops
    pub fn parse_binops(&mut self) -> Gc<Expr> {
        let prefix_expr = self.parse_prefix_expr();
        self.parse_more_binops(prefix_expr, 0)
    }

    /// Parse an expression of binops of at least min_prec precedence
    pub fn parse_more_binops(&mut self, lhs: Gc<Expr>,
                             min_prec: uint) -> Gc<Expr> {
        if self.expr_is_complete(lhs) { return lhs; }

        // Prevent dynamic borrow errors later on by limiting the
        // scope of the borrows.
        {
            let token: &token::Token = &self.token;
            let restriction: &restriction = &self.restriction;
            match (token, restriction) {
                (&token::BINOP(token::OR), &RESTRICT_NO_BAR_OP) => return lhs,
                (&token::BINOP(token::OR),
                 &RESTRICT_NO_BAR_OR_DOUBLEBAR_OP) => return lhs,
                (&token::OROR, &RESTRICT_NO_BAR_OR_DOUBLEBAR_OP) => return lhs,
                _ => { }
            }
        }

        let cur_opt = token_to_binop(&self.token);
        match cur_opt {
            Some(cur_op) => {
                let cur_prec = operator_prec(cur_op);
                if cur_prec > min_prec {
                    self.bump();
                    let expr = self.parse_prefix_expr();
                    let rhs = self.parse_more_binops(expr, cur_prec);
                    let binary = self.mk_binary(cur_op, lhs, rhs);
                    let bin = self.mk_expr(lhs.span.lo, rhs.span.hi, binary);
                    self.parse_more_binops(bin, min_prec)
                } else {
                    lhs
                }
            }
            None => {
                if as_prec > min_prec && self.eat_keyword(keywords::As) {
                    let rhs = self.parse_ty(false);
                    let _as = self.mk_expr(lhs.span.lo,
                                           rhs.span.hi,
                                           ExprCast(lhs, rhs));
                    self.parse_more_binops(_as, min_prec)
                } else {
                    lhs
                }
            }
        }
    }

    /// Parse an assignment expression....
    /// actually, this seems to be the main entry point for
    /// parsing an arbitrary expression.
    pub fn parse_assign_expr(&mut self) -> Gc<Expr> {
        let lo = self.span.lo;
        let lhs = self.parse_binops();
        match self.token {
          token::EQ => {
              self.bump();
              let rhs = self.parse_expr();
              self.mk_expr(lo, rhs.span.hi, ExprAssign(lhs, rhs))
          }
          token::BINOPEQ(op) => {
              self.bump();
              let rhs = self.parse_expr();
              let aop = match op {
                  token::PLUS =>    BiAdd,
                  token::MINUS =>   BiSub,
                  token::STAR =>    BiMul,
                  token::SLASH =>   BiDiv,
                  token::PERCENT => BiRem,
                  token::CARET =>   BiBitXor,
                  token::AND =>     BiBitAnd,
                  token::OR =>      BiBitOr,
                  token::SHL =>     BiShl,
                  token::SHR =>     BiShr
              };
              let assign_op = self.mk_assign_op(aop, lhs, rhs);
              self.mk_expr(lo, rhs.span.hi, assign_op)
          }
          _ => {
              lhs
          }
        }
    }

    /// Parse an 'if' expression ('if' token already eaten)
    pub fn parse_if_expr(&mut self) -> Gc<Expr> {
        let lo = self.last_span.lo;
        let cond = self.parse_expr_res(RESTRICT_NO_STRUCT_LITERAL);
        let thn = self.parse_block();
        let mut els: Option<Gc<Expr>> = None;
        let mut hi = thn.span.hi;
        if self.eat_keyword(keywords::Else) {
            let elexpr = self.parse_else_expr();
            els = Some(elexpr);
            hi = elexpr.span.hi;
        }
        self.mk_expr(lo, hi, ExprIf(cond, thn, els))
    }

    // `|args| expr`
    pub fn parse_lambda_expr(&mut self, capture_clause: CaptureClause)
                             -> Gc<Expr> {
        let lo = self.span.lo;
        let (decl, optional_unboxed_closure_kind) =
            self.parse_fn_block_decl();
        let body = self.parse_expr();
        let fakeblock = P(ast::Block {
            view_items: Vec::new(),
            stmts: Vec::new(),
            expr: Some(body),
            id: ast::DUMMY_NODE_ID,
            rules: DefaultBlock,
            span: body.span,
        });

        match optional_unboxed_closure_kind {
            Some(unboxed_closure_kind) => {
                self.mk_expr(lo,
                             body.span.hi,
                             ExprUnboxedFn(capture_clause,
                                           unboxed_closure_kind,
                                           decl,
                                           fakeblock))
            }
            None => {
                self.mk_expr(lo,
                             body.span.hi,
                             ExprFnBlock(capture_clause, decl, fakeblock))
            }
        }
    }

    pub fn parse_else_expr(&mut self) -> Gc<Expr> {
        if self.eat_keyword(keywords::If) {
            return self.parse_if_expr();
        } else {
            let blk = self.parse_block();
            return self.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk));
        }
    }

    /// Parse a 'for' .. 'in' expression ('for' token already eaten)
    pub fn parse_for_expr(&mut self, opt_ident: Option<ast::Ident>) -> Gc<Expr> {
        // Parse: `for <src_pat> in <src_expr> <src_loop_block>`

        let lo = self.last_span.lo;
        let pat = self.parse_pat();
        self.expect_keyword(keywords::In);
        let expr = self.parse_expr_res(RESTRICT_NO_STRUCT_LITERAL);
        let loop_block = self.parse_block();
        let hi = self.span.hi;

        self.mk_expr(lo, hi, ExprForLoop(pat, expr, loop_block, opt_ident))
    }

    pub fn parse_while_expr(&mut self) -> Gc<Expr> {
        let lo = self.last_span.lo;
        let cond = self.parse_expr_res(RESTRICT_NO_STRUCT_LITERAL);
        let body = self.parse_block();
        let hi = body.span.hi;
        return self.mk_expr(lo, hi, ExprWhile(cond, body));
    }

    pub fn parse_loop_expr(&mut self, opt_ident: Option<ast::Ident>) -> Gc<Expr> {
        let lo = self.last_span.lo;
        let body = self.parse_block();
        let hi = body.span.hi;
        self.mk_expr(lo, hi, ExprLoop(body, opt_ident))
    }

    fn parse_match_expr(&mut self) -> Gc<Expr> {
        let lo = self.last_span.lo;
        let discriminant = self.parse_expr_res(RESTRICT_NO_STRUCT_LITERAL);
        self.commit_expr_expecting(discriminant, token::LBRACE);
        let mut arms: Vec<Arm> = Vec::new();
        while self.token != token::RBRACE {
            arms.push(self.parse_arm());
        }
        let hi = self.span.hi;
        self.bump();
        return self.mk_expr(lo, hi, ExprMatch(discriminant, arms));
    }

    pub fn parse_arm(&mut self) -> Arm {
        let attrs = self.parse_outer_attributes();
        let pats = self.parse_pats();
        let mut guard = None;
        if self.eat_keyword(keywords::If) {
            guard = Some(self.parse_expr());
        }
        self.expect(&token::FAT_ARROW);
        let expr = self.parse_expr_res(RESTRICT_STMT_EXPR);

        let require_comma =
            !classify::expr_is_simple_block(expr)
            && self.token != token::RBRACE;

        if require_comma {
            self.commit_expr(expr, &[token::COMMA], &[token::RBRACE]);
        } else {
            self.eat(&token::COMMA);
        }

        ast::Arm {
            attrs: attrs,
            pats: pats,
            guard: guard,
            body: expr,
        }
    }

    /// Parse an expression
    pub fn parse_expr(&mut self) -> Gc<Expr> {
        return self.parse_expr_res(UNRESTRICTED);
    }

    /// Parse an expression, subject to the given restriction
    pub fn parse_expr_res(&mut self, r: restriction) -> Gc<Expr> {
        let old = self.restriction;
        self.restriction = r;
        let e = self.parse_assign_expr();
        self.restriction = old;
        return e;
    }

    /// Parse the RHS of a local variable declaration (e.g. '= 14;')
    fn parse_initializer(&mut self) -> Option<Gc<Expr>> {
        if self.token == token::EQ {
            self.bump();
            Some(self.parse_expr())
        } else {
            None
        }
    }

    /// Parse patterns, separated by '|' s
    fn parse_pats(&mut self) -> Vec<Gc<Pat>> {
        let mut pats = Vec::new();
        loop {
            pats.push(self.parse_pat());
            if self.token == token::BINOP(token::OR) { self.bump(); }
            else { return pats; }
        };
    }

    fn parse_pat_vec_elements(
        &mut self,
    ) -> (Vec<Gc<Pat>> , Option<Gc<Pat>>, Vec<Gc<Pat>> ) {
        let mut before = Vec::new();
        let mut slice = None;
        let mut after = Vec::new();
        let mut first = true;
        let mut before_slice = true;

        while self.token != token::RBRACKET {
            if first { first = false; }
            else { self.expect(&token::COMMA); }

            let mut is_slice = false;
            if before_slice {
                if self.token == token::DOTDOT {
                    self.bump();
                    is_slice = true;
                    before_slice = false;
                }
            }

            if is_slice {
                if self.token == token::COMMA || self.token == token::RBRACKET {
                    slice = Some(box(GC) ast::Pat {
                        id: ast::DUMMY_NODE_ID,
                        node: PatWild(PatWildMulti),
                        span: self.span,
                    })
                } else {
                    let subpat = self.parse_pat();
                    match *subpat {
                        ast::Pat { node: PatIdent(_, _, _), .. } => {
                            slice = Some(subpat);
                        }
                        ast::Pat { span, .. } => self.span_fatal(
                            span, "expected an identifier or nothing"
                        )
                    }
                }
            } else {
                let subpat = self.parse_pat();
                if before_slice {
                    before.push(subpat);
                } else {
                    after.push(subpat);
                }
            }
        }

        (before, slice, after)
    }

    /// Parse the fields of a struct-like pattern
    fn parse_pat_fields(&mut self) -> (Vec<ast::FieldPat> , bool) {
        let mut fields = Vec::new();
        let mut etc = false;
        let mut first = true;
        while self.token != token::RBRACE {
            if first {
                first = false;
            } else {
                self.expect(&token::COMMA);
                // accept trailing commas
                if self.token == token::RBRACE { break }
            }

            if self.token == token::DOTDOT {
                self.bump();
                if self.token != token::RBRACE {
                    let token_str = self.this_token_to_string();
                    self.fatal(format!("expected `{}`, found `{}`", "}",
                                       token_str).as_slice())
                }
                etc = true;
                break;
            }

            let bind_type = if self.eat_keyword(keywords::Mut) {
                BindByValue(MutMutable)
            } else if self.eat_keyword(keywords::Ref) {
                BindByRef(self.parse_mutability())
            } else {
                BindByValue(MutImmutable)
            };

            let fieldname = self.parse_ident();

            let subpat = if self.token == token::COLON {
                match bind_type {
                    BindByRef(..) | BindByValue(MutMutable) => {
                        let token_str = self.this_token_to_string();
                        self.fatal(format!("unexpected `{}`",
                                           token_str).as_slice())
                    }
                    _ => {}
                }

                self.bump();
                self.parse_pat()
            } else {
                let fieldpath = codemap::Spanned{span:self.last_span, node: fieldname};
                box(GC) ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    node: PatIdent(bind_type, fieldpath, None),
                    span: self.last_span
                }
            };
            fields.push(ast::FieldPat { ident: fieldname, pat: subpat });
        }
        return (fields, etc);
    }

    /// Parse a pattern.
    pub fn parse_pat(&mut self) -> Gc<Pat> {
        maybe_whole!(self, NtPat);

        let lo = self.span.lo;
        let mut hi;
        let pat;
        match self.token {
            // parse _
          token::UNDERSCORE => {
            self.bump();
            pat = PatWild(PatWildSingle);
            hi = self.last_span.hi;
            return box(GC) ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          token::TILDE => {
            // parse ~pat
            self.bump();
            let sub = self.parse_pat();
            pat = PatBox(sub);
            let last_span = self.last_span;
            hi = last_span.hi;
            self.obsolete(last_span, ObsoleteOwnedPattern);
            return box(GC) ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          token::BINOP(token::AND) | token::ANDAND => {
            // parse &pat
            let lo = self.span.lo;
            self.expect_and();
            let sub = self.parse_pat();
            pat = PatRegion(sub);
            hi = self.last_span.hi;
            return box(GC) ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          token::LPAREN => {
            // parse (pat,pat,pat,...) as tuple
            self.bump();
            if self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = box(GC) codemap::Spanned {
                    node: LitNil,
                    span: mk_sp(lo, hi)};
                let expr = self.mk_expr(lo, hi, ExprLit(lit));
                pat = PatLit(expr);
            } else {
                let mut fields = vec!(self.parse_pat());
                if self.look_ahead(1, |t| *t != token::RPAREN) {
                    while self.token == token::COMMA {
                        self.bump();
                        if self.token == token::RPAREN { break; }
                        fields.push(self.parse_pat());
                    }
                }
                if fields.len() == 1 { self.expect(&token::COMMA); }
                self.expect(&token::RPAREN);
                pat = PatTup(fields);
            }
            hi = self.last_span.hi;
            return box(GC) ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          token::LBRACKET => {
            // parse [pat,pat,...] as vector pattern
            self.bump();
            let (before, slice, after) =
                self.parse_pat_vec_elements();

            self.expect(&token::RBRACKET);
            pat = ast::PatVec(before, slice, after);
            hi = self.last_span.hi;
            return box(GC) ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          _ => {}
        }
        // at this point, token != _, ~, &, &&, (, [

        if (!is_ident_or_path(&self.token) && self.token != token::MOD_SEP)
                || self.is_keyword(keywords::True)
                || self.is_keyword(keywords::False) {
            // Parse an expression pattern or exp .. exp.
            //
            // These expressions are limited to literals (possibly
            // preceded by unary-minus) or identifiers.
            let val = self.parse_literal_maybe_minus();
            if self.eat(&token::DOTDOT) {
                let end = if is_ident_or_path(&self.token) {
                    let path = self.parse_path(LifetimeAndTypesWithColons)
                                   .path;
                    let hi = self.span.hi;
                    self.mk_expr(lo, hi, ExprPath(path))
                } else {
                    self.parse_literal_maybe_minus()
                };
                pat = PatRange(val, end);
            } else {
                pat = PatLit(val);
            }
        } else if self.eat_keyword(keywords::Mut) {
            pat = self.parse_pat_ident(BindByValue(MutMutable));
        } else if self.eat_keyword(keywords::Ref) {
            // parse ref pat
            let mutbl = self.parse_mutability();
            pat = self.parse_pat_ident(BindByRef(mutbl));
        } else if self.eat_keyword(keywords::Box) {
            // `box PAT`
            //
            // FIXME(#13910): Rename to `PatBox` and extend to full DST
            // support.
            let sub = self.parse_pat();
            pat = PatBox(sub);
            hi = self.last_span.hi;
            return box(GC) ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
        } else {
            let can_be_enum_or_struct = self.look_ahead(1, |t| {
                match *t {
                    token::LPAREN | token::LBRACKET | token::LT |
                    token::LBRACE | token::MOD_SEP => true,
                    _ => false,
                }
            });

            if self.look_ahead(1, |t| *t == token::DOTDOT) {
                let start = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                self.eat(&token::DOTDOT);
                let end = self.parse_expr_res(RESTRICT_NO_BAR_OP);
                pat = PatRange(start, end);
            } else if is_plain_ident(&self.token) && !can_be_enum_or_struct {
                let id = self.parse_ident();
                let id_span = self.last_span;
                let pth1 = codemap::Spanned{span:id_span, node: id};
                if self.eat(&token::NOT) {
                    // macro invocation
                    let ket = token::close_delimiter_for(&self.token)
                                    .unwrap_or_else(|| self.fatal("expected open delimiter"));
                    self.bump();

                    let tts = self.parse_seq_to_end(&ket,
                                                    seq_sep_none(),
                                                    |p| p.parse_token_tree());

                    let mac = MacInvocTT(ident_to_path(id_span,id), tts, EMPTY_CTXT);
                    pat = ast::PatMac(codemap::Spanned {node: mac, span: self.span});
                } else {
                    let sub = if self.eat(&token::AT) {
                        // parse foo @ pat
                        Some(self.parse_pat())
                    } else {
                        // or just foo
                        None
                    };
                    pat = PatIdent(BindByValue(MutImmutable), pth1, sub);
                }
            } else {
                // parse an enum pat
                let enum_path = self.parse_path(LifetimeAndTypesWithColons)
                                    .path;
                match self.token {
                    token::LBRACE => {
                        self.bump();
                        let (fields, etc) =
                            self.parse_pat_fields();
                        self.bump();
                        pat = PatStruct(enum_path, fields, etc);
                    }
                    _ => {
                        let mut args: Vec<Gc<Pat>> = Vec::new();
                        match self.token {
                          token::LPAREN => {
                            let is_dotdot = self.look_ahead(1, |t| {
                                match *t {
                                    token::DOTDOT => true,
                                    _ => false,
                                }
                            });
                            if is_dotdot {
                                // This is a "top constructor only" pat
                                self.bump();
                                self.bump();
                                self.expect(&token::RPAREN);
                                pat = PatEnum(enum_path, None);
                            } else {
                                args = self.parse_enum_variant_seq(
                                    &token::LPAREN,
                                    &token::RPAREN,
                                    seq_sep_trailing_allowed(token::COMMA),
                                    |p| p.parse_pat()
                                );
                                pat = PatEnum(enum_path, Some(args));
                            }
                          },
                          _ => {
                              if !enum_path.global &&
                                    enum_path.segments.len() == 1 &&
                                    enum_path.segments
                                             .get(0)
                                             .lifetimes
                                             .len() == 0 &&
                                    enum_path.segments
                                             .get(0)
                                             .types
                                             .len() == 0 {
                                  // it could still be either an enum
                                  // or an identifier pattern, resolve
                                  // will sort it out:
                                  pat = PatIdent(BindByValue(MutImmutable),
                                                 codemap::Spanned{
                                                    span: enum_path.span,
                                                    node: enum_path.segments.get(0)
                                                           .identifier},
                                                 None);
                              } else {
                                  pat = PatEnum(enum_path, Some(args));
                              }
                          }
                        }
                    }
                }
            }
        }
        hi = self.last_span.hi;
        box(GC) ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: pat,
            span: mk_sp(lo, hi),
        }
    }

    /// Parse ident or ident @ pat
    /// used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like ref foo(a,b)
    fn parse_pat_ident(&mut self,
                       binding_mode: ast::BindingMode)
                       -> ast::Pat_ {
        if !is_plain_ident(&self.token) {
            let last_span = self.last_span;
            self.span_fatal(last_span,
                            "expected identifier, found path");
        }
        let ident = self.parse_ident();
        let last_span = self.last_span;
        let name = codemap::Spanned{span: last_span, node: ident};
        let sub = if self.eat(&token::AT) {
            Some(self.parse_pat())
        } else {
            None
        };

        // just to be friendly, if they write something like
        //   ref Some(i)
        // we end up here with ( as the current token.  This shortly
        // leads to a parse error.  Note that if there is no explicit
        // binding mode then we do not end up here, because the lookahead
        // will direct us over to parse_enum_variant()
        if self.token == token::LPAREN {
            let last_span = self.last_span;
            self.span_fatal(
                last_span,
                "expected identifier, found enum pattern");
        }

        PatIdent(binding_mode, name, sub)
    }

    /// Parse a local variable declaration
    fn parse_local(&mut self) -> Gc<Local> {
        let lo = self.span.lo;
        let pat = self.parse_pat();

        let mut ty = P(Ty {
            id: ast::DUMMY_NODE_ID,
            node: TyInfer,
            span: mk_sp(lo, lo),
        });
        if self.eat(&token::COLON) {
            ty = self.parse_ty(true);
        }
        let init = self.parse_initializer();
        box(GC) ast::Local {
            ty: ty,
            pat: pat,
            init: init,
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, self.last_span.hi),
            source: LocalLet,
        }
    }

    /// Parse a "let" stmt
    fn parse_let(&mut self) -> Gc<Decl> {
        let lo = self.span.lo;
        let local = self.parse_local();
        box(GC) spanned(lo, self.last_span.hi, DeclLocal(local))
    }

    /// Parse a structure field
    fn parse_name_and_ty(&mut self, pr: Visibility,
                         attrs: Vec<Attribute> ) -> StructField {
        let lo = self.span.lo;
        if !is_plain_ident(&self.token) {
            self.fatal("expected ident");
        }
        let name = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(true);
        spanned(lo, self.last_span.hi, ast::StructField_ {
            kind: NamedField(name, pr),
            id: ast::DUMMY_NODE_ID,
            ty: ty,
            attrs: attrs,
        })
    }

    /// Parse a statement. may include decl.
    /// Precondition: any attributes are parsed already
    pub fn parse_stmt(&mut self, item_attrs: Vec<Attribute>) -> Gc<Stmt> {
        maybe_whole!(self, NtStmt);

        fn check_expected_item(p: &mut Parser, found_attrs: bool) {
            // If we have attributes then we should have an item
            if found_attrs {
                let last_span = p.last_span;
                p.span_err(last_span, "expected item after attributes");
            }
        }

        let lo = self.span.lo;
        if self.is_keyword(keywords::Let) {
            check_expected_item(self, !item_attrs.is_empty());
            self.expect_keyword(keywords::Let);
            let decl = self.parse_let();
            return box(GC) spanned(lo, decl.span.hi, StmtDecl(decl, ast::DUMMY_NODE_ID));
        } else if is_ident(&self.token)
            && !token::is_any_keyword(&self.token)
            && self.look_ahead(1, |t| *t == token::NOT) {
            // it's a macro invocation:

            check_expected_item(self, !item_attrs.is_empty());

            // Potential trouble: if we allow macros with paths instead of
            // idents, we'd need to look ahead past the whole path here...
            let pth = self.parse_path(NoTypesAllowed).path;
            self.bump();

            let id = if token::close_delimiter_for(&self.token).is_some() {
                token::special_idents::invalid // no special identifier
            } else {
                self.parse_ident()
            };

            // check that we're pointing at delimiters (need to check
            // again after the `if`, because of `parse_ident`
            // consuming more tokens).
            let (bra, ket) = match token::close_delimiter_for(&self.token) {
                Some(ket) => (self.token.clone(), ket),
                None      => {
                    // we only expect an ident if we didn't parse one
                    // above.
                    let ident_str = if id.name == token::special_idents::invalid.name {
                        "identifier, "
                    } else {
                        ""
                    };
                    let tok_str = self.this_token_to_string();
                    self.fatal(format!("expected {}`(` or `{{`, found `{}`",
                                       ident_str,
                                       tok_str).as_slice())
                }
            };

            let tts = self.parse_unspanned_seq(
                &bra,
                &ket,
                seq_sep_none(),
                |p| p.parse_token_tree()
            );
            let hi = self.span.hi;

            if id.name == token::special_idents::invalid.name {
                return box(GC) spanned(lo, hi, StmtMac(
                    spanned(lo, hi, MacInvocTT(pth, tts, EMPTY_CTXT)), false));
            } else {
                // if it has a special ident, it's definitely an item
                return box(GC) spanned(lo, hi, StmtDecl(
                    box(GC) spanned(lo, hi, DeclItem(
                        self.mk_item(
                            lo, hi, id /*id is good here*/,
                            ItemMac(spanned(lo, hi, MacInvocTT(pth, tts, EMPTY_CTXT))),
                            Inherited, Vec::new(/*no attrs*/)))),
                    ast::DUMMY_NODE_ID));
            }

        } else {
            let found_attrs = !item_attrs.is_empty();
            match self.parse_item_or_view_item(item_attrs, false) {
                IoviItem(i) => {
                    let hi = i.span.hi;
                    let decl = box(GC) spanned(lo, hi, DeclItem(i));
                    return box(GC) spanned(lo, hi, StmtDecl(decl, ast::DUMMY_NODE_ID));
                }
                IoviViewItem(vi) => {
                    self.span_fatal(vi.span,
                                    "view items must be declared at the top of the block");
                }
                IoviForeignItem(_) => {
                    self.fatal("foreign items are not allowed here");
                }
                IoviNone(_) => { /* fallthrough */ }
            }

            check_expected_item(self, found_attrs);

            // Remainder are line-expr stmts.
            let e = self.parse_expr_res(RESTRICT_STMT_EXPR);
            return box(GC) spanned(lo, e.span.hi, StmtExpr(e, ast::DUMMY_NODE_ID));
        }
    }

    /// Is this expression a successfully-parsed statement?
    fn expr_is_complete(&mut self, e: Gc<Expr>) -> bool {
        return self.restriction == RESTRICT_STMT_EXPR &&
            !classify::expr_requires_semi_to_be_stmt(e);
    }

    /// Parse a block. No inner attrs are allowed.
    pub fn parse_block(&mut self) -> P<Block> {
        maybe_whole!(no_clone self, NtBlock);

        let lo = self.span.lo;
        self.expect(&token::LBRACE);

        return self.parse_block_tail_(lo, DefaultBlock, Vec::new());
    }

    /// Parse a block. Inner attrs are allowed.
    fn parse_inner_attrs_and_block(&mut self)
        -> (Vec<Attribute> , P<Block>) {

        maybe_whole!(pair_empty self, NtBlock);

        let lo = self.span.lo;
        self.expect(&token::LBRACE);
        let (inner, next) = self.parse_inner_attrs_and_next();

        (inner, self.parse_block_tail_(lo, DefaultBlock, next))
    }

    /// Precondition: already parsed the '{' or '#{'
    /// I guess that also means "already parsed the 'impure'" if
    /// necessary, and this should take a qualifier.
    /// Some blocks start with "#{"...
    fn parse_block_tail(&mut self, lo: BytePos, s: BlockCheckMode) -> P<Block> {
        self.parse_block_tail_(lo, s, Vec::new())
    }

    /// Parse the rest of a block expression or function body
    fn parse_block_tail_(&mut self, lo: BytePos, s: BlockCheckMode,
                         first_item_attrs: Vec<Attribute> ) -> P<Block> {
        let mut stmts = Vec::new();
        let mut expr = None;

        // wouldn't it be more uniform to parse view items only, here?
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: items,
            ..
        } = self.parse_items_and_view_items(first_item_attrs,
                                            false, false);

        for item in items.iter() {
            let decl = box(GC) spanned(item.span.lo, item.span.hi, DeclItem(*item));
            stmts.push(box(GC) spanned(item.span.lo, item.span.hi,
                                StmtDecl(decl, ast::DUMMY_NODE_ID)));
        }

        let mut attributes_box = attrs_remaining;

        while self.token != token::RBRACE {
            // parsing items even when they're not allowed lets us give
            // better error messages and recover more gracefully.
            attributes_box.push_all(self.parse_outer_attributes().as_slice());
            match self.token {
                token::SEMI => {
                    if !attributes_box.is_empty() {
                        let last_span = self.last_span;
                        self.span_err(last_span, "expected item after attributes");
                        attributes_box = Vec::new();
                    }
                    self.bump(); // empty
                }
                token::RBRACE => {
                    // fall through and out.
                }
                _ => {
                    let stmt = self.parse_stmt(attributes_box);
                    attributes_box = Vec::new();
                    match stmt.node {
                        StmtExpr(e, stmt_id) => {
                            // expression without semicolon
                            if classify::stmt_ends_with_semi(&*stmt) {
                                // Just check for errors and recover; do not eat semicolon yet.
                                self.commit_stmt(stmt, &[], &[token::SEMI, token::RBRACE]);
                            }

                            match self.token {
                                token::SEMI => {
                                    self.bump();
                                    let span_with_semi = Span {
                                        lo: stmt.span.lo,
                                        hi: self.last_span.hi,
                                        expn_info: stmt.span.expn_info,
                                    };
                                    stmts.push(box(GC) codemap::Spanned {
                                        node: StmtSemi(e, stmt_id),
                                        span: span_with_semi,
                                    });
                                }
                                token::RBRACE => {
                                    expr = Some(e);
                                }
                                _ => {
                                    stmts.push(stmt);
                                }
                            }
                        }
                        StmtMac(ref m, _) => {
                            // statement macro; might be an expr
                            match self.token {
                                token::SEMI => {
                                    self.bump();
                                    stmts.push(box(GC) codemap::Spanned {
                                        node: StmtMac((*m).clone(), true),
                                        span: stmt.span,
                                    });
                                }
                                token::RBRACE => {
                                    // if a block ends in `m!(arg)` without
                                    // a `;`, it must be an expr
                                    expr = Some(
                                        self.mk_mac_expr(stmt.span.lo,
                                                         stmt.span.hi,
                                                         m.node.clone()));
                                }
                                _ => {
                                    stmts.push(stmt);
                                }
                            }
                        }
                        _ => { // all other kinds of statements:
                            stmts.push(stmt.clone());

                            if classify::stmt_ends_with_semi(&*stmt) {
                                self.commit_stmt_expecting(stmt, token::SEMI);
                            }
                        }
                    }
                }
            }
        }

        if !attributes_box.is_empty() {
            let last_span = self.last_span;
            self.span_err(last_span, "expected item after attributes");
        }

        let hi = self.span.hi;
        self.bump();
        P(ast::Block {
            view_items: view_items,
            stmts: stmts,
            expr: expr,
            id: ast::DUMMY_NODE_ID,
            rules: s,
            span: mk_sp(lo, hi),
        })
    }

    fn parse_unboxed_function_type(&mut self) -> UnboxedFnTy {
        let (optional_unboxed_closure_kind, inputs) =
            if self.eat(&token::OROR) {
                (None, Vec::new())
            } else {
                self.expect_or();

                let optional_unboxed_closure_kind =
                    self.parse_optional_unboxed_closure_kind();

                let inputs = self.parse_seq_to_before_or(&token::COMMA,
                                                         |p| {
                    p.parse_arg_general(false)
                });
                self.expect_or();
                (optional_unboxed_closure_kind, inputs)
            };

        let (return_style, output) = self.parse_ret_ty();
        UnboxedFnTy {
            decl: P(FnDecl {
                inputs: inputs,
                output: output,
                cf: return_style,
                variadic: false,
            }),
            kind: match optional_unboxed_closure_kind {
                Some(kind) => kind,
                None => FnMutUnboxedClosureKind,
            },
        }
    }

    /// matches optbounds = ( ( : ( boundseq )? )? )
    /// where   boundseq  = ( bound + boundseq ) | bound
    /// and     bound     = 'static | ty
    /// Returns "None" if there's no colon (e.g. "T");
    /// Returns "Some(Empty)" if there's a colon but nothing after (e.g. "T:")
    /// Returns "Some(stuff)" otherwise (e.g. "T:stuff").
    /// NB: The None/Some distinction is important for issue #7264.
    ///
    /// Note that the `allow_any_lifetime` argument is a hack for now while the
    /// AST doesn't support arbitrary lifetimes in bounds on type parameters. In
    /// the future, this flag should be removed, and the return value of this
    /// function should be Option<~[TyParamBound]>
    fn parse_ty_param_bounds(&mut self, allow_any_lifetime: bool)
                             -> (Option<ast::Lifetime>,
                                 OwnedSlice<TyParamBound>) {
        let mut ret_lifetime = None;
        let mut result = vec!();
        loop {
            match self.token {
                token::LIFETIME(lifetime) => {
                    let lifetime_interned_string = token::get_ident(lifetime);
                    if lifetime_interned_string.equiv(&("'static")) {
                        result.push(StaticRegionTyParamBound);
                        if allow_any_lifetime && ret_lifetime.is_none() {
                            ret_lifetime = Some(ast::Lifetime {
                                id: ast::DUMMY_NODE_ID,
                                span: self.span,
                                name: lifetime.name
                            });
                        }
                    } else if allow_any_lifetime && ret_lifetime.is_none() {
                        ret_lifetime = Some(ast::Lifetime {
                            id: ast::DUMMY_NODE_ID,
                            span: self.span,
                            name: lifetime.name
                        });
                    } else {
                        result.push(OtherRegionTyParamBound(self.span));
                    }
                    self.bump();
                }
                token::MOD_SEP | token::IDENT(..) => {
                    let tref = self.parse_trait_ref();
                    result.push(TraitTyParamBound(tref));
                }
                token::BINOP(token::OR) | token::OROR => {
                    let unboxed_function_type =
                        self.parse_unboxed_function_type();
                    result.push(UnboxedFnTyParamBound(unboxed_function_type));
                }
                _ => break,
            }

            if !self.eat(&token::BINOP(token::PLUS)) {
                break;
            }
        }

        return (ret_lifetime, OwnedSlice::from_vec(result));
    }

    fn trait_ref_from_ident(ident: Ident, span: Span) -> ast::TraitRef {
        let segment = ast::PathSegment {
            identifier: ident,
            lifetimes: Vec::new(),
            types: OwnedSlice::empty(),
        };
        let path = ast::Path {
            span: span,
            global: false,
            segments: vec![segment],
        };
        ast::TraitRef {
            path: path,
            ref_id: ast::DUMMY_NODE_ID,
        }
    }

    /// Matches typaram = (unbound`?`)? IDENT optbounds ( EQ ty )?
    fn parse_ty_param(&mut self) -> TyParam {
        // This is a bit hacky. Currently we are only interested in a single
        // unbound, and it may only be `Sized`. To avoid backtracking and other
        // complications, we parse an ident, then check for `?`. If we find it,
        // we use the ident as the unbound, otherwise, we use it as the name of
        // type param.
        let mut span = self.span;
        let mut ident = self.parse_ident();
        let mut unbound = None;
        if self.eat(&token::QUESTION) {
            let tref = Parser::trait_ref_from_ident(ident, span);
            unbound = Some(TraitTyParamBound(tref));
            span = self.span;
            ident = self.parse_ident();
        }

        let opt_bounds = {
            if self.eat(&token::COLON) {
                let (_, bounds) = self.parse_ty_param_bounds(false);
                Some(bounds)
            } else {
                None
            }
        };
        // For typarams we don't care about the difference b/w "<T>" and "<T:>".
        let bounds = opt_bounds.unwrap_or_default();

        let default = if self.token == token::EQ {
            self.bump();
            Some(self.parse_ty(true))
        }
        else { None };

        TyParam {
            ident: ident,
            id: ast::DUMMY_NODE_ID,
            bounds: bounds,
            unbound: unbound,
            default: default,
            span: span,
        }
    }

    /// Parse a set of optional generic type parameter declarations. Where
    /// clauses are not parsed here, and must be added later via
    /// `parse_where_clause()`.
    ///
    /// matches generics = ( ) | ( < > ) | ( < typaramseq ( , )? > ) | ( < lifetimes ( , )? > )
    ///                  | ( < lifetimes , typaramseq ( , )? > )
    /// where   typaramseq = ( typaram ) | ( typaram , typaramseq )
    pub fn parse_generics(&mut self) -> ast::Generics {
        if self.eat(&token::LT) {
            let lifetime_defs = self.parse_lifetime_defs();
            let mut seen_default = false;
            let ty_params = self.parse_seq_to_gt(Some(token::COMMA), |p| {
                p.forbid_lifetime();
                let ty_param = p.parse_ty_param();
                if ty_param.default.is_some() {
                    seen_default = true;
                } else if seen_default {
                    let last_span = p.last_span;
                    p.span_err(last_span,
                               "type parameters with a default must be trailing");
                }
                ty_param
            });
            ast::Generics {
                lifetimes: lifetime_defs,
                ty_params: ty_params,
                where_clause: WhereClause {
                    id: ast::DUMMY_NODE_ID,
                    predicates: Vec::new(),
                }
            }
        } else {
            ast_util::empty_generics()
        }
    }

    fn parse_generic_values_after_lt(&mut self) -> (Vec<ast::Lifetime>, Vec<P<Ty>> ) {
        let lifetimes = self.parse_lifetimes(token::COMMA);
        let result = self.parse_seq_to_gt(
            Some(token::COMMA),
            |p| {
                p.forbid_lifetime();
                p.parse_ty(true)
            }
        );
        (lifetimes, result.into_vec())
    }

    fn forbid_lifetime(&mut self) {
        if Parser::token_is_lifetime(&self.token) {
            let span = self.span;
            self.span_fatal(span, "lifetime parameters must be declared \
                                        prior to type parameters");
        }
    }

    /// Parses an optional `where` clause and places it in `generics`.
    fn parse_where_clause(&mut self, generics: &mut ast::Generics) {
        if !self.eat_keyword(keywords::Where) {
            return
        }

        let mut parsed_something = false;
        loop {
            let lo = self.span.lo;
            let ident = match self.token {
                token::IDENT(..) => self.parse_ident(),
                _ => break,
            };
            self.expect(&token::COLON);

            let (_, bounds) = self.parse_ty_param_bounds(false);
            let hi = self.span.hi;
            let span = mk_sp(lo, hi);

            if bounds.len() == 0 {
                self.span_err(span,
                              "each predicate in a `where` clause must have \
                               at least one bound in it");
            }

            generics.where_clause.predicates.push(ast::WherePredicate {
                id: ast::DUMMY_NODE_ID,
                span: span,
                ident: ident,
                bounds: bounds,
            });
            parsed_something = true;

            if !self.eat(&token::COMMA) {
                break
            }
        }

        if !parsed_something {
            let last_span = self.last_span;
            self.span_err(last_span,
                          "a `where` clause must have at least one predicate \
                           in it");
        }
    }

    fn parse_fn_args(&mut self, named_args: bool, allow_variadic: bool)
                     -> (Vec<Arg> , bool) {
        let sp = self.span;
        let mut args: Vec<Option<Arg>> =
            self.parse_unspanned_seq(
                &token::LPAREN,
                &token::RPAREN,
                seq_sep_trailing_allowed(token::COMMA),
                |p| {
                    if p.token == token::DOTDOTDOT {
                        p.bump();
                        if allow_variadic {
                            if p.token != token::RPAREN {
                                let span = p.span;
                                p.span_fatal(span,
                                    "`...` must be last in argument list for variadic function");
                            }
                        } else {
                            let span = p.span;
                            p.span_fatal(span,
                                         "only foreign functions are allowed to be variadic");
                        }
                        None
                    } else {
                        Some(p.parse_arg_general(named_args))
                    }
                }
            );

        let variadic = match args.pop() {
            Some(None) => true,
            Some(x) => {
                // Need to put back that last arg
                args.push(x);
                false
            }
            None => false
        };

        if variadic && args.is_empty() {
            self.span_err(sp,
                          "variadic function must be declared with at least one named argument");
        }

        let args = args.move_iter().map(|x| x.unwrap()).collect();

        (args, variadic)
    }

    /// Parse the argument list and result type of a function declaration
    pub fn parse_fn_decl(&mut self, allow_variadic: bool) -> P<FnDecl> {

        let (args, variadic) = self.parse_fn_args(true, allow_variadic);
        let (ret_style, ret_ty) = self.parse_ret_ty();

        P(FnDecl {
            inputs: args,
            output: ret_ty,
            cf: ret_style,
            variadic: variadic
        })
    }

    fn is_self_ident(&mut self) -> bool {
        match self.token {
          token::IDENT(id, false) => id.name == special_idents::self_.name,
          _ => false
        }
    }

    fn expect_self_ident(&mut self) -> ast::Ident {
        match self.token {
            token::IDENT(id, false) if id.name == special_idents::self_.name => {
                self.bump();
                id
            },
            _ => {
                let token_str = self.this_token_to_string();
                self.fatal(format!("expected `self`, found `{}`",
                                   token_str).as_slice())
            }
        }
    }

    /// Parse the argument list and result type of a function
    /// that may have a self type.
    fn parse_fn_decl_with_self(&mut self, parse_arg_fn: |&mut Parser| -> Arg)
                               -> (ExplicitSelf, P<FnDecl>) {
        fn maybe_parse_borrowed_explicit_self(this: &mut Parser)
                                              -> ast::ExplicitSelf_ {
            // The following things are possible to see here:
            //
            //     fn(&mut self)
            //     fn(&mut self)
            //     fn(&'lt self)
            //     fn(&'lt mut self)
            //
            // We already know that the current token is `&`.

            if this.look_ahead(1, |t| token::is_keyword(keywords::Self, t)) {
                this.bump();
                SelfRegion(None, MutImmutable, this.expect_self_ident())
            } else if this.look_ahead(1, |t| Parser::token_is_mutability(t)) &&
                    this.look_ahead(2,
                                    |t| token::is_keyword(keywords::Self,
                                                          t)) {
                this.bump();
                let mutability = this.parse_mutability();
                SelfRegion(None, mutability, this.expect_self_ident())
            } else if this.look_ahead(1, |t| Parser::token_is_lifetime(t)) &&
                       this.look_ahead(2,
                                       |t| token::is_keyword(keywords::Self,
                                                             t)) {
                this.bump();
                let lifetime = this.parse_lifetime();
                SelfRegion(Some(lifetime), MutImmutable, this.expect_self_ident())
            } else if this.look_ahead(1, |t| Parser::token_is_lifetime(t)) &&
                      this.look_ahead(2, |t| {
                          Parser::token_is_mutability(t)
                      }) &&
                      this.look_ahead(3, |t| token::is_keyword(keywords::Self,
                                                               t)) {
                this.bump();
                let lifetime = this.parse_lifetime();
                let mutability = this.parse_mutability();
                SelfRegion(Some(lifetime), mutability, this.expect_self_ident())
            } else {
                SelfStatic
            }
        }

        self.expect(&token::LPAREN);

        // A bit of complexity and lookahead is needed here in order to be
        // backwards compatible.
        let lo = self.span.lo;
        let mut mutbl_self = MutImmutable;
        let explicit_self = match self.token {
            token::BINOP(token::AND) => {
                maybe_parse_borrowed_explicit_self(self)
            }
            token::TILDE => {
                // We need to make sure it isn't a type
                if self.look_ahead(1, |t| token::is_keyword(keywords::Self, t)) {
                    self.bump();
                    drop(self.expect_self_ident());
                    let last_span = self.last_span;
                    self.obsolete(last_span, ObsoleteOwnedSelf)
                }
                SelfStatic
            }
            token::BINOP(token::STAR) => {
                // Possibly "*self" or "*mut self" -- not supported. Try to avoid
                // emitting cryptic "unexpected token" errors.
                self.bump();
                let _mutability = if Parser::token_is_mutability(&self.token) {
                    self.parse_mutability()
                } else {
                    MutImmutable
                };
                if self.is_self_ident() {
                    let span = self.span;
                    self.span_err(span, "cannot pass self by unsafe pointer");
                    self.bump();
                }
                // error case, making bogus self ident:
                SelfValue(special_idents::self_)
            }
            token::IDENT(..) => {
                if self.is_self_ident() {
                    let self_ident = self.expect_self_ident();

                    // Determine whether this is the fully explicit form, `self:
                    // TYPE`.
                    if self.eat(&token::COLON) {
                        SelfExplicit(self.parse_ty(false), self_ident)
                    } else {
                        SelfValue(self_ident)
                    }
                } else if Parser::token_is_mutability(&self.token) &&
                        self.look_ahead(1, |t| {
                            token::is_keyword(keywords::Self, t)
                        }) {
                    mutbl_self = self.parse_mutability();
                    let self_ident = self.expect_self_ident();

                    // Determine whether this is the fully explicit form,
                    // `self: TYPE`.
                    if self.eat(&token::COLON) {
                        SelfExplicit(self.parse_ty(false), self_ident)
                    } else {
                        SelfValue(self_ident)
                    }
                } else if Parser::token_is_mutability(&self.token) &&
                        self.look_ahead(1, |t| *t == token::TILDE) &&
                        self.look_ahead(2, |t| {
                            token::is_keyword(keywords::Self, t)
                        }) {
                    mutbl_self = self.parse_mutability();
                    self.bump();
                    drop(self.expect_self_ident());
                    let last_span = self.last_span;
                    self.obsolete(last_span, ObsoleteOwnedSelf);
                    SelfStatic
                } else {
                    SelfStatic
                }
            }
            _ => SelfStatic,
        };

        let explicit_self_sp = mk_sp(lo, self.span.hi);

        // shared fall-through for the three cases below. borrowing prevents simply
        // writing this as a closure
        macro_rules! parse_remaining_arguments {
            ($self_id:ident) =>
            {
            // If we parsed a self type, expect a comma before the argument list.
            match self.token {
                token::COMMA => {
                    self.bump();
                    let sep = seq_sep_trailing_allowed(token::COMMA);
                    let mut fn_inputs = self.parse_seq_to_before_end(
                        &token::RPAREN,
                        sep,
                        parse_arg_fn
                    );
                    fn_inputs.unshift(Arg::new_self(explicit_self_sp, mutbl_self, $self_id));
                    fn_inputs
                }
                token::RPAREN => {
                    vec!(Arg::new_self(explicit_self_sp, mutbl_self, $self_id))
                }
                _ => {
                    let token_str = self.this_token_to_string();
                    self.fatal(format!("expected `,` or `)`, found `{}`",
                                       token_str).as_slice())
                }
            }
            }
        }

        let fn_inputs = match explicit_self {
            SelfStatic =>  {
                let sep = seq_sep_trailing_allowed(token::COMMA);
                self.parse_seq_to_before_end(&token::RPAREN, sep, parse_arg_fn)
            }
            SelfValue(id) => parse_remaining_arguments!(id),
            SelfRegion(_,_,id) => parse_remaining_arguments!(id),
            SelfExplicit(_,id) => parse_remaining_arguments!(id),
        };


        self.expect(&token::RPAREN);

        let hi = self.span.hi;

        let (ret_style, ret_ty) = self.parse_ret_ty();

        let fn_decl = P(FnDecl {
            inputs: fn_inputs,
            output: ret_ty,
            cf: ret_style,
            variadic: false
        });

        (spanned(lo, hi, explicit_self), fn_decl)
    }

    // parse the |arg, arg| header on a lambda
    fn parse_fn_block_decl(&mut self)
                           -> (P<FnDecl>, Option<UnboxedClosureKind>) {
        let (optional_unboxed_closure_kind, inputs_captures) = {
            if self.eat(&token::OROR) {
                (None, Vec::new())
            } else {
                self.expect(&token::BINOP(token::OR));
                let optional_unboxed_closure_kind =
                    self.parse_optional_unboxed_closure_kind();
                let args = self.parse_seq_to_before_end(
                    &token::BINOP(token::OR),
                    seq_sep_trailing_allowed(token::COMMA),
                    |p| p.parse_fn_block_arg()
                );
                self.bump();
                (optional_unboxed_closure_kind, args)
            }
        };
        let output = if self.eat(&token::RARROW) {
            self.parse_ty(true)
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyInfer,
                span: self.span,
            })
        };

        (P(FnDecl {
            inputs: inputs_captures,
            output: output,
            cf: Return,
            variadic: false
        }), optional_unboxed_closure_kind)
    }

    /// Parses the `(arg, arg) -> return_type` header on a procedure.
    fn parse_proc_decl(&mut self) -> P<FnDecl> {
        let inputs =
            self.parse_unspanned_seq(&token::LPAREN,
                                     &token::RPAREN,
                                     seq_sep_trailing_allowed(token::COMMA),
                                     |p| p.parse_fn_block_arg());

        let output = if self.eat(&token::RARROW) {
            self.parse_ty(true)
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyInfer,
                span: self.span,
            })
        };

        P(FnDecl {
            inputs: inputs,
            output: output,
            cf: Return,
            variadic: false
        })
    }

    /// Parse the name and optional generic types of a function header.
    fn parse_fn_header(&mut self) -> (Ident, ast::Generics) {
        let id = self.parse_ident();
        let generics = self.parse_generics();
        (id, generics)
    }

    fn mk_item(&mut self, lo: BytePos, hi: BytePos, ident: Ident,
               node: Item_, vis: Visibility,
               attrs: Vec<Attribute>) -> Gc<Item> {
        box(GC) Item {
            ident: ident,
            attrs: attrs,
            id: ast::DUMMY_NODE_ID,
            node: node,
            vis: vis,
            span: mk_sp(lo, hi)
        }
    }

    /// Parse an item-position function declaration.
    fn parse_item_fn(&mut self, fn_style: FnStyle, abi: abi::Abi) -> ItemInfo {
        let (ident, mut generics) = self.parse_fn_header();
        let decl = self.parse_fn_decl(false);
        self.parse_where_clause(&mut generics);
        let (inner_attrs, body) = self.parse_inner_attrs_and_block();
        (ident, ItemFn(decl, fn_style, abi, generics, body), Some(inner_attrs))
    }

    /// Parse a method in a trait impl, starting with `attrs` attributes.
    pub fn parse_method(&mut self,
                        already_parsed_attrs: Option<Vec<Attribute>>)
                        -> Gc<Method> {
        let next_attrs = self.parse_outer_attributes();
        let attrs = match already_parsed_attrs {
            Some(mut a) => { a.push_all_move(next_attrs); a }
            None => next_attrs
        };

        let lo = self.span.lo;

        // code copied from parse_macro_use_or_failure... abstraction!
        let (method_, hi, new_attrs) = {
            if !token::is_any_keyword(&self.token)
                && self.look_ahead(1, |t| *t == token::NOT)
                && (self.look_ahead(2, |t| *t == token::LPAREN)
                    || self.look_ahead(2, |t| *t == token::LBRACE)) {
                // method macro.
                let pth = self.parse_path(NoTypesAllowed).path;
                self.expect(&token::NOT);

                // eat a matched-delimiter token tree:
                let tts = match token::close_delimiter_for(&self.token) {
                    Some(ket) => {
                        self.bump();
                        self.parse_seq_to_end(&ket,
                                              seq_sep_none(),
                                              |p| p.parse_token_tree())
                    }
                    None => self.fatal("expected open delimiter")
                };
                let m_ = ast::MacInvocTT(pth, tts, EMPTY_CTXT);
                let m: ast::Mac = codemap::Spanned { node: m_,
                                                 span: mk_sp(self.span.lo,
                                                             self.span.hi) };
                (ast::MethMac(m), self.span.hi, attrs)
            } else {
                let visa = self.parse_visibility();
                let abi = if self.eat_keyword(keywords::Extern) {
                    self.parse_opt_abi().unwrap_or(abi::C)
                } else if attr::contains_name(attrs.as_slice(),
                                              "rust_call_abi_hack") {
                    // FIXME(stage0, pcwalton): Remove this awful hack after a
                    // snapshot, and change to `extern "rust-call" fn`.
                    abi::RustCall
                } else {
                    abi::Rust
                };
                let fn_style = self.parse_fn_style();
                let ident = self.parse_ident();
                let mut generics = self.parse_generics();
                let (explicit_self, decl) = self.parse_fn_decl_with_self(|p| {
                        p.parse_arg()
                    });
                self.parse_where_clause(&mut generics);
                let (inner_attrs, body) = self.parse_inner_attrs_and_block();
                let new_attrs = attrs.append(inner_attrs.as_slice());
                (ast::MethDecl(ident,
                               generics,
                               abi,
                               explicit_self,
                               fn_style,
                               decl,
                               body,
                               visa),
                 body.span.hi, new_attrs)
            }
        };
        box(GC) ast::Method {
            attrs: new_attrs,
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, hi),
            node: method_,
        }
    }

    /// Parse trait Foo { ... }
    fn parse_item_trait(&mut self) -> ItemInfo {
        let ident = self.parse_ident();
        let mut tps = self.parse_generics();
        let sized = self.parse_for_sized();

        // Parse traits, if necessary.
        let traits;
        if self.token == token::COLON {
            self.bump();
            traits = self.parse_trait_ref_list(&token::LBRACE);
        } else {
            traits = Vec::new();
        }

        self.parse_where_clause(&mut tps);

        let meths = self.parse_trait_methods();
        (ident, ItemTrait(tps, sized, traits, meths), None)
    }

    fn parse_impl_items(&mut self) -> (Vec<ImplItem>, Vec<Attribute>) {
        let mut impl_items = Vec::new();
        self.expect(&token::LBRACE);
        let (inner_attrs, next) = self.parse_inner_attrs_and_next();
        let mut method_attrs = Some(next);
        while !self.eat(&token::RBRACE) {
            impl_items.push(MethodImplItem(self.parse_method(method_attrs)));
            method_attrs = None;
        }
        (impl_items, inner_attrs)
    }

    /// Parses two variants (with the region/type params always optional):
    ///    impl<T> Foo { ... }
    ///    impl<T> ToString for ~[T] { ... }
    fn parse_item_impl(&mut self) -> ItemInfo {
        // First, parse type parameters if necessary.
        let mut generics = self.parse_generics();

        // Special case: if the next identifier that follows is '(', don't
        // allow this to be parsed as a trait.
        let could_be_trait = self.token != token::LPAREN;

        // Parse the trait.
        let mut ty = self.parse_ty(true);

        // Parse traits, if necessary.
        let opt_trait = if could_be_trait && self.eat_keyword(keywords::For) {
            // New-style trait. Reinterpret the type as a trait.
            let opt_trait_ref = match ty.node {
                TyPath(ref path, None, node_id) => {
                    Some(TraitRef {
                        path: /* bad */ (*path).clone(),
                        ref_id: node_id
                    })
                }
                TyPath(..) => {
                    self.span_err(ty.span,
                                  "bounded traits are only valid in type position");
                    None
                }
                _ => {
                    self.span_err(ty.span, "not a trait");
                    None
                }
            };

            ty = self.parse_ty(true);
            opt_trait_ref
        } else {
            None
        };

        self.parse_where_clause(&mut generics);
        let (impl_items, attrs) = self.parse_impl_items();

        let ident = ast_util::impl_pretty_name(&opt_trait, &*ty);

        (ident,
         ItemImpl(generics, opt_trait, ty, impl_items),
         Some(attrs))
    }

    /// Parse a::B<String,int>
    fn parse_trait_ref(&mut self) -> TraitRef {
        ast::TraitRef {
            path: self.parse_path(LifetimeAndTypesWithoutColons).path,
            ref_id: ast::DUMMY_NODE_ID,
        }
    }

    /// Parse B + C<String,int> + D
    fn parse_trait_ref_list(&mut self, ket: &token::Token) -> Vec<TraitRef> {
        self.parse_seq_to_before_end(
            ket,
            seq_sep_trailing_disallowed(token::BINOP(token::PLUS)),
            |p| p.parse_trait_ref()
        )
    }

    /// Parse struct Foo { ... }
    fn parse_item_struct(&mut self, is_virtual: bool) -> ItemInfo {
        let class_name = self.parse_ident();
        let mut generics = self.parse_generics();

        let super_struct = if self.eat(&token::COLON) {
            let ty = self.parse_ty(true);
            match ty.node {
                TyPath(_, None, _) => {
                    Some(ty)
                }
                _ => {
                    self.span_err(ty.span, "not a struct");
                    None
                }
            }
        } else {
            None
        };

        self.parse_where_clause(&mut generics);

        let mut fields: Vec<StructField>;
        let is_tuple_like;

        if self.eat(&token::LBRACE) {
            // It's a record-like struct.
            is_tuple_like = false;
            fields = Vec::new();
            while self.token != token::RBRACE {
                fields.push(self.parse_struct_decl_field());
            }
            if fields.len() == 0 {
                self.fatal(format!("unit-like struct definition should be \
                                    written as `struct {};`",
                                   token::get_ident(class_name)).as_slice());
            }
            self.bump();
        } else if self.token == token::LPAREN {
            // It's a tuple-like struct.
            is_tuple_like = true;
            fields = self.parse_unspanned_seq(
                &token::LPAREN,
                &token::RPAREN,
                seq_sep_trailing_allowed(token::COMMA),
                |p| {
                let attrs = p.parse_outer_attributes();
                let lo = p.span.lo;
                let struct_field_ = ast::StructField_ {
                    kind: UnnamedField(p.parse_visibility()),
                    id: ast::DUMMY_NODE_ID,
                    ty: p.parse_ty(true),
                    attrs: attrs,
                };
                spanned(lo, p.span.hi, struct_field_)
            });
            if fields.len() == 0 {
                self.fatal(format!("unit-like struct definition should be \
                                    written as `struct {};`",
                                   token::get_ident(class_name)).as_slice());
            }
            self.expect(&token::SEMI);
        } else if self.eat(&token::SEMI) {
            // It's a unit-like struct.
            is_tuple_like = true;
            fields = Vec::new();
        } else {
            let token_str = self.this_token_to_string();
            self.fatal(format!("expected `{}`, `(`, or `;` after struct \
                                name, found `{}`", "{",
                               token_str).as_slice())
        }

        let _ = ast::DUMMY_NODE_ID;  // FIXME: Workaround for crazy bug.
        let new_id = ast::DUMMY_NODE_ID;
        (class_name,
         ItemStruct(box(GC) ast::StructDef {
             fields: fields,
             ctor_id: if is_tuple_like { Some(new_id) } else { None },
             super_struct: super_struct,
             is_virtual: is_virtual,
         }, generics),
         None)
    }

    /// Parse a structure field declaration
    pub fn parse_single_struct_field(&mut self,
                                     vis: Visibility,
                                     attrs: Vec<Attribute> )
                                     -> StructField {
        let a_var = self.parse_name_and_ty(vis, attrs);
        match self.token {
            token::COMMA => {
                self.bump();
            }
            token::RBRACE => {}
            _ => {
                let span = self.span;
                let token_str = self.this_token_to_string();
                self.span_fatal(span,
                                format!("expected `,`, or `}}`, found `{}`",
                                        token_str).as_slice())
            }
        }
        a_var
    }

    /// Parse an element of a struct definition
    fn parse_struct_decl_field(&mut self) -> StructField {

        let attrs = self.parse_outer_attributes();

        if self.eat_keyword(keywords::Pub) {
           return self.parse_single_struct_field(Public, attrs);
        }

        return self.parse_single_struct_field(Inherited, attrs);
    }

    /// Parse visibility: PUB, PRIV, or nothing
    fn parse_visibility(&mut self) -> Visibility {
        if self.eat_keyword(keywords::Pub) { Public }
        else { Inherited }
    }

    fn parse_for_sized(&mut self) -> Option<ast::TyParamBound> {
        if self.eat_keyword(keywords::For) {
            let span = self.span;
            let ident = self.parse_ident();
            if !self.eat(&token::QUESTION) {
                self.span_err(span,
                    "expected 'Sized?' after `for` in trait item");
                return None;
            }
            let tref = Parser::trait_ref_from_ident(ident, span);
            Some(TraitTyParamBound(tref))
        } else {
            None
        }
    }

    /// Given a termination token and a vector of already-parsed
    /// attributes (of length 0 or 1), parse all of the items in a module
    fn parse_mod_items(&mut self,
                       term: token::Token,
                       first_item_attrs: Vec<Attribute>,
                       inner_lo: BytePos)
                       -> Mod {
        // parse all of the items up to closing or an attribute.
        // view items are legal here.
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: starting_items,
            ..
        } = self.parse_items_and_view_items(first_item_attrs, true, true);
        let mut items: Vec<Gc<Item>> = starting_items;
        let attrs_remaining_len = attrs_remaining.len();

        // don't think this other loop is even necessary....

        let mut first = true;
        while self.token != term {
            let mut attrs = self.parse_outer_attributes();
            if first {
                attrs = attrs_remaining.clone().append(attrs.as_slice());
                first = false;
            }
            debug!("parse_mod_items: parse_item_or_view_item(attrs={:?})",
                   attrs);
            match self.parse_item_or_view_item(attrs,
                                               true /* macros allowed */) {
              IoviItem(item) => items.push(item),
              IoviViewItem(view_item) => {
                self.span_fatal(view_item.span,
                                "view items must be declared at the top of \
                                 the module");
              }
              _ => {
                  let token_str = self.this_token_to_string();
                  self.fatal(format!("expected item, found `{}`",
                                     token_str).as_slice())
              }
            }
        }

        if first && attrs_remaining_len > 0u {
            // We parsed attributes for the first item but didn't find it
            let last_span = self.last_span;
            self.span_err(last_span, "expected item after attributes");
        }

        ast::Mod {
            inner: mk_sp(inner_lo, self.span.lo),
            view_items: view_items,
            items: items
        }
    }

    fn parse_item_const(&mut self) -> ItemInfo {
        let m = if self.eat_keyword(keywords::Mut) {MutMutable} else {MutImmutable};
        let id = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(true);
        self.expect(&token::EQ);
        let e = self.parse_expr();
        self.commit_expr_expecting(e, token::SEMI);
        (id, ItemStatic(ty, m, e), None)
    }

    /// Parse a `mod <foo> { ... }` or `mod <foo>;` item
    fn parse_item_mod(&mut self, outer_attrs: &[Attribute]) -> ItemInfo {
        let id_span = self.span;
        let id = self.parse_ident();
        if self.token == token::SEMI {
            self.bump();
            // This mod is in an external file. Let's go get it!
            let (m, attrs) = self.eval_src_mod(id, outer_attrs, id_span);
            (id, m, Some(attrs))
        } else {
            self.push_mod_path(id, outer_attrs);
            self.expect(&token::LBRACE);
            let mod_inner_lo = self.span.lo;
            let old_owns_directory = self.owns_directory;
            self.owns_directory = true;
            let (inner, next) = self.parse_inner_attrs_and_next();
            let m = self.parse_mod_items(token::RBRACE, next, mod_inner_lo);
            self.expect(&token::RBRACE);
            self.owns_directory = old_owns_directory;
            self.pop_mod_path();
            (id, ItemMod(m), Some(inner))
        }
    }

    fn push_mod_path(&mut self, id: Ident, attrs: &[Attribute]) {
        let default_path = self.id_to_interned_str(id);
        let file_path = match ::attr::first_attr_value_str_by_name(attrs,
                                                                   "path") {
            Some(d) => d,
            None => default_path,
        };
        self.mod_path_stack.push(file_path)
    }

    fn pop_mod_path(&mut self) {
        self.mod_path_stack.pop().unwrap();
    }

    /// Read a module from a source file.
    fn eval_src_mod(&mut self,
                    id: ast::Ident,
                    outer_attrs: &[ast::Attribute],
                    id_sp: Span)
                    -> (ast::Item_, Vec<ast::Attribute> ) {
        let mut prefix = Path::new(self.sess.span_diagnostic.cm.span_to_filename(self.span));
        prefix.pop();
        let mod_path = Path::new(".").join_many(self.mod_path_stack.as_slice());
        let dir_path = prefix.join(&mod_path);
        let mod_string = token::get_ident(id);
        let (file_path, owns_directory) = match ::attr::first_attr_value_str_by_name(
                outer_attrs, "path") {
            Some(d) => (dir_path.join(d), true),
            None => {
                let mod_name = mod_string.get().to_string();
                let default_path_str = format!("{}.rs", mod_name);
                let secondary_path_str = format!("{}/mod.rs", mod_name);
                let default_path = dir_path.join(default_path_str.as_slice());
                let secondary_path = dir_path.join(secondary_path_str.as_slice());
                let default_exists = default_path.exists();
                let secondary_exists = secondary_path.exists();

                if !self.owns_directory {
                    self.span_err(id_sp,
                                  "cannot declare a new module at this location");
                    let this_module = match self.mod_path_stack.last() {
                        Some(name) => name.get().to_string(),
                        None => self.root_module_name.get_ref().clone(),
                    };
                    self.span_note(id_sp,
                                   format!("maybe move this module `{0}` \
                                            to its own directory via \
                                            `{0}/mod.rs`",
                                           this_module).as_slice());
                    if default_exists || secondary_exists {
                        self.span_note(id_sp,
                                       format!("... or maybe `use` the module \
                                                `{}` instead of possibly \
                                                redeclaring it",
                                               mod_name).as_slice());
                    }
                    self.abort_if_errors();
                }

                match (default_exists, secondary_exists) {
                    (true, false) => (default_path, false),
                    (false, true) => (secondary_path, true),
                    (false, false) => {
                        self.span_fatal(id_sp,
                                        format!("file not found for module \
                                                 `{}`",
                                                 mod_name).as_slice());
                    }
                    (true, true) => {
                        self.span_fatal(
                            id_sp,
                            format!("file for module `{}` found at both {} \
                                     and {}",
                                    mod_name,
                                    default_path_str,
                                    secondary_path_str).as_slice());
                    }
                }
            }
        };

        self.eval_src_mod_from_path(file_path, owns_directory,
                                    mod_string.get().to_string(), id_sp)
    }

    fn eval_src_mod_from_path(&mut self,
                              path: Path,
                              owns_directory: bool,
                              name: String,
                              id_sp: Span) -> (ast::Item_, Vec<ast::Attribute> ) {
        let mut included_mod_stack = self.sess.included_mod_stack.borrow_mut();
        match included_mod_stack.iter().position(|p| *p == path) {
            Some(i) => {
                let mut err = String::from_str("circular modules: ");
                let len = included_mod_stack.len();
                for p in included_mod_stack.slice(i, len).iter() {
                    err.push_str(p.display().as_maybe_owned().as_slice());
                    err.push_str(" -> ");
                }
                err.push_str(path.display().as_maybe_owned().as_slice());
                self.span_fatal(id_sp, err.as_slice());
            }
            None => ()
        }
        included_mod_stack.push(path.clone());
        drop(included_mod_stack);

        let mut p0 =
            new_sub_parser_from_file(self.sess,
                                     self.cfg.clone(),
                                     &path,
                                     owns_directory,
                                     Some(name),
                                     id_sp);
        let mod_inner_lo = p0.span.lo;
        let (mod_attrs, next) = p0.parse_inner_attrs_and_next();
        let first_item_outer_attrs = next;
        let m0 = p0.parse_mod_items(token::EOF, first_item_outer_attrs, mod_inner_lo);
        self.sess.included_mod_stack.borrow_mut().pop();
        return (ast::ItemMod(m0), mod_attrs);
    }

    /// Parse a function declaration from a foreign module
    fn parse_item_foreign_fn(&mut self, vis: ast::Visibility,
                             attrs: Vec<Attribute>) -> Gc<ForeignItem> {
        let lo = self.span.lo;
        self.expect_keyword(keywords::Fn);

        let (ident, mut generics) = self.parse_fn_header();
        let decl = self.parse_fn_decl(true);
        self.parse_where_clause(&mut generics);
        let hi = self.span.hi;
        self.expect(&token::SEMI);
        box(GC) ast::ForeignItem { ident: ident,
                                   attrs: attrs,
                                   node: ForeignItemFn(decl, generics),
                                   id: ast::DUMMY_NODE_ID,
                                   span: mk_sp(lo, hi),
                                   vis: vis }
    }

    /// Parse a static item from a foreign module
    fn parse_item_foreign_static(&mut self, vis: ast::Visibility,
                                 attrs: Vec<Attribute> ) -> Gc<ForeignItem> {
        let lo = self.span.lo;

        self.expect_keyword(keywords::Static);
        let mutbl = self.eat_keyword(keywords::Mut);

        let ident = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(true);
        let hi = self.span.hi;
        self.expect(&token::SEMI);
        box(GC) ast::ForeignItem {
            ident: ident,
            attrs: attrs,
            node: ForeignItemStatic(ty, mutbl),
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, hi),
            vis: vis,
        }
    }

    /// Parse safe/unsafe and fn
    fn parse_fn_style(&mut self) -> FnStyle {
        if self.eat_keyword(keywords::Fn) { NormalFn }
        else if self.eat_keyword(keywords::Unsafe) {
            self.expect_keyword(keywords::Fn);
            UnsafeFn
        }
        else { self.unexpected(); }
    }


    /// At this point, this is essentially a wrapper for
    /// parse_foreign_items.
    fn parse_foreign_mod_items(&mut self,
                               abi: abi::Abi,
                               first_item_attrs: Vec<Attribute> )
                               -> ForeignMod {
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: _,
            foreign_items: foreign_items
        } = self.parse_foreign_items(first_item_attrs, true);
        if ! attrs_remaining.is_empty() {
            let last_span = self.last_span;
            self.span_err(last_span,
                          "expected item after attributes");
        }
        assert!(self.token == token::RBRACE);
        ast::ForeignMod {
            abi: abi,
            view_items: view_items,
            items: foreign_items
        }
    }

    /// Parse extern crate links
    ///
    /// # Example
    ///
    /// extern crate url;
    /// extern crate foo = "bar";
    fn parse_item_extern_crate(&mut self,
                                lo: BytePos,
                                visibility: Visibility,
                                attrs: Vec<Attribute> )
                                -> ItemOrViewItem {

        let (maybe_path, ident) = match self.token {
            token::IDENT(..) => {
                let the_ident = self.parse_ident();
                self.expect_one_of(&[], &[token::EQ, token::SEMI]);
                let path = if self.token == token::EQ {
                    self.bump();
                    Some(self.parse_str())
                } else {None};

                self.expect(&token::SEMI);
                (path, the_ident)
            }
            _ => {
                let span = self.span;
                let token_str = self.this_token_to_string();
                self.span_fatal(span,
                                format!("expected extern crate name but \
                                         found `{}`",
                                        token_str).as_slice());
            }
        };

        IoviViewItem(ast::ViewItem {
                node: ViewItemExternCrate(ident, maybe_path, ast::DUMMY_NODE_ID),
                attrs: attrs,
                vis: visibility,
                span: mk_sp(lo, self.last_span.hi)
            })
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
                              lo: BytePos,
                              opt_abi: Option<abi::Abi>,
                              visibility: Visibility,
                              attrs: Vec<Attribute> )
                              -> ItemOrViewItem {

        self.expect(&token::LBRACE);

        let abi = opt_abi.unwrap_or(abi::C);

        let (inner, next) = self.parse_inner_attrs_and_next();
        let m = self.parse_foreign_mod_items(abi, next);
        self.expect(&token::RBRACE);

        let last_span = self.last_span;
        let item = self.mk_item(lo,
                                last_span.hi,
                                special_idents::invalid,
                                ItemForeignMod(m),
                                visibility,
                                maybe_append(attrs, Some(inner)));
        return IoviItem(item);
    }

    /// Parse type Foo = Bar;
    fn parse_item_type(&mut self) -> ItemInfo {
        let ident = self.parse_ident();
        let mut tps = self.parse_generics();
        self.parse_where_clause(&mut tps);
        self.expect(&token::EQ);
        let ty = self.parse_ty(true);
        self.expect(&token::SEMI);
        (ident, ItemTy(ty, tps), None)
    }

    /// Parse a structure-like enum variant definition
    /// this should probably be renamed or refactored...
    fn parse_struct_def(&mut self) -> Gc<StructDef> {
        let mut fields: Vec<StructField> = Vec::new();
        while self.token != token::RBRACE {
            fields.push(self.parse_struct_decl_field());
        }
        self.bump();

        return box(GC) ast::StructDef {
            fields: fields,
            ctor_id: None,
            super_struct: None,
            is_virtual: false,
        };
    }

    /// Parse the part of an "enum" decl following the '{'
    fn parse_enum_def(&mut self, _generics: &ast::Generics) -> EnumDef {
        let mut variants = Vec::new();
        let mut all_nullary = true;
        let mut have_disr = false;
        while self.token != token::RBRACE {
            let variant_attrs = self.parse_outer_attributes();
            let vlo = self.span.lo;

            let vis = self.parse_visibility();

            let ident;
            let kind;
            let mut args = Vec::new();
            let mut disr_expr = None;
            ident = self.parse_ident();
            if self.eat(&token::LBRACE) {
                // Parse a struct variant.
                all_nullary = false;
                kind = StructVariantKind(self.parse_struct_def());
            } else if self.token == token::LPAREN {
                all_nullary = false;
                let arg_tys = self.parse_enum_variant_seq(
                    &token::LPAREN,
                    &token::RPAREN,
                    seq_sep_trailing_allowed(token::COMMA),
                    |p| p.parse_ty(true)
                );
                for ty in arg_tys.move_iter() {
                    args.push(ast::VariantArg {
                        ty: ty,
                        id: ast::DUMMY_NODE_ID,
                    });
                }
                kind = TupleVariantKind(args);
            } else if self.eat(&token::EQ) {
                have_disr = true;
                disr_expr = Some(self.parse_expr());
                kind = TupleVariantKind(args);
            } else {
                kind = TupleVariantKind(Vec::new());
            }

            let vr = ast::Variant_ {
                name: ident,
                attrs: variant_attrs,
                kind: kind,
                id: ast::DUMMY_NODE_ID,
                disr_expr: disr_expr,
                vis: vis,
            };
            variants.push(P(spanned(vlo, self.last_span.hi, vr)));

            if !self.eat(&token::COMMA) { break; }
        }
        self.expect(&token::RBRACE);
        if have_disr && !all_nullary {
            self.fatal("discriminator values can only be used with a c-like \
                        enum");
        }

        ast::EnumDef { variants: variants }
    }

    /// Parse an "enum" declaration
    fn parse_item_enum(&mut self) -> ItemInfo {
        let id = self.parse_ident();
        let mut generics = self.parse_generics();
        self.parse_where_clause(&mut generics);
        self.expect(&token::LBRACE);

        let enum_definition = self.parse_enum_def(&generics);
        (id, ItemEnum(enum_definition, generics), None)
    }

    fn fn_expr_lookahead(tok: &token::Token) -> bool {
        match *tok {
          token::LPAREN | token::AT | token::TILDE | token::BINOP(_) => true,
          _ => false
        }
    }

    /// Parses a string as an ABI spec on an extern type or module. Consumes
    /// the `extern` keyword, if one is found.
    fn parse_opt_abi(&mut self) -> Option<abi::Abi> {
        match self.token {
            token::LIT_STR(s) | token::LIT_STR_RAW(s, _) => {
                self.bump();
                let the_string = s.as_str();
                match abi::lookup(the_string) {
                    Some(abi) => Some(abi),
                    None => {
                        let last_span = self.last_span;
                        self.span_err(
                            last_span,
                            format!("illegal ABI: expected one of [{}], \
                                     found `{}`",
                                    abi::all_names().connect(", "),
                                    the_string).as_slice());
                        None
                    }
                }
            }

            _ => None,
        }
    }

    /// Parse one of the items or view items allowed by the
    /// flags; on failure, return IoviNone.
    /// NB: this function no longer parses the items inside an
    /// extern crate.
    fn parse_item_or_view_item(&mut self,
                               attrs: Vec<Attribute> ,
                               macros_allowed: bool)
                               -> ItemOrViewItem {
        match self.token {
            INTERPOLATED(token::NtItem(item)) => {
                self.bump();
                let new_attrs = attrs.append(item.attrs.as_slice());
                return IoviItem(box(GC) Item {
                    attrs: new_attrs,
                    ..(*item).clone()
                });
            }
            _ => {}
        }

        let lo = self.span.lo;

        let visibility = self.parse_visibility();

        // must be a view item:
        if self.eat_keyword(keywords::Use) {
            // USE ITEM (IoviViewItem)
            let view_item = self.parse_use();
            self.expect(&token::SEMI);
            return IoviViewItem(ast::ViewItem {
                node: view_item,
                attrs: attrs,
                vis: visibility,
                span: mk_sp(lo, self.last_span.hi)
            });
        }
        // either a view item or an item:
        if self.eat_keyword(keywords::Extern) {
            let next_is_mod = self.eat_keyword(keywords::Mod);

            if next_is_mod || self.eat_keyword(keywords::Crate) {
                if next_is_mod {
                    let last_span = self.last_span;
                    self.span_err(mk_sp(lo, last_span.hi),
                                 format!("`extern mod` is obsolete, use \
                                          `extern crate` instead \
                                          to refer to external \
                                          crates.").as_slice())
                }
                return self.parse_item_extern_crate(lo, visibility, attrs);
            }

            let opt_abi = self.parse_opt_abi();

            if self.eat_keyword(keywords::Fn) {
                // EXTERN FUNCTION ITEM
                let abi = opt_abi.unwrap_or(abi::C);
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(NormalFn, abi);
                let last_span = self.last_span;
                let item = self.mk_item(lo,
                                        last_span.hi,
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return IoviItem(item);
            } else if self.token == token::LBRACE {
                return self.parse_item_foreign_mod(lo, opt_abi, visibility, attrs);
            }

            let span = self.span;
            let token_str = self.this_token_to_string();
            self.span_fatal(span,
                            format!("expected `{}` or `fn`, found `{}`", "{",
                                    token_str).as_slice());
        }

        let is_virtual = self.eat_keyword(keywords::Virtual);
        if is_virtual && !self.is_keyword(keywords::Struct) {
            let span = self.span;
            self.span_err(span,
                          "`virtual` keyword may only be used with `struct`");
        }

        // the rest are all guaranteed to be items:
        if self.is_keyword(keywords::Static) {
            // STATIC ITEM
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_const();
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.is_keyword(keywords::Fn) &&
                self.look_ahead(1, |f| !Parser::fn_expr_lookahead(f)) {
            // FUNCTION ITEM
            self.bump();
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(NormalFn, abi::Rust);
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.is_keyword(keywords::Unsafe)
            && self.look_ahead(1u, |t| *t != token::LBRACE) {
            // UNSAFE FUNCTION ITEM
            self.bump();
            let abi = if self.eat_keyword(keywords::Extern) {
                self.parse_opt_abi().unwrap_or(abi::C)
            } else {
                abi::Rust
            };
            self.expect_keyword(keywords::Fn);
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(UnsafeFn, abi);
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Mod) {
            // MODULE ITEM
            let (ident, item_, extra_attrs) =
                self.parse_item_mod(attrs.as_slice());
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Type) {
            // TYPE ITEM
            let (ident, item_, extra_attrs) = self.parse_item_type();
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Enum) {
            // ENUM ITEM
            let (ident, item_, extra_attrs) = self.parse_item_enum();
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Trait) {
            // TRAIT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_trait();
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Impl) {
            // IMPL ITEM
            let (ident, item_, extra_attrs) = self.parse_item_impl();
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Struct) {
            // STRUCT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_struct(is_virtual);
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        self.parse_macro_use_or_failure(attrs,macros_allowed,lo,visibility)
    }

    /// Parse a foreign item; on failure, return IoviNone.
    fn parse_foreign_item(&mut self,
                          attrs: Vec<Attribute> ,
                          macros_allowed: bool)
                          -> ItemOrViewItem {
        maybe_whole!(iovi self, NtItem);
        let lo = self.span.lo;

        let visibility = self.parse_visibility();

        if self.is_keyword(keywords::Static) {
            // FOREIGN STATIC ITEM
            let item = self.parse_item_foreign_static(visibility, attrs);
            return IoviForeignItem(item);
        }
        if self.is_keyword(keywords::Fn) || self.is_keyword(keywords::Unsafe) {
            // FOREIGN FUNCTION ITEM
            let item = self.parse_item_foreign_fn(visibility, attrs);
            return IoviForeignItem(item);
        }
        self.parse_macro_use_or_failure(attrs,macros_allowed,lo,visibility)
    }

    /// This is the fall-through for parsing items.
    fn parse_macro_use_or_failure(
        &mut self,
        attrs: Vec<Attribute> ,
        macros_allowed: bool,
        lo: BytePos,
        visibility: Visibility
    ) -> ItemOrViewItem {
        if macros_allowed && !token::is_any_keyword(&self.token)
                && self.look_ahead(1, |t| *t == token::NOT)
                && (self.look_ahead(2, |t| is_plain_ident(t))
                    || self.look_ahead(2, |t| *t == token::LPAREN)
                    || self.look_ahead(2, |t| *t == token::LBRACE)) {
            // MACRO INVOCATION ITEM

            // item macro.
            let pth = self.parse_path(NoTypesAllowed).path;
            self.expect(&token::NOT);

            // a 'special' identifier (like what `macro_rules!` uses)
            // is optional. We should eventually unify invoc syntax
            // and remove this.
            let id = if is_plain_ident(&self.token) {
                self.parse_ident()
            } else {
                token::special_idents::invalid // no special identifier
            };
            // eat a matched-delimiter token tree:
            let tts = match token::close_delimiter_for(&self.token) {
                Some(ket) => {
                    self.bump();
                    self.parse_seq_to_end(&ket,
                                          seq_sep_none(),
                                          |p| p.parse_token_tree())
                }
                None => self.fatal("expected open delimiter")
            };
            // single-variant-enum... :
            let m = ast::MacInvocTT(pth, tts, EMPTY_CTXT);
            let m: ast::Mac = codemap::Spanned { node: m,
                                             span: mk_sp(self.span.lo,
                                                         self.span.hi) };
            let item_ = ItemMac(m);
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    id,
                                    item_,
                                    visibility,
                                    attrs);
            return IoviItem(item);
        }

        // FAILURE TO PARSE ITEM
        if visibility != Inherited {
            let mut s = String::from_str("unmatched visibility `");
            if visibility == Public {
                s.push_str("pub")
            } else {
                s.push_str("priv")
            }
            s.push_char('`');
            let last_span = self.last_span;
            self.span_fatal(last_span, s.as_slice());
        }
        return IoviNone(attrs);
    }

    pub fn parse_item_with_outer_attributes(&mut self) -> Option<Gc<Item>> {
        let attrs = self.parse_outer_attributes();
        self.parse_item(attrs)
    }

    pub fn parse_item(&mut self, attrs: Vec<Attribute> ) -> Option<Gc<Item>> {
        match self.parse_item_or_view_item(attrs, true) {
            IoviNone(_) => None,
            IoviViewItem(_) =>
                self.fatal("view items are not allowed here"),
            IoviForeignItem(_) =>
                self.fatal("foreign items are not allowed here"),
            IoviItem(item) => Some(item)
        }
    }

    /// Parse, e.g., "use a::b::{z,y}"
    fn parse_use(&mut self) -> ViewItem_ {
        return ViewItemUse(self.parse_view_path());
    }


    /// Matches view_path : MOD? IDENT EQ non_global_path
    /// | MOD? non_global_path MOD_SEP LBRACE RBRACE
    /// | MOD? non_global_path MOD_SEP LBRACE ident_seq RBRACE
    /// | MOD? non_global_path MOD_SEP STAR
    /// | MOD? non_global_path
    fn parse_view_path(&mut self) -> Gc<ViewPath> {
        let lo = self.span.lo;

        if self.token == token::LBRACE {
            // use {foo,bar}
            let idents = self.parse_unspanned_seq(
                &token::LBRACE, &token::RBRACE,
                seq_sep_trailing_allowed(token::COMMA),
                |p| p.parse_path_list_item());
            let path = ast::Path {
                span: mk_sp(lo, self.span.hi),
                global: false,
                segments: Vec::new()
            };
            return box(GC) spanned(lo, self.span.hi,
                            ViewPathList(path, idents, ast::DUMMY_NODE_ID));
        }

        let first_ident = self.parse_ident();
        let mut path = vec!(first_ident);
        match self.token {
          token::EQ => {
            // x = foo::bar
            self.bump();
            let path_lo = self.span.lo;
            path = vec!(self.parse_ident());
            while self.token == token::MOD_SEP {
                self.bump();
                let id = self.parse_ident();
                path.push(id);
            }
            let span = mk_sp(path_lo, self.span.hi);
            self.obsolete(span, ObsoleteImportRenaming);
            let path = ast::Path {
                span: span,
                global: false,
                segments: path.move_iter().map(|identifier| {
                    ast::PathSegment {
                        identifier: identifier,
                        lifetimes: Vec::new(),
                        types: OwnedSlice::empty(),
                    }
                }).collect()
            };
            return box(GC) spanned(lo, self.span.hi,
                            ViewPathSimple(first_ident, path,
                                           ast::DUMMY_NODE_ID));
          }

          token::MOD_SEP => {
            // foo::bar or foo::{a,b,c} or foo::*
            while self.token == token::MOD_SEP {
                self.bump();

                match self.token {
                  token::IDENT(i, _) => {
                    self.bump();
                    path.push(i);
                  }

                  // foo::bar::{a,b,c}
                  token::LBRACE => {
                    let idents = self.parse_unspanned_seq(
                        &token::LBRACE,
                        &token::RBRACE,
                        seq_sep_trailing_allowed(token::COMMA),
                        |p| p.parse_path_list_item()
                    );
                    let path = ast::Path {
                        span: mk_sp(lo, self.span.hi),
                        global: false,
                        segments: path.move_iter().map(|identifier| {
                            ast::PathSegment {
                                identifier: identifier,
                                lifetimes: Vec::new(),
                                types: OwnedSlice::empty(),
                            }
                        }).collect()
                    };
                    return box(GC) spanned(lo, self.span.hi,
                                    ViewPathList(path, idents, ast::DUMMY_NODE_ID));
                  }

                  // foo::bar::*
                  token::BINOP(token::STAR) => {
                    self.bump();
                    let path = ast::Path {
                        span: mk_sp(lo, self.span.hi),
                        global: false,
                        segments: path.move_iter().map(|identifier| {
                            ast::PathSegment {
                                identifier: identifier,
                                lifetimes: Vec::new(),
                                types: OwnedSlice::empty(),
                            }
                        }).collect()
                    };
                    return box(GC) spanned(lo, self.span.hi,
                                    ViewPathGlob(path, ast::DUMMY_NODE_ID));
                  }

                  _ => break
                }
            }
          }
          _ => ()
        }
        let mut rename_to = *path.get(path.len() - 1u);
        let path = ast::Path {
            span: mk_sp(lo, self.span.hi),
            global: false,
            segments: path.move_iter().map(|identifier| {
                ast::PathSegment {
                    identifier: identifier,
                    lifetimes: Vec::new(),
                    types: OwnedSlice::empty(),
                }
            }).collect()
        };
        if self.eat_keyword(keywords::As) {
            rename_to = self.parse_ident()
        }
        return box(GC) spanned(lo,
                        self.last_span.hi,
                        ViewPathSimple(rename_to, path, ast::DUMMY_NODE_ID));
    }

    /// Parses a sequence of items. Stops when it finds program
    /// text that can't be parsed as an item
    /// - mod_items uses extern_mod_allowed = true
    /// - block_tail_ uses extern_mod_allowed = false
    fn parse_items_and_view_items(&mut self,
                                  first_item_attrs: Vec<Attribute> ,
                                  mut extern_mod_allowed: bool,
                                  macros_allowed: bool)
                                  -> ParsedItemsAndViewItems {
        let mut attrs = first_item_attrs.append(self.parse_outer_attributes().as_slice());
        // First, parse view items.
        let mut view_items : Vec<ast::ViewItem> = Vec::new();
        let mut items = Vec::new();

        // I think this code would probably read better as a single
        // loop with a mutable three-state-variable (for extern crates,
        // view items, and regular items) ... except that because
        // of macros, I'd like to delay that entire check until later.
        loop {
            match self.parse_item_or_view_item(attrs, macros_allowed) {
                IoviNone(attrs) => {
                    return ParsedItemsAndViewItems {
                        attrs_remaining: attrs,
                        view_items: view_items,
                        items: items,
                        foreign_items: Vec::new()
                    }
                }
                IoviViewItem(view_item) => {
                    match view_item.node {
                        ViewItemUse(..) => {
                            // `extern crate` must precede `use`.
                            extern_mod_allowed = false;
                        }
                        ViewItemExternCrate(..) if !extern_mod_allowed => {
                            self.span_err(view_item.span,
                                          "\"extern crate\" declarations are \
                                           not allowed here");
                        }
                        ViewItemExternCrate(..) => {}
                    }
                    view_items.push(view_item);
                }
                IoviItem(item) => {
                    items.push(item);
                    attrs = self.parse_outer_attributes();
                    break;
                }
                IoviForeignItem(_) => {
                    fail!();
                }
            }
            attrs = self.parse_outer_attributes();
        }

        // Next, parse items.
        loop {
            match self.parse_item_or_view_item(attrs, macros_allowed) {
                IoviNone(returned_attrs) => {
                    attrs = returned_attrs;
                    break
                }
                IoviViewItem(view_item) => {
                    attrs = self.parse_outer_attributes();
                    self.span_err(view_item.span,
                                  "`use` and `extern crate` declarations must precede items");
                }
                IoviItem(item) => {
                    attrs = self.parse_outer_attributes();
                    items.push(item)
                }
                IoviForeignItem(_) => {
                    fail!();
                }
            }
        }

        ParsedItemsAndViewItems {
            attrs_remaining: attrs,
            view_items: view_items,
            items: items,
            foreign_items: Vec::new()
        }
    }

    /// Parses a sequence of foreign items. Stops when it finds program
    /// text that can't be parsed as an item
    fn parse_foreign_items(&mut self, first_item_attrs: Vec<Attribute> ,
                           macros_allowed: bool)
        -> ParsedItemsAndViewItems {
        let mut attrs = first_item_attrs.append(self.parse_outer_attributes().as_slice());
        let mut foreign_items = Vec::new();
        loop {
            match self.parse_foreign_item(attrs, macros_allowed) {
                IoviNone(returned_attrs) => {
                    if self.token == token::RBRACE {
                        attrs = returned_attrs;
                        break
                    }
                    self.unexpected();
                },
                IoviViewItem(view_item) => {
                    // I think this can't occur:
                    self.span_err(view_item.span,
                                  "`use` and `extern crate` declarations must precede items");
                }
                IoviItem(item) => {
                    // FIXME #5668: this will occur for a macro invocation:
                    self.span_fatal(item.span, "macros cannot expand to foreign items");
                }
                IoviForeignItem(foreign_item) => {
                    foreign_items.push(foreign_item);
                }
            }
            attrs = self.parse_outer_attributes();
        }

        ParsedItemsAndViewItems {
            attrs_remaining: attrs,
            view_items: Vec::new(),
            items: Vec::new(),
            foreign_items: foreign_items
        }
    }

    /// Parses a source module as a crate. This is the main
    /// entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> Crate {
        let lo = self.span.lo;
        // parse the crate's inner attrs, maybe (oops) one
        // of the attrs of an item:
        let (inner, next) = self.parse_inner_attrs_and_next();
        let first_item_outer_attrs = next;
        // parse the items inside the crate:
        let m = self.parse_mod_items(token::EOF, first_item_outer_attrs, lo);

        ast::Crate {
            module: m,
            attrs: inner,
            config: self.cfg.clone(),
            span: mk_sp(lo, self.span.lo),
            exported_macros: Vec::new(),
        }
    }

    pub fn parse_optional_str(&mut self)
                              -> Option<(InternedString, ast::StrStyle)> {
        let (s, style) = match self.token {
            token::LIT_STR(s) => (self.id_to_interned_str(s.ident()), ast::CookedStr),
            token::LIT_STR_RAW(s, n) => {
                (self.id_to_interned_str(s.ident()), ast::RawStr(n))
            }
            _ => return None
        };
        self.bump();
        Some((s, style))
    }

    pub fn parse_str(&mut self) -> (InternedString, StrStyle) {
        match self.parse_optional_str() {
            Some(s) => { s }
            _ =>  self.fatal("expected string literal")
        }
    }
}

