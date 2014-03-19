// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_escape];

use abi;
use abi::AbiSet;
use ast::{Sigil, BorrowedSigil, ManagedSigil, OwnedSigil};
use ast::{BareFnTy, ClosureTy};
use ast::{RegionTyParamBound, TraitTyParamBound};
use ast::{Provided, Public, Purity};
use ast::{Mod, BiAdd, Arg, Arm, Attribute, BindByRef, BindByValue};
use ast::{BiBitAnd, BiBitOr, BiBitXor, Block};
use ast::{BlockCheckMode, UnBox};
use ast::{Crate, CrateConfig, Decl, DeclItem};
use ast::{DeclLocal, DefaultBlock, UnDeref, BiDiv, EMPTY_CTXT, EnumDef, ExplicitSelf};
use ast::{Expr, Expr_, ExprAddrOf, ExprMatch, ExprAgain};
use ast::{ExprAssign, ExprAssignOp, ExprBinary, ExprBlock, ExprBox};
use ast::{ExprBreak, ExprCall, ExprCast};
use ast::{ExprField, ExprFnBlock, ExprIf, ExprIndex};
use ast::{ExprLit, ExprLoop, ExprMac};
use ast::{ExprMethodCall, ExprParen, ExprPath, ExprProc};
use ast::{ExprRepeat, ExprRet, ExprStruct, ExprTup, ExprUnary};
use ast::{ExprVec, ExprVstore, ExprVstoreSlice};
use ast::{ExprVstoreMutSlice, ExprWhile, ExprForLoop, ExternFn, Field, FnDecl};
use ast::{ExprVstoreUniq, Onceness, Once, Many};
use ast::{ForeignItem, ForeignItemStatic, ForeignItemFn, ForeignMod};
use ast::{Ident, ImpureFn, Inherited, Item, Item_, ItemStatic};
use ast::{ItemEnum, ItemFn, ItemForeignMod, ItemImpl};
use ast::{ItemMac, ItemMod, ItemStruct, ItemTrait, ItemTy, Lit, Lit_};
use ast::{LitBool, LitFloat, LitFloatUnsuffixed, LitInt, LitChar};
use ast::{LitIntUnsuffixed, LitNil, LitStr, LitUint, Local};
use ast::{MutImmutable, MutMutable, Mac_, MacInvocTT, Matcher, MatchNonterminal};
use ast::{MatchSeq, MatchTok, Method, MutTy, BiMul, Mutability};
use ast::{NamedField, UnNeg, NoReturn, UnNot, P, Pat, PatEnum};
use ast::{PatIdent, PatLit, PatRange, PatRegion, PatStruct};
use ast::{PatTup, PatUniq, PatWild, PatWildMulti, Private};
use ast::{BiRem, Required};
use ast::{RetStyle, Return, BiShl, BiShr, Stmt, StmtDecl};
use ast::{StmtExpr, StmtSemi, StmtMac, StructDef, StructField};
use ast::{StructVariantKind, BiSub};
use ast::StrStyle;
use ast::{SelfRegion, SelfStatic, SelfUniq, SelfValue};
use ast::{TokenTree, TraitMethod, TraitRef, TTDelim, TTSeq, TTTok};
use ast::{TTNonterminal, TupleVariantKind, Ty, Ty_, TyBot, TyBox};
use ast::{TypeField, TyFixedLengthVec, TyClosure, TyBareFn, TyTypeof};
use ast::{TyInfer, TypeMethod};
use ast::{TyNil, TyParam, TyParamBound, TyPath, TyPtr, TyRptr};
use ast::{TyTup, TyU32, TyUniq, TyVec, UnUniq};
use ast::{UnnamedField, UnsafeBlock, UnsafeFn, ViewItem};
use ast::{ViewItem_, ViewItemExternCrate, ViewItemUse};
use ast::{ViewPath, ViewPathGlob, ViewPathList, ViewPathSimple};
use ast::Visibility;
use ast;
use ast_util::{as_prec, lit_is_str, operator_prec};
use ast_util;
use codemap::{Span, BytePos, Spanned, spanned, mk_sp};
use codemap;
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
use opt_vec;
use opt_vec::OptVec;

use std::cell::Cell;
use collections::HashSet;
use std::kinds::marker;
use std::mem::replace;
use std::vec_ng::Vec;
use std::vec_ng;

#[allow(non_camel_case_types)]
#[deriving(Eq)]
pub enum restriction {
    UNRESTRICTED,
    RESTRICT_STMT_EXPR,
    RESTRICT_NO_BAR_OP,
    RESTRICT_NO_BAR_OR_DOUBLEBAR_OP,
}

type ItemInfo = (Ident, Item_, Option<Vec<Attribute> >);

/// How to parse a path. There are four different kinds of paths, all of which
/// are parsed somewhat differently.
#[deriving(Eq)]
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
    /// set of type parameters only; e.g. `foo::bar<'a>::Baz:X+Y<T>` This
    /// form does not use extra double colons.
    LifetimeAndTypesAndBounds,
}

/// A pair of a path segment and group of type parameter bounds. (See `ast.rs`
/// for the definition of a path segment.)
struct PathSegmentAndBoundSet {
    segment: ast::PathSegment,
    bound_set: Option<OptVec<TyParamBound>>,
}

/// A path paired with optional type bounds.
pub struct PathAndBounds {
    path: ast::Path,
    bounds: Option<OptVec<TyParamBound>>,
}

enum ItemOrViewItem {
    // Indicates a failure to parse any kind of item. The attributes are
    // returned.
    IoviNone(Vec<Attribute> ),
    IoviItem(@Item),
    IoviForeignItem(@ForeignItem),
    IoviViewItem(ViewItem)
}

/* The expr situation is not as complex as I thought it would be.
The important thing is to make sure that lookahead doesn't balk
at INTERPOLATED tokens */
macro_rules! maybe_whole_expr (
    ($p:expr) => (
        {
            let mut maybe_path = match ($p).token {
                INTERPOLATED(token::NtPath(ref pt)) => Some((**pt).clone()),
                _ => None,
            };
            let ret = match ($p).token {
                INTERPOLATED(token::NtExpr(e)) => {
                    Some(e)
                }
                INTERPOLATED(token::NtPath(_)) => {
                    let pt = maybe_path.take_unwrap();
                    Some($p.mk_expr(($p).span.lo, ($p).span.hi, ExprPath(pt)))
                }
                _ => None
            };
            match ret {
                Some(e) => {
                    $p.bump();
                    return e;
                }
                None => ()
            }
        }
    )
)

macro_rules! maybe_whole (
    ($p:expr, $constructor:ident) => (
        {
            let __found__ = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match __found__ {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return x.clone()
                }
                _ => {}
            }
        }
    );
    (no_clone $p:expr, $constructor:ident) => (
        {
            let __found__ = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match __found__ {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return x
                }
                _ => {}
            }
        }
    );
    (deref $p:expr, $constructor:ident) => (
        {
            let __found__ = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match __found__ {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return (*x).clone()
                }
                _ => {}
            }
        }
    );
    (Some $p:expr, $constructor:ident) => (
        {
            let __found__ = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match __found__ {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return Some(x.clone()),
                }
                _ => {}
            }
        }
    );
    (iovi $p:expr, $constructor:ident) => (
        {
            let __found__ = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match __found__ {
                Some(INTERPOLATED(token::$constructor(x))) => {
                    return IoviItem(x.clone())
                }
                _ => {}
            }
        }
    );
    (pair_empty $p:expr, $constructor:ident) => (
        {
            let __found__ = match ($p).token {
                INTERPOLATED(token::$constructor(_)) => {
                    Some(($p).bump_and_get())
                }
                _ => None
            };
            match __found__ {
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
        Some(ref attrs) => vec_ng::append(lhs, attrs.as_slice())
    }
}


struct ParsedItemsAndViewItems {
    attrs_remaining: Vec<Attribute> ,
    view_items: Vec<ViewItem> ,
    items: Vec<@Item> ,
    foreign_items: Vec<@ForeignItem> }

/* ident is handled by common.rs */

pub fn Parser<'a>(sess: &'a ParseSess, cfg: ast::CrateConfig, rdr: ~Reader:)
              -> Parser<'a> {
    let tok0 = rdr.next_token();
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
        nopod: marker::NoPod
    }
}

pub struct Parser<'a> {
    sess: &'a ParseSess,
    cfg: CrateConfig,
    // the current token:
    token: token::Token,
    // the span of the current token:
    span: Span,
    // the span of the prior token:
    last_span: Span,
    // the previous token or None (only stashed sometimes).
    last_token: Option<~token::Token>,
    buffer: [TokenAndSpan, ..4],
    buffer_start: int,
    buffer_end: int,
    tokens_consumed: uint,
    restriction: restriction,
    quote_depth: uint, // not (yet) related to the quasiquoter
    reader: ~Reader:,
    interner: @token::IdentInterner,
    /// The set of seen errors about obsolete syntax. Used to suppress
    /// extra detail when the same error is seen twice
    obsolete_set: HashSet<ObsoleteSyntax>,
    /// Used to determine the path to externally loaded source files
    mod_path_stack: Vec<InternedString> ,
    /// Stack of spans of open delimiters. Used for error message.
    open_braces: Vec<Span> ,
    /* do not copy the parser; its state is tied to outside state */
    priv nopod: marker::NoPod
}

fn is_plain_ident_or_underscore(t: &token::Token) -> bool {
    is_plain_ident(t) || *t == token::UNDERSCORE
}

impl<'a> Parser<'a> {
    // convert a token to a string using self's reader
    pub fn token_to_str(token: &token::Token) -> ~str {
        token::to_str(token)
    }

    // convert the current token to a string using self's reader
    pub fn this_token_to_str(&mut self) -> ~str {
        Parser::token_to_str(&self.token)
    }

    pub fn unexpected_last(&mut self, t: &token::Token) -> ! {
        let token_str = Parser::token_to_str(t);
        self.span_fatal(self.last_span, format!("unexpected token: `{}`",
                                                token_str));
    }

    pub fn unexpected(&mut self) -> ! {
        let this_token = self.this_token_to_str();
        self.fatal(format!("unexpected token: `{}`", this_token));
    }

    // expect and consume the token t. Signal an error if
    // the next token is not t.
    pub fn expect(&mut self, t: &token::Token) {
        if self.token == *t {
            self.bump();
        } else {
            let token_str = Parser::token_to_str(t);
            let this_token_str = self.this_token_to_str();
            self.fatal(format!("expected `{}` but found `{}`",
                               token_str,
                               this_token_str))
        }
    }

    // Expect next token to be edible or inedible token.  If edible,
    // then consume it; if inedible, then return without consuming
    // anything.  Signal a fatal error if next token is unexpected.
    pub fn expect_one_of(&mut self,
                         edible: &[token::Token],
                         inedible: &[token::Token]) {
        fn tokens_to_str(tokens: &[token::Token]) -> ~str {
            let mut i = tokens.iter();
            // This might be a sign we need a connect method on Iterator.
            let b = i.next().map_or(~"", |t| Parser::token_to_str(t));
            i.fold(b, |b,a| b + "`, `" + Parser::token_to_str(a))
        }
        if edible.contains(&self.token) {
            self.bump();
        } else if inedible.contains(&self.token) {
            // leave it in the input
        } else {
            let expected = vec_ng::append(edible.iter()
                                                .map(|x| (*x).clone())
                                                .collect(),
                                          inedible);
            let expect = tokens_to_str(expected.as_slice());
            let actual = self.this_token_to_str();
            self.fatal(
                if expected.len() != 1 {
                    format!("expected one of `{}` but found `{}`", expect, actual)
                } else {
                    format!("expected `{}` but found `{}`", expect, actual)
                }
            )
        }
    }

    // Check for erroneous `ident { }`; if matches, signal error and
    // recover (without consuming any expected input token).  Returns
    // true if and only if input was consumed for recovery.
    pub fn check_for_erroneous_unit_struct_expecting(&mut self, expected: &[token::Token]) -> bool {
        if self.token == token::LBRACE
            && expected.iter().all(|t| *t != token::LBRACE)
            && self.look_ahead(1, |t| *t == token::RBRACE) {
            // matched; signal non-fatal error and recover.
            self.span_err(self.span,
                          "unit-like struct construction is written with no trailing `{ }`");
            self.eat(&token::LBRACE);
            self.eat(&token::RBRACE);
            true
        } else {
            false
        }
    }

    // Commit to parsing a complete expression `e` expected to be
    // followed by some token from the set edible + inedible.  Recover
    // from anticipated input errors, discarding erroneous characters.
    pub fn commit_expr(&mut self, e: @Expr, edible: &[token::Token], inedible: &[token::Token]) {
        debug!("commit_expr {:?}", e);
        match e.node {
            ExprPath(..) => {
                // might be unit-struct construction; check for recoverableinput error.
                let expected = vec_ng::append(edible.iter()
                                                    .map(|x| (*x).clone())
                                                    .collect(),
                                              inedible);
                self.check_for_erroneous_unit_struct_expecting(
                    expected.as_slice());
            }
            _ => {}
        }
        self.expect_one_of(edible, inedible)
    }

    pub fn commit_expr_expecting(&mut self, e: @Expr, edible: token::Token) {
        self.commit_expr(e, &[edible], &[])
    }

    // Commit to parsing a complete statement `s`, which expects to be
    // followed by some token from the set edible + inedible.  Check
    // for recoverable input errors, discarding erroneous characters.
    pub fn commit_stmt(&mut self, s: @Stmt, edible: &[token::Token], inedible: &[token::Token]) {
        debug!("commit_stmt {:?}", s);
        let _s = s; // unused, but future checks might want to inspect `s`.
        if self.last_token.as_ref().map_or(false, |t| is_ident_or_path(*t)) {
            let expected = vec_ng::append(edible.iter()
                                                .map(|x| (*x).clone())
                                                .collect(),
                                          inedible.as_slice());
            self.check_for_erroneous_unit_struct_expecting(
                expected.as_slice());
        }
        self.expect_one_of(edible, inedible)
    }

    pub fn commit_stmt_expecting(&mut self, s: @Stmt, edible: token::Token) {
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
                let token_str = self.this_token_to_str();
                self.fatal(format!( "expected ident, found `{}`", token_str))
            }
        }
    }

    pub fn parse_path_list_ident(&mut self) -> ast::PathListIdent {
        let lo = self.span.lo;
        let ident = self.parse_ident();
        let hi = self.last_span.hi;
        spanned(lo, hi, ast::PathListIdent_ { name: ident,
                                              id: ast::DUMMY_NODE_ID })
    }

    // consume token 'tok' if it exists. Returns true if the given
    // token was present, false otherwise.
    pub fn eat(&mut self, tok: &token::Token) -> bool {
        let is_present = self.token == *tok;
        if is_present { self.bump() }
        is_present
    }

    pub fn is_keyword(&mut self, kw: keywords::Keyword) -> bool {
        token::is_keyword(kw, &self.token)
    }

    // if the next token is the given keyword, eat it and return
    // true. Otherwise, return false.
    pub fn eat_keyword(&mut self, kw: keywords::Keyword) -> bool {
        let is_kw = match self.token {
            token::IDENT(sid, false) => kw.to_ident().name == sid.name,
            _ => false
        };
        if is_kw { self.bump() }
        is_kw
    }

    // if the given word is not a keyword, signal an error.
    // if the next token is not the given word, signal an error.
    // otherwise, eat it.
    pub fn expect_keyword(&mut self, kw: keywords::Keyword) {
        if !self.eat_keyword(kw) {
            let id_interned_str = token::get_ident(kw.to_ident());
            let token_str = self.this_token_to_str();
            self.fatal(format!("expected `{}`, found `{}`",
                               id_interned_str, token_str))
        }
    }

    // signal an error if the given string is a strict keyword
    pub fn check_strict_keywords(&mut self) {
        if token::is_strict_keyword(&self.token) {
            let token_str = self.this_token_to_str();
            self.span_err(self.span,
                          format!("found `{}` in ident position", token_str));
        }
    }

    // signal an error if the current token is a reserved keyword
    pub fn check_reserved_keywords(&mut self) {
        if token::is_reserved_keyword(&self.token) {
            let token_str = self.this_token_to_str();
            self.fatal(format!("`{}` is a reserved keyword", token_str))
        }
    }

    // Expect and consume a `|`. If `||` is seen, replace it with a single
    // `|` and continue. If a `|` is not seen, signal an error.
    fn expect_or(&mut self) {
        match self.token {
            token::BINOP(token::OR) => self.bump(),
            token::OROR => {
                let lo = self.span.lo + BytePos(1);
                self.replace_token(token::BINOP(token::OR), lo, self.span.hi)
            }
            _ => {
                let token_str = self.this_token_to_str();
                let found_token =
                    Parser::token_to_str(&token::BINOP(token::OR));
                self.fatal(format!("expected `{}`, found `{}`",
                                   found_token,
                                   token_str))
            }
        }
    }

    // Parse a sequence bracketed by `|` and `|`, stopping before the `|`.
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

    // expect and consume a GT. if a >> is seen, replace it
    // with a single > and continue. If a GT is not seen,
    // signal an error.
    pub fn expect_gt(&mut self) {
        match self.token {
            token::GT => self.bump(),
            token::BINOP(token::SHR) => {
                let lo = self.span.lo + BytePos(1);
                self.replace_token(token::GT, lo, self.span.hi)
            }
            _ => {
                let gt_str = Parser::token_to_str(&token::GT);
                let this_token_str = self.this_token_to_str();
                self.fatal(format!("expected `{}`, found `{}`",
                                   gt_str,
                                   this_token_str))
            }
        }
    }

    // parse a sequence bracketed by '<' and '>', stopping
    // before the '>'.
    pub fn parse_seq_to_before_gt<T>(
                                  &mut self,
                                  sep: Option<token::Token>,
                                  f: |&mut Parser| -> T)
                                  -> OptVec<T> {
        let mut first = true;
        let mut v = opt_vec::Empty;
        while self.token != token::GT
            && self.token != token::BINOP(token::SHR) {
            match sep {
              Some(ref t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            v.push(f(self));
        }
        return v;
    }

    pub fn parse_seq_to_gt<T>(
                           &mut self,
                           sep: Option<token::Token>,
                           f: |&mut Parser| -> T)
                           -> OptVec<T> {
        let v = self.parse_seq_to_before_gt(sep, f);
        self.expect_gt();
        return v;
    }

    // parse a sequence, including the closing delimiter. The function
    // f must consume tokens until reaching the next separator or
    // closing bracket.
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

    // parse a sequence, not including the closing delimiter. The function
    // f must consume tokens until reaching the next separator or
    // closing bracket.
    pub fn parse_seq_to_before_end<T>(
                                   &mut self,
                                   ket: &token::Token,
                                   sep: SeqSep,
                                   f: |&mut Parser| -> T)
                                   -> Vec<T> {
        let mut first: bool = true;
        let mut v: Vec<T> = Vec::new();
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

    // parse a sequence, including the closing delimiter. The function
    // f must consume tokens until reaching the next separator or
    // closing bracket.
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

    // parse a sequence parameter of enum variant. For consistency purposes,
    // these should not be empty.
    pub fn parse_enum_variant_seq<T>(
                               &mut self,
                               bra: &token::Token,
                               ket: &token::Token,
                               sep: SeqSep,
                               f: |&mut Parser| -> T)
                               -> Vec<T> {
        let result = self.parse_unspanned_seq(bra, ket, sep, f);
        if result.is_empty() {
            self.span_err(self.last_span,
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

    // advance the parser by one token
    pub fn bump(&mut self) {
        self.last_span = self.span;
        // Stash token for error recovery (sometimes; clone is not necessarily cheap).
        self.last_token = if is_ident_or_path(&self.token) {
            Some(~self.token.clone())
        } else {
            None
        };
        let next = if self.buffer_start == self.buffer_end {
            self.reader.next_token()
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

    // Advance the parser by one token and return the bumped token.
    pub fn bump_and_get(&mut self) -> token::Token {
        let old_token = replace(&mut self.token, token::UNDERSCORE);
        self.bump();
        old_token
    }

    // EFFECT: replace the current token and span with the given one
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
            self.buffer[self.buffer_end] = self.reader.next_token();
            self.buffer_end = (self.buffer_end + 1) & 3;
        }
        f(&self.buffer[(self.buffer_start + dist - 1) & 3].tok)
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
    pub fn span_err(&mut self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_err(sp, m)
    }
    pub fn abort_if_errors(&mut self) {
        self.sess.span_diagnostic.handler().abort_if_errors();
    }

    pub fn id_to_interned_str(&mut self, id: Ident) -> InternedString {
        token::get_ident(id)
    }

    // Is the current token one of the keywords that signals a bare function
    // type?
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

    // Is the current token one of the keywords that signals a closure type?
    pub fn token_is_closure_keyword(&mut self) -> bool {
        token::is_keyword(keywords::Unsafe, &self.token) ||
            token::is_keyword(keywords::Once, &self.token)
    }

    // Is the current token one of the keywords that signals an old-style
    // closure type (with explicit sigil)?
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

    // parse a TyBareFn type:
    pub fn parse_ty_bare_fn(&mut self) -> Ty_ {
        /*

        [extern "ABI"] [unsafe] fn <'lt> (S) -> T
                ^~~~^  ^~~~~~~^    ^~~~^ ^~^    ^
                  |      |           |    |     |
                  |      |           |    |   Return type
                  |      |           |  Argument types
                  |      |       Lifetimes
                  |      |
                  |    Purity
                 ABI

        */

        let abis = if self.eat_keyword(keywords::Extern) {
            self.parse_opt_abis().unwrap_or(AbiSet::C())
        } else {
            AbiSet::Rust()
        };

        let purity = self.parse_unsafety();
        self.expect_keyword(keywords::Fn);
        let (decl, lifetimes) = self.parse_ty_fn_decl(true);
        return TyBareFn(@BareFnTy {
            abis: abis,
            purity: purity,
            lifetimes: lifetimes,
            decl: decl
        });
    }

    // Parses a procedure type (`proc`). The initial `proc` keyword must
    // already have been parsed.
    pub fn parse_proc_type(&mut self) -> Ty_ {
        let bounds = self.parse_optional_ty_param_bounds();
        let (decl, lifetimes) = self.parse_ty_fn_decl(false);
        TyClosure(@ClosureTy {
            sigil: OwnedSigil,
            region: None,
            purity: ImpureFn,
            onceness: Once,
            bounds: bounds,
            decl: decl,
            lifetimes: lifetimes,
        })
    }

    // parse a TyClosure type
    pub fn parse_ty_closure(&mut self,
                            opt_sigil: Option<ast::Sigil>,
                            mut region: Option<ast::Lifetime>)
                            -> Ty_ {
        /*

        (&|~|@) ['r] [unsafe] [once] fn [:Bounds] <'lt> (S) -> T
        ^~~~~~^ ^~~^ ^~~~~~~^ ^~~~~^    ^~~~~~~~^ ^~~~^ ^~^    ^
           |     |     |        |           |       |    |     |
           |     |     |        |           |       |    |   Return type
           |     |     |        |           |       |  Argument types
           |     |     |        |           |   Lifetimes
           |     |     |        |       Closure bounds
           |     |     |     Once-ness (a.k.a., affine)
           |     |   Purity
           | Lifetime bound
        Allocation type

        */

        // At this point, the allocation type and lifetime bound have been
        // parsed.

        let purity = self.parse_unsafety();
        let onceness = parse_onceness(self);

        let (sigil, decl, lifetimes, bounds) = match opt_sigil {
            Some(sigil) => {
                // Old-style closure syntax (`fn(A)->B`).
                self.expect_keyword(keywords::Fn);
                let bounds = self.parse_optional_ty_param_bounds();
                let (decl, lifetimes) = self.parse_ty_fn_decl(false);
                (sigil, decl, lifetimes, bounds)
            }
            None => {
                // New-style closure syntax (`<'lt>|A|:K -> B`).
                let lifetimes = if self.eat(&token::LT) {
                    let lifetimes = self.parse_lifetimes();
                    self.expect_gt();

                    // Re-parse the region here. What a hack.
                    if region.is_some() {
                        self.span_err(self.last_span,
                                      "lifetime declarations must precede \
                                       the lifetime associated with a \
                                       closure");
                    }
                    region = self.parse_opt_lifetime();

                    lifetimes
                } else {
                    Vec::new()
                };

                let inputs = if self.eat(&token::OROR) {
                    Vec::new()
                } else {
                    self.expect_or();
                    let inputs = self.parse_seq_to_before_or(
                        &token::COMMA,
                        |p| p.parse_arg_general(false));
                    self.expect_or();
                    inputs
                };

                let bounds = self.parse_optional_ty_param_bounds();

                let (return_style, output) = self.parse_ret_ty();
                let decl = P(FnDecl {
                    inputs: inputs,
                    output: output,
                    cf: return_style,
                    variadic: false
                });

                (BorrowedSigil, decl, lifetimes, bounds)
            }
        };

        return TyClosure(@ClosureTy {
            sigil: sigil,
            region: region,
            purity: purity,
            onceness: onceness,
            bounds: bounds,
            decl: decl,
            lifetimes: lifetimes,
        });

        fn parse_onceness(this: &mut Parser) -> Onceness {
            if this.eat_keyword(keywords::Once) {
                Once
            } else {
                Many
            }
        }
    }

    pub fn parse_unsafety(&mut self) -> Purity {
        if self.eat_keyword(keywords::Unsafe) {
            return UnsafeFn;
        } else {
            return ImpureFn;
        }
    }

    // parse a function type (following the 'fn')
    pub fn parse_ty_fn_decl(&mut self, allow_variadic: bool)
                            -> (P<FnDecl>, Vec<ast::Lifetime>) {
        /*

        (fn) <'lt> (S) -> T
             ^~~~^ ^~^    ^
               |    |     |
               |    |   Return type
               |  Argument types
           Lifetimes

        */
        let lifetimes = if self.eat(&token::LT) {
            let lifetimes = self.parse_lifetimes();
            self.expect_gt();
            lifetimes
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
        (decl, lifetimes)
    }

    // parse the methods in a trait declaration
    pub fn parse_trait_methods(&mut self) -> Vec<TraitMethod> {
        self.parse_unspanned_seq(
            &token::LBRACE,
            &token::RBRACE,
            seq_sep_none(),
            |p| {
            let attrs = p.parse_outer_attributes();
            let lo = p.span.lo;

            let vis_span = p.span;
            let vis = p.parse_visibility();
            let pur = p.parse_fn_purity();
            // NB: at the moment, trait methods are public by default; this
            // could change.
            let ident = p.parse_ident();

            let generics = p.parse_generics();

            let (explicit_self, d) = p.parse_fn_decl_with_self(|p| {
                // This is somewhat dubious; We don't want to allow argument
                // names to be left off if there is a definition...
                p.parse_arg_general(false)
            });

            let hi = p.last_span.hi;
            match p.token {
              token::SEMI => {
                p.bump();
                debug!("parse_trait_methods(): parsing required method");
                // NB: at the moment, visibility annotations on required
                // methods are ignored; this could change.
                if vis != ast::Inherited {
                    p.obsolete(vis_span, ObsoleteTraitFuncVisibility);
                }
                Required(TypeMethod {
                    ident: ident,
                    attrs: attrs,
                    purity: pur,
                    decl: d,
                    generics: generics,
                    explicit_self: explicit_self,
                    id: ast::DUMMY_NODE_ID,
                    span: mk_sp(lo, hi)
                })
              }
              token::LBRACE => {
                debug!("parse_trait_methods(): parsing provided method");
                let (inner_attrs, body) =
                    p.parse_inner_attrs_and_block();
                let attrs = vec_ng::append(attrs, inner_attrs.as_slice());
                Provided(@ast::Method {
                    ident: ident,
                    attrs: attrs,
                    generics: generics,
                    explicit_self: explicit_self,
                    purity: pur,
                    decl: d,
                    body: body,
                    id: ast::DUMMY_NODE_ID,
                    span: mk_sp(lo, hi),
                    vis: vis,
                })
              }

              _ => {
                  let token_str = p.this_token_to_str();
                  p.fatal(format!("expected `;` or `\\{` but found `{}`",
                                  token_str))
              }
            }
        })
    }

    // parse a possibly mutable type
    pub fn parse_mt(&mut self) -> MutTy {
        let mutbl = self.parse_mutability();
        let t = self.parse_ty(false);
        MutTy { ty: t, mutbl: mutbl }
    }

    // parse [mut/const/imm] ID : TY
    // now used only by obsolete record syntax parser...
    pub fn parse_ty_field(&mut self) -> TypeField {
        let lo = self.span.lo;
        let mutbl = self.parse_mutability();
        let id = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        let hi = ty.span.hi;
        ast::TypeField {
            ident: id,
            mt: MutTy { ty: ty, mutbl: mutbl },
            span: mk_sp(lo, hi),
        }
    }

    // parse optional return type [ -> TY ] in function decl
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
                (Return, self.parse_ty(false))
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

    // parse a type.
    // Useless second parameter for compatibility with quasiquote macros.
    // Bleh!
    pub fn parse_ty(&mut self, _: bool) -> P<Ty> {
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
                let mut ts = vec!(self.parse_ty(false));
                let mut one_tuple = false;
                while self.token == token::COMMA {
                    self.bump();
                    if self.token != token::RPAREN {
                        ts.push(self.parse_ty(false));
                    }
                    else {
                        one_tuple = true;
                    }
                }

                if ts.len() == 1 && !one_tuple {
                    self.expect(&token::RPAREN);
                    return *ts.get(0)
                }

                let t = TyTup(ts);
                self.expect(&token::RPAREN);
                t
            }
        } else if self.token == token::AT {
            // MANAGED POINTER
            self.bump();
            self.parse_box_or_uniq_pointee(ManagedSigil)
        } else if self.token == token::TILDE {
            // OWNED POINTER
            self.bump();
            self.parse_box_or_uniq_pointee(OwnedSigil)
        } else if self.token == token::BINOP(token::STAR) {
            // STAR POINTER (bare pointer?)
            self.bump();
            TyPtr(self.parse_mt())
        } else if self.token == token::LBRACKET {
            // VECTOR
            self.expect(&token::LBRACKET);
            let t = self.parse_ty(false);

            // Parse the `, ..e` in `[ int, ..e ]`
            // where `e` is a const expression
            let t = match self.maybe_parse_fixed_vstore() {
                None => TyVec(t),
                Some(suffix) => TyFixedLengthVec(t, suffix)
            };
            self.expect(&token::RBRACKET);
            t
        } else if self.token == token::BINOP(token::AND) {
            // BORROWED POINTER
            self.bump();
            self.parse_borrowed_pointee()
        } else if self.is_keyword(keywords::Extern) ||
                self.token_is_bare_fn_keyword() {
            // BARE FUNCTION
            self.parse_ty_bare_fn()
        } else if self.token_is_closure_keyword() ||
                self.token == token::BINOP(token::OR) ||
                self.token == token::OROR ||
                self.token == token::LT ||
                Parser::token_is_lifetime(&self.token) {
            // CLOSURE
            //
            // FIXME(pcwalton): Eventually `token::LT` will not unambiguously
            // introduce a closure, once procs can have lifetime bounds. We
            // will need to refactor the grammar a little bit at that point.

            let lifetime = self.parse_opt_lifetime();
            let result = self.parse_ty_closure(None, lifetime);
            result
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
            let PathAndBounds {
                path,
                bounds
            } = self.parse_path(LifetimeAndTypesAndBounds);
            TyPath(path, bounds, ast::DUMMY_NODE_ID)
        } else if self.eat(&token::UNDERSCORE) {
            // TYPE TO BE INFERRED
            TyInfer
        } else {
            let msg = format!("expected type, found token {:?}", self.token);
            self.fatal(msg);
        };

        let sp = mk_sp(lo, self.last_span.hi);
        P(Ty {id: ast::DUMMY_NODE_ID, node: t, span: sp})
    }

    // parse the type following a @ or a ~
    pub fn parse_box_or_uniq_pointee(&mut self,
                                     sigil: ast::Sigil)
                                     -> Ty_ {
        // ~'foo fn() or ~fn() are parsed directly as obsolete fn types:
        match self.token {
            token::LIFETIME(..) => {
                let lifetime = self.parse_lifetime();
                self.obsolete(self.last_span, ObsoleteBoxedClosure);
                return self.parse_ty_closure(Some(sigil), Some(lifetime));
            }

            token::IDENT(..) => {
                if self.token_is_old_style_closure_keyword() {
                    self.obsolete(self.last_span, ObsoleteBoxedClosure);
                    return self.parse_ty_closure(Some(sigil), None);
                }
            }
            _ => {}
        }

        // other things are parsed as @/~ + a type.  Note that constructs like
        // ~[] and ~str will be resolved during typeck to slices and so forth,
        // rather than boxed ptrs.  But the special casing of str/vec is not
        // reflected in the AST type.
        if sigil == OwnedSigil {
            TyUniq(self.parse_ty(false))
        } else {
            TyBox(self.parse_ty(false))
        }
    }

    pub fn parse_borrowed_pointee(&mut self) -> Ty_ {
        // look for `&'lt` or `&'foo ` and interpret `foo` as the region name:
        let opt_lifetime = self.parse_opt_lifetime();

        if self.token_is_old_style_closure_keyword() {
            self.obsolete(self.last_span, ObsoleteClosureType);
            return self.parse_ty_closure(Some(BorrowedSigil), opt_lifetime);
        }

        let mt = self.parse_mt();
        return TyRptr(opt_lifetime, mt);
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

    // This version of parse arg doesn't necessarily require
    // identifier names.
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

        let t = self.parse_ty(false);

        Arg {
            ty: t,
            pat: pat,
            id: ast::DUMMY_NODE_ID,
        }
    }

    // parse a single function argument
    pub fn parse_arg(&mut self) -> Arg {
        self.parse_arg_general(true)
    }

    // parse an argument in a lambda header e.g. |arg, arg|
    pub fn parse_fn_block_arg(&mut self) -> Arg {
        let pat = self.parse_pat();
        let t = if self.eat(&token::COLON) {
            self.parse_ty(false)
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

    pub fn maybe_parse_fixed_vstore(&mut self) -> Option<@ast::Expr> {
        if self.token == token::COMMA &&
                self.look_ahead(1, |t| *t == token::DOTDOT) {
            self.bump();
            self.bump();
            Some(self.parse_expr())
        } else {
            None
        }
    }

    // matches token_lit = LIT_INT | ...
    pub fn lit_from_token(&mut self, tok: &token::Token) -> Lit_ {
        match *tok {
            token::LIT_CHAR(i) => LitChar(i),
            token::LIT_INT(i, it) => LitInt(i, it),
            token::LIT_UINT(u, ut) => LitUint(u, ut),
            token::LIT_INT_UNSUFFIXED(i) => LitIntUnsuffixed(i),
            token::LIT_FLOAT(s, ft) => {
                LitFloat(self.id_to_interned_str(s), ft)
            }
            token::LIT_FLOAT_UNSUFFIXED(s) => {
                LitFloatUnsuffixed(self.id_to_interned_str(s))
            }
            token::LIT_STR(s) => {
                LitStr(self.id_to_interned_str(s), ast::CookedStr)
            }
            token::LIT_STR_RAW(s, n) => {
                LitStr(self.id_to_interned_str(s), ast::RawStr(n))
            }
            token::LPAREN => { self.expect(&token::RPAREN); LitNil },
            _ => { self.unexpected_last(tok); }
        }
    }

    // matches lit = true | false | token_lit
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

    // matches '-' lit | lit
    pub fn parse_literal_maybe_minus(&mut self) -> @Expr {
        let minus_lo = self.span.lo;
        let minus_present = self.eat(&token::BINOP(token::MINUS));

        let lo = self.span.lo;
        let literal = @self.parse_lit();
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
            Some(INTERPOLATED(token::NtPath(~path))) => {
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

            // Next, parse a colon and bounded type parameters, if applicable.
            let bound_set = if mode == LifetimeAndTypesAndBounds {
                self.parse_optional_ty_param_bounds()
            } else {
                None
            };

            // Parse the '::' before type parameters if it's required. If
            // it is required and wasn't present, then we're done.
            if mode == LifetimeAndTypesWithColons &&
                    !self.eat(&token::MOD_SEP) {
                segments.push(PathSegmentAndBoundSet {
                    segment: ast::PathSegment {
                        identifier: identifier,
                        lifetimes: Vec::new(),
                        types: opt_vec::Empty,
                    },
                    bound_set: bound_set
                });
                break
            }

            // Parse the `<` before the lifetime and types, if applicable.
            let (any_lifetime_or_types, lifetimes, types) = {
                if mode != NoTypesAllowed && self.eat(&token::LT) {
                    let (lifetimes, types) =
                        self.parse_generic_values_after_lt();
                    (true, lifetimes, opt_vec::from(types))
                } else {
                    (false, Vec::new(), opt_vec::Empty)
                }
            };

            // Assemble and push the result.
            segments.push(PathSegmentAndBoundSet {
                segment: ast::PathSegment {
                    identifier: identifier,
                    lifetimes: lifetimes,
                    types: types,
                },
                bound_set: bound_set
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

        // Assemble the span.
        let span = mk_sp(lo, self.last_span.hi);

        // Assemble the path segments.
        let mut path_segments = Vec::new();
        let mut bounds = None;
        let last_segment_index = segments.len() - 1;
        for (i, segment_and_bounds) in segments.move_iter().enumerate() {
            let PathSegmentAndBoundSet {
                segment: segment,
                bound_set: bound_set
            } = segment_and_bounds;
            path_segments.push(segment);

            if bound_set.is_some() {
                if i != last_segment_index {
                    self.span_err(span,
                                  "type parameter bounds are allowed only \
                                   before the last segment in a path")
                }

                bounds = bound_set
            }
        }

        // Assemble the result.
        let path_and_bounds = PathAndBounds {
            path: ast::Path {
                span: span,
                global: is_global,
                segments: path_segments,
            },
            bounds: bounds,
        };

        path_and_bounds
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
    // matches lifetime = LIFETIME
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
                self.fatal(format!("expected a lifetime name"));
            }
        }
    }

    // matches lifetimes = ( lifetime ) | ( lifetime , lifetimes )
    // actually, it matches the empty one too, but putting that in there
    // messes up the grammar....
    pub fn parse_lifetimes(&mut self) -> Vec<ast::Lifetime> {
        /*!
         *
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

            match self.token {
                token::COMMA => { self.bump();}
                token::GT => { return res; }
                token::BINOP(token::SHR) => { return res; }
                _ => {
                    let msg = format!("expected `,` or `>` after lifetime \
                                      name, got: {:?}",
                                      self.token);
                    self.fatal(msg);
                }
            }
        }
    }

    pub fn token_is_mutability(tok: &token::Token) -> bool {
        token::is_keyword(keywords::Mut, tok) ||
        token::is_keyword(keywords::Const, tok)
    }

    // parse mutability declaration (mut/const/imm)
    pub fn parse_mutability(&mut self) -> Mutability {
        if self.eat_keyword(keywords::Mut) {
            MutMutable
        } else if self.eat_keyword(keywords::Const) {
            self.obsolete(self.last_span, ObsoleteConstPointer);
            MutImmutable
        } else {
            MutImmutable
        }
    }

    // parse ident COLON expr
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

    pub fn mk_expr(&mut self, lo: BytePos, hi: BytePos, node: Expr_) -> @Expr {
        @Expr {
            id: ast::DUMMY_NODE_ID,
            node: node,
            span: mk_sp(lo, hi),
        }
    }

    pub fn mk_unary(&mut self, unop: ast::UnOp, expr: @Expr) -> ast::Expr_ {
        ExprUnary(unop, expr)
    }

    pub fn mk_binary(&mut self, binop: ast::BinOp, lhs: @Expr, rhs: @Expr) -> ast::Expr_ {
        ExprBinary(binop, lhs, rhs)
    }

    pub fn mk_call(&mut self, f: @Expr, args: Vec<@Expr> ) -> ast::Expr_ {
        ExprCall(f, args)
    }

    fn mk_method_call(&mut self, ident: Ident, tps: Vec<P<Ty>> , args: Vec<@Expr> ) -> ast::Expr_ {
        ExprMethodCall(ident, tps, args)
    }

    pub fn mk_index(&mut self, expr: @Expr, idx: @Expr) -> ast::Expr_ {
        ExprIndex(expr, idx)
    }

    pub fn mk_field(&mut self, expr: @Expr, ident: Ident, tys: Vec<P<Ty>> ) -> ast::Expr_ {
        ExprField(expr, ident, tys)
    }

    pub fn mk_assign_op(&mut self, binop: ast::BinOp, lhs: @Expr, rhs: @Expr) -> ast::Expr_ {
        ExprAssignOp(binop, lhs, rhs)
    }

    pub fn mk_mac_expr(&mut self, lo: BytePos, hi: BytePos, m: Mac_) -> @Expr {
        @Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprMac(codemap::Spanned {node: m, span: mk_sp(lo, hi)}),
            span: mk_sp(lo, hi),
        }
    }

    pub fn mk_lit_u32(&mut self, i: u32) -> @Expr {
        let span = &self.span;
        let lv_lit = @codemap::Spanned {
            node: LitUint(i as u64, TyU32),
            span: *span
        };

        @Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprLit(lv_lit),
            span: *span,
        }
    }

    // at the bottom (top?) of the precedence hierarchy,
    // parse things like parenthesized exprs,
    // macros, return, etc.
    pub fn parse_bottom_expr(&mut self) -> @Expr {
        maybe_whole_expr!(self);

        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let ex: Expr_;

        if self.token == token::LPAREN {
            self.bump();
            // (e) is parenthesized e
            // (e,) is a tuple with only one field, e
            let mut trailing_comma = false;
            if self.token == token::RPAREN {
                hi = self.span.hi;
                self.bump();
                let lit = @spanned(lo, hi, LitNil);
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
        } else if self.token == token::LBRACE {
            self.bump();
            let blk = self.parse_block_tail(lo, DefaultBlock);
            return self.mk_expr(blk.span.lo, blk.span.hi,
                                 ExprBlock(blk));
        } else if token::is_bar(&self.token) {
            return self.parse_lambda_expr();
        } else if self.eat_keyword(keywords::Proc) {
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
        } else if self.eat_keyword(keywords::Self) {
            let path = ast_util::ident_to_path(mk_sp(lo, hi), special_idents::self_);
            ex = ExprPath(path);
            hi = self.last_span.hi;
        } else if self.eat_keyword(keywords::If) {
            return self.parse_if_expr();
        } else if self.eat_keyword(keywords::For) {
            return self.parse_for_expr(None);
        } else if self.eat_keyword(keywords::While) {
            return self.parse_while_expr();
        } else if Parser::token_is_lifetime(&self.token) {
            let lifetime = self.get_lifetime();
            self.bump();
            self.expect(&token::COLON);
            if self.eat_keyword(keywords::For) {
                return self.parse_for_expr(Some(lifetime))
            } else if self.eat_keyword(keywords::Loop) {
                return self.parse_loop_expr(Some(lifetime))
            } else {
                self.fatal("expected `for` or `loop` after a label")
            }
        } else if self.eat_keyword(keywords::Loop) {
            return self.parse_loop_expr(None);
        } else if self.eat_keyword(keywords::Continue) {
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
        } else if self.eat_keyword(keywords::Match) {
            return self.parse_match_expr();
        } else if self.eat_keyword(keywords::Unsafe) {
            return self.parse_block_expr(lo, UnsafeBlock(ast::UserProvided));
        } else if self.token == token::LBRACKET {
            self.bump();
            let mutbl = MutImmutable;

            if self.token == token::RBRACKET {
                // Empty vector.
                self.bump();
                ex = ExprVec(Vec::new(), mutbl);
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
                    ex = ExprRepeat(first_expr, count, mutbl);
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
                    ex = ExprVec(exprs, mutbl);
                } else {
                    // Vector with one element.
                    self.expect(&token::RBRACKET);
                    ex = ExprVec(vec!(first_expr), mutbl);
                }
            }
            hi = self.last_span.hi;
        } else if self.eat_keyword(keywords::Return) {
            // RETURN expression
            if can_begin_expr(&self.token) {
                let e = self.parse_expr();
                hi = e.span.hi;
                ex = ExprRet(Some(e));
            } else { ex = ExprRet(None); }
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
                is_ident(&self.token) && !self.is_keyword(keywords::True) &&
                !self.is_keyword(keywords::False) {
            let pth = self.parse_path(LifetimeAndTypesWithColons).path;

            // `!`, as an operator, is prefix, so we know this isn't that
            if self.token == token::NOT {
                // MACRO INVOCATION expression
                self.bump();
                match self.token {
                    token::LPAREN | token::LBRACE => {}
                    _ => self.fatal("expected open delimiter")
                };

                let ket = token::flip_delimiter(&self.token);
                self.bump();

                let tts = self.parse_seq_to_end(&ket,
                                                seq_sep_none(),
                                                |p| p.parse_token_tree());
                let hi = self.span.hi;

                return self.mk_mac_expr(lo, hi, MacInvocTT(pth, tts, EMPTY_CTXT));
            } else if self.token == token::LBRACE {
                // This might be a struct literal.
                if self.looking_at_struct_literal() {
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
                                         &[token::COMMA], &[token::RBRACE]);
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
            ex = ExprLit(@lit);
        }

        return self.mk_expr(lo, hi, ex);
    }

    // parse a block or unsafe block
    pub fn parse_block_expr(&mut self, lo: BytePos, blk_mode: BlockCheckMode)
                            -> @Expr {
        self.expect(&token::LBRACE);
        let blk = self.parse_block_tail(lo, blk_mode);
        return self.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk));
    }

    // parse a.b or a(13) or a[4] or just a
    pub fn parse_dot_or_call_expr(&mut self) -> @Expr {
        let b = self.parse_bottom_expr();
        self.parse_dot_or_call_expr_with(b)
    }

    pub fn parse_dot_or_call_expr_with(&mut self, e0: @Expr) -> @Expr {
        let mut e = e0;
        let lo = e.span.lo;
        let mut hi;
        loop {
            // expr.f
            if self.eat(&token::DOT) {
                match self.token {
                  token::IDENT(i, _) => {
                    hi = self.span.hi;
                    self.bump();
                    let (_, tys) = if self.eat(&token::MOD_SEP) {
                        self.expect(&token::LT);
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
                                seq_sep_trailing_disallowed(token::COMMA),
                                |p| p.parse_expr()
                            );
                            hi = self.last_span.hi;

                            es.unshift(e);
                            let nd = self.mk_method_call(i, tys, es);
                            e = self.mk_expr(lo, hi, nd);
                        }
                        _ => {
                            let field = self.mk_field(e, i, tys);
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

    // parse an optional separator followed by a kleene-style
    // repetition token (+ or *).
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

    // parse a single token tree from the input.
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
                  let token_str = p.this_token_to_str();
                  p.fatal(format!("incorrect close delimiter: `{}`",
                                  token_str))
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
                    TTSeq(mk_sp(sp.lo, p.span.hi), @seq, s, z)
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

        match self.token {
            token::EOF => {
                let open_braces = self.open_braces.clone();
                for sp in open_braces.iter() {
                    self.span_note(*sp, "Did you mean to close this delimiter?");
                }
                // There shouldn't really be a span, but it's easier for the test runner
                // if we give it one
                self.fatal("this file contains an un-closed delimiter ");
            }
            token::LPAREN | token::LBRACE | token::LBRACKET => {
                let close_delim = token::flip_delimiter(&self.token);

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

                TTDelim(@result)
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
        let name_idx = @Cell::new(0u);
        match self.token {
            token::LBRACE | token::LPAREN | token::LBRACKET => {
                let other_delimiter = token::flip_delimiter(&self.token);
                self.bump();
                self.parse_matcher_subseq_upto(name_idx, &other_delimiter)
            }
            _ => self.fatal("expected open delimiter")
        }
    }

    // This goofy function is necessary to correctly match parens in Matcher's.
    // Otherwise, `$( ( )` would be a valid Matcher, and `$( () )` would be
    // invalid. It's similar to common::parse_seq.
    pub fn parse_matcher_subseq_upto(&mut self,
                                     name_idx: @Cell<uint>,
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

    pub fn parse_matcher(&mut self, name_idx: @Cell<uint>) -> Matcher {
        let lo = self.span.lo;

        let m = if self.token == token::DOLLAR {
            self.bump();
            if self.token == token::LPAREN {
                let name_idx_lo = name_idx.get();
                self.bump();
                let ms = self.parse_matcher_subseq_upto(name_idx,
                                                        &token::RPAREN);
                if ms.len() == 0u {
                    self.fatal("repetition body must be nonempty");
                }
                let (sep, zerok) = self.parse_sep_and_zerok();
                MatchSeq(ms, sep, zerok, name_idx_lo, name_idx.get())
            } else {
                let bound_to = self.parse_ident();
                self.expect(&token::COLON);
                let nt_name = self.parse_ident();
                let m = MatchNonterminal(bound_to, nt_name, name_idx.get());
                name_idx.set(name_idx.get() + 1u);
                m
            }
        } else {
            MatchTok(self.bump_and_get())
        };

        return spanned(lo, self.span.hi, m);
    }

    // parse a prefix-operator expr
    pub fn parse_prefix_expr(&mut self) -> @Expr {
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
          token::BINOP(b) => {
            match b {
              token::MINUS => {
                self.bump();
                let e = self.parse_prefix_expr();
                hi = e.span.hi;
                ex = self.mk_unary(UnNeg, e);
              }
              token::STAR => {
                self.bump();
                let e = self.parse_prefix_expr();
                hi = e.span.hi;
                ex = self.mk_unary(UnDeref, e);
              }
              token::AND => {
                self.bump();
                let _lt = self.parse_opt_lifetime();
                let m = self.parse_mutability();
                let e = self.parse_prefix_expr();
                hi = e.span.hi;
                // HACK: turn &[...] into a &-vec
                ex = match e.node {
                  ExprVec(..) if m == MutImmutable => {
                    ExprVstore(e, ExprVstoreSlice)
                  }
                  ExprLit(lit) if lit_is_str(lit) && m == MutImmutable => {
                    ExprVstore(e, ExprVstoreSlice)
                  }
                  ExprVec(..) if m == MutMutable => {
                    ExprVstore(e, ExprVstoreMutSlice)
                  }
                  _ => ExprAddrOf(m, e)
                };
              }
              _ => return self.parse_dot_or_call_expr()
            }
          }
          token::AT => {
            self.bump();
            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            // HACK: pretending @[] is a (removed) @-vec
            ex = match e.node {
              ExprVec(..) |
              ExprRepeat(..) => {
                  self.obsolete(e.span, ObsoleteManagedVec);
                  // the above error means that no-one will know we're
                  // lying... hopefully.
                  ExprVstore(e, ExprVstoreUniq)
              }
              ExprLit(lit) if lit_is_str(lit) => {
                  self.obsolete(self.last_span, ObsoleteManagedString);
                  ExprVstore(e, ExprVstoreUniq)
              }
              _ => self.mk_unary(UnBox, e)
            };
          }
          token::TILDE => {
            self.bump();

            let e = self.parse_prefix_expr();
            hi = e.span.hi;
            // HACK: turn ~[...] into a ~-vec
            ex = match e.node {
              ExprVec(..) | ExprRepeat(..) => ExprVstore(e, ExprVstoreUniq),
              ExprLit(lit) if lit_is_str(lit) => {
                  ExprVstore(e, ExprVstoreUniq)
              }
              _ => self.mk_unary(UnUniq, e)
            };
          }
          token::IDENT(_, _) if self.is_keyword(keywords::Box) => {
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
            // HACK: turn `box [...]` into a boxed-vec
            ex = match subexpression.node {
                ExprVec(..) | ExprRepeat(..) => {
                    ExprVstore(subexpression, ExprVstoreUniq)
                }
                ExprLit(lit) if lit_is_str(lit) => {
                    ExprVstore(subexpression, ExprVstoreUniq)
                }
                _ => self.mk_unary(UnUniq, subexpression)
            };
          }
          _ => return self.parse_dot_or_call_expr()
        }
        return self.mk_expr(lo, hi, ex);
    }

    // parse an expression of binops
    pub fn parse_binops(&mut self) -> @Expr {
        let prefix_expr = self.parse_prefix_expr();
        self.parse_more_binops(prefix_expr, 0)
    }

    // parse an expression of binops of at least min_prec precedence
    pub fn parse_more_binops(&mut self, lhs: @Expr, min_prec: uint) -> @Expr {
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
                    let rhs = self.parse_ty(true);
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

    // parse an assignment expression....
    // actually, this seems to be the main entry point for
    // parsing an arbitrary expression.
    pub fn parse_assign_expr(&mut self) -> @Expr {
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
          token::DARROW => {
            self.obsolete(self.span, ObsoleteSwap);
            self.bump();
            // Ignore what we get, this is an error anyway
            self.parse_expr();
            self.mk_expr(lo, self.span.hi, ExprBreak(None))
          }
          _ => {
              lhs
          }
        }
    }

    // parse an 'if' expression ('if' token already eaten)
    pub fn parse_if_expr(&mut self) -> @Expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let thn = self.parse_block();
        let mut els: Option<@Expr> = None;
        let mut hi = thn.span.hi;
        if self.eat_keyword(keywords::Else) {
            let elexpr = self.parse_else_expr();
            els = Some(elexpr);
            hi = elexpr.span.hi;
        }
        self.mk_expr(lo, hi, ExprIf(cond, thn, els))
    }

    // `|args| { ... }` or `{ ...}` like in `do` expressions
    pub fn parse_lambda_block_expr(&mut self) -> @Expr {
        self.parse_lambda_expr_(
            |p| {
                match p.token {
                    token::BINOP(token::OR) | token::OROR => {
                        p.parse_fn_block_decl()
                    }
                    _ => {
                        // No argument list - `do foo {`
                        P(FnDecl {
                            inputs: Vec::new(),
                            output: P(Ty {
                                id: ast::DUMMY_NODE_ID,
                                node: TyInfer,
                                span: p.span
                            }),
                            cf: Return,
                            variadic: false
                        })
                    }
                }
            },
            |p| {
                let blk = p.parse_block();
                p.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk))
            })
    }

    // `|args| expr`
    pub fn parse_lambda_expr(&mut self) -> @Expr {
        self.parse_lambda_expr_(|p| p.parse_fn_block_decl(),
                                |p| p.parse_expr())
    }

    // parse something of the form |args| expr
    // this is used both in parsing a lambda expr
    // and in parsing a block expr as e.g. in for...
    pub fn parse_lambda_expr_(&mut self,
                              parse_decl: |&mut Parser| -> P<FnDecl>,
                              parse_body: |&mut Parser| -> @Expr)
                              -> @Expr {
        let lo = self.span.lo;
        let decl = parse_decl(self);
        let body = parse_body(self);
        let fakeblock = P(ast::Block {
            view_items: Vec::new(),
            stmts: Vec::new(),
            expr: Some(body),
            id: ast::DUMMY_NODE_ID,
            rules: DefaultBlock,
            span: body.span,
        });

        return self.mk_expr(lo, body.span.hi, ExprFnBlock(decl, fakeblock));
    }

    pub fn parse_else_expr(&mut self) -> @Expr {
        if self.eat_keyword(keywords::If) {
            return self.parse_if_expr();
        } else {
            let blk = self.parse_block();
            return self.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk));
        }
    }

    // parse a 'for' .. 'in' expression ('for' token already eaten)
    pub fn parse_for_expr(&mut self, opt_ident: Option<ast::Ident>) -> @Expr {
        // Parse: `for <src_pat> in <src_expr> <src_loop_block>`

        let lo = self.last_span.lo;
        let pat = self.parse_pat();
        self.expect_keyword(keywords::In);
        let expr = self.parse_expr();
        let loop_block = self.parse_block();
        let hi = self.span.hi;

        self.mk_expr(lo, hi, ExprForLoop(pat, expr, loop_block, opt_ident))
    }

    pub fn parse_while_expr(&mut self) -> @Expr {
        let lo = self.last_span.lo;
        let cond = self.parse_expr();
        let body = self.parse_block();
        let hi = body.span.hi;
        return self.mk_expr(lo, hi, ExprWhile(cond, body));
    }

    pub fn parse_loop_expr(&mut self, opt_ident: Option<ast::Ident>) -> @Expr {
        // loop headers look like 'loop {' or 'loop unsafe {'
        let is_loop_header =
            self.token == token::LBRACE
            || (is_ident(&self.token)
                && self.look_ahead(1, |t| *t == token::LBRACE));

        if is_loop_header {
            // This is a loop body
            let lo = self.last_span.lo;
            let body = self.parse_block();
            let hi = body.span.hi;
            return self.mk_expr(lo, hi, ExprLoop(body, opt_ident));
        } else {
            // This is an obsolete 'continue' expression
            if opt_ident.is_some() {
                self.span_err(self.last_span,
                              "a label may not be used with a `loop` expression");
            }

            self.obsolete(self.last_span, ObsoleteLoopAsContinue);
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
    }

    // For distingishing between struct literals and blocks
    fn looking_at_struct_literal(&mut self) -> bool {
        self.token == token::LBRACE &&
        ((self.look_ahead(1, |t| token::is_plain_ident(t)) &&
          self.look_ahead(2, |t| *t == token::COLON))
         || self.look_ahead(1, |t| *t == token::DOTDOT))
    }

    fn parse_match_expr(&mut self) -> @Expr {
        let lo = self.last_span.lo;
        let discriminant = self.parse_expr();
        self.commit_expr_expecting(discriminant, token::LBRACE);
        let mut arms: Vec<Arm> = Vec::new();
        while self.token != token::RBRACE {
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

            arms.push(ast::Arm { pats: pats, guard: guard, body: expr });
        }
        let hi = self.span.hi;
        self.bump();
        return self.mk_expr(lo, hi, ExprMatch(discriminant, arms));
    }

    // parse an expression
    pub fn parse_expr(&mut self) -> @Expr {
        return self.parse_expr_res(UNRESTRICTED);
    }

    // parse an expression, subject to the given restriction
    fn parse_expr_res(&mut self, r: restriction) -> @Expr {
        let old = self.restriction;
        self.restriction = r;
        let e = self.parse_assign_expr();
        self.restriction = old;
        return e;
    }

    // parse the RHS of a local variable declaration (e.g. '= 14;')
    fn parse_initializer(&mut self) -> Option<@Expr> {
        if self.token == token::EQ {
            self.bump();
            Some(self.parse_expr())
        } else {
            None
        }
    }

    // parse patterns, separated by '|' s
    fn parse_pats(&mut self) -> Vec<@Pat> {
        let mut pats = Vec::new();
        loop {
            pats.push(self.parse_pat());
            if self.token == token::BINOP(token::OR) { self.bump(); }
            else { return pats; }
        };
    }

    fn parse_pat_vec_elements(
        &mut self,
    ) -> (Vec<@Pat> , Option<@Pat>, Vec<@Pat> ) {
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
                    slice = Some(@ast::Pat {
                        id: ast::DUMMY_NODE_ID,
                        node: PatWildMulti,
                        span: self.span,
                    })
                } else {
                    let subpat = self.parse_pat();
                    match *subpat {
                        ast::Pat { id, node: PatWild, span } => {
                            self.obsolete(self.span, ObsoleteVecDotDotWildcard);
                            slice = Some(@ast::Pat {
                                id: id,
                                node: PatWildMulti,
                                span: span
                            })
                        },
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

    // parse the fields of a struct-like pattern
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

            etc = self.token == token::UNDERSCORE || self.token == token::DOTDOT;
            if self.token == token::UNDERSCORE {
                self.obsolete(self.span, ObsoleteStructWildcard);
            }
            if etc {
                self.bump();
                if self.token != token::RBRACE {
                    let token_str = self.this_token_to_str();
                    self.fatal(format!("expected `\\}`, found `{}`",
                                       token_str))
                }
                etc = true;
                break;
            }

            let lo1 = self.last_span.lo;
            let bind_type = if self.eat_keyword(keywords::Mut) {
                BindByValue(MutMutable)
            } else if self.eat_keyword(keywords::Ref) {
                BindByRef(self.parse_mutability())
            } else {
                BindByValue(MutImmutable)
            };

            let fieldname = self.parse_ident();
            let hi1 = self.last_span.lo;
            let fieldpath = ast_util::ident_to_path(mk_sp(lo1, hi1),
                                                    fieldname);
            let subpat;
            if self.token == token::COLON {
                match bind_type {
                    BindByRef(..) | BindByValue(MutMutable) => {
                        let token_str = self.this_token_to_str();
                        self.fatal(format!("unexpected `{}`", token_str))
                    }
                    _ => {}
                }

                self.bump();
                subpat = self.parse_pat();
            } else {
                subpat = @ast::Pat {
                    id: ast::DUMMY_NODE_ID,
                    node: PatIdent(bind_type, fieldpath, None),
                    span: self.last_span
                };
            }
            fields.push(ast::FieldPat { ident: fieldname, pat: subpat });
        }
        return (fields, etc);
    }

    // parse a pattern.
    pub fn parse_pat(&mut self) -> @Pat {
        maybe_whole!(self, NtPat);

        let lo = self.span.lo;
        let mut hi;
        let pat;
        match self.token {
            // parse _
          token::UNDERSCORE => {
            self.bump();
            pat = PatWild;
            hi = self.last_span.hi;
            return @ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          // parse @pat
          token::AT => {
            self.bump();
            let sub = self.parse_pat();
            self.obsolete(self.span, ObsoleteManagedPattern);
            let hi = self.last_span.hi;
            return @ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: PatUniq(sub),
                span: mk_sp(lo, hi)
            }
          }
          token::TILDE => {
            // parse ~pat
            self.bump();
            let sub = self.parse_pat();
            pat = PatUniq(sub);
            hi = self.last_span.hi;
            return @ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          token::BINOP(token::AND) => {
              // parse &pat
              let lo = self.span.lo;
              self.bump();
              let sub = self.parse_pat();
              hi = sub.span.hi;
              // HACK: parse &"..." as a literal of a borrowed str
              pat = match sub.node {
                  PatLit(e) => {
                      match e.node {
                        ExprLit(lit) if lit_is_str(lit) => {
                          let vst = @Expr {
                              id: ast::DUMMY_NODE_ID,
                              node: ExprVstore(e, ExprVstoreSlice),
                              span: mk_sp(lo, hi)
                          };
                          PatLit(vst)
                        }
                        _ => PatRegion(sub),
                      }
                  }
                  _ => PatRegion(sub),
            };
            hi = self.last_span.hi;
            return @ast::Pat {
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
                let lit = @codemap::Spanned {
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
            return @ast::Pat {
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
            return @ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: pat,
                span: mk_sp(lo, hi)
            }
          }
          _ => {}
        }

        if !is_ident_or_path(&self.token)
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
                let name = self.parse_path(NoTypesAllowed).path;
                let sub;
                if self.eat(&token::AT) {
                    // parse foo @ pat
                    sub = Some(self.parse_pat());
                } else {
                    // or just foo
                    sub = None;
                }
                pat = PatIdent(BindByValue(MutImmutable), name, sub);
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
                        let mut args: Vec<@Pat> = Vec::new();
                        match self.token {
                          token::LPAREN => {
                            let is_star = self.look_ahead(1, |t| {
                                match *t {
                                    token::BINOP(token::STAR) => true,
                                    _ => false,
                                }
                            });
                            let is_dotdot = self.look_ahead(1, |t| {
                                match *t {
                                    token::DOTDOT => true,
                                    _ => false,
                                }
                            });
                            if is_star | is_dotdot {
                                // This is a "top constructor only" pat
                                self.bump();
                                if is_star {
                                    self.obsolete(self.span, ObsoleteEnumWildcard);
                                }
                                self.bump();
                                self.expect(&token::RPAREN);
                                pat = PatEnum(enum_path, None);
                            } else {
                                args = self.parse_enum_variant_seq(
                                    &token::LPAREN,
                                    &token::RPAREN,
                                    seq_sep_trailing_disallowed(token::COMMA),
                                    |p| p.parse_pat()
                                );
                                pat = PatEnum(enum_path, Some(args));
                            }
                          },
                          _ => {
                              if enum_path.segments.len() == 1 {
                                  // it could still be either an enum
                                  // or an identifier pattern, resolve
                                  // will sort it out:
                                  pat = PatIdent(BindByValue(MutImmutable),
                                                  enum_path,
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
        @ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: pat,
            span: mk_sp(lo, hi),
        }
    }

    // parse ident or ident @ pat
    // used by the copy foo and ref foo patterns to give a good
    // error message when parsing mistakes like ref foo(a,b)
    fn parse_pat_ident(&mut self,
                       binding_mode: ast::BindingMode)
                       -> ast::Pat_ {
        if !is_plain_ident(&self.token) {
            self.span_fatal(self.last_span,
                            "expected identifier, found path");
        }
        // why a path here, and not just an identifier?
        let name = self.parse_path(NoTypesAllowed).path;
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
            self.span_fatal(
                self.last_span,
                "expected identifier, found enum pattern");
        }

        PatIdent(binding_mode, name, sub)
    }

    // parse a local variable declaration
    fn parse_local(&mut self) -> @Local {
        let lo = self.span.lo;
        let pat = self.parse_pat();

        let mut ty = P(Ty {
            id: ast::DUMMY_NODE_ID,
            node: TyInfer,
            span: mk_sp(lo, lo),
        });
        if self.eat(&token::COLON) { ty = self.parse_ty(false); }
        let init = self.parse_initializer();
        @ast::Local {
            ty: ty,
            pat: pat,
            init: init,
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, self.last_span.hi),
        }
    }

    // parse a "let" stmt
    fn parse_let(&mut self) -> @Decl {
        let lo = self.span.lo;
        let local = self.parse_local();
        while self.eat(&token::COMMA) {
            let _ = self.parse_local();
            self.obsolete(self.span, ObsoleteMultipleLocalDecl);
        }
        return @spanned(lo, self.last_span.hi, DeclLocal(local));
    }

    // parse a structure field
    fn parse_name_and_ty(&mut self, pr: Visibility,
                         attrs: Vec<Attribute> ) -> StructField {
        let lo = self.span.lo;
        if !is_plain_ident(&self.token) {
            self.fatal("expected ident");
        }
        let name = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        spanned(lo, self.last_span.hi, ast::StructField_ {
            kind: NamedField(name, pr),
            id: ast::DUMMY_NODE_ID,
            ty: ty,
            attrs: attrs,
        })
    }

    // parse a statement. may include decl.
    // precondition: any attributes are parsed already
    pub fn parse_stmt(&mut self, item_attrs: Vec<Attribute> ) -> @Stmt {
        maybe_whole!(self, NtStmt);

        fn check_expected_item(p: &mut Parser, found_attrs: bool) {
            // If we have attributes then we should have an item
            if found_attrs {
                p.span_err(p.last_span, "expected item after attributes");
            }
        }

        let lo = self.span.lo;
        if self.is_keyword(keywords::Let) {
            check_expected_item(self, !item_attrs.is_empty());
            self.expect_keyword(keywords::Let);
            let decl = self.parse_let();
            return @spanned(lo, decl.span.hi, StmtDecl(decl, ast::DUMMY_NODE_ID));
        } else if is_ident(&self.token)
            && !token::is_any_keyword(&self.token)
            && self.look_ahead(1, |t| *t == token::NOT) {
            // parse a macro invocation. Looks like there's serious
            // overlap here; if this clause doesn't catch it (and it
            // won't, for brace-delimited macros) it will fall through
            // to the macro clause of parse_item_or_view_item. This
            // could use some cleanup, it appears to me.

            // whoops! I now have a guess: I'm guessing the "parens-only"
            // rule here is deliberate, to allow macro users to use parens
            // for things that should be parsed as stmt_mac, and braces
            // for things that should expand into items. Tricky, and
            // somewhat awkward... and probably undocumented. Of course,
            // I could just be wrong.

            check_expected_item(self, !item_attrs.is_empty());

            // Potential trouble: if we allow macros with paths instead of
            // idents, we'd need to look ahead past the whole path here...
            let pth = self.parse_path(NoTypesAllowed).path;
            self.bump();

            let id = if self.token == token::LPAREN || self.token == token::LBRACE {
                token::special_idents::invalid // no special identifier
            } else {
                self.parse_ident()
            };

            // check that we're pointing at delimiters (need to check
            // again after the `if`, because of `parse_ident`
            // consuming more tokens).
            let (bra, ket) = match self.token {
                token::LPAREN => (token::LPAREN, token::RPAREN),
                token::LBRACE => (token::LBRACE, token::RBRACE),
                _ => {
                    // we only expect an ident if we didn't parse one
                    // above.
                    let ident_str = if id == token::special_idents::invalid {
                        "identifier, "
                    } else {
                        ""
                    };
                    let tok_str = self.this_token_to_str();
                    self.fatal(format!("expected {}`(` or `\\{`, but found `{}`",
                                       ident_str, tok_str))
                }
            };

            let tts = self.parse_unspanned_seq(
                &bra,
                &ket,
                seq_sep_none(),
                |p| p.parse_token_tree()
            );
            let hi = self.span.hi;

            if id == token::special_idents::invalid {
                return @spanned(lo, hi, StmtMac(
                    spanned(lo, hi, MacInvocTT(pth, tts, EMPTY_CTXT)), false));
            } else {
                // if it has a special ident, it's definitely an item
                return @spanned(lo, hi, StmtDecl(
                    @spanned(lo, hi, DeclItem(
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
                    let decl = @spanned(lo, hi, DeclItem(i));
                    return @spanned(lo, hi, StmtDecl(decl, ast::DUMMY_NODE_ID));
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
            return @spanned(lo, e.span.hi, StmtExpr(e, ast::DUMMY_NODE_ID));
        }
    }

    // is this expression a successfully-parsed statement?
    fn expr_is_complete(&mut self, e: @Expr) -> bool {
        return self.restriction == RESTRICT_STMT_EXPR &&
            !classify::expr_requires_semi_to_be_stmt(e);
    }

    // parse a block. No inner attrs are allowed.
    pub fn parse_block(&mut self) -> P<Block> {
        maybe_whole!(no_clone self, NtBlock);

        let lo = self.span.lo;
        if self.eat_keyword(keywords::Unsafe) {
            self.obsolete(self.span, ObsoleteUnsafeBlock);
        }
        self.expect(&token::LBRACE);

        return self.parse_block_tail_(lo, DefaultBlock, Vec::new());
    }

    // parse a block. Inner attrs are allowed.
    fn parse_inner_attrs_and_block(&mut self)
        -> (Vec<Attribute> , P<Block>) {

        maybe_whole!(pair_empty self, NtBlock);

        let lo = self.span.lo;
        if self.eat_keyword(keywords::Unsafe) {
            self.obsolete(self.span, ObsoleteUnsafeBlock);
        }
        self.expect(&token::LBRACE);
        let (inner, next) = self.parse_inner_attrs_and_next();

        (inner, self.parse_block_tail_(lo, DefaultBlock, next))
    }

    // Precondition: already parsed the '{' or '#{'
    // I guess that also means "already parsed the 'impure'" if
    // necessary, and this should take a qualifier.
    // some blocks start with "#{"...
    fn parse_block_tail(&mut self, lo: BytePos, s: BlockCheckMode) -> P<Block> {
        self.parse_block_tail_(lo, s, Vec::new())
    }

    // parse the rest of a block expression or function body
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
            let decl = @spanned(item.span.lo, item.span.hi, DeclItem(*item));
            stmts.push(@spanned(item.span.lo, item.span.hi,
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
                        self.span_err(self.last_span, "expected item after attributes");
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
                            if classify::stmt_ends_with_semi(stmt) {
                                // Just check for errors and recover; do not eat semicolon yet.
                                self.commit_stmt(stmt, &[], &[token::SEMI, token::RBRACE]);
                            }

                            match self.token {
                                token::SEMI => {
                                    self.bump();
                                    stmts.push(@codemap::Spanned {
                                        node: StmtSemi(e, stmt_id),
                                        span: stmt.span,
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
                            let has_semi;
                            match self.token {
                                token::SEMI => {
                                    has_semi = true;
                                }
                                token::RBRACE => {
                                    // if a block ends in `m!(arg)` without
                                    // a `;`, it must be an expr
                                    has_semi = false;
                                    expr = Some(
                                        self.mk_mac_expr(stmt.span.lo,
                                                         stmt.span.hi,
                                                         m.node.clone()));
                                }
                                _ => {
                                    has_semi = false;
                                    stmts.push(stmt);
                                }
                            }

                            if has_semi {
                                self.bump();
                                stmts.push(@codemap::Spanned {
                                    node: StmtMac((*m).clone(), true),
                                    span: stmt.span,
                                });
                            }
                        }
                        _ => { // all other kinds of statements:
                            stmts.push(stmt);

                            if classify::stmt_ends_with_semi(stmt) {
                                self.commit_stmt_expecting(stmt, token::SEMI);
                            }
                        }
                    }
                }
            }
        }

        if !attributes_box.is_empty() {
            self.span_err(self.last_span, "expected item after attributes");
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

    // matches optbounds = ( ( : ( boundseq )? )? )
    // where   boundseq  = ( bound + boundseq ) | bound
    // and     bound     = 'static | ty
    // Returns "None" if there's no colon (e.g. "T");
    // Returns "Some(Empty)" if there's a colon but nothing after (e.g. "T:")
    // Returns "Some(stuff)" otherwise (e.g. "T:stuff").
    // NB: The None/Some distinction is important for issue #7264.
    fn parse_optional_ty_param_bounds(&mut self) -> Option<OptVec<TyParamBound>> {
        if !self.eat(&token::COLON) {
            return None;
        }

        let mut result = opt_vec::Empty;
        loop {
            match self.token {
                token::LIFETIME(lifetime) => {
                    let lifetime_interned_string = token::get_ident(lifetime);
                    if lifetime_interned_string.equiv(&("static")) {
                        result.push(RegionTyParamBound);
                    } else {
                        self.span_err(self.span,
                                      "`'static` is the only permissible region bound here");
                    }
                    self.bump();
                }
                token::MOD_SEP | token::IDENT(..) => {
                    let tref = self.parse_trait_ref();
                    result.push(TraitTyParamBound(tref));
                }
                _ => break,
            }

            if !self.eat(&token::BINOP(token::PLUS)) {
                break;
            }
        }

        return Some(result);
    }

    // matches typaram = IDENT optbounds ( EQ ty )?
    fn parse_ty_param(&mut self) -> TyParam {
        let ident = self.parse_ident();
        let opt_bounds = self.parse_optional_ty_param_bounds();
        // For typarams we don't care about the difference b/w "<T>" and "<T:>".
        let bounds = opt_bounds.unwrap_or_default();

        let default = if self.token == token::EQ {
            self.bump();
            Some(self.parse_ty(false))
        }
        else { None };

        TyParam {
            ident: ident,
            id: ast::DUMMY_NODE_ID,
            bounds: bounds,
            default: default
        }
    }

    // parse a set of optional generic type parameter declarations
    // matches generics = ( ) | ( < > ) | ( < typaramseq ( , )? > ) | ( < lifetimes ( , )? > )
    //                  | ( < lifetimes , typaramseq ( , )? > )
    // where   typaramseq = ( typaram ) | ( typaram , typaramseq )
    pub fn parse_generics(&mut self) -> ast::Generics {
        if self.eat(&token::LT) {
            let lifetimes = self.parse_lifetimes();
            let mut seen_default = false;
            let ty_params = self.parse_seq_to_gt(Some(token::COMMA), |p| {
                let ty_param = p.parse_ty_param();
                if ty_param.default.is_some() {
                    seen_default = true;
                } else if seen_default {
                    p.span_err(p.last_span,
                               "type parameters with a default must be trailing");
                }
                ty_param
            });
            ast::Generics { lifetimes: lifetimes, ty_params: ty_params }
        } else {
            ast_util::empty_generics()
        }
    }

    fn parse_generic_values_after_lt(&mut self) -> (Vec<ast::Lifetime>, Vec<P<Ty>> ) {
        let lifetimes = self.parse_lifetimes();
        let result = self.parse_seq_to_gt(
            Some(token::COMMA),
            |p| p.parse_ty(false));
        (lifetimes, opt_vec::take_vec(result))
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
                                p.span_fatal(p.span,
                                    "`...` must be last in argument list for variadic function");
                            }
                        } else {
                            p.span_fatal(p.span,
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

    // parse the argument list and result type of a function declaration
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

    fn expect_self_ident(&mut self) {
        if !self.is_self_ident() {
            let token_str = self.this_token_to_str();
            self.fatal(format!("expected `self` but found `{}`", token_str))
        }
        self.bump();
    }

    // parse the argument list and result type of a function
    // that may have a self type.
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
                this.expect_self_ident();
                SelfRegion(None, MutImmutable)
            } else if this.look_ahead(1, |t| Parser::token_is_mutability(t)) &&
                    this.look_ahead(2,
                                    |t| token::is_keyword(keywords::Self,
                                                          t)) {
                this.bump();
                let mutability = this.parse_mutability();
                this.expect_self_ident();
                SelfRegion(None, mutability)
            } else if this.look_ahead(1, |t| Parser::token_is_lifetime(t)) &&
                       this.look_ahead(2,
                                       |t| token::is_keyword(keywords::Self,
                                                             t)) {
                this.bump();
                let lifetime = this.parse_lifetime();
                this.expect_self_ident();
                SelfRegion(Some(lifetime), MutImmutable)
            } else if this.look_ahead(1, |t| Parser::token_is_lifetime(t)) &&
                      this.look_ahead(2, |t| {
                          Parser::token_is_mutability(t)
                      }) &&
                      this.look_ahead(3, |t| token::is_keyword(keywords::Self,
                                                               t)) {
                this.bump();
                let lifetime = this.parse_lifetime();
                let mutability = this.parse_mutability();
                this.expect_self_ident();
                SelfRegion(Some(lifetime), mutability)
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
                    self.expect_self_ident();
                    SelfUniq
                } else {
                    SelfStatic
                }
            }
            token::IDENT(..) if self.is_self_ident() => {
                self.bump();
                SelfValue
            }
            token::BINOP(token::STAR) => {
                // Possibly "*self" or "*mut self" -- not supported. Try to avoid
                // emitting cryptic "unexpected token" errors.
                self.bump();
                let _mutability = if Parser::token_is_mutability(&self.token) {
                    self.parse_mutability()
                } else { MutImmutable };
                if self.is_self_ident() {
                    self.span_err(self.span, "cannot pass self by unsafe pointer");
                    self.bump();
                }
                SelfValue
            }
            _ if Parser::token_is_mutability(&self.token) &&
                    self.look_ahead(1, |t| token::is_keyword(keywords::Self, t)) => {
                mutbl_self = self.parse_mutability();
                self.expect_self_ident();
                SelfValue
            }
            _ if Parser::token_is_mutability(&self.token) &&
                    self.look_ahead(1, |t| *t == token::TILDE) &&
                    self.look_ahead(2, |t| token::is_keyword(keywords::Self, t)) => {
                mutbl_self = self.parse_mutability();
                self.bump();
                self.expect_self_ident();
                SelfUniq
            }
            _ => SelfStatic
        };

        let explicit_self_sp = mk_sp(lo, self.span.hi);

        // If we parsed a self type, expect a comma before the argument list.
        let fn_inputs = if explicit_self != SelfStatic {
            match self.token {
                token::COMMA => {
                    self.bump();
                    let sep = seq_sep_trailing_disallowed(token::COMMA);
                    let mut fn_inputs = self.parse_seq_to_before_end(
                        &token::RPAREN,
                        sep,
                        parse_arg_fn
                    );
                    fn_inputs.unshift(Arg::new_self(explicit_self_sp, mutbl_self));
                    fn_inputs
                }
                token::RPAREN => {
                    vec!(Arg::new_self(explicit_self_sp, mutbl_self))
                }
                _ => {
                    let token_str = self.this_token_to_str();
                    self.fatal(format!("expected `,` or `)`, found `{}`",
                                       token_str))
                }
            }
        } else {
            let sep = seq_sep_trailing_disallowed(token::COMMA);
            self.parse_seq_to_before_end(&token::RPAREN, sep, parse_arg_fn)
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
    fn parse_fn_block_decl(&mut self) -> P<FnDecl> {
        let inputs_captures = {
            if self.eat(&token::OROR) {
                Vec::new()
            } else {
                self.parse_unspanned_seq(
                    &token::BINOP(token::OR),
                    &token::BINOP(token::OR),
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_fn_block_arg()
                )
            }
        };
        let output = if self.eat(&token::RARROW) {
            self.parse_ty(false)
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyInfer,
                span: self.span,
            })
        };

        P(FnDecl {
            inputs: inputs_captures,
            output: output,
            cf: Return,
            variadic: false
        })
    }

    // Parses the `(arg, arg) -> return_type` header on a procedure.
    fn parse_proc_decl(&mut self) -> P<FnDecl> {
        let inputs =
            self.parse_unspanned_seq(&token::LPAREN,
                                     &token::RPAREN,
                                     seq_sep_trailing_allowed(token::COMMA),
                                     |p| p.parse_fn_block_arg());

        let output = if self.eat(&token::RARROW) {
            self.parse_ty(false)
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

    // parse the name and optional generic types of a function header.
    fn parse_fn_header(&mut self) -> (Ident, ast::Generics) {
        let id = self.parse_ident();
        let generics = self.parse_generics();
        (id, generics)
    }

    fn mk_item(&mut self, lo: BytePos, hi: BytePos, ident: Ident,
               node: Item_, vis: Visibility,
               attrs: Vec<Attribute> ) -> @Item {
        @Item {
            ident: ident,
            attrs: attrs,
            id: ast::DUMMY_NODE_ID,
            node: node,
            vis: vis,
            span: mk_sp(lo, hi)
        }
    }

    // parse an item-position function declaration.
    fn parse_item_fn(&mut self, purity: Purity, abis: AbiSet) -> ItemInfo {
        let (ident, generics) = self.parse_fn_header();
        let decl = self.parse_fn_decl(false);
        let (inner_attrs, body) = self.parse_inner_attrs_and_block();
        (ident, ItemFn(decl, purity, abis, generics, body), Some(inner_attrs))
    }

    // parse a method in a trait impl, starting with `attrs` attributes.
    fn parse_method(&mut self, already_parsed_attrs: Option<Vec<Attribute> >) -> @Method {
        let next_attrs = self.parse_outer_attributes();
        let attrs = match already_parsed_attrs {
            Some(mut a) => { a.push_all_move(next_attrs); a }
            None => next_attrs
        };

        let lo = self.span.lo;

        let visa = self.parse_visibility();
        let pur = self.parse_fn_purity();
        let ident = self.parse_ident();
        let generics = self.parse_generics();
        let (explicit_self, decl) = self.parse_fn_decl_with_self(|p| {
            p.parse_arg()
        });

        let (inner_attrs, body) = self.parse_inner_attrs_and_block();
        let hi = body.span.hi;
        let attrs = vec_ng::append(attrs, inner_attrs.as_slice());
        @ast::Method {
            ident: ident,
            attrs: attrs,
            generics: generics,
            explicit_self: explicit_self,
            purity: pur,
            decl: decl,
            body: body,
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, hi),
            vis: visa,
        }
    }

    // parse trait Foo { ... }
    fn parse_item_trait(&mut self) -> ItemInfo {
        let ident = self.parse_ident();
        let tps = self.parse_generics();

        // Parse traits, if necessary.
        let traits;
        if self.token == token::COLON {
            self.bump();
            traits = self.parse_trait_ref_list(&token::LBRACE);
        } else {
            traits = Vec::new();
        }

        let meths = self.parse_trait_methods();
        (ident, ItemTrait(tps, traits, meths), None)
    }

    // Parses two variants (with the region/type params always optional):
    //    impl<T> Foo { ... }
    //    impl<T> ToStr for ~[T] { ... }
    fn parse_item_impl(&mut self) -> ItemInfo {
        // First, parse type parameters if necessary.
        let generics = self.parse_generics();

        // Special case: if the next identifier that follows is '(', don't
        // allow this to be parsed as a trait.
        let could_be_trait = self.token != token::LPAREN;

        // Parse the trait.
        let mut ty = self.parse_ty(false);

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

            ty = self.parse_ty(false);
            opt_trait_ref
        } else {
            None
        };

        let mut meths = Vec::new();
        self.expect(&token::LBRACE);
        let (inner_attrs, next) = self.parse_inner_attrs_and_next();
        let mut method_attrs = Some(next);
        while !self.eat(&token::RBRACE) {
            meths.push(self.parse_method(method_attrs));
            method_attrs = None;
        }

        let ident = ast_util::impl_pretty_name(&opt_trait, ty);

        (ident, ItemImpl(generics, opt_trait, ty, meths), Some(inner_attrs))
    }

    // parse a::B<~str,int>
    fn parse_trait_ref(&mut self) -> TraitRef {
        ast::TraitRef {
            path: self.parse_path(LifetimeAndTypesWithoutColons).path,
            ref_id: ast::DUMMY_NODE_ID,
        }
    }

    // parse B + C<~str,int> + D
    fn parse_trait_ref_list(&mut self, ket: &token::Token) -> Vec<TraitRef> {
        self.parse_seq_to_before_end(
            ket,
            seq_sep_trailing_disallowed(token::BINOP(token::PLUS)),
            |p| p.parse_trait_ref()
        )
    }

    // parse struct Foo { ... }
    fn parse_item_struct(&mut self) -> ItemInfo {
        let class_name = self.parse_ident();
        let generics = self.parse_generics();

        let mut fields: Vec<StructField> ;
        let is_tuple_like;

        if self.eat(&token::LBRACE) {
            // It's a record-like struct.
            is_tuple_like = false;
            fields = Vec::new();
            while self.token != token::RBRACE {
                fields.push(self.parse_struct_decl_field());
            }
            if fields.len() == 0 {
                self.fatal(format!("unit-like struct definition should be written as `struct {};`",
                                   token::get_ident(class_name)));
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
                    kind: UnnamedField,
                    id: ast::DUMMY_NODE_ID,
                    ty: p.parse_ty(false),
                    attrs: attrs,
                };
                spanned(lo, p.span.hi, struct_field_)
            });
            self.expect(&token::SEMI);
        } else if self.eat(&token::SEMI) {
            // It's a unit-like struct.
            is_tuple_like = true;
            fields = Vec::new();
        } else {
            let token_str = self.this_token_to_str();
            self.fatal(format!("expected `\\{`, `(`, or `;` after struct \
                                name but found `{}`",
                               token_str))
        }

        let _ = ast::DUMMY_NODE_ID;  // FIXME: Workaround for crazy bug.
        let new_id = ast::DUMMY_NODE_ID;
        (class_name,
         ItemStruct(@ast::StructDef {
             fields: fields,
             ctor_id: if is_tuple_like { Some(new_id) } else { None }
         }, generics),
         None)
    }

    // parse a structure field declaration
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
                let token_str = self.this_token_to_str();
                self.span_fatal(self.span,
                                format!("expected `,`, or `\\}` but found `{}`",
                                        token_str))
            }
        }
        a_var
    }

    // parse an element of a struct definition
    fn parse_struct_decl_field(&mut self) -> StructField {

        let attrs = self.parse_outer_attributes();

        if self.eat_keyword(keywords::Priv) {
            return self.parse_single_struct_field(Private, attrs);
        }

        if self.eat_keyword(keywords::Pub) {
           return self.parse_single_struct_field(Public, attrs);
        }

        return self.parse_single_struct_field(Inherited, attrs);
    }

    // parse visiility: PUB, PRIV, or nothing
    fn parse_visibility(&mut self) -> Visibility {
        if self.eat_keyword(keywords::Pub) { Public }
        else if self.eat_keyword(keywords::Priv) { Private }
        else { Inherited }
    }

    // given a termination token and a vector of already-parsed
    // attributes (of length 0 or 1), parse all of the items in a module
    fn parse_mod_items(&mut self,
                       term: token::Token,
                       first_item_attrs: Vec<Attribute> )
                       -> Mod {
        // parse all of the items up to closing or an attribute.
        // view items are legal here.
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: starting_items,
            ..
        } = self.parse_items_and_view_items(first_item_attrs, true, true);
        let mut items: Vec<@Item> = starting_items;
        let attrs_remaining_len = attrs_remaining.len();

        // don't think this other loop is even necessary....

        let mut first = true;
        while self.token != term {
            let mut attrs = self.parse_outer_attributes();
            if first {
                attrs = vec_ng::append(attrs_remaining.clone(),
                                       attrs.as_slice());
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
                  let token_str = self.this_token_to_str();
                  self.fatal(format!("expected item but found `{}`",
                                     token_str))
              }
            }
        }

        if first && attrs_remaining_len > 0u {
            // We parsed attributes for the first item but didn't find it
            self.span_err(self.last_span, "expected item after attributes");
        }

        ast::Mod { view_items: view_items, items: items }
    }

    fn parse_item_const(&mut self) -> ItemInfo {
        let m = if self.eat_keyword(keywords::Mut) {MutMutable} else {MutImmutable};
        let id = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        self.expect(&token::EQ);
        let e = self.parse_expr();
        self.commit_expr_expecting(e, token::SEMI);
        (id, ItemStatic(ty, m, e), None)
    }

    // parse a `mod <foo> { ... }` or `mod <foo>;` item
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
            let (inner, next) = self.parse_inner_attrs_and_next();
            let m = self.parse_mod_items(token::RBRACE, next);
            self.expect(&token::RBRACE);
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

    // read a module from a source file.
    fn eval_src_mod(&mut self,
                    id: ast::Ident,
                    outer_attrs: &[ast::Attribute],
                    id_sp: Span)
                    -> (ast::Item_, Vec<ast::Attribute> ) {
        let mut prefix = Path::new(self.sess.span_diagnostic.cm.span_to_filename(self.span));
        prefix.pop();
        let mod_path = Path::new(".").join_many(self.mod_path_stack.as_slice());
        let dir_path = prefix.join(&mod_path);
        let file_path = match ::attr::first_attr_value_str_by_name(
                outer_attrs, "path") {
            Some(d) => dir_path.join(d),
            None => {
                let mod_string = token::get_ident(id);
                let mod_name = mod_string.get().to_owned();
                let default_path_str = mod_name + ".rs";
                let secondary_path_str = mod_name + "/mod.rs";
                let default_path = dir_path.join(default_path_str.as_slice());
                let secondary_path = dir_path.join(secondary_path_str.as_slice());
                let default_exists = default_path.exists();
                let secondary_exists = secondary_path.exists();
                match (default_exists, secondary_exists) {
                    (true, false) => default_path,
                    (false, true) => secondary_path,
                    (false, false) => {
                        self.span_fatal(id_sp, format!("file not found for module `{}`", mod_name));
                    }
                    (true, true) => {
                        self.span_fatal(id_sp,
                                        format!("file for module `{}` found at both {} and {}",
                                             mod_name, default_path_str, secondary_path_str));
                    }
                }
            }
        };

        self.eval_src_mod_from_path(file_path,
                                    outer_attrs.iter().map(|x| *x).collect(),
                                    id_sp)
    }

    fn eval_src_mod_from_path(&mut self,
                              path: Path,
                              outer_attrs: Vec<ast::Attribute> ,
                              id_sp: Span) -> (ast::Item_, Vec<ast::Attribute> ) {
        {
            let mut included_mod_stack = self.sess
                                             .included_mod_stack
                                             .borrow_mut();
            let maybe_i = included_mod_stack.get()
                                            .iter()
                                            .position(|p| *p == path);
            match maybe_i {
                Some(i) => {
                    let mut err = ~"circular modules: ";
                    let len = included_mod_stack.get().len();
                    for p in included_mod_stack.get().slice(i, len).iter() {
                        err.push_str(p.display().as_maybe_owned().as_slice());
                        err.push_str(" -> ");
                    }
                    err.push_str(path.display().as_maybe_owned().as_slice());
                    self.span_fatal(id_sp, err);
                }
                None => ()
            }
            included_mod_stack.get().push(path.clone());
        }

        let mut p0 =
            new_sub_parser_from_file(self.sess,
                                     self.cfg.clone(),
                                     &path,
                                     id_sp);
        let (inner, next) = p0.parse_inner_attrs_and_next();
        let mod_attrs = vec_ng::append(outer_attrs, inner.as_slice());
        let first_item_outer_attrs = next;
        let m0 = p0.parse_mod_items(token::EOF, first_item_outer_attrs);
        {
            let mut included_mod_stack = self.sess
                                             .included_mod_stack
                                             .borrow_mut();
            included_mod_stack.get().pop();
        }
        return (ast::ItemMod(m0), mod_attrs);
    }

    // parse a function declaration from a foreign module
    fn parse_item_foreign_fn(&mut self, vis: ast::Visibility,
                             attrs: Vec<Attribute> ) -> @ForeignItem {
        let lo = self.span.lo;

        // Parse obsolete purity.
        let purity = self.parse_fn_purity();
        if purity != ImpureFn {
            self.obsolete(self.last_span, ObsoleteUnsafeExternFn);
        }

        let (ident, generics) = self.parse_fn_header();
        let decl = self.parse_fn_decl(true);
        let hi = self.span.hi;
        self.expect(&token::SEMI);
        @ast::ForeignItem { ident: ident,
                            attrs: attrs,
                            node: ForeignItemFn(decl, generics),
                            id: ast::DUMMY_NODE_ID,
                            span: mk_sp(lo, hi),
                            vis: vis }
    }

    // parse a static item from a foreign module
    fn parse_item_foreign_static(&mut self, vis: ast::Visibility,
                                 attrs: Vec<Attribute> ) -> @ForeignItem {
        let lo = self.span.lo;

        self.expect_keyword(keywords::Static);
        let mutbl = self.eat_keyword(keywords::Mut);

        let ident = self.parse_ident();
        self.expect(&token::COLON);
        let ty = self.parse_ty(false);
        let hi = self.span.hi;
        self.expect(&token::SEMI);
        @ast::ForeignItem { ident: ident,
                            attrs: attrs,
                            node: ForeignItemStatic(ty, mutbl),
                            id: ast::DUMMY_NODE_ID,
                            span: mk_sp(lo, hi),
                            vis: vis }
    }

    // parse safe/unsafe and fn
    fn parse_fn_purity(&mut self) -> Purity {
        if self.eat_keyword(keywords::Fn) { ImpureFn }
        else if self.eat_keyword(keywords::Unsafe) {
            self.expect_keyword(keywords::Fn);
            UnsafeFn
        }
        else { self.unexpected(); }
    }


    // at this point, this is essentially a wrapper for
    // parse_foreign_items.
    fn parse_foreign_mod_items(&mut self,
                               abis: AbiSet,
                               first_item_attrs: Vec<Attribute> )
                               -> ForeignMod {
        let ParsedItemsAndViewItems {
            attrs_remaining: attrs_remaining,
            view_items: view_items,
            items: _,
            foreign_items: foreign_items
        } = self.parse_foreign_items(first_item_attrs, true);
        if ! attrs_remaining.is_empty() {
            self.span_err(self.last_span,
                          "expected item after attributes");
        }
        assert!(self.token == token::RBRACE);
        ast::ForeignMod {
            abis: abis,
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
                let token_str = self.this_token_to_str();
                self.span_fatal(self.span,
                                format!("expected extern crate name but found `{}`",
                                        token_str));
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
                              opt_abis: Option<AbiSet>,
                              visibility: Visibility,
                              attrs: Vec<Attribute> )
                              -> ItemOrViewItem {

        self.expect(&token::LBRACE);

        let abis = opt_abis.unwrap_or(AbiSet::C());

        let (inner, next) = self.parse_inner_attrs_and_next();
        let m = self.parse_foreign_mod_items(abis, next);
        self.expect(&token::RBRACE);

        let item = self.mk_item(lo,
                                self.last_span.hi,
                                special_idents::invalid,
                                ItemForeignMod(m),
                                visibility,
                                maybe_append(attrs, Some(inner)));
        return IoviItem(item);
    }

    // parse type Foo = Bar;
    fn parse_item_type(&mut self) -> ItemInfo {
        let ident = self.parse_ident();
        let tps = self.parse_generics();
        self.expect(&token::EQ);
        let ty = self.parse_ty(false);
        self.expect(&token::SEMI);
        (ident, ItemTy(ty, tps), None)
    }

    // parse a structure-like enum variant definition
    // this should probably be renamed or refactored...
    fn parse_struct_def(&mut self) -> @StructDef {
        let mut fields: Vec<StructField> = Vec::new();
        while self.token != token::RBRACE {
            fields.push(self.parse_struct_decl_field());
        }
        self.bump();

        return @ast::StructDef {
            fields: fields,
            ctor_id: None
        };
    }

    // parse the part of an "enum" decl following the '{'
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
                    seq_sep_trailing_disallowed(token::COMMA),
                    |p| p.parse_ty(false)
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

    // parse an "enum" declaration
    fn parse_item_enum(&mut self) -> ItemInfo {
        let id = self.parse_ident();
        let generics = self.parse_generics();
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

    // Parses a string as an ABI spec on an extern type or module. Consumes
    // the `extern` keyword, if one is found.
    fn parse_opt_abis(&mut self) -> Option<AbiSet> {
        match self.token {
            token::LIT_STR(s)
            | token::LIT_STR_RAW(s, _) => {
                self.bump();
                let identifier_string = token::get_ident(s);
                let the_string = identifier_string.get();
                let mut abis = AbiSet::empty();
                for word in the_string.words() {
                    match abi::lookup(word) {
                        Some(abi) => {
                            if abis.contains(abi) {
                                self.span_err(
                                    self.span,
                                    format!("ABI `{}` appears twice",
                                         word));
                            } else {
                                abis.add(abi);
                            }
                        }

                        None => {
                            self.span_err(
                                self.span,
                                format!("illegal ABI: \
                                      expected one of [{}], \
                                      found `{}`",
                                     abi::all_names().connect(", "),
                                     word));
                        }
                     }
                 }
                Some(abis)
            }

            _ => {
                None
             }
         }
    }

    // parse one of the items or view items allowed by the
    // flags; on failure, return IoviNone.
    // NB: this function no longer parses the items inside an
    // extern crate.
    fn parse_item_or_view_item(&mut self,
                               attrs: Vec<Attribute> ,
                               macros_allowed: bool)
                               -> ItemOrViewItem {
        match self.token {
            INTERPOLATED(token::NtItem(item)) => {
                self.bump();
                let new_attrs = vec_ng::append(attrs, item.attrs.as_slice());
                return IoviItem(@Item {
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
                   self.span_err(mk_sp(lo, self.last_span.hi),
                                 format!("`extern mod` is obsolete, use \
                                          `extern crate` instead \
                                          to refer to external crates."))
                }
                return self.parse_item_extern_crate(lo, visibility, attrs);
            }

            let opt_abis = self.parse_opt_abis();

            if self.eat_keyword(keywords::Fn) {
                // EXTERN FUNCTION ITEM
                let abis = opt_abis.unwrap_or(AbiSet::C());
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(ExternFn, abis);
                let item = self.mk_item(lo,
                                        self.last_span.hi,
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return IoviItem(item);
            } else if self.token == token::LBRACE {
                return self.parse_item_foreign_mod(lo, opt_abis, visibility, attrs);
            }

            let token_str = self.this_token_to_str();
            self.span_fatal(self.span,
                            format!("expected `\\{` or `fn` but found `{}`", token_str));
        }

        // the rest are all guaranteed to be items:
        if self.is_keyword(keywords::Static) {
            // STATIC ITEM
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_const();
            let item = self.mk_item(lo,
                                    self.last_span.hi,
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
                self.parse_item_fn(ImpureFn, AbiSet::Rust());
            let item = self.mk_item(lo,
                                    self.last_span.hi,
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
            self.expect_keyword(keywords::Fn);
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(UnsafeFn, AbiSet::Rust());
            let item = self.mk_item(lo,
                                    self.last_span.hi,
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
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Type) {
            // TYPE ITEM
            let (ident, item_, extra_attrs) = self.parse_item_type();
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Enum) {
            // ENUM ITEM
            let (ident, item_, extra_attrs) = self.parse_item_enum();
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Trait) {
            // TRAIT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_trait();
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Impl) {
            // IMPL ITEM
            let (ident, item_, extra_attrs) = self.parse_item_impl();
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        if self.eat_keyword(keywords::Struct) {
            // STRUCT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_struct();
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return IoviItem(item);
        }
        self.parse_macro_use_or_failure(attrs,macros_allowed,lo,visibility)
    }

    // parse a foreign item; on failure, return IoviNone.
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

    // this is the fall-through for parsing items.
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
            let tts = match self.token {
                token::LPAREN | token::LBRACE => {
                    let ket = token::flip_delimiter(&self.token);
                    self.bump();
                    self.parse_seq_to_end(&ket,
                                          seq_sep_none(),
                                          |p| p.parse_token_tree())
                }
                _ => self.fatal("expected open delimiter")
            };
            // single-variant-enum... :
            let m = ast::MacInvocTT(pth, tts, EMPTY_CTXT);
            let m: ast::Mac = codemap::Spanned { node: m,
                                             span: mk_sp(self.span.lo,
                                                         self.span.hi) };
            let item_ = ItemMac(m);
            let item = self.mk_item(lo,
                                    self.last_span.hi,
                                    id,
                                    item_,
                                    visibility,
                                    attrs);
            return IoviItem(item);
        }

        // FAILURE TO PARSE ITEM
        if visibility != Inherited {
            let mut s = ~"unmatched visibility `";
            if visibility == Public {
                s.push_str("pub")
            } else {
                s.push_str("priv")
            }
            s.push_char('`');
            self.span_fatal(self.last_span, s);
        }
        return IoviNone(attrs);
    }

    pub fn parse_item(&mut self, attrs: Vec<Attribute> ) -> Option<@Item> {
        match self.parse_item_or_view_item(attrs, true) {
            IoviNone(_) => None,
            IoviViewItem(_) =>
                self.fatal("view items are not allowed here"),
            IoviForeignItem(_) =>
                self.fatal("foreign items are not allowed here"),
            IoviItem(item) => Some(item)
        }
    }

    // parse, e.g., "use a::b::{z,y}"
    fn parse_use(&mut self) -> ViewItem_ {
        return ViewItemUse(self.parse_view_paths());
    }


    // matches view_path : MOD? IDENT EQ non_global_path
    // | MOD? non_global_path MOD_SEP LBRACE RBRACE
    // | MOD? non_global_path MOD_SEP LBRACE ident_seq RBRACE
    // | MOD? non_global_path MOD_SEP STAR
    // | MOD? non_global_path
    fn parse_view_path(&mut self) -> @ViewPath {
        let lo = self.span.lo;

        if self.token == token::LBRACE {
            // use {foo,bar}
            let idents = self.parse_unspanned_seq(
                &token::LBRACE, &token::RBRACE,
                seq_sep_trailing_allowed(token::COMMA),
                |p| p.parse_path_list_ident());
            let path = ast::Path {
                span: mk_sp(lo, self.span.hi),
                global: false,
                segments: Vec::new()
            };
            return @spanned(lo, self.span.hi,
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
            let path = ast::Path {
                span: mk_sp(path_lo, self.span.hi),
                global: false,
                segments: path.move_iter().map(|identifier| {
                    ast::PathSegment {
                        identifier: identifier,
                        lifetimes: Vec::new(),
                        types: opt_vec::Empty,
                    }
                }).collect()
            };
            return @spanned(lo, self.span.hi,
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
                        |p| p.parse_path_list_ident()
                    );
                    let path = ast::Path {
                        span: mk_sp(lo, self.span.hi),
                        global: false,
                        segments: path.move_iter().map(|identifier| {
                            ast::PathSegment {
                                identifier: identifier,
                                lifetimes: Vec::new(),
                                types: opt_vec::Empty,
                            }
                        }).collect()
                    };
                    return @spanned(lo, self.span.hi,
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
                                types: opt_vec::Empty,
                            }
                        }).collect()
                    };
                    return @spanned(lo, self.span.hi,
                                    ViewPathGlob(path, ast::DUMMY_NODE_ID));
                  }

                  _ => break
                }
            }
          }
          _ => ()
        }
        let last = *path.get(path.len() - 1u);
        let path = ast::Path {
            span: mk_sp(lo, self.span.hi),
            global: false,
            segments: path.move_iter().map(|identifier| {
                ast::PathSegment {
                    identifier: identifier,
                    lifetimes: Vec::new(),
                    types: opt_vec::Empty,
                }
            }).collect()
        };
        return @spanned(lo,
                        self.last_span.hi,
                        ViewPathSimple(last, path, ast::DUMMY_NODE_ID));
    }

    // matches view_paths = view_path | view_path , view_paths
    fn parse_view_paths(&mut self) -> Vec<@ViewPath> {
        let mut vp = vec!(self.parse_view_path());
        while self.token == token::COMMA {
            self.bump();
            self.obsolete(self.last_span, ObsoleteMultipleImport);
            vp.push(self.parse_view_path());
        }
        return vp;
    }

    // Parses a sequence of items. Stops when it finds program
    // text that can't be parsed as an item
    // - mod_items uses extern_mod_allowed = true
    // - block_tail_ uses extern_mod_allowed = false
    fn parse_items_and_view_items(&mut self,
                                  first_item_attrs: Vec<Attribute> ,
                                  mut extern_mod_allowed: bool,
                                  macros_allowed: bool)
                                  -> ParsedItemsAndViewItems {
        let mut attrs = vec_ng::append(first_item_attrs,
                                       self.parse_outer_attributes()
                                           .as_slice());
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
                                          "\"extern crate\" declarations are not allowed here");
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

    // Parses a sequence of foreign items. Stops when it finds program
    // text that can't be parsed as an item
    fn parse_foreign_items(&mut self, first_item_attrs: Vec<Attribute> ,
                           macros_allowed: bool)
        -> ParsedItemsAndViewItems {
        let mut attrs = vec_ng::append(first_item_attrs,
                                       self.parse_outer_attributes()
                                           .as_slice());
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

    // Parses a source module as a crate. This is the main
    // entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> Crate {
        let lo = self.span.lo;
        // parse the crate's inner attrs, maybe (oops) one
        // of the attrs of an item:
        let (inner, next) = self.parse_inner_attrs_and_next();
        let first_item_outer_attrs = next;
        // parse the items inside the crate:
        let m = self.parse_mod_items(token::EOF, first_item_outer_attrs);

        ast::Crate {
            module: m,
            attrs: inner,
            config: self.cfg.clone(),
            span: mk_sp(lo, self.span.lo)
        }
    }

    pub fn parse_optional_str(&mut self)
                              -> Option<(InternedString, ast::StrStyle)> {
        let (s, style) = match self.token {
            token::LIT_STR(s) => (self.id_to_interned_str(s), ast::CookedStr),
            token::LIT_STR_RAW(s, n) => {
                (self.id_to_interned_str(s), ast::RawStr(n))
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
