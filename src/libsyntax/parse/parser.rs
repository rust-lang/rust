// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::PathParsingMode::*;

use abi;
use ast::BareFnTy;
use ast::{RegionTyParamBound, TraitTyParamBound, TraitBoundModifier};
use ast::{Public, Unsafety};
use ast::{Mod, BiAdd, Arg, Arm, Attribute, BindByRef, BindByValue};
use ast::{BiBitAnd, BiBitOr, BiBitXor, BiRem, BiLt, Block};
use ast::{BlockCheckMode, CaptureByRef, CaptureByValue, CaptureClause};
use ast::{Constness, ConstImplItem, ConstTraitItem, Crate, CrateConfig};
use ast::{Decl, DeclItem, DeclLocal, DefaultBlock, DefaultReturn};
use ast::{UnDeref, BiDiv, EMPTY_CTXT, EnumDef, ExplicitSelf};
use ast::{Expr, Expr_, ExprAddrOf, ExprMatch, ExprAgain};
use ast::{ExprAssign, ExprAssignOp, ExprBinary, ExprBlock, ExprBox};
use ast::{ExprBreak, ExprCall, ExprCast, ExprInPlace};
use ast::{ExprField, ExprTupField, ExprClosure, ExprIf, ExprIfLet, ExprIndex};
use ast::{ExprLit, ExprLoop, ExprMac, ExprRange};
use ast::{ExprMethodCall, ExprParen, ExprPath};
use ast::{ExprRepeat, ExprRet, ExprStruct, ExprTup, ExprUnary};
use ast::{ExprVec, ExprWhile, ExprWhileLet, ExprForLoop, Field, FnDecl};
use ast::{ForeignItem, ForeignItemStatic, ForeignItemFn, ForeignMod, FunctionRetTy};
use ast::{Ident, Inherited, ImplItem, Item, Item_, ItemStatic};
use ast::{ItemEnum, ItemFn, ItemForeignMod, ItemImpl, ItemConst};
use ast::{ItemMac, ItemMod, ItemStruct, ItemTrait, ItemTy, ItemDefaultImpl};
use ast::{ItemExternCrate, ItemUse};
use ast::{LifetimeDef, Lit, Lit_};
use ast::{LitBool, LitChar, LitByte, LitByteStr};
use ast::{LitStr, LitInt, Local};
use ast::{MacStmtWithBraces, MacStmtWithSemicolon, MacStmtWithoutBraces};
use ast::{MutImmutable, MutMutable, Mac_};
use ast::{MutTy, BiMul, Mutability};
use ast::{MethodImplItem, NamedField, UnNeg, NoReturn, UnNot};
use ast::{Pat, PatBox, PatEnum, PatIdent, PatLit, PatQPath, PatMac, PatRange};
use ast::{PatRegion, PatStruct, PatTup, PatVec, PatWild};
use ast::{PolyTraitRef, QSelf};
use ast::{Return, BiShl, BiShr, Stmt, StmtDecl};
use ast::{StmtExpr, StmtSemi, StmtMac, VariantData, StructField};
use ast::{BiSub, StrStyle};
use ast::{SelfExplicit, SelfRegion, SelfStatic, SelfValue};
use ast::{Delimited, SequenceRepetition, TokenTree, TraitItem, TraitRef};
use ast::{Ty, Ty_, TypeBinding, TyMac};
use ast::{TyFixedLengthVec, TyBareFn, TyTypeof, TyInfer};
use ast::{TyParam, TyParamBound, TyParen, TyPath, TyPolyTraitRef, TyPtr};
use ast::{TyRptr, TyTup, TyU32, TyVec};
use ast::{TypeImplItem, TypeTraitItem};
use ast::{UnnamedField, UnsafeBlock};
use ast::{ViewPath, ViewPathGlob, ViewPathList, ViewPathSimple};
use ast::{Visibility, WhereClause};
use ast;
use ast_util::{self, ident_to_path};
use attr;
use codemap::{self, Span, BytePos, Spanned, spanned, mk_sp, CodeMap};
use diagnostic;
use ext::tt::macro_parser;
use parse;
use parse::classify;
use parse::common::{SeqSep, seq_sep_none, seq_sep_trailing_allowed};
use parse::lexer::{Reader, TokenAndSpan};
use parse::obsolete::{ParserObsoleteMethods, ObsoleteSyntax};
use parse::token::{self, MatchNt, SubstNt, SpecialVarNt, InternedString};
use parse::token::{keywords, special_idents, SpecialMacroVar};
use parse::{new_sub_parser_from_file, ParseSess};
use util::parser::{AssocOp, Fixity};
use print::pprust;
use ptr::P;
use owned_slice::OwnedSlice;
use parse::PResult;
use diagnostic::FatalError;

use std::collections::HashSet;
use std::io::prelude::*;
use std::mem;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::slice;

bitflags! {
    flags Restrictions: u8 {
        const RESTRICTION_STMT_EXPR         = 1 << 0,
        const RESTRICTION_NO_STRUCT_LITERAL = 1 << 1,
    }
}

type ItemInfo = (Ident, Item_, Option<Vec<Attribute> >);

/// How to parse a path. There are four different kinds of paths, all of which
/// are parsed somewhat differently.
#[derive(Copy, Clone, PartialEq)]
pub enum PathParsingMode {
    /// A path with no type parameters; e.g. `foo::bar::Baz`
    NoTypesAllowed,
    /// A path with a lifetime and type parameters, with no double colons
    /// before the type parameters; e.g. `foo::bar<'a>::Baz<T>`
    LifetimeAndTypesWithoutColons,
    /// A path with a lifetime and type parameters with double colons before
    /// the type parameters; e.g. `foo::bar::<'a>::Baz::<T>`
    LifetimeAndTypesWithColons,
}

/// How to parse a bound, whether to allow bound modifiers such as `?`.
#[derive(Copy, Clone, PartialEq)]
pub enum BoundParsingMode {
    Bare,
    Modified,
}

/// `pub` should be parsed in struct fields and not parsed in variant fields
#[derive(Clone, Copy, PartialEq)]
pub enum ParsePub {
    Yes,
    No,
}

/// Possibly accept an `token::Interpolated` expression (a pre-parsed expression
/// dropped into the token stream, which happens while parsing the result of
/// macro expansion). Placement of these is not as complex as I feared it would
/// be. The important thing is to make sure that lookahead doesn't balk at
/// `token::Interpolated` tokens.
macro_rules! maybe_whole_expr {
    ($p:expr) => (
        {
            let found = match $p.token {
                token::Interpolated(token::NtExpr(ref e)) => {
                    Some((*e).clone())
                }
                token::Interpolated(token::NtPath(_)) => {
                    // FIXME: The following avoids an issue with lexical borrowck scopes,
                    // but the clone is unfortunate.
                    let pt = match $p.token {
                        token::Interpolated(token::NtPath(ref pt)) => (**pt).clone(),
                        _ => unreachable!()
                    };
                    let span = $p.span;
                    Some($p.mk_expr(span.lo, span.hi, ExprPath(None, pt)))
                }
                token::Interpolated(token::NtBlock(_)) => {
                    // FIXME: The following avoids an issue with lexical borrowck scopes,
                    // but the clone is unfortunate.
                    let b = match $p.token {
                        token::Interpolated(token::NtBlock(ref b)) => (*b).clone(),
                        _ => unreachable!()
                    };
                    let span = $p.span;
                    Some($p.mk_expr(span.lo, span.hi, ExprBlock(b)))
                }
                _ => None
            };
            match found {
                Some(e) => {
                    try!($p.bump());
                    return Ok(e);
                }
                None => ()
            }
        }
    )
}

/// As maybe_whole_expr, but for things other than expressions
macro_rules! maybe_whole {
    ($p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                token::Interpolated(token::$constructor(_)) => {
                    Some(try!(($p).bump_and_get()))
                }
                _ => None
            };
            if let Some(token::Interpolated(token::$constructor(x))) = found {
                return Ok(x.clone());
            }
        }
    );
    (no_clone $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                token::Interpolated(token::$constructor(_)) => {
                    Some(try!(($p).bump_and_get()))
                }
                _ => None
            };
            if let Some(token::Interpolated(token::$constructor(x))) = found {
                return Ok(x);
            }
        }
    );
    (deref $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                token::Interpolated(token::$constructor(_)) => {
                    Some(try!(($p).bump_and_get()))
                }
                _ => None
            };
            if let Some(token::Interpolated(token::$constructor(x))) = found {
                return Ok((*x).clone());
            }
        }
    );
    (Some deref $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                token::Interpolated(token::$constructor(_)) => {
                    Some(try!(($p).bump_and_get()))
                }
                _ => None
            };
            if let Some(token::Interpolated(token::$constructor(x))) = found {
                return Ok(Some((*x).clone()));
            }
        }
    );
    (pair_empty $p:expr, $constructor:ident) => (
        {
            let found = match ($p).token {
                token::Interpolated(token::$constructor(_)) => {
                    Some(try!(($p).bump_and_get()))
                }
                _ => None
            };
            if let Some(token::Interpolated(token::$constructor(x))) = found {
                return Ok((Vec::new(), x));
            }
        }
    )
}


fn maybe_append(mut lhs: Vec<Attribute>, rhs: Option<Vec<Attribute>>)
                -> Vec<Attribute> {
    if let Some(ref attrs) = rhs {
        lhs.extend(attrs.iter().cloned())
    }
    lhs
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
    pub buffer: [TokenAndSpan; 4],
    pub buffer_start: isize,
    pub buffer_end: isize,
    pub tokens_consumed: usize,
    pub restrictions: Restrictions,
    pub quote_depth: usize, // not (yet) related to the quasiquoter
    pub reader: Box<Reader+'a>,
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
    pub expected_tokens: Vec<TokenType>,
}

#[derive(PartialEq, Eq, Clone)]
pub enum TokenType {
    Token(token::Token),
    Keyword(keywords::Keyword),
    Operator,
}

impl TokenType {
    fn to_string(&self) -> String {
        match *self {
            TokenType::Token(ref t) => format!("`{}`", Parser::token_to_string(t)),
            TokenType::Operator => "an operator".to_string(),
            TokenType::Keyword(kw) => format!("`{}`", kw.to_name()),
        }
    }
}

fn is_plain_ident_or_underscore(t: &token::Token) -> bool {
    t.is_plain_ident() || *t == token::Underscore
}

/// Information about the path to a module.
pub struct ModulePath {
    pub name: String,
    pub path_exists: bool,
    pub result: Result<ModulePathSuccess, ModulePathError>,
}

pub struct ModulePathSuccess {
    pub path: ::std::path::PathBuf,
    pub owns_directory: bool,
}

pub struct ModulePathError {
    pub err_msg: String,
    pub help_msg: String,
}


impl<'a> Parser<'a> {
    pub fn new(sess: &'a ParseSess,
               cfg: ast::CrateConfig,
               mut rdr: Box<Reader+'a>)
               -> Parser<'a>
    {
        let tok0 = rdr.real_token();
        let span = tok0.sp;
        let placeholder = TokenAndSpan {
            tok: token::Underscore,
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
            restrictions: Restrictions::empty(),
            quote_depth: 0,
            obsolete_set: HashSet::new(),
            mod_path_stack: Vec::new(),
            open_braces: Vec::new(),
            owns_directory: true,
            root_module_name: None,
            expected_tokens: Vec::new(),
        }
    }

    // Panicing fns (for now!)
    // These functions are used by the quote_*!() syntax extensions, but shouldn't
    // be used otherwise.
    pub fn parse_expr_panic(&mut self) -> P<Expr> {
        panictry!(self.parse_expr())
    }

    pub fn parse_item_panic(&mut self) -> Option<P<Item>> {
        panictry!(self.parse_item())
    }

    pub fn parse_pat_panic(&mut self) -> P<Pat> {
        panictry!(self.parse_pat())
    }

    pub fn parse_arm_panic(&mut self) -> Arm {
        panictry!(self.parse_arm())
    }

    pub fn parse_ty_panic(&mut self) -> P<Ty> {
        panictry!(self.parse_ty())
    }

    pub fn parse_stmt_panic(&mut self) -> Option<P<Stmt>> {
        panictry!(self.parse_stmt())
    }

    pub fn parse_attribute_panic(&mut self, permit_inner: bool) -> ast::Attribute {
        panictry!(self.parse_attribute(permit_inner))
    }

    pub fn parse_arg_panic(&mut self) -> Arg {
        panictry!(self.parse_arg())
    }

    pub fn parse_block_panic(&mut self) -> P<Block> {
        panictry!(self.parse_block())
    }

    pub fn parse_meta_item_panic(&mut self) -> P<ast::MetaItem> {
        panictry!(self.parse_meta_item())
    }

    pub fn parse_path_panic(&mut self, mode: PathParsingMode) -> ast::Path {
        panictry!(self.parse_path(mode))
    }

    /// Convert a token to a string using self's reader
    pub fn token_to_string(token: &token::Token) -> String {
        pprust::token_to_string(token)
    }

    /// Convert the current token to a string using self's reader
    pub fn this_token_to_string(&self) -> String {
        Parser::token_to_string(&self.token)
    }

    pub fn unexpected_last(&self, t: &token::Token) -> FatalError {
        let token_str = Parser::token_to_string(t);
        let last_span = self.last_span;
        self.span_fatal(last_span, &format!("unexpected token: `{}`",
                                                token_str))
    }

    pub fn unexpected(&mut self) -> FatalError {
        match self.expect_one_of(&[], &[]) {
            Err(e) => e,
            Ok(_) => unreachable!()
        }
    }

    /// Expect and consume the token t. Signal an error if
    /// the next token is not t.
    pub fn expect(&mut self, t: &token::Token) -> PResult<()> {
        if self.expected_tokens.is_empty() {
            if self.token == *t {
                self.bump()
            } else {
                let token_str = Parser::token_to_string(t);
                let this_token_str = self.this_token_to_string();
                Err(self.fatal(&format!("expected `{}`, found `{}`",
                                   token_str,
                                   this_token_str)))
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
                         inedible: &[token::Token]) -> PResult<()>{
        fn tokens_to_string(tokens: &[TokenType]) -> String {
            let mut i = tokens.iter();
            // This might be a sign we need a connect method on Iterator.
            let b = i.next()
                     .map_or("".to_string(), |t| t.to_string());
            i.enumerate().fold(b, |mut b, (i, ref a)| {
                if tokens.len() > 2 && i == tokens.len() - 2 {
                    b.push_str(", or ");
                } else if tokens.len() == 2 && i == tokens.len() - 2 {
                    b.push_str(" or ");
                } else {
                    b.push_str(", ");
                }
                b.push_str(&*a.to_string());
                b
            })
        }
        if edible.contains(&self.token) {
            self.bump()
        } else if inedible.contains(&self.token) {
            // leave it in the input
            Ok(())
        } else {
            let mut expected = edible.iter()
                .map(|x| TokenType::Token(x.clone()))
                .chain(inedible.iter().map(|x| TokenType::Token(x.clone())))
                .chain(self.expected_tokens.iter().cloned())
                .collect::<Vec<_>>();
            expected.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
            expected.dedup();
            let expect = tokens_to_string(&expected[..]);
            let actual = self.this_token_to_string();
            Err(self.fatal(
                &(if expected.len() > 1 {
                    (format!("expected one of {}, found `{}`",
                             expect,
                             actual))
                } else if expected.is_empty() {
                    (format!("unexpected token: `{}`",
                             actual))
                } else {
                    (format!("expected {}, found `{}`",
                             expect,
                             actual))
                })[..]
            ))
        }
    }

    /// Check for erroneous `ident { }`; if matches, signal error and
    /// recover (without consuming any expected input token).  Returns
    /// true if and only if input was consumed for recovery.
    pub fn check_for_erroneous_unit_struct_expecting(&mut self,
                                                     expected: &[token::Token])
                                                     -> PResult<bool> {
        if self.token == token::OpenDelim(token::Brace)
            && expected.iter().all(|t| *t != token::OpenDelim(token::Brace))
            && self.look_ahead(1, |t| *t == token::CloseDelim(token::Brace)) {
            // matched; signal non-fatal error and recover.
            let span = self.span;
            self.span_err(span,
                          "unit-like struct construction is written with no trailing `{ }`");
            try!(self.eat(&token::OpenDelim(token::Brace)));
            try!(self.eat(&token::CloseDelim(token::Brace)));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Commit to parsing a complete expression `e` expected to be
    /// followed by some token from the set edible + inedible.  Recover
    /// from anticipated input errors, discarding erroneous characters.
    pub fn commit_expr(&mut self, e: &Expr, edible: &[token::Token],
                       inedible: &[token::Token]) -> PResult<()> {
        debug!("commit_expr {:?}", e);
        if let ExprPath(..) = e.node {
            // might be unit-struct construction; check for recoverableinput error.
            let expected = edible.iter()
                .cloned()
                .chain(inedible.iter().cloned())
                .collect::<Vec<_>>();
            try!(self.check_for_erroneous_unit_struct_expecting(&expected[..]));
        }
        self.expect_one_of(edible, inedible)
    }

    pub fn commit_expr_expecting(&mut self, e: &Expr, edible: token::Token) -> PResult<()> {
        self.commit_expr(e, &[edible], &[])
    }

    /// Commit to parsing a complete statement `s`, which expects to be
    /// followed by some token from the set edible + inedible.  Check
    /// for recoverable input errors, discarding erroneous characters.
    pub fn commit_stmt(&mut self, edible: &[token::Token],
                       inedible: &[token::Token]) -> PResult<()> {
        if self.last_token
               .as_ref()
               .map_or(false, |t| t.is_ident() || t.is_path()) {
            let expected = edible.iter()
                .cloned()
                .chain(inedible.iter().cloned())
                .collect::<Vec<_>>();
            try!(self.check_for_erroneous_unit_struct_expecting(&expected));
        }
        self.expect_one_of(edible, inedible)
    }

    pub fn commit_stmt_expecting(&mut self, edible: token::Token) -> PResult<()> {
        self.commit_stmt(&[edible], &[])
    }

    pub fn parse_ident(&mut self) -> PResult<ast::Ident> {
        self.check_strict_keywords();
        try!(self.check_reserved_keywords());
        match self.token {
            token::Ident(i, _) => {
                try!(self.bump());
                Ok(i)
            }
            token::Interpolated(token::NtIdent(..)) => {
                self.bug("ident interpolation not converted to real token");
            }
            _ => {
                let token_str = self.this_token_to_string();
                Err(self.fatal(&format!("expected ident, found `{}`",
                                    token_str)))
            }
        }
    }

    pub fn parse_ident_or_self_type(&mut self) -> PResult<ast::Ident> {
        if self.is_self_type_ident() {
            self.expect_self_type_ident()
        } else {
            self.parse_ident()
        }
    }

    pub fn parse_path_list_item(&mut self) -> PResult<ast::PathListItem> {
        let lo = self.span.lo;
        let node = if try!(self.eat_keyword(keywords::SelfValue)) {
            let rename = try!(self.parse_rename());
            ast::PathListMod { id: ast::DUMMY_NODE_ID, rename: rename }
        } else {
            let ident = try!(self.parse_ident());
            let rename = try!(self.parse_rename());
            ast::PathListIdent { name: ident, rename: rename, id: ast::DUMMY_NODE_ID }
        };
        let hi = self.last_span.hi;
        Ok(spanned(lo, hi, node))
    }

    /// Check if the next token is `tok`, and return `true` if so.
    ///
    /// This method is will automatically add `tok` to `expected_tokens` if `tok` is not
    /// encountered.
    pub fn check(&mut self, tok: &token::Token) -> bool {
        let is_present = self.token == *tok;
        if !is_present { self.expected_tokens.push(TokenType::Token(tok.clone())); }
        is_present
    }

    /// Consume token 'tok' if it exists. Returns true if the given
    /// token was present, false otherwise.
    pub fn eat(&mut self, tok: &token::Token) -> PResult<bool> {
        let is_present = self.check(tok);
        if is_present { try!(self.bump())}
        Ok(is_present)
    }

    pub fn check_keyword(&mut self, kw: keywords::Keyword) -> bool {
        self.expected_tokens.push(TokenType::Keyword(kw));
        self.token.is_keyword(kw)
    }

    /// If the next token is the given keyword, eat it and return
    /// true. Otherwise, return false.
    pub fn eat_keyword(&mut self, kw: keywords::Keyword) -> PResult<bool> {
        if self.check_keyword(kw) {
            try!(self.bump());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn eat_keyword_noexpect(&mut self, kw: keywords::Keyword) -> PResult<bool> {
        if self.token.is_keyword(kw) {
            try!(self.bump());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// If the given word is not a keyword, signal an error.
    /// If the next token is not the given word, signal an error.
    /// Otherwise, eat it.
    pub fn expect_keyword(&mut self, kw: keywords::Keyword) -> PResult<()> {
        if !try!(self.eat_keyword(kw) ){
            self.expect_one_of(&[], &[])
        } else {
            Ok(())
        }
    }

    /// Signal an error if the given string is a strict keyword
    pub fn check_strict_keywords(&mut self) {
        if self.token.is_strict_keyword() {
            let token_str = self.this_token_to_string();
            let span = self.span;
            self.span_err(span,
                          &format!("expected identifier, found keyword `{}`",
                                  token_str));
        }
    }

    /// Signal an error if the current token is a reserved keyword
    pub fn check_reserved_keywords(&mut self) -> PResult<()>{
        if self.token.is_reserved_keyword() {
            let token_str = self.this_token_to_string();
            Err(self.fatal(&format!("`{}` is a reserved keyword",
                               token_str)))
        } else {
            Ok(())
        }
    }

    /// Expect and consume an `&`. If `&&` is seen, replace it with a single
    /// `&` and continue. If an `&` is not seen, signal an error.
    fn expect_and(&mut self) -> PResult<()> {
        self.expected_tokens.push(TokenType::Token(token::BinOp(token::And)));
        match self.token {
            token::BinOp(token::And) => self.bump(),
            token::AndAnd => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                Ok(self.replace_token(token::BinOp(token::And), lo, span.hi))
            }
            _ => self.expect_one_of(&[], &[])
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
                self.span_err(sp, &*format!("{} with a suffix is invalid", kind));
            }
        }
    }


    /// Attempt to consume a `<`. If `<<` is seen, replace it with a single
    /// `<` and continue. If a `<` is not seen, return false.
    ///
    /// This is meant to be used when parsing generics on a path to get the
    /// starting token.
    fn eat_lt(&mut self) -> PResult<bool> {
        self.expected_tokens.push(TokenType::Token(token::Lt));
        match self.token {
            token::Lt => { try!(self.bump()); Ok(true)}
            token::BinOp(token::Shl) => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                self.replace_token(token::Lt, lo, span.hi);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn expect_lt(&mut self) -> PResult<()> {
        if !try!(self.eat_lt()) {
            self.expect_one_of(&[], &[])
        } else {
            Ok(())
        }
    }

    /// Expect and consume a GT. if a >> is seen, replace it
    /// with a single > and continue. If a GT is not seen,
    /// signal an error.
    pub fn expect_gt(&mut self) -> PResult<()> {
        self.expected_tokens.push(TokenType::Token(token::Gt));
        match self.token {
            token::Gt => self.bump(),
            token::BinOp(token::Shr) => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                Ok(self.replace_token(token::Gt, lo, span.hi))
            }
            token::BinOpEq(token::Shr) => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                Ok(self.replace_token(token::Ge, lo, span.hi))
            }
            token::Ge => {
                let span = self.span;
                let lo = span.lo + BytePos(1);
                Ok(self.replace_token(token::Eq, lo, span.hi))
            }
            _ => {
                let gt_str = Parser::token_to_string(&token::Gt);
                let this_token_str = self.this_token_to_string();
                Err(self.fatal(&format!("expected `{}`, found `{}`",
                                   gt_str,
                                   this_token_str)))
            }
        }
    }

    pub fn parse_seq_to_before_gt_or_return<T, F>(&mut self,
                                                  sep: Option<token::Token>,
                                                  mut f: F)
                                                  -> PResult<(OwnedSlice<T>, bool)> where
        F: FnMut(&mut Parser) -> PResult<Option<T>>,
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
                match try!(f(self)) {
                    Some(result) => v.push(result),
                    None => return Ok((OwnedSlice::from_vec(v), true))
                }
            } else {
                if let Some(t) = sep.as_ref() {
                    try!(self.expect(t));
                }

            }
        }
        return Ok((OwnedSlice::from_vec(v), false));
    }

    /// Parse a sequence bracketed by '<' and '>', stopping
    /// before the '>'.
    pub fn parse_seq_to_before_gt<T, F>(&mut self,
                                        sep: Option<token::Token>,
                                        mut f: F)
                                        -> PResult<OwnedSlice<T>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        let (result, returned) = try!(self.parse_seq_to_before_gt_or_return(sep,
                                                    |p| Ok(Some(try!(f(p))))));
        assert!(!returned);
        return Ok(result);
    }

    pub fn parse_seq_to_gt<T, F>(&mut self,
                                 sep: Option<token::Token>,
                                 f: F)
                                 -> PResult<OwnedSlice<T>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        let v = try!(self.parse_seq_to_before_gt(sep, f));
        try!(self.expect_gt());
        return Ok(v);
    }

    pub fn parse_seq_to_gt_or_return<T, F>(&mut self,
                                           sep: Option<token::Token>,
                                           f: F)
                                           -> PResult<(OwnedSlice<T>, bool)> where
        F: FnMut(&mut Parser) -> PResult<Option<T>>,
    {
        let (v, returned) = try!(self.parse_seq_to_before_gt_or_return(sep, f));
        if !returned {
            try!(self.expect_gt());
        }
        return Ok((v, returned));
    }

    /// Parse a sequence, including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_end<T, F>(&mut self,
                                  ket: &token::Token,
                                  sep: SeqSep,
                                  f: F)
                                  -> PResult<Vec<T>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        let val = try!(self.parse_seq_to_before_end(ket, sep, f));
        try!(self.bump());
        Ok(val)
    }

    /// Parse a sequence, not including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_seq_to_before_end<T, F>(&mut self,
                                         ket: &token::Token,
                                         sep: SeqSep,
                                         mut f: F)
                                         -> PResult<Vec<T>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        let mut first: bool = true;
        let mut v = vec!();
        while self.token != *ket {
            match sep.sep {
              Some(ref t) => {
                if first { first = false; }
                else { try!(self.expect(t)); }
              }
              _ => ()
            }
            if sep.trailing_sep_allowed && self.check(ket) { break; }
            v.push(try!(f(self)));
        }
        return Ok(v);
    }

    /// Parse a sequence, including the closing delimiter. The function
    /// f must consume tokens until reaching the next separator or
    /// closing bracket.
    pub fn parse_unspanned_seq<T, F>(&mut self,
                                     bra: &token::Token,
                                     ket: &token::Token,
                                     sep: SeqSep,
                                     f: F)
                                     -> PResult<Vec<T>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        try!(self.expect(bra));
        let result = try!(self.parse_seq_to_before_end(ket, sep, f));
        try!(self.bump());
        Ok(result)
    }

    /// Parse a sequence parameter of enum variant. For consistency purposes,
    /// these should not be empty.
    pub fn parse_enum_variant_seq<T, F>(&mut self,
                                        bra: &token::Token,
                                        ket: &token::Token,
                                        sep: SeqSep,
                                        f: F)
                                        -> PResult<Vec<T>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        let result = try!(self.parse_unspanned_seq(bra, ket, sep, f));
        if result.is_empty() {
            let last_span = self.last_span;
            self.span_err(last_span,
            "nullary enum variants are written with no trailing `( )`");
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
                           -> PResult<Spanned<Vec<T>>> where
        F: FnMut(&mut Parser) -> PResult<T>,
    {
        let lo = self.span.lo;
        try!(self.expect(bra));
        let result = try!(self.parse_seq_to_before_end(ket, sep, f));
        let hi = self.span.hi;
        try!(self.bump());
        Ok(spanned(lo, hi, result))
    }

    /// Advance the parser by one token
    pub fn bump(&mut self) -> PResult<()> {
        self.last_span = self.span;
        // Stash token for error recovery (sometimes; clone is not necessarily cheap).
        self.last_token = if self.token.is_ident() ||
                          self.token.is_path() ||
                          self.token == token::Comma {
            Some(Box::new(self.token.clone()))
        } else {
            None
        };
        let next = if self.buffer_start == self.buffer_end {
            self.reader.real_token()
        } else {
            // Avoid token copies with `replace`.
            let buffer_start = self.buffer_start as usize;
            let next_index = (buffer_start + 1) & 3;
            self.buffer_start = next_index as isize;

            let placeholder = TokenAndSpan {
                tok: token::Underscore,
                sp: self.span,
            };
            mem::replace(&mut self.buffer[buffer_start], placeholder)
        };
        self.span = next.sp;
        self.token = next.tok;
        self.tokens_consumed += 1;
        self.expected_tokens.clear();
        // check after each token
        self.check_unknown_macro_variable()
    }

    /// Advance the parser by one token and return the bumped token.
    pub fn bump_and_get(&mut self) -> PResult<token::Token> {
        let old_token = mem::replace(&mut self.token, token::Underscore);
        try!(self.bump());
        Ok(old_token)
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
    pub fn buffer_length(&mut self) -> isize {
        if self.buffer_start <= self.buffer_end {
            return self.buffer_end - self.buffer_start;
        }
        return (4 - self.buffer_start) + self.buffer_end;
    }
    pub fn look_ahead<R, F>(&mut self, distance: usize, f: F) -> R where
        F: FnOnce(&token::Token) -> R,
    {
        let dist = distance as isize;
        while self.buffer_length() < dist {
            self.buffer[self.buffer_end as usize] = self.reader.real_token();
            self.buffer_end = (self.buffer_end + 1) & 3;
        }
        f(&self.buffer[((self.buffer_start + dist - 1) & 3) as usize].tok)
    }
    pub fn fatal(&self, m: &str) -> diagnostic::FatalError {
        self.sess.span_diagnostic.span_fatal(self.span, m)
    }
    pub fn span_fatal(&self, sp: Span, m: &str) -> diagnostic::FatalError {
        self.sess.span_diagnostic.span_fatal(sp, m)
    }
    pub fn span_fatal_help(&self, sp: Span, m: &str, help: &str) -> diagnostic::FatalError {
        self.span_err(sp, m);
        self.fileline_help(sp, help);
        diagnostic::FatalError
    }
    pub fn span_note(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_note(sp, m)
    }
    pub fn span_help(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_help(sp, m)
    }
    pub fn span_suggestion(&self, sp: Span, m: &str, n: String) {
        self.sess.span_diagnostic.span_suggestion(sp, m, n)
    }
    pub fn fileline_help(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.fileline_help(sp, m)
    }
    pub fn bug(&self, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(self.span, m)
    }
    pub fn warn(&self, m: &str) {
        self.sess.span_diagnostic.span_warn(self.span, m)
    }
    pub fn span_warn(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_warn(sp, m)
    }
    pub fn span_err(&self, sp: Span, m: &str) {
        self.sess.span_diagnostic.span_err(sp, m)
    }
    pub fn span_bug(&self, sp: Span, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(sp, m)
    }
    pub fn abort_if_errors(&self) {
        self.sess.span_diagnostic.handler().abort_if_errors();
    }

    pub fn id_to_interned_str(&mut self, id: Ident) -> InternedString {
        id.name.as_str()
    }

    /// Is the current token one of the keywords that signals a bare function
    /// type?
    pub fn token_is_bare_fn_keyword(&mut self) -> bool {
        self.check_keyword(keywords::Fn) ||
            self.check_keyword(keywords::Unsafe) ||
            self.check_keyword(keywords::Extern)
    }

    pub fn get_lifetime(&mut self) -> ast::Ident {
        match self.token {
            token::Lifetime(ref ident) => *ident,
            _ => self.bug("not a lifetime"),
        }
    }

    pub fn parse_for_in_type(&mut self) -> PResult<Ty_> {
        /*
        Parses whatever can come after a `for` keyword in a type.
        The `for` has already been consumed.

        Deprecated:

        - for <'lt> |S| -> T

        Eventually:

        - for <'lt> [unsafe] [extern "ABI"] fn (S) -> T
        - for <'lt> path::foo(a, b)

        */

        // parse <'lt>
        let lo = self.span.lo;

        let lifetime_defs = try!(self.parse_late_bound_lifetime_defs());

        // examine next token to decide to do
        if self.token_is_bare_fn_keyword() {
            self.parse_ty_bare_fn(lifetime_defs)
        } else {
            let hi = self.span.hi;
            let trait_ref = try!(self.parse_trait_ref());
            let poly_trait_ref = ast::PolyTraitRef { bound_lifetimes: lifetime_defs,
                                                     trait_ref: trait_ref,
                                                     span: mk_sp(lo, hi)};
            let other_bounds = if try!(self.eat(&token::BinOp(token::Plus)) ){
                try!(self.parse_ty_param_bounds(BoundParsingMode::Bare))
            } else {
                OwnedSlice::empty()
            };
            let all_bounds =
                Some(TraitTyParamBound(poly_trait_ref, TraitBoundModifier::None)).into_iter()
                .chain(other_bounds.into_vec())
                .collect();
            Ok(ast::TyPolyTraitRef(all_bounds))
        }
    }

    pub fn parse_ty_path(&mut self) -> PResult<Ty_> {
        Ok(TyPath(None, try!(self.parse_path(LifetimeAndTypesWithoutColons))))
    }

    /// parse a TyBareFn type:
    pub fn parse_ty_bare_fn(&mut self, lifetime_defs: Vec<ast::LifetimeDef>) -> PResult<Ty_> {
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

        let unsafety = try!(self.parse_unsafety());
        let abi = if try!(self.eat_keyword(keywords::Extern) ){
            try!(self.parse_opt_abi()).unwrap_or(abi::C)
        } else {
            abi::Rust
        };

        try!(self.expect_keyword(keywords::Fn));
        let (inputs, variadic) = try!(self.parse_fn_args(false, true));
        let ret_ty = try!(self.parse_ret_ty());
        let decl = P(FnDecl {
            inputs: inputs,
            output: ret_ty,
            variadic: variadic
        });
        Ok(TyBareFn(P(BareFnTy {
            abi: abi,
            unsafety: unsafety,
            lifetimes: lifetime_defs,
            decl: decl
        })))
    }

    /// Parses an obsolete closure kind (`&:`, `&mut:`, or `:`).
    pub fn parse_obsolete_closure_kind(&mut self) -> PResult<()> {
         let lo = self.span.lo;
        if
            self.check(&token::BinOp(token::And)) &&
            self.look_ahead(1, |t| t.is_keyword(keywords::Mut)) &&
            self.look_ahead(2, |t| *t == token::Colon)
        {
            try!(self.bump());
            try!(self.bump());
            try!(self.bump());
        } else if
            self.token == token::BinOp(token::And) &&
            self.look_ahead(1, |t| *t == token::Colon)
        {
            try!(self.bump());
            try!(self.bump());
        } else if
            try!(self.eat(&token::Colon))
        {
            /* nothing */
        } else {
            return Ok(());
        }

        let span = mk_sp(lo, self.span.hi);
        self.obsolete(span, ObsoleteSyntax::ClosureKind);
        Ok(())
    }

    pub fn parse_unsafety(&mut self) -> PResult<Unsafety> {
        if try!(self.eat_keyword(keywords::Unsafe)) {
            return Ok(Unsafety::Unsafe);
        } else {
            return Ok(Unsafety::Normal);
        }
    }

    /// Parse the items in a trait declaration
    pub fn parse_trait_items(&mut self) -> PResult<Vec<P<TraitItem>>> {
        self.parse_unspanned_seq(
            &token::OpenDelim(token::Brace),
            &token::CloseDelim(token::Brace),
            seq_sep_none(),
            |p| -> PResult<P<TraitItem>> {
            maybe_whole!(no_clone p, NtTraitItem);
            let mut attrs = try!(p.parse_outer_attributes());
            let lo = p.span.lo;

            let (name, node) = if try!(p.eat_keyword(keywords::Type)) {
                let TyParam {ident, bounds, default, ..} = try!(p.parse_ty_param());
                try!(p.expect(&token::Semi));
                (ident, TypeTraitItem(bounds, default))
            } else if p.is_const_item() {
                try!(p.expect_keyword(keywords::Const));
                let ident = try!(p.parse_ident());
                try!(p.expect(&token::Colon));
                let ty = try!(p.parse_ty_sum());
                let default = if p.check(&token::Eq) {
                    try!(p.bump());
                    let expr = try!(p.parse_expr());
                    try!(p.commit_expr_expecting(&expr, token::Semi));
                    Some(expr)
                } else {
                    try!(p.expect(&token::Semi));
                    None
                };
                (ident, ConstTraitItem(ty, default))
            } else {
                let (constness, unsafety, abi) = try!(p.parse_fn_front_matter());

                let ident = try!(p.parse_ident());
                let mut generics = try!(p.parse_generics());

                let (explicit_self, d) = try!(p.parse_fn_decl_with_self(|p|{
                    // This is somewhat dubious; We don't want to allow
                    // argument names to be left off if there is a
                    // definition...
                    p.parse_arg_general(false)
                }));

                generics.where_clause = try!(p.parse_where_clause());
                let sig = ast::MethodSig {
                    unsafety: unsafety,
                    constness: constness,
                    decl: d,
                    generics: generics,
                    abi: abi,
                    explicit_self: explicit_self,
                };

                let body = match p.token {
                  token::Semi => {
                    try!(p.bump());
                    debug!("parse_trait_methods(): parsing required method");
                    None
                  }
                  token::OpenDelim(token::Brace) => {
                    debug!("parse_trait_methods(): parsing provided method");
                    let (inner_attrs, body) =
                        try!(p.parse_inner_attrs_and_block());
                    attrs.extend(inner_attrs.iter().cloned());
                    Some(body)
                  }

                  _ => {
                      let token_str = p.this_token_to_string();
                      return Err(p.fatal(&format!("expected `;` or `{{`, found `{}`",
                                       token_str)[..]))
                  }
                };
                (ident, ast::MethodTraitItem(sig, body))
            };

            Ok(P(TraitItem {
                id: ast::DUMMY_NODE_ID,
                ident: name,
                attrs: attrs,
                node: node,
                span: mk_sp(lo, p.last_span.hi),
            }))
        })
    }

    /// Parse a possibly mutable type
    pub fn parse_mt(&mut self) -> PResult<MutTy> {
        let mutbl = try!(self.parse_mutability());
        let t = try!(self.parse_ty());
        Ok(MutTy { ty: t, mutbl: mutbl })
    }

    /// Parse optional return type [ -> TY ] in function decl
    pub fn parse_ret_ty(&mut self) -> PResult<FunctionRetTy> {
        if try!(self.eat(&token::RArrow) ){
            if try!(self.eat(&token::Not) ){
                Ok(NoReturn(self.last_span))
            } else {
                Ok(Return(try!(self.parse_ty())))
            }
        } else {
            let pos = self.span.lo;
            Ok(DefaultReturn(mk_sp(pos, pos)))
        }
    }

    /// Parse a type in a context where `T1+T2` is allowed.
    pub fn parse_ty_sum(&mut self) -> PResult<P<Ty>> {
        let lo = self.span.lo;
        let lhs = try!(self.parse_ty());

        if !try!(self.eat(&token::BinOp(token::Plus)) ){
            return Ok(lhs);
        }

        let bounds = try!(self.parse_ty_param_bounds(BoundParsingMode::Bare));

        // In type grammar, `+` is treated like a binary operator,
        // and hence both L and R side are required.
        if bounds.is_empty() {
            let last_span = self.last_span;
            self.span_err(last_span,
                          "at least one type parameter bound \
                          must be specified");
        }

        let sp = mk_sp(lo, self.last_span.hi);
        let sum = ast::TyObjectSum(lhs, bounds);
        Ok(P(Ty {id: ast::DUMMY_NODE_ID, node: sum, span: sp}))
    }

    /// Parse a type.
    pub fn parse_ty(&mut self) -> PResult<P<Ty>> {
        maybe_whole!(no_clone self, NtTy);

        let lo = self.span.lo;

        let t = if self.check(&token::OpenDelim(token::Paren)) {
            try!(self.bump());

            // (t) is a parenthesized ty
            // (t,) is the type of a tuple with only one field,
            // of type t
            let mut ts = vec![];
            let mut last_comma = false;
            while self.token != token::CloseDelim(token::Paren) {
                ts.push(try!(self.parse_ty_sum()));
                if self.check(&token::Comma) {
                    last_comma = true;
                    try!(self.bump());
                } else {
                    last_comma = false;
                    break;
                }
            }

            try!(self.expect(&token::CloseDelim(token::Paren)));
            if ts.len() == 1 && !last_comma {
                TyParen(ts.into_iter().nth(0).unwrap())
            } else {
                TyTup(ts)
            }
        } else if self.check(&token::BinOp(token::Star)) {
            // STAR POINTER (bare pointer?)
            try!(self.bump());
            TyPtr(try!(self.parse_ptr()))
        } else if self.check(&token::OpenDelim(token::Bracket)) {
            // VECTOR
            try!(self.expect(&token::OpenDelim(token::Bracket)));
            let t = try!(self.parse_ty_sum());

            // Parse the `; e` in `[ i32; e ]`
            // where `e` is a const expression
            let t = match try!(self.maybe_parse_fixed_length_of_vec()) {
                None => TyVec(t),
                Some(suffix) => TyFixedLengthVec(t, suffix)
            };
            try!(self.expect(&token::CloseDelim(token::Bracket)));
            t
        } else if self.check(&token::BinOp(token::And)) ||
                  self.token == token::AndAnd {
            // BORROWED POINTER
            try!(self.expect_and());
            try!(self.parse_borrowed_pointee())
        } else if self.check_keyword(keywords::For) {
            try!(self.parse_for_in_type())
        } else if self.token_is_bare_fn_keyword() {
            // BARE FUNCTION
            try!(self.parse_ty_bare_fn(Vec::new()))
        } else if try!(self.eat_keyword_noexpect(keywords::Typeof)) {
            // TYPEOF
            // In order to not be ambiguous, the type must be surrounded by parens.
            try!(self.expect(&token::OpenDelim(token::Paren)));
            let e = try!(self.parse_expr());
            try!(self.expect(&token::CloseDelim(token::Paren)));
            TyTypeof(e)
        } else if try!(self.eat_lt()) {

            let (qself, path) =
                 try!(self.parse_qualified_path(NoTypesAllowed));

            TyPath(Some(qself), path)
        } else if self.check(&token::ModSep) ||
                  self.token.is_ident() ||
                  self.token.is_path() {
            let path = try!(self.parse_path(LifetimeAndTypesWithoutColons));
            if self.check(&token::Not) {
                // MACRO INVOCATION
                try!(self.bump());
                let delim = try!(self.expect_open_delim());
                let tts = try!(self.parse_seq_to_end(&token::CloseDelim(delim),
                                                     seq_sep_none(),
                                                     |p| p.parse_token_tree()));
                let hi = self.span.hi;
                TyMac(spanned(lo, hi, Mac_ { path: path, tts: tts, ctxt: EMPTY_CTXT }))
            } else {
                // NAMED TYPE
                TyPath(None, path)
            }
        } else if try!(self.eat(&token::Underscore) ){
            // TYPE TO BE INFERRED
            TyInfer
        } else {
            let this_token_str = self.this_token_to_string();
            let msg = format!("expected type, found `{}`", this_token_str);
            return Err(self.fatal(&msg[..]));
        };

        let sp = mk_sp(lo, self.last_span.hi);
        Ok(P(Ty {id: ast::DUMMY_NODE_ID, node: t, span: sp}))
    }

    pub fn parse_borrowed_pointee(&mut self) -> PResult<Ty_> {
        // look for `&'lt` or `&'foo ` and interpret `foo` as the region name:
        let opt_lifetime = try!(self.parse_opt_lifetime());

        let mt = try!(self.parse_mt());
        return Ok(TyRptr(opt_lifetime, mt));
    }

    pub fn parse_ptr(&mut self) -> PResult<MutTy> {
        let mutbl = if try!(self.eat_keyword(keywords::Mut) ){
            MutMutable
        } else if try!(self.eat_keyword(keywords::Const) ){
            MutImmutable
        } else {
            let span = self.last_span;
            self.span_err(span,
                          "bare raw pointers are no longer allowed, you should \
                           likely use `*mut T`, but otherwise `*T` is now \
                           known as `*const T`");
            MutImmutable
        };
        let t = try!(self.parse_ty());
        Ok(MutTy { ty: t, mutbl: mutbl })
    }

    pub fn is_named_argument(&mut self) -> bool {
        let offset = match self.token {
            token::BinOp(token::And) => 1,
            token::AndAnd => 1,
            _ if self.token.is_keyword(keywords::Mut) => 1,
            _ => 0
        };

        debug!("parser is_named_argument offset:{}", offset);

        if offset == 0 {
            is_plain_ident_or_underscore(&self.token)
                && self.look_ahead(1, |t| *t == token::Colon)
        } else {
            self.look_ahead(offset, |t| is_plain_ident_or_underscore(t))
                && self.look_ahead(offset + 1, |t| *t == token::Colon)
        }
    }

    /// This version of parse arg doesn't necessarily require
    /// identifier names.
    pub fn parse_arg_general(&mut self, require_name: bool) -> PResult<Arg> {
        maybe_whole!(no_clone self, NtArg);

        let pat = if require_name || self.is_named_argument() {
            debug!("parse_arg_general parse_pat (require_name:{})",
                   require_name);
            let pat = try!(self.parse_pat());

            try!(self.expect(&token::Colon));
            pat
        } else {
            debug!("parse_arg_general ident_to_pat");
            ast_util::ident_to_pat(ast::DUMMY_NODE_ID,
                                   self.last_span,
                                   special_idents::invalid)
        };

        let t = try!(self.parse_ty_sum());

        Ok(Arg {
            ty: t,
            pat: pat,
            id: ast::DUMMY_NODE_ID,
        })
    }

    /// Parse a single function argument
    pub fn parse_arg(&mut self) -> PResult<Arg> {
        self.parse_arg_general(true)
    }

    /// Parse an argument in a lambda header e.g. |arg, arg|
    pub fn parse_fn_block_arg(&mut self) -> PResult<Arg> {
        let pat = try!(self.parse_pat());
        let t = if try!(self.eat(&token::Colon) ){
            try!(self.parse_ty_sum())
        } else {
            P(Ty {
                id: ast::DUMMY_NODE_ID,
                node: TyInfer,
                span: mk_sp(self.span.lo, self.span.hi),
            })
        };
        Ok(Arg {
            ty: t,
            pat: pat,
            id: ast::DUMMY_NODE_ID
        })
    }

    pub fn maybe_parse_fixed_length_of_vec(&mut self) -> PResult<Option<P<ast::Expr>>> {
        if self.check(&token::Semi) {
            try!(self.bump());
            Ok(Some(try!(self.parse_expr())))
        } else {
            Ok(None)
        }
    }

    /// Matches token_lit = LIT_INTEGER | ...
    pub fn lit_from_token(&self, tok: &token::Token) -> PResult<Lit_> {
        match *tok {
            token::Interpolated(token::NtExpr(ref v)) => {
                match v.node {
                    ExprLit(ref lit) => { Ok(lit.node.clone()) }
                    _ => { return Err(self.unexpected_last(tok)); }
                }
            }
            token::Literal(lit, suf) => {
                let (suffix_illegal, out) = match lit {
                    token::Byte(i) => (true, LitByte(parse::byte_lit(&i.as_str()).0)),
                    token::Char(i) => (true, LitChar(parse::char_lit(&i.as_str()).0)),

                    // there are some valid suffixes for integer and
                    // float literals, so all the handling is done
                    // internally.
                    token::Integer(s) => {
                        (false, parse::integer_lit(&s.as_str(),
                                                   suf.as_ref().map(|s| s.as_str()),
                                                   &self.sess.span_diagnostic,
                                                   self.last_span))
                    }
                    token::Float(s) => {
                        (false, parse::float_lit(&s.as_str(),
                                                 suf.as_ref().map(|s| s.as_str()),
                                                  &self.sess.span_diagnostic,
                                                 self.last_span))
                    }

                    token::Str_(s) => {
                        (true,
                         LitStr(token::intern_and_get_ident(&parse::str_lit(&s.as_str())),
                                ast::CookedStr))
                    }
                    token::StrRaw(s, n) => {
                        (true,
                         LitStr(
                            token::intern_and_get_ident(&parse::raw_str_lit(&s.as_str())),
                            ast::RawStr(n)))
                    }
                    token::ByteStr(i) =>
                        (true, LitByteStr(parse::byte_str_lit(&i.as_str()))),
                    token::ByteStrRaw(i, _) =>
                        (true,
                         LitByteStr(Rc::new(i.to_string().into_bytes()))),
                };

                if suffix_illegal {
                    let sp = self.last_span;
                    self.expect_no_suffix(sp, &*format!("{} literal", lit.short_name()), suf)
                }

                Ok(out)
            }
            _ => { return Err(self.unexpected_last(tok)); }
        }
    }

    /// Matches lit = true | false | token_lit
    pub fn parse_lit(&mut self) -> PResult<Lit> {
        let lo = self.span.lo;
        let lit = if try!(self.eat_keyword(keywords::True) ){
            LitBool(true)
        } else if try!(self.eat_keyword(keywords::False) ){
            LitBool(false)
        } else {
            let token = try!(self.bump_and_get());
            let lit = try!(self.lit_from_token(&token));
            lit
        };
        Ok(codemap::Spanned { node: lit, span: mk_sp(lo, self.last_span.hi) })
    }

    /// matches '-' lit | lit
    pub fn parse_literal_maybe_minus(&mut self) -> PResult<P<Expr>> {
        let minus_lo = self.span.lo;
        let minus_present = try!(self.eat(&token::BinOp(token::Minus)));

        let lo = self.span.lo;
        let literal = P(try!(self.parse_lit()));
        let hi = self.last_span.hi;
        let expr = self.mk_expr(lo, hi, ExprLit(literal));

        if minus_present {
            let minus_hi = self.last_span.hi;
            let unary = self.mk_unary(UnNeg, expr);
            Ok(self.mk_expr(minus_lo, minus_hi, unary))
        } else {
            Ok(expr)
        }
    }

    /// Parses qualified path.
    ///
    /// Assumes that the leading `<` has been parsed already.
    ///
    /// Qualifed paths are a part of the universal function call
    /// syntax (UFCS).
    ///
    /// `qualified_path = <type [as trait_ref]>::path`
    ///
    /// See `parse_path` for `mode` meaning.
    ///
    /// # Examples:
    ///
    /// `<T as U>::a`
    /// `<T as U>::F::a::<S>`
    pub fn parse_qualified_path(&mut self, mode: PathParsingMode)
                                -> PResult<(QSelf, ast::Path)> {
        let span = self.last_span;
        let self_type = try!(self.parse_ty_sum());
        let mut path = if try!(self.eat_keyword(keywords::As)) {
            try!(self.parse_path(LifetimeAndTypesWithoutColons))
        } else {
            ast::Path {
                span: span,
                global: false,
                segments: vec![]
            }
        };

        let qself = QSelf {
            ty: self_type,
            position: path.segments.len()
        };

        try!(self.expect(&token::Gt));
        try!(self.expect(&token::ModSep));

        let segments = match mode {
            LifetimeAndTypesWithoutColons => {
                try!(self.parse_path_segments_without_colons())
            }
            LifetimeAndTypesWithColons => {
                try!(self.parse_path_segments_with_colons())
            }
            NoTypesAllowed => {
                try!(self.parse_path_segments_without_types())
            }
        };
        path.segments.extend(segments);

        path.span.hi = self.last_span.hi;

        Ok((qself, path))
    }

    /// Parses a path and optional type parameter bounds, depending on the
    /// mode. The `mode` parameter determines whether lifetimes, types, and/or
    /// bounds are permitted and whether `::` must precede type parameter
    /// groups.
    pub fn parse_path(&mut self, mode: PathParsingMode) -> PResult<ast::Path> {
        // Check for a whole path...
        let found = match self.token {
            token::Interpolated(token::NtPath(_)) => Some(try!(self.bump_and_get())),
            _ => None,
        };
        if let Some(token::Interpolated(token::NtPath(path))) = found {
            return Ok(*path);
        }

        let lo = self.span.lo;
        let is_global = try!(self.eat(&token::ModSep));

        // Parse any number of segments and bound sets. A segment is an
        // identifier followed by an optional lifetime and a set of types.
        // A bound set is a set of type parameter bounds.
        let segments = match mode {
            LifetimeAndTypesWithoutColons => {
                try!(self.parse_path_segments_without_colons())
            }
            LifetimeAndTypesWithColons => {
                try!(self.parse_path_segments_with_colons())
            }
            NoTypesAllowed => {
                try!(self.parse_path_segments_without_types())
            }
        };

        // Assemble the span.
        let span = mk_sp(lo, self.last_span.hi);

        // Assemble the result.
        Ok(ast::Path {
            span: span,
            global: is_global,
            segments: segments,
        })
    }

    /// Examples:
    /// - `a::b<T,U>::c<V,W>`
    /// - `a::b<T,U>::c(V) -> W`
    /// - `a::b<T,U>::c(V)`
    pub fn parse_path_segments_without_colons(&mut self) -> PResult<Vec<ast::PathSegment>> {
        let mut segments = Vec::new();
        loop {
            // First, parse an identifier.
            let identifier = try!(self.parse_ident_or_self_type());

            // Parse types, optionally.
            let parameters = if try!(self.eat_lt() ){
                let (lifetimes, types, bindings) = try!(self.parse_generic_values_after_lt());

                ast::AngleBracketedParameters(ast::AngleBracketedParameterData {
                    lifetimes: lifetimes,
                    types: OwnedSlice::from_vec(types),
                    bindings: OwnedSlice::from_vec(bindings),
                })
            } else if try!(self.eat(&token::OpenDelim(token::Paren)) ){
                let lo = self.last_span.lo;

                let inputs = try!(self.parse_seq_to_end(
                    &token::CloseDelim(token::Paren),
                    seq_sep_trailing_allowed(token::Comma),
                    |p| p.parse_ty_sum()));

                let output_ty = if try!(self.eat(&token::RArrow) ){
                    Some(try!(self.parse_ty()))
                } else {
                    None
                };

                let hi = self.last_span.hi;

                ast::ParenthesizedParameters(ast::ParenthesizedParameterData {
                    span: mk_sp(lo, hi),
                    inputs: inputs,
                    output: output_ty,
                })
            } else {
                ast::PathParameters::none()
            };

            // Assemble and push the result.
            segments.push(ast::PathSegment { identifier: identifier,
                                             parameters: parameters });

            // Continue only if we see a `::`
            if !try!(self.eat(&token::ModSep) ){
                return Ok(segments);
            }
        }
    }

    /// Examples:
    /// - `a::b::<T,U>::c`
    pub fn parse_path_segments_with_colons(&mut self) -> PResult<Vec<ast::PathSegment>> {
        let mut segments = Vec::new();
        loop {
            // First, parse an identifier.
            let identifier = try!(self.parse_ident_or_self_type());

            // If we do not see a `::`, stop.
            if !try!(self.eat(&token::ModSep) ){
                segments.push(ast::PathSegment {
                    identifier: identifier,
                    parameters: ast::PathParameters::none()
                });
                return Ok(segments);
            }

            // Check for a type segment.
            if try!(self.eat_lt() ){
                // Consumed `a::b::<`, go look for types
                let (lifetimes, types, bindings) = try!(self.parse_generic_values_after_lt());
                segments.push(ast::PathSegment {
                    identifier: identifier,
                    parameters: ast::AngleBracketedParameters(ast::AngleBracketedParameterData {
                        lifetimes: lifetimes,
                        types: OwnedSlice::from_vec(types),
                        bindings: OwnedSlice::from_vec(bindings),
                    }),
                });

                // Consumed `a::b::<T,U>`, check for `::` before proceeding
                if !try!(self.eat(&token::ModSep) ){
                    return Ok(segments);
                }
            } else {
                // Consumed `a::`, go look for `b`
                segments.push(ast::PathSegment {
                    identifier: identifier,
                    parameters: ast::PathParameters::none(),
                });
            }
        }
    }


    /// Examples:
    /// - `a::b::c`
    pub fn parse_path_segments_without_types(&mut self) -> PResult<Vec<ast::PathSegment>> {
        let mut segments = Vec::new();
        loop {
            // First, parse an identifier.
            let identifier = try!(self.parse_ident_or_self_type());

            // Assemble and push the result.
            segments.push(ast::PathSegment {
                identifier: identifier,
                parameters: ast::PathParameters::none()
            });

            // If we do not see a `::`, stop.
            if !try!(self.eat(&token::ModSep) ){
                return Ok(segments);
            }
        }
    }

    /// parses 0 or 1 lifetime
    pub fn parse_opt_lifetime(&mut self) -> PResult<Option<ast::Lifetime>> {
        match self.token {
            token::Lifetime(..) => {
                Ok(Some(try!(self.parse_lifetime())))
            }
            _ => {
                Ok(None)
            }
        }
    }

    /// Parses a single lifetime
    /// Matches lifetime = LIFETIME
    pub fn parse_lifetime(&mut self) -> PResult<ast::Lifetime> {
        match self.token {
            token::Lifetime(i) => {
                let span = self.span;
                try!(self.bump());
                return Ok(ast::Lifetime {
                    id: ast::DUMMY_NODE_ID,
                    span: span,
                    name: i.name
                });
            }
            _ => {
                return Err(self.fatal("expected a lifetime name"));
            }
        }
    }

    /// Parses `lifetime_defs = [ lifetime_defs { ',' lifetime_defs } ]` where `lifetime_def  =
    /// lifetime [':' lifetimes]`
    pub fn parse_lifetime_defs(&mut self) -> PResult<Vec<ast::LifetimeDef>> {

        let mut res = Vec::new();
        loop {
            match self.token {
                token::Lifetime(_) => {
                    let lifetime = try!(self.parse_lifetime());
                    let bounds =
                        if try!(self.eat(&token::Colon) ){
                            try!(self.parse_lifetimes(token::BinOp(token::Plus)))
                        } else {
                            Vec::new()
                        };
                    res.push(ast::LifetimeDef { lifetime: lifetime,
                                                bounds: bounds });
                }

                _ => {
                    return Ok(res);
                }
            }

            match self.token {
                token::Comma => { try!(self.bump());}
                token::Gt => { return Ok(res); }
                token::BinOp(token::Shr) => { return Ok(res); }
                _ => {
                    let this_token_str = self.this_token_to_string();
                    let msg = format!("expected `,` or `>` after lifetime \
                                      name, found `{}`",
                                      this_token_str);
                    return Err(self.fatal(&msg[..]));
                }
            }
        }
    }

    /// matches lifetimes = ( lifetime ) | ( lifetime , lifetimes ) actually, it matches the empty
    /// one too, but putting that in there messes up the grammar....
    ///
    /// Parses zero or more comma separated lifetimes. Expects each lifetime to be followed by
    /// either a comma or `>`.  Used when parsing type parameter lists, where we expect something
    /// like `<'a, 'b, T>`.
    pub fn parse_lifetimes(&mut self, sep: token::Token) -> PResult<Vec<ast::Lifetime>> {

        let mut res = Vec::new();
        loop {
            match self.token {
                token::Lifetime(_) => {
                    res.push(try!(self.parse_lifetime()));
                }
                _ => {
                    return Ok(res);
                }
            }

            if self.token != sep {
                return Ok(res);
            }

            try!(self.bump());
        }
    }

    /// Parse mutability declaration (mut/const/imm)
    pub fn parse_mutability(&mut self) -> PResult<Mutability> {
        if try!(self.eat_keyword(keywords::Mut) ){
            Ok(MutMutable)
        } else {
            Ok(MutImmutable)
        }
    }

    /// Parse ident COLON expr
    pub fn parse_field(&mut self) -> PResult<Field> {
        let lo = self.span.lo;
        let i = try!(self.parse_ident());
        let hi = self.last_span.hi;
        try!(self.expect(&token::Colon));
        let e = try!(self.parse_expr());
        Ok(ast::Field {
            ident: spanned(lo, hi, i),
            span: mk_sp(lo, e.span.hi),
            expr: e,
        })
    }

    pub fn mk_expr(&mut self, lo: BytePos, hi: BytePos, node: Expr_) -> P<Expr> {
        P(Expr {
            id: ast::DUMMY_NODE_ID,
            node: node,
            span: mk_sp(lo, hi),
        })
    }

    pub fn mk_unary(&mut self, unop: ast::UnOp, expr: P<Expr>) -> ast::Expr_ {
        ExprUnary(unop, expr)
    }

    pub fn mk_binary(&mut self, binop: ast::BinOp, lhs: P<Expr>, rhs: P<Expr>) -> ast::Expr_ {
        ExprBinary(binop, lhs, rhs)
    }

    pub fn mk_call(&mut self, f: P<Expr>, args: Vec<P<Expr>>) -> ast::Expr_ {
        ExprCall(f, args)
    }

    fn mk_method_call(&mut self,
                      ident: ast::SpannedIdent,
                      tps: Vec<P<Ty>>,
                      args: Vec<P<Expr>>)
                      -> ast::Expr_ {
        ExprMethodCall(ident, tps, args)
    }

    pub fn mk_index(&mut self, expr: P<Expr>, idx: P<Expr>) -> ast::Expr_ {
        ExprIndex(expr, idx)
    }

    pub fn mk_range(&mut self,
                    start: Option<P<Expr>>,
                    end: Option<P<Expr>>)
                    -> ast::Expr_ {
        ExprRange(start, end)
    }

    pub fn mk_field(&mut self, expr: P<Expr>, ident: ast::SpannedIdent) -> ast::Expr_ {
        ExprField(expr, ident)
    }

    pub fn mk_tup_field(&mut self, expr: P<Expr>, idx: codemap::Spanned<usize>) -> ast::Expr_ {
        ExprTupField(expr, idx)
    }

    pub fn mk_assign_op(&mut self, binop: ast::BinOp,
                        lhs: P<Expr>, rhs: P<Expr>) -> ast::Expr_ {
        ExprAssignOp(binop, lhs, rhs)
    }

    pub fn mk_mac_expr(&mut self, lo: BytePos, hi: BytePos, m: Mac_) -> P<Expr> {
        P(Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprMac(codemap::Spanned {node: m, span: mk_sp(lo, hi)}),
            span: mk_sp(lo, hi),
        })
    }

    pub fn mk_lit_u32(&mut self, i: u32) -> P<Expr> {
        let span = &self.span;
        let lv_lit = P(codemap::Spanned {
            node: LitInt(i as u64, ast::UnsignedIntLit(TyU32)),
            span: *span
        });

        P(Expr {
            id: ast::DUMMY_NODE_ID,
            node: ExprLit(lv_lit),
            span: *span,
        })
    }

    fn expect_open_delim(&mut self) -> PResult<token::DelimToken> {
        self.expected_tokens.push(TokenType::Token(token::Gt));
        match self.token {
            token::OpenDelim(delim) => {
                try!(self.bump());
                Ok(delim)
            },
            _ => Err(self.fatal("expected open delimiter")),
        }
    }

    /// At the bottom (top?) of the precedence hierarchy,
    /// parse things like parenthesized exprs,
    /// macros, return, etc.
    pub fn parse_bottom_expr(&mut self) -> PResult<P<Expr>> {
        maybe_whole_expr!(self);

        let lo = self.span.lo;
        let mut hi = self.span.hi;

        let ex: Expr_;

        // Note: when adding new syntax here, don't forget to adjust Token::can_begin_expr().
        match self.token {
            token::OpenDelim(token::Paren) => {
                try!(self.bump());

                // (e) is parenthesized e
                // (e,) is a tuple with only one field, e
                let mut es = vec![];
                let mut trailing_comma = false;
                while self.token != token::CloseDelim(token::Paren) {
                    es.push(try!(self.parse_expr()));
                    try!(self.commit_expr(&**es.last().unwrap(), &[],
                                     &[token::Comma, token::CloseDelim(token::Paren)]));
                    if self.check(&token::Comma) {
                        trailing_comma = true;

                        try!(self.bump());
                    } else {
                        trailing_comma = false;
                        break;
                    }
                }
                try!(self.bump());

                hi = self.last_span.hi;
                return if es.len() == 1 && !trailing_comma {
                    Ok(self.mk_expr(lo, hi, ExprParen(es.into_iter().nth(0).unwrap())))
                } else {
                    Ok(self.mk_expr(lo, hi, ExprTup(es)))
                }
            },
            token::OpenDelim(token::Brace) => {
                return self.parse_block_expr(lo, DefaultBlock);
            },
            token::BinOp(token::Or) |  token::OrOr => {
                let lo = self.span.lo;
                return self.parse_lambda_expr(lo, CaptureByRef);
            },
            token::Ident(id @ ast::Ident {
                            name: token::SELF_KEYWORD_NAME,
                            ctxt: _
                         }, token::Plain) => {
                try!(self.bump());
                let path = ast_util::ident_to_path(mk_sp(lo, hi), id);
                ex = ExprPath(None, path);
                hi = self.last_span.hi;
            }
            token::OpenDelim(token::Bracket) => {
                try!(self.bump());

                if self.check(&token::CloseDelim(token::Bracket)) {
                    // Empty vector.
                    try!(self.bump());
                    ex = ExprVec(Vec::new());
                } else {
                    // Nonempty vector.
                    let first_expr = try!(self.parse_expr());
                    if self.check(&token::Semi) {
                        // Repeating array syntax: [ 0; 512 ]
                        try!(self.bump());
                        let count = try!(self.parse_expr());
                        try!(self.expect(&token::CloseDelim(token::Bracket)));
                        ex = ExprRepeat(first_expr, count);
                    } else if self.check(&token::Comma) {
                        // Vector with two or more elements.
                        try!(self.bump());
                        let remaining_exprs = try!(self.parse_seq_to_end(
                            &token::CloseDelim(token::Bracket),
                            seq_sep_trailing_allowed(token::Comma),
                            |p| Ok(try!(p.parse_expr()))
                                ));
                        let mut exprs = vec!(first_expr);
                        exprs.extend(remaining_exprs);
                        ex = ExprVec(exprs);
                    } else {
                        // Vector with one element.
                        try!(self.expect(&token::CloseDelim(token::Bracket)));
                        ex = ExprVec(vec!(first_expr));
                    }
                }
                hi = self.last_span.hi;
            }
            _ => {
                if try!(self.eat_lt()){
                    let (qself, path) =
                        try!(self.parse_qualified_path(LifetimeAndTypesWithColons));
                    hi = path.span.hi;
                    return Ok(self.mk_expr(lo, hi, ExprPath(Some(qself), path)));
                }
                if try!(self.eat_keyword(keywords::Move) ){
                    let lo = self.last_span.lo;
                    return self.parse_lambda_expr(lo, CaptureByValue);
                }
                if try!(self.eat_keyword(keywords::If)) {
                    return self.parse_if_expr();
                }
                if try!(self.eat_keyword(keywords::For) ){
                    let lo = self.last_span.lo;
                    return self.parse_for_expr(None, lo);
                }
                if try!(self.eat_keyword(keywords::While) ){
                    let lo = self.last_span.lo;
                    return self.parse_while_expr(None, lo);
                }
                if self.token.is_lifetime() {
                    let lifetime = self.get_lifetime();
                    let lo = self.span.lo;
                    try!(self.bump());
                    try!(self.expect(&token::Colon));
                    if try!(self.eat_keyword(keywords::While) ){
                        return self.parse_while_expr(Some(lifetime), lo)
                    }
                    if try!(self.eat_keyword(keywords::For) ){
                        return self.parse_for_expr(Some(lifetime), lo)
                    }
                    if try!(self.eat_keyword(keywords::Loop) ){
                        return self.parse_loop_expr(Some(lifetime), lo)
                    }
                    return Err(self.fatal("expected `while`, `for`, or `loop` after a label"))
                }
                if try!(self.eat_keyword(keywords::Loop) ){
                    let lo = self.last_span.lo;
                    return self.parse_loop_expr(None, lo);
                }
                if try!(self.eat_keyword(keywords::Continue) ){
                    let ex = if self.token.is_lifetime() {
                        let ex = ExprAgain(Some(Spanned{
                            node: self.get_lifetime(),
                            span: self.span
                        }));
                        try!(self.bump());
                        ex
                    } else {
                        ExprAgain(None)
                    };
                    let hi = self.last_span.hi;
                    return Ok(self.mk_expr(lo, hi, ex));
                }
                if try!(self.eat_keyword(keywords::Match) ){
                    return self.parse_match_expr();
                }
                if try!(self.eat_keyword(keywords::Unsafe) ){
                    return self.parse_block_expr(
                        lo,
                        UnsafeBlock(ast::UserProvided));
                }
                if try!(self.eat_keyword(keywords::Return) ){
                    if self.token.can_begin_expr() {
                        let e = try!(self.parse_expr());
                        hi = e.span.hi;
                        ex = ExprRet(Some(e));
                    } else {
                        ex = ExprRet(None);
                    }
                } else if try!(self.eat_keyword(keywords::Break) ){
                    if self.token.is_lifetime() {
                        ex = ExprBreak(Some(Spanned {
                            node: self.get_lifetime(),
                            span: self.span
                        }));
                        try!(self.bump());
                    } else {
                        ex = ExprBreak(None);
                    }
                    hi = self.last_span.hi;
                } else if self.check(&token::ModSep) ||
                        self.token.is_ident() &&
                        !self.check_keyword(keywords::True) &&
                        !self.check_keyword(keywords::False) {
                    let pth =
                        try!(self.parse_path(LifetimeAndTypesWithColons));

                    // `!`, as an operator, is prefix, so we know this isn't that
                    if self.check(&token::Not) {
                        // MACRO INVOCATION expression
                        try!(self.bump());

                        let delim = try!(self.expect_open_delim());
                        let tts = try!(self.parse_seq_to_end(
                            &token::CloseDelim(delim),
                            seq_sep_none(),
                            |p| p.parse_token_tree()));
                        let hi = self.last_span.hi;

                        return Ok(self.mk_mac_expr(lo,
                                                   hi,
                                                   Mac_ { path: pth, tts: tts, ctxt: EMPTY_CTXT }));
                    }
                    if self.check(&token::OpenDelim(token::Brace)) {
                        // This is a struct literal, unless we're prohibited
                        // from parsing struct literals here.
                        let prohibited = self.restrictions.contains(
                            Restrictions::RESTRICTION_NO_STRUCT_LITERAL
                        );
                        if !prohibited {
                            // It's a struct literal.
                            try!(self.bump());
                            let mut fields = Vec::new();
                            let mut base = None;

                            while self.token != token::CloseDelim(token::Brace) {
                                if try!(self.eat(&token::DotDot) ){
                                    base = Some(try!(self.parse_expr()));
                                    break;
                                }

                                fields.push(try!(self.parse_field()));
                                try!(self.commit_expr(&*fields.last().unwrap().expr,
                                                 &[token::Comma],
                                                 &[token::CloseDelim(token::Brace)]));
                            }

                            hi = self.span.hi;
                            try!(self.expect(&token::CloseDelim(token::Brace)));
                            ex = ExprStruct(pth, fields, base);
                            return Ok(self.mk_expr(lo, hi, ex));
                        }
                    }

                    hi = pth.span.hi;
                    ex = ExprPath(None, pth);
                } else {
                    // other literal expression
                    let lit = try!(self.parse_lit());
                    hi = lit.span.hi;
                    ex = ExprLit(P(lit));
                }
            }
        }

        return Ok(self.mk_expr(lo, hi, ex));
    }

    /// Parse a block or unsafe block
    pub fn parse_block_expr(&mut self, lo: BytePos, blk_mode: BlockCheckMode)
                            -> PResult<P<Expr>> {
        try!(self.expect(&token::OpenDelim(token::Brace)));
        let blk = try!(self.parse_block_tail(lo, blk_mode));
        return Ok(self.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk)));
    }

    /// parse a.b or a(13) or a[4] or just a
    pub fn parse_dot_or_call_expr(&mut self) -> PResult<P<Expr>> {
        let b = try!(self.parse_bottom_expr());
        self.parse_dot_or_call_expr_with(b)
    }

    pub fn parse_dot_or_call_expr_with(&mut self, e0: P<Expr>) -> PResult<P<Expr>> {
        let mut e = e0;
        let lo = e.span.lo;
        let mut hi;
        loop {
            // expr.f
            if try!(self.eat(&token::Dot) ){
                match self.token {
                  token::Ident(i, _) => {
                    let dot = self.last_span.hi;
                    hi = self.span.hi;
                    try!(self.bump());
                    let (_, tys, bindings) = if try!(self.eat(&token::ModSep) ){
                        try!(self.expect_lt());
                        try!(self.parse_generic_values_after_lt())
                    } else {
                        (Vec::new(), Vec::new(), Vec::new())
                    };

                    if !bindings.is_empty() {
                        let last_span = self.last_span;
                        self.span_err(last_span, "type bindings are only permitted on trait paths");
                    }

                    // expr.f() method call
                    match self.token {
                        token::OpenDelim(token::Paren) => {
                            let mut es = try!(self.parse_unspanned_seq(
                                &token::OpenDelim(token::Paren),
                                &token::CloseDelim(token::Paren),
                                seq_sep_trailing_allowed(token::Comma),
                                |p| Ok(try!(p.parse_expr()))
                            ));
                            hi = self.last_span.hi;

                            es.insert(0, e);
                            let id = spanned(dot, hi, i);
                            let nd = self.mk_method_call(id, tys, es);
                            e = self.mk_expr(lo, hi, nd);
                        }
                        _ => {
                            if !tys.is_empty() {
                                let last_span = self.last_span;
                                self.span_err(last_span,
                                              "field expressions may not \
                                               have type parameters");
                            }

                            let id = spanned(dot, hi, i);
                            let field = self.mk_field(e, id);
                            e = self.mk_expr(lo, hi, field);
                        }
                    }
                  }
                  token::Literal(token::Integer(n), suf) => {
                    let sp = self.span;

                    // A tuple index may not have a suffix
                    self.expect_no_suffix(sp, "tuple index", suf);

                    let dot = self.last_span.hi;
                    hi = self.span.hi;
                    try!(self.bump());

                    let index = n.as_str().parse::<usize>().ok();
                    match index {
                        Some(n) => {
                            let id = spanned(dot, hi, n);
                            let field = self.mk_tup_field(e, id);
                            e = self.mk_expr(lo, hi, field);
                        }
                        None => {
                            let last_span = self.last_span;
                            self.span_err(last_span, "invalid tuple or tuple struct index");
                        }
                    }
                  }
                  token::Literal(token::Float(n), _suf) => {
                    try!(self.bump());
                    let last_span = self.last_span;
                    let fstr = n.as_str();
                    self.span_err(last_span,
                                  &format!("unexpected token: `{}`", n.as_str()));
                    if fstr.chars().all(|x| "0123456789.".contains(x)) {
                        let float = match fstr.parse::<f64>().ok() {
                            Some(f) => f,
                            None => continue,
                        };
                        self.fileline_help(last_span,
                            &format!("try parenthesizing the first index; e.g., `(foo.{}){}`",
                                    float.trunc() as usize,
                                    format!(".{}", fstr.splitn(2, ".").last().unwrap())));
                    }
                    self.abort_if_errors();

                  }
                  _ => return Err(self.unexpected())
                }
                continue;
            }
            if self.expr_is_complete(&*e) { break; }
            match self.token {
              // expr(...)
              token::OpenDelim(token::Paren) => {
                let es = try!(self.parse_unspanned_seq(
                    &token::OpenDelim(token::Paren),
                    &token::CloseDelim(token::Paren),
                    seq_sep_trailing_allowed(token::Comma),
                    |p| Ok(try!(p.parse_expr()))
                ));
                hi = self.last_span.hi;

                let nd = self.mk_call(e, es);
                e = self.mk_expr(lo, hi, nd);
              }

              // expr[...]
              // Could be either an index expression or a slicing expression.
              token::OpenDelim(token::Bracket) => {
                try!(self.bump());
                let ix = try!(self.parse_expr());
                hi = self.span.hi;
                try!(self.commit_expr_expecting(&*ix, token::CloseDelim(token::Bracket)));
                let index = self.mk_index(e, ix);
                e = self.mk_expr(lo, hi, index)
              }
              _ => return Ok(e)
            }
        }
        return Ok(e);
    }

    // Parse unquoted tokens after a `$` in a token tree
    fn parse_unquoted(&mut self) -> PResult<TokenTree> {
        let mut sp = self.span;
        let (name, namep) = match self.token {
            token::Dollar => {
                try!(self.bump());

                if self.token == token::OpenDelim(token::Paren) {
                    let Spanned { node: seq, span: seq_span } = try!(self.parse_seq(
                        &token::OpenDelim(token::Paren),
                        &token::CloseDelim(token::Paren),
                        seq_sep_none(),
                        |p| p.parse_token_tree()
                    ));
                    let (sep, repeat) = try!(self.parse_sep_and_kleene_op());
                    let name_num = macro_parser::count_names(&seq);
                    return Ok(TokenTree::Sequence(mk_sp(sp.lo, seq_span.hi),
                                      Rc::new(SequenceRepetition {
                                          tts: seq,
                                          separator: sep,
                                          op: repeat,
                                          num_captures: name_num
                                      })));
                } else if self.token.is_keyword_allow_following_colon(keywords::Crate) {
                    try!(self.bump());
                    return Ok(TokenTree::Token(sp, SpecialVarNt(SpecialMacroVar::CrateMacroVar)));
                } else {
                    sp = mk_sp(sp.lo, self.span.hi);
                    let namep = match self.token { token::Ident(_, p) => p, _ => token::Plain };
                    let name = try!(self.parse_ident());
                    (name, namep)
                }
            }
            token::SubstNt(name, namep) => {
                try!(self.bump());
                (name, namep)
            }
            _ => unreachable!()
        };
        // continue by trying to parse the `:ident` after `$name`
        if self.token == token::Colon && self.look_ahead(1, |t| t.is_ident() &&
                                                                !t.is_strict_keyword() &&
                                                                !t.is_reserved_keyword()) {
            try!(self.bump());
            sp = mk_sp(sp.lo, self.span.hi);
            let kindp = match self.token { token::Ident(_, p) => p, _ => token::Plain };
            let nt_kind = try!(self.parse_ident());
            Ok(TokenTree::Token(sp, MatchNt(name, nt_kind, namep, kindp)))
        } else {
            Ok(TokenTree::Token(sp, SubstNt(name, namep)))
        }
    }

    pub fn check_unknown_macro_variable(&mut self) -> PResult<()> {
        if self.quote_depth == 0 {
            match self.token {
                token::SubstNt(name, _) =>
                    return Err(self.fatal(&format!("unknown macro variable `{}`",
                                       name))),
                _ => {}
            }
        }
        Ok(())
    }

    /// Parse an optional separator followed by a Kleene-style
    /// repetition token (+ or *).
    pub fn parse_sep_and_kleene_op(&mut self) -> PResult<(Option<token::Token>, ast::KleeneOp)> {
        fn parse_kleene_op(parser: &mut Parser) -> PResult<Option<ast::KleeneOp>> {
            match parser.token {
                token::BinOp(token::Star) => {
                    try!(parser.bump());
                    Ok(Some(ast::ZeroOrMore))
                },
                token::BinOp(token::Plus) => {
                    try!(parser.bump());
                    Ok(Some(ast::OneOrMore))
                },
                _ => Ok(None)
            }
        };

        match try!(parse_kleene_op(self)) {
            Some(kleene_op) => return Ok((None, kleene_op)),
            None => {}
        }

        let separator = try!(self.bump_and_get());
        match try!(parse_kleene_op(self)) {
            Some(zerok) => Ok((Some(separator), zerok)),
            None => return Err(self.fatal("expected `*` or `+`"))
        }
    }

    /// parse a single token tree from the input.
    pub fn parse_token_tree(&mut self) -> PResult<TokenTree> {
        // FIXME #6994: currently, this is too eager. It
        // parses token trees but also identifies TokenType::Sequence's
        // and token::SubstNt's; it's too early to know yet
        // whether something will be a nonterminal or a seq
        // yet.
        maybe_whole!(deref self, NtTT);

        // this is the fall-through for the 'match' below.
        // invariants: the current token is not a left-delimiter,
        // not an EOF, and not the desired right-delimiter (if
        // it were, parse_seq_to_before_end would have prevented
        // reaching this point.
        fn parse_non_delim_tt_tok(p: &mut Parser) -> PResult<TokenTree> {
            maybe_whole!(deref p, NtTT);
            match p.token {
                token::CloseDelim(_) => {
                    // This is a conservative error: only report the last unclosed delimiter. The
                    // previous unclosed delimiters could actually be closed! The parser just hasn't
                    // gotten to them yet.
                    match p.open_braces.last() {
                        None => {}
                        Some(&sp) => p.span_note(sp, "unclosed delimiter"),
                    };
                    let token_str = p.this_token_to_string();
                    Err(p.fatal(&format!("incorrect close delimiter: `{}`",
                                    token_str)))
                },
                /* we ought to allow different depths of unquotation */
                token::Dollar | token::SubstNt(..) if p.quote_depth > 0 => {
                    p.parse_unquoted()
                }
                _ => {
                    Ok(TokenTree::Token(p.span, try!(p.bump_and_get())))
                }
            }
        }

        match self.token {
            token::Eof => {
                let open_braces = self.open_braces.clone();
                for sp in &open_braces {
                    self.span_help(*sp, "did you mean to close this delimiter?");
                }
                // There shouldn't really be a span, but it's easier for the test runner
                // if we give it one
                return Err(self.fatal("this file contains an un-closed delimiter "));
            },
            token::OpenDelim(delim) => {
                // The span for beginning of the delimited section
                let pre_span = self.span;

                // Parse the open delimiter.
                self.open_braces.push(self.span);
                let open_span = self.span;
                try!(self.bump());

                // Parse the token trees within the delimiters
                let tts = try!(self.parse_seq_to_before_end(
                    &token::CloseDelim(delim),
                    seq_sep_none(),
                    |p| p.parse_token_tree()
                ));

                // Parse the close delimiter.
                let close_span = self.span;
                try!(self.bump());
                self.open_braces.pop().unwrap();

                // Expand to cover the entire delimited token tree
                let span = Span { hi: close_span.hi, ..pre_span };

                Ok(TokenTree::Delimited(span, Rc::new(Delimited {
                    delim: delim,
                    open_span: open_span,
                    tts: tts,
                    close_span: close_span,
                })))
            },
            _ => parse_non_delim_tt_tok(self),
        }
    }

    // parse a stream of tokens into a list of TokenTree's,
    // up to EOF.
    pub fn parse_all_token_trees(&mut self) -> PResult<Vec<TokenTree>> {
        let mut tts = Vec::new();
        while self.token != token::Eof {
            tts.push(try!(self.parse_token_tree()));
        }
        Ok(tts)
    }

    /// Parse a prefix-unary-operator expr
    pub fn parse_prefix_expr(&mut self) -> PResult<P<Expr>> {
        let lo = self.span.lo;
        let hi;
        // Note: when adding new unary operators, don't forget to adjust Token::can_begin_expr()
        let ex = match self.token {
            token::Not => {
                try!(self.bump());
                let e = try!(self.parse_prefix_expr());
                hi = e.span.hi;
                self.mk_unary(UnNot, e)
            }
            token::BinOp(token::Minus) => {
                try!(self.bump());
                let e = try!(self.parse_prefix_expr());
                hi = e.span.hi;
                self.mk_unary(UnNeg, e)
            }
            token::BinOp(token::Star) => {
                try!(self.bump());
                let e = try!(self.parse_prefix_expr());
                hi = e.span.hi;
                self.mk_unary(UnDeref, e)
            }
            token::BinOp(token::And) | token::AndAnd => {
                try!(self.expect_and());
                let m = try!(self.parse_mutability());
                let e = try!(self.parse_prefix_expr());
                hi = e.span.hi;
                ExprAddrOf(m, e)
            }
            token::Ident(..) if self.token.is_keyword(keywords::In) => {
                try!(self.bump());
                let place = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
                let blk = try!(self.parse_block());
                let span = blk.span;
                hi = span.hi;
                let blk_expr = self.mk_expr(span.lo, span.hi, ExprBlock(blk));
                ExprInPlace(place, blk_expr)
            }
            token::Ident(..) if self.token.is_keyword(keywords::Box) => {
                try!(self.bump());
                let subexpression = try!(self.parse_prefix_expr());
                hi = subexpression.span.hi;
                ExprBox(subexpression)
            }
            _ => return self.parse_dot_or_call_expr()
        };
        return Ok(self.mk_expr(lo, hi, ex));
    }

    /// Parse an associative expression
    ///
    /// This parses an expression accounting for associativity and precedence of the operators in
    /// the expression.
    pub fn parse_assoc_expr(&mut self) -> PResult<P<Expr>> {
        self.parse_assoc_expr_with(0, None)
    }

    /// Parse an associative expression with operators of at least `min_prec` precedence
    pub fn parse_assoc_expr_with(&mut self,
                                 min_prec: usize,
                                 lhs: Option<P<Expr>>)
                                 -> PResult<P<Expr>> {
        let mut lhs = if lhs.is_some() {
            lhs.unwrap()
        } else if self.token == token::DotDot {
            return self.parse_prefix_range_expr();
        } else {
            try!(self.parse_prefix_expr())
        };
        if self.expr_is_complete(&*lhs) {
            // Semi-statement forms are odd. See https://github.com/rust-lang/rust/issues/29071
            return Ok(lhs);
        }
        self.expected_tokens.push(TokenType::Operator);
        while let Some(op) = AssocOp::from_token(&self.token) {
            let cur_op_span = self.span;
            let restrictions = if op.is_assign_like() {
                self.restrictions & Restrictions::RESTRICTION_NO_STRUCT_LITERAL
            } else {
                self.restrictions
            };
            if op.precedence() < min_prec {
                break;
            }
            try!(self.bump());
            if op.is_comparison() {
                self.check_no_chained_comparison(&*lhs, &op);
            }
            // Special cases:
            if op == AssocOp::As {
                let rhs = try!(self.parse_ty());
                lhs = self.mk_expr(lhs.span.lo, rhs.span.hi, ExprCast(lhs, rhs));
                continue
            } else if op == AssocOp::DotDot {
                    // If we didn’t have to handle `x..`, it would be pretty easy to generalise
                    // it to the Fixity::None code.
                    //
                    // We have 2 alternatives here: `x..y` and `x..` The other two variants are
                    // handled with `parse_prefix_range_expr` call above.
                    let rhs = if self.is_at_start_of_range_notation_rhs() {
                        self.parse_assoc_expr_with(op.precedence() + 1, None).ok()
                    } else {
                        None
                    };
                    let (lhs_span, rhs_span) = (lhs.span, if let Some(ref x) = rhs {
                        x.span
                    } else {
                        cur_op_span
                    });
                    let r = self.mk_range(Some(lhs), rhs);
                    lhs = self.mk_expr(lhs_span.lo, rhs_span.hi, r);
                    break
            }


            let rhs = try!(match op.fixity() {
                Fixity::Right => self.with_res(restrictions, |this|{
                    this.parse_assoc_expr_with(op.precedence(), None)
                }),
                Fixity::Left => self.with_res(restrictions, |this|{
                    this.parse_assoc_expr_with(op.precedence() + 1, None)
                }),
                // We currently have no non-associative operators that are not handled above by
                // the special cases. The code is here only for future convenience.
                Fixity::None => self.with_res(restrictions, |this|{
                    this.parse_assoc_expr_with(op.precedence() + 1, None)
                }),
            });

            lhs = match op {
                AssocOp::Add | AssocOp::Subtract | AssocOp::Multiply | AssocOp::Divide |
                AssocOp::Modulus | AssocOp::LAnd | AssocOp::LOr | AssocOp::BitXor |
                AssocOp::BitAnd | AssocOp::BitOr | AssocOp::ShiftLeft | AssocOp::ShiftRight |
                AssocOp::Equal | AssocOp::Less | AssocOp::LessEqual | AssocOp::NotEqual |
                AssocOp::Greater | AssocOp::GreaterEqual => {
                    let ast_op = op.to_ast_binop().unwrap();
                    let (lhs_span, rhs_span) = (lhs.span, rhs.span);
                    let binary = self.mk_binary(codemap::respan(cur_op_span, ast_op), lhs, rhs);
                    self.mk_expr(lhs_span.lo, rhs_span.hi, binary)
                }
                AssocOp::Assign =>
                    self.mk_expr(lhs.span.lo, rhs.span.hi, ExprAssign(lhs, rhs)),
                AssocOp::Inplace =>
                    self.mk_expr(lhs.span.lo, rhs.span.hi, ExprInPlace(lhs, rhs)),
                AssocOp::AssignOp(k) => {
                    let aop = match k {
                        token::Plus =>    BiAdd,
                        token::Minus =>   BiSub,
                        token::Star =>    BiMul,
                        token::Slash =>   BiDiv,
                        token::Percent => BiRem,
                        token::Caret =>   BiBitXor,
                        token::And =>     BiBitAnd,
                        token::Or =>      BiBitOr,
                        token::Shl =>     BiShl,
                        token::Shr =>     BiShr
                    };
                    let (lhs_span, rhs_span) = (lhs.span, rhs.span);
                    let aopexpr = self.mk_assign_op(codemap::respan(cur_op_span, aop), lhs, rhs);
                    self.mk_expr(lhs_span.lo, rhs_span.hi, aopexpr)
                }
                AssocOp::As | AssocOp::DotDot => self.bug("As or DotDot branch reached")
            };

            if op.fixity() == Fixity::None { break }
        }
        Ok(lhs)
    }

    /// Produce an error if comparison operators are chained (RFC #558).
    /// We only need to check lhs, not rhs, because all comparison ops
    /// have same precedence and are left-associative
    fn check_no_chained_comparison(&mut self, lhs: &Expr, outer_op: &AssocOp) {
        debug_assert!(outer_op.is_comparison());
        match lhs.node {
            ExprBinary(op, _, _) if ast_util::is_comparison_binop(op.node) => {
                // respan to include both operators
                let op_span = mk_sp(op.span.lo, self.span.hi);
                self.span_err(op_span,
                    "chained comparison operators require parentheses");
                if op.node == BiLt && *outer_op == AssocOp::Greater {
                    self.fileline_help(op_span,
                        "use `::<...>` instead of `<...>` if you meant to specify type arguments");
                }
            }
            _ => {}
        }
    }

    /// Parse prefix-forms of range notation: `..expr` and `..`
    fn parse_prefix_range_expr(&mut self) -> PResult<P<Expr>> {
        debug_assert!(self.token == token::DotDot);
        let lo = self.span.lo;
        let mut hi = self.span.hi;
        try!(self.bump());
        let opt_end = if self.is_at_start_of_range_notation_rhs() {
            // RHS must be parsed with more associativity than DotDot.
            let next_prec = AssocOp::from_token(&token::DotDot).unwrap().precedence() + 1;
            Some(try!(self.parse_assoc_expr_with(next_prec, None).map(|x|{
                hi = x.span.hi;
                x
            })))
         } else {
            None
        };
        let r = self.mk_range(None, opt_end);
        Ok(self.mk_expr(lo, hi, r))
    }

    fn is_at_start_of_range_notation_rhs(&self) -> bool {
        if self.token.can_begin_expr() {
            // parse `for i in 1.. { }` as infinite loop, not as `for i in (1..{})`.
            if self.token == token::OpenDelim(token::Brace) {
                return !self.restrictions.contains(Restrictions::RESTRICTION_NO_STRUCT_LITERAL);
            }
            true
        } else {
            false
        }
    }

    /// Parse an 'if' or 'if let' expression ('if' token already eaten)
    pub fn parse_if_expr(&mut self) -> PResult<P<Expr>> {
        if self.check_keyword(keywords::Let) {
            return self.parse_if_let_expr();
        }
        let lo = self.last_span.lo;
        let cond = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
        let thn = try!(self.parse_block());
        let mut els: Option<P<Expr>> = None;
        let mut hi = thn.span.hi;
        if try!(self.eat_keyword(keywords::Else) ){
            let elexpr = try!(self.parse_else_expr());
            hi = elexpr.span.hi;
            els = Some(elexpr);
        }
        Ok(self.mk_expr(lo, hi, ExprIf(cond, thn, els)))
    }

    /// Parse an 'if let' expression ('if' token already eaten)
    pub fn parse_if_let_expr(&mut self) -> PResult<P<Expr>> {
        let lo = self.last_span.lo;
        try!(self.expect_keyword(keywords::Let));
        let pat = try!(self.parse_pat());
        try!(self.expect(&token::Eq));
        let expr = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
        let thn = try!(self.parse_block());
        let (hi, els) = if try!(self.eat_keyword(keywords::Else) ){
            let expr = try!(self.parse_else_expr());
            (expr.span.hi, Some(expr))
        } else {
            (thn.span.hi, None)
        };
        Ok(self.mk_expr(lo, hi, ExprIfLet(pat, expr, thn, els)))
    }

    // `|args| expr`
    pub fn parse_lambda_expr(&mut self, lo: BytePos, capture_clause: CaptureClause)
                             -> PResult<P<Expr>>
    {
        let decl = try!(self.parse_fn_block_decl());
        let body = match decl.output {
            DefaultReturn(_) => {
                // If no explicit return type is given, parse any
                // expr and wrap it up in a dummy block:
                let body_expr = try!(self.parse_expr());
                P(ast::Block {
                    id: ast::DUMMY_NODE_ID,
                    stmts: vec![],
                    span: body_expr.span,
                    expr: Some(body_expr),
                    rules: DefaultBlock,
                })
            }
            _ => {
                // If an explicit return type is given, require a
                // block to appear (RFC 968).
                try!(self.parse_block())
            }
        };

        Ok(self.mk_expr(
            lo,
            body.span.hi,
            ExprClosure(capture_clause, decl, body)))
    }

    pub fn parse_else_expr(&mut self) -> PResult<P<Expr>> {
        if try!(self.eat_keyword(keywords::If) ){
            return self.parse_if_expr();
        } else {
            let blk = try!(self.parse_block());
            return Ok(self.mk_expr(blk.span.lo, blk.span.hi, ExprBlock(blk)));
        }
    }

    /// Parse a 'for' .. 'in' expression ('for' token already eaten)
    pub fn parse_for_expr(&mut self, opt_ident: Option<ast::Ident>,
                          span_lo: BytePos) -> PResult<P<Expr>> {
        // Parse: `for <src_pat> in <src_expr> <src_loop_block>`

        let pat = try!(self.parse_pat());
        try!(self.expect_keyword(keywords::In));
        let expr = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
        let loop_block = try!(self.parse_block());
        let hi = self.last_span.hi;

        Ok(self.mk_expr(span_lo, hi, ExprForLoop(pat, expr, loop_block, opt_ident)))
    }

    /// Parse a 'while' or 'while let' expression ('while' token already eaten)
    pub fn parse_while_expr(&mut self, opt_ident: Option<ast::Ident>,
                            span_lo: BytePos) -> PResult<P<Expr>> {
        if self.token.is_keyword(keywords::Let) {
            return self.parse_while_let_expr(opt_ident, span_lo);
        }
        let cond = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
        let body = try!(self.parse_block());
        let hi = body.span.hi;
        return Ok(self.mk_expr(span_lo, hi, ExprWhile(cond, body, opt_ident)));
    }

    /// Parse a 'while let' expression ('while' token already eaten)
    pub fn parse_while_let_expr(&mut self, opt_ident: Option<ast::Ident>,
                                span_lo: BytePos) -> PResult<P<Expr>> {
        try!(self.expect_keyword(keywords::Let));
        let pat = try!(self.parse_pat());
        try!(self.expect(&token::Eq));
        let expr = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
        let body = try!(self.parse_block());
        let hi = body.span.hi;
        return Ok(self.mk_expr(span_lo, hi, ExprWhileLet(pat, expr, body, opt_ident)));
    }

    pub fn parse_loop_expr(&mut self, opt_ident: Option<ast::Ident>,
                           span_lo: BytePos) -> PResult<P<Expr>> {
        let body = try!(self.parse_block());
        let hi = body.span.hi;
        Ok(self.mk_expr(span_lo, hi, ExprLoop(body, opt_ident)))
    }

    fn parse_match_expr(&mut self) -> PResult<P<Expr>> {
        let match_span = self.last_span;
        let lo = self.last_span.lo;
        let discriminant = try!(self.parse_expr_res(Restrictions::RESTRICTION_NO_STRUCT_LITERAL));
        if let Err(e) = self.commit_expr_expecting(&*discriminant, token::OpenDelim(token::Brace)) {
            if self.token == token::Token::Semi {
                self.span_note(match_span, "did you mean to remove this `match` keyword?");
            }
            return Err(e)
        }
        let mut arms: Vec<Arm> = Vec::new();
        while self.token != token::CloseDelim(token::Brace) {
            arms.push(try!(self.parse_arm()));
        }
        let hi = self.span.hi;
        try!(self.bump());
        return Ok(self.mk_expr(lo, hi, ExprMatch(discriminant, arms)));
    }

    pub fn parse_arm(&mut self) -> PResult<Arm> {
        maybe_whole!(no_clone self, NtArm);

        let attrs = try!(self.parse_outer_attributes());
        let pats = try!(self.parse_pats());
        let mut guard = None;
        if try!(self.eat_keyword(keywords::If) ){
            guard = Some(try!(self.parse_expr()));
        }
        try!(self.expect(&token::FatArrow));
        let expr = try!(self.parse_expr_res(Restrictions::RESTRICTION_STMT_EXPR));

        let require_comma =
            !classify::expr_is_simple_block(&*expr)
            && self.token != token::CloseDelim(token::Brace);

        if require_comma {
            try!(self.commit_expr(&*expr, &[token::Comma], &[token::CloseDelim(token::Brace)]));
        } else {
            try!(self.eat(&token::Comma));
        }

        Ok(ast::Arm {
            attrs: attrs,
            pats: pats,
            guard: guard,
            body: expr,
        })
    }

    /// Parse an expression
    pub fn parse_expr(&mut self) -> PResult<P<Expr>> {
        self.parse_expr_res(Restrictions::empty())
    }

    /// Evaluate the closure with restrictions in place.
    ///
    /// After the closure is evaluated, restrictions are reset.
    pub fn with_res<F>(&mut self, r: Restrictions, f: F) -> PResult<P<Expr>>
    where F: FnOnce(&mut Self) -> PResult<P<Expr>> {
        let old = self.restrictions;
        self.restrictions = r;
        let r = f(self);
        self.restrictions = old;
        return r;

    }

    /// Parse an expression, subject to the given restrictions
    pub fn parse_expr_res(&mut self, r: Restrictions) -> PResult<P<Expr>> {
        self.with_res(r, |this| this.parse_assoc_expr())
    }

    /// Parse the RHS of a local variable declaration (e.g. '= 14;')
    fn parse_initializer(&mut self) -> PResult<Option<P<Expr>>> {
        if self.check(&token::Eq) {
            try!(self.bump());
            Ok(Some(try!(self.parse_expr())))
        } else {
            Ok(None)
        }
    }

    /// Parse patterns, separated by '|' s
    fn parse_pats(&mut self) -> PResult<Vec<P<Pat>>> {
        let mut pats = Vec::new();
        loop {
            pats.push(try!(self.parse_pat()));
            if self.check(&token::BinOp(token::Or)) { try!(self.bump());}
            else { return Ok(pats); }
        };
    }

    fn parse_pat_tuple_elements(&mut self) -> PResult<Vec<P<Pat>>> {
        let mut fields = vec![];
        if !self.check(&token::CloseDelim(token::Paren)) {
            fields.push(try!(self.parse_pat()));
            if self.look_ahead(1, |t| *t != token::CloseDelim(token::Paren)) {
                while try!(self.eat(&token::Comma)) &&
                      !self.check(&token::CloseDelim(token::Paren)) {
                    fields.push(try!(self.parse_pat()));
                }
            }
            if fields.len() == 1 {
                try!(self.expect(&token::Comma));
            }
        }
        Ok(fields)
    }

    fn parse_pat_vec_elements(
        &mut self,
    ) -> PResult<(Vec<P<Pat>>, Option<P<Pat>>, Vec<P<Pat>>)> {
        let mut before = Vec::new();
        let mut slice = None;
        let mut after = Vec::new();
        let mut first = true;
        let mut before_slice = true;

        while self.token != token::CloseDelim(token::Bracket) {
            if first {
                first = false;
            } else {
                try!(self.expect(&token::Comma));

                if self.token == token::CloseDelim(token::Bracket)
                        && (before_slice || !after.is_empty()) {
                    break
                }
            }

            if before_slice {
                if self.check(&token::DotDot) {
                    try!(self.bump());

                    if self.check(&token::Comma) ||
                            self.check(&token::CloseDelim(token::Bracket)) {
                        slice = Some(P(ast::Pat {
                            id: ast::DUMMY_NODE_ID,
                            node: PatWild,
                            span: self.span,
                        }));
                        before_slice = false;
                    }
                    continue
                }
            }

            let subpat = try!(self.parse_pat());
            if before_slice && self.check(&token::DotDot) {
                try!(self.bump());
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
    fn parse_pat_fields(&mut self) -> PResult<(Vec<codemap::Spanned<ast::FieldPat>> , bool)> {
        let mut fields = Vec::new();
        let mut etc = false;
        let mut first = true;
        while self.token != token::CloseDelim(token::Brace) {
            if first {
                first = false;
            } else {
                try!(self.expect(&token::Comma));
                // accept trailing commas
                if self.check(&token::CloseDelim(token::Brace)) { break }
            }

            let lo = self.span.lo;
            let hi;

            if self.check(&token::DotDot) {
                try!(self.bump());
                if self.token != token::CloseDelim(token::Brace) {
                    let token_str = self.this_token_to_string();
                    return Err(self.fatal(&format!("expected `{}`, found `{}`", "}",
                                       token_str)))
                }
                etc = true;
                break;
            }

            // Check if a colon exists one ahead. This means we're parsing a fieldname.
            let (subpat, fieldname, is_shorthand) = if self.look_ahead(1, |t| t == &token::Colon) {
                // Parsing a pattern of the form "fieldname: pat"
                let fieldname = try!(self.parse_ident());
                try!(self.bump());
                let pat = try!(self.parse_pat());
                hi = pat.span.hi;
                (pat, fieldname, false)
            } else {
                // Parsing a pattern of the form "(box) (ref) (mut) fieldname"
                let is_box = try!(self.eat_keyword(keywords::Box));
                let boxed_span_lo = self.span.lo;
                let is_ref = try!(self.eat_keyword(keywords::Ref));
                let is_mut = try!(self.eat_keyword(keywords::Mut));
                let fieldname = try!(self.parse_ident());
                hi = self.last_span.hi;

                let bind_type = match (is_ref, is_mut) {
                    (true, true) => BindByRef(MutMutable),
                    (true, false) => BindByRef(MutImmutable),
                    (false, true) => BindByValue(MutMutable),
                    (false, false) => BindByValue(MutImmutable),
                };
                let fieldpath = codemap::Spanned{span:self.last_span, node:fieldname};
                let fieldpat = P(ast::Pat{
                    id: ast::DUMMY_NODE_ID,
                    node: PatIdent(bind_type, fieldpath, None),
                    span: mk_sp(boxed_span_lo, hi),
                });

                let subpat = if is_box {
                    P(ast::Pat{
                        id: ast::DUMMY_NODE_ID,
                        node: PatBox(fieldpat),
                        span: mk_sp(lo, hi),
                    })
                } else {
                    fieldpat
                };
                (subpat, fieldname, true)
            };

            fields.push(codemap::Spanned { span: mk_sp(lo, hi),
                                           node: ast::FieldPat { ident: fieldname,
                                                                 pat: subpat,
                                                                 is_shorthand: is_shorthand }});
        }
        return Ok((fields, etc));
    }

    fn parse_pat_range_end(&mut self) -> PResult<P<Expr>> {
        if self.is_path_start() {
            let lo = self.span.lo;
            let (qself, path) = if try!(self.eat_lt()) {
                // Parse a qualified path
                let (qself, path) =
                    try!(self.parse_qualified_path(NoTypesAllowed));
                (Some(qself), path)
            } else {
                // Parse an unqualified path
                (None, try!(self.parse_path(LifetimeAndTypesWithColons)))
            };
            let hi = self.last_span.hi;
            Ok(self.mk_expr(lo, hi, ExprPath(qself, path)))
        } else {
            self.parse_literal_maybe_minus()
        }
    }

    fn is_path_start(&self) -> bool {
        (self.token == token::Lt || self.token == token::ModSep
            || self.token.is_ident() || self.token.is_path())
            && !self.token.is_keyword(keywords::True) && !self.token.is_keyword(keywords::False)
    }

    /// Parse a pattern.
    pub fn parse_pat(&mut self) -> PResult<P<Pat>> {
        maybe_whole!(self, NtPat);

        let lo = self.span.lo;
        let pat;
        match self.token {
          token::Underscore => {
            // Parse _
            try!(self.bump());
            pat = PatWild;
          }
          token::BinOp(token::And) | token::AndAnd => {
            // Parse &pat / &mut pat
            try!(self.expect_and());
            let mutbl = try!(self.parse_mutability());
            if let token::Lifetime(ident) = self.token {
                return Err(self.fatal(&format!("unexpected lifetime `{}` in pattern", ident)));
            }

            let subpat = try!(self.parse_pat());
            pat = PatRegion(subpat, mutbl);
          }
          token::OpenDelim(token::Paren) => {
            // Parse (pat,pat,pat,...) as tuple pattern
            try!(self.bump());
            let fields = try!(self.parse_pat_tuple_elements());
            try!(self.expect(&token::CloseDelim(token::Paren)));
            pat = PatTup(fields);
          }
          token::OpenDelim(token::Bracket) => {
            // Parse [pat,pat,...] as slice pattern
            try!(self.bump());
            let (before, slice, after) = try!(self.parse_pat_vec_elements());
            try!(self.expect(&token::CloseDelim(token::Bracket)));
            pat = PatVec(before, slice, after);
          }
          _ => {
            // At this point, token != _, &, &&, (, [
            if try!(self.eat_keyword(keywords::Mut)) {
                // Parse mut ident @ pat
                pat = try!(self.parse_pat_ident(BindByValue(MutMutable)));
            } else if try!(self.eat_keyword(keywords::Ref)) {
                // Parse ref ident @ pat / ref mut ident @ pat
                let mutbl = try!(self.parse_mutability());
                pat = try!(self.parse_pat_ident(BindByRef(mutbl)));
            } else if try!(self.eat_keyword(keywords::Box)) {
                // Parse box pat
                let subpat = try!(self.parse_pat());
                pat = PatBox(subpat);
            } else if self.is_path_start() {
                // Parse pattern starting with a path
                if self.token.is_plain_ident() && self.look_ahead(1, |t| *t != token::DotDotDot &&
                        *t != token::OpenDelim(token::Brace) &&
                        *t != token::OpenDelim(token::Paren) &&
                        // Contrary to its definition, a plain ident can be followed by :: in macros
                        *t != token::ModSep) {
                    // Plain idents have some extra abilities here compared to general paths
                    if self.look_ahead(1, |t| *t == token::Not) {
                        // Parse macro invocation
                        let ident = try!(self.parse_ident());
                        let ident_span = self.last_span;
                        let path = ident_to_path(ident_span, ident);
                        try!(self.bump());
                        let delim = try!(self.expect_open_delim());
                        let tts = try!(self.parse_seq_to_end(&token::CloseDelim(delim),
                                seq_sep_none(), |p| p.parse_token_tree()));
                        let mac = Mac_ { path: path, tts: tts, ctxt: EMPTY_CTXT };
                        pat = PatMac(codemap::Spanned {node: mac, span: self.span});
                    } else {
                        // Parse ident @ pat
                        // This can give false positives and parse nullary enums,
                        // they are dealt with later in resolve
                        pat = try!(self.parse_pat_ident(BindByValue(MutImmutable)));
                    }
                } else {
                    let (qself, path) = if try!(self.eat_lt()) {
                        // Parse a qualified path
                        let (qself, path) =
                            try!(self.parse_qualified_path(NoTypesAllowed));
                        (Some(qself), path)
                    } else {
                        // Parse an unqualified path
                        (None, try!(self.parse_path(LifetimeAndTypesWithColons)))
                    };
                    match self.token {
                      token::DotDotDot => {
                        // Parse range
                        let hi = self.last_span.hi;
                        let begin = self.mk_expr(lo, hi, ExprPath(qself, path));
                        try!(self.bump());
                        let end = try!(self.parse_pat_range_end());
                        pat = PatRange(begin, end);
                      }
                      token::OpenDelim(token::Brace) => {
                         if qself.is_some() {
                            return Err(self.fatal("unexpected `{` after qualified path"));
                        }
                        // Parse struct pattern
                        try!(self.bump());
                        let (fields, etc) = try!(self.parse_pat_fields());
                        try!(self.bump());
                        pat = PatStruct(path, fields, etc);
                      }
                      token::OpenDelim(token::Paren) => {
                        if qself.is_some() {
                            return Err(self.fatal("unexpected `(` after qualified path"));
                        }
                        // Parse tuple struct or enum pattern
                        if self.look_ahead(1, |t| *t == token::DotDot) {
                            // This is a "top constructor only" pat
                            try!(self.bump());
                            try!(self.bump());
                            try!(self.expect(&token::CloseDelim(token::Paren)));
                            pat = PatEnum(path, None);
                        } else {
                            let args = try!(self.parse_enum_variant_seq(
                                    &token::OpenDelim(token::Paren),
                                    &token::CloseDelim(token::Paren),
                                    seq_sep_trailing_allowed(token::Comma),
                                    |p| p.parse_pat()));
                            pat = PatEnum(path, Some(args));
                        }
                      }
                      _ => {
                        pat = match qself {
                            // Parse qualified path
                            Some(qself) => PatQPath(qself, path),
                            // Parse nullary enum
                            None => PatEnum(path, Some(vec![]))
                        };
                      }
                    }
                }
            } else {
                // Try to parse everything else as literal with optional minus
                let begin = try!(self.parse_literal_maybe_minus());
                if try!(self.eat(&token::DotDotDot)) {
                    let end = try!(self.parse_pat_range_end());
                    pat = PatRange(begin, end);
                } else {
                    pat = PatLit(begin);
                }
            }
          }
        }

        let hi = self.last_span.hi;
        Ok(P(ast::Pat {
            id: ast::DUMMY_NODE_ID,
            node: pat,
            span: mk_sp(lo, hi),
        }))
    }

    /// Parse ident or ident @ pat
    /// used by the copy foo and ref foo patterns to give a good
    /// error message when parsing mistakes like ref foo(a,b)
    fn parse_pat_ident(&mut self,
                       binding_mode: ast::BindingMode)
                       -> PResult<ast::Pat_> {
        if !self.token.is_plain_ident() {
            let span = self.span;
            let tok_str = self.this_token_to_string();
            return Err(self.span_fatal(span,
                            &format!("expected identifier, found `{}`", tok_str)))
        }
        let ident = try!(self.parse_ident());
        let last_span = self.last_span;
        let name = codemap::Spanned{span: last_span, node: ident};
        let sub = if try!(self.eat(&token::At) ){
            Some(try!(self.parse_pat()))
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
            let last_span = self.last_span;
            return Err(self.span_fatal(
                last_span,
                "expected identifier, found enum pattern"))
        }

        Ok(PatIdent(binding_mode, name, sub))
    }

    /// Parse a local variable declaration
    fn parse_local(&mut self) -> PResult<P<Local>> {
        let lo = self.span.lo;
        let pat = try!(self.parse_pat());

        let mut ty = None;
        if try!(self.eat(&token::Colon) ){
            ty = Some(try!(self.parse_ty_sum()));
        }
        let init = try!(self.parse_initializer());
        Ok(P(ast::Local {
            ty: ty,
            pat: pat,
            init: init,
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, self.last_span.hi),
        }))
    }

    /// Parse a "let" stmt
    fn parse_let(&mut self) -> PResult<P<Decl>> {
        let lo = self.span.lo;
        let local = try!(self.parse_local());
        Ok(P(spanned(lo, self.last_span.hi, DeclLocal(local))))
    }

    /// Parse a structure field
    fn parse_name_and_ty(&mut self, pr: Visibility,
                         attrs: Vec<Attribute> ) -> PResult<StructField> {
        let lo = match pr {
            Inherited => self.span.lo,
            Public => self.last_span.lo,
        };
        if !self.token.is_plain_ident() {
            return Err(self.fatal("expected ident"));
        }
        let name = try!(self.parse_ident());
        try!(self.expect(&token::Colon));
        let ty = try!(self.parse_ty_sum());
        Ok(spanned(lo, self.last_span.hi, ast::StructField_ {
            kind: NamedField(name, pr),
            id: ast::DUMMY_NODE_ID,
            ty: ty,
            attrs: attrs,
        }))
    }

    /// Emit an expected item after attributes error.
    fn expected_item_err(&self, attrs: &[Attribute]) {
        let message = match attrs.last() {
            Some(&Attribute { node: ast::Attribute_ { is_sugared_doc: true, .. }, .. }) => {
                "expected item after doc comment"
            }
            _ => "expected item after attributes",
        };

        self.span_err(self.last_span, message);
    }

    /// Parse a statement. may include decl.
    pub fn parse_stmt(&mut self) -> PResult<Option<P<Stmt>>> {
        Ok(try!(self.parse_stmt_()).map(P))
    }

    fn parse_stmt_(&mut self) -> PResult<Option<Stmt>> {
        maybe_whole!(Some deref self, NtStmt);

        fn check_expected_item(p: &mut Parser, attrs: &[Attribute]) {
            // If we have attributes then we should have an item
            if !attrs.is_empty() {
                p.expected_item_err(attrs);
            }
        }

        let attrs = try!(self.parse_outer_attributes());
        let lo = self.span.lo;

        Ok(Some(if self.check_keyword(keywords::Let) {
            check_expected_item(self, &attrs);
            try!(self.expect_keyword(keywords::Let));
            let decl = try!(self.parse_let());
            spanned(lo, decl.span.hi, StmtDecl(decl, ast::DUMMY_NODE_ID))
        } else if self.token.is_ident()
            && !self.token.is_any_keyword()
            && self.look_ahead(1, |t| *t == token::Not) {
            // it's a macro invocation:

            check_expected_item(self, &attrs);

            // Potential trouble: if we allow macros with paths instead of
            // idents, we'd need to look ahead past the whole path here...
            let pth = try!(self.parse_path(NoTypesAllowed));
            try!(self.bump());

            let id = match self.token {
                token::OpenDelim(_) => token::special_idents::invalid, // no special identifier
                _ => try!(self.parse_ident()),
            };

            // check that we're pointing at delimiters (need to check
            // again after the `if`, because of `parse_ident`
            // consuming more tokens).
            let delim = match self.token {
                token::OpenDelim(delim) => delim,
                _ => {
                    // we only expect an ident if we didn't parse one
                    // above.
                    let ident_str = if id.name == token::special_idents::invalid.name {
                        "identifier, "
                    } else {
                        ""
                    };
                    let tok_str = self.this_token_to_string();
                    return Err(self.fatal(&format!("expected {}`(` or `{{`, found `{}`",
                                       ident_str,
                                       tok_str)))
                },
            };

            let tts = try!(self.parse_unspanned_seq(
                &token::OpenDelim(delim),
                &token::CloseDelim(delim),
                seq_sep_none(),
                |p| p.parse_token_tree()
            ));
            let hi = self.last_span.hi;

            let style = if delim == token::Brace {
                MacStmtWithBraces
            } else {
                MacStmtWithoutBraces
            };

            if id.name == token::special_idents::invalid.name {
                spanned(lo, hi,
                        StmtMac(P(spanned(lo,
                                          hi,
                                          Mac_ { path: pth, tts: tts, ctxt: EMPTY_CTXT })),
                                  style))
            } else {
                // if it has a special ident, it's definitely an item
                //
                // Require a semicolon or braces.
                if style != MacStmtWithBraces {
                    if !try!(self.eat(&token::Semi) ){
                        let last_span = self.last_span;
                        self.span_err(last_span,
                                      "macros that expand to items must \
                                       either be surrounded with braces or \
                                       followed by a semicolon");
                    }
                }
                spanned(lo, hi, StmtDecl(
                    P(spanned(lo, hi, DeclItem(
                        self.mk_item(
                            lo, hi, id /*id is good here*/,
                            ItemMac(spanned(lo, hi,
                                            Mac_ { path: pth, tts: tts, ctxt: EMPTY_CTXT })),
                            Inherited, Vec::new(/*no attrs*/))))),
                    ast::DUMMY_NODE_ID))
            }
        } else {
            match try!(self.parse_item_(attrs, false)) {
                Some(i) => {
                    let hi = i.span.hi;
                    let decl = P(spanned(lo, hi, DeclItem(i)));
                    spanned(lo, hi, StmtDecl(decl, ast::DUMMY_NODE_ID))
                }
                None => {
                    // Do not attempt to parse an expression if we're done here.
                    if self.token == token::Semi {
                        try!(self.bump());
                        return Ok(None);
                    }

                    if self.token == token::CloseDelim(token::Brace) {
                        return Ok(None);
                    }

                    // Remainder are line-expr stmts.
                    let e = try!(self.parse_expr_res(Restrictions::RESTRICTION_STMT_EXPR));
                    spanned(lo, e.span.hi, StmtExpr(e, ast::DUMMY_NODE_ID))
                }
            }
        }))
    }

    /// Is this expression a successfully-parsed statement?
    fn expr_is_complete(&mut self, e: &Expr) -> bool {
        self.restrictions.contains(Restrictions::RESTRICTION_STMT_EXPR) &&
            !classify::expr_requires_semi_to_be_stmt(e)
    }

    /// Parse a block. No inner attrs are allowed.
    pub fn parse_block(&mut self) -> PResult<P<Block>> {
        maybe_whole!(no_clone self, NtBlock);

        let lo = self.span.lo;

        if !try!(self.eat(&token::OpenDelim(token::Brace)) ){
            let sp = self.span;
            let tok = self.this_token_to_string();
            return Err(self.span_fatal_help(sp,
                                 &format!("expected `{{`, found `{}`", tok),
                                 "place this code inside a block"));
        }

        self.parse_block_tail(lo, DefaultBlock)
    }

    /// Parse a block. Inner attrs are allowed.
    fn parse_inner_attrs_and_block(&mut self) -> PResult<(Vec<Attribute>, P<Block>)> {
        maybe_whole!(pair_empty self, NtBlock);

        let lo = self.span.lo;
        try!(self.expect(&token::OpenDelim(token::Brace)));
        Ok((try!(self.parse_inner_attributes()),
         try!(self.parse_block_tail(lo, DefaultBlock))))
    }

    /// Parse the rest of a block expression or function body
    /// Precondition: already parsed the '{'.
    fn parse_block_tail(&mut self, lo: BytePos, s: BlockCheckMode) -> PResult<P<Block>> {
        let mut stmts = vec![];
        let mut expr = None;

        while !try!(self.eat(&token::CloseDelim(token::Brace))) {
            let Spanned {node, span} = if let Some(s) = try!(self.parse_stmt_()) {
                s
            } else {
                // Found only `;` or `}`.
                continue;
            };
            match node {
                StmtExpr(e, _) => {
                    try!(self.handle_expression_like_statement(e, span, &mut stmts, &mut expr));
                }
                StmtMac(mac, MacStmtWithoutBraces) => {
                    // statement macro without braces; might be an
                    // expr depending on whether a semicolon follows
                    match self.token {
                        token::Semi => {
                            stmts.push(P(Spanned {
                                node: StmtMac(mac, MacStmtWithSemicolon),
                                span: mk_sp(span.lo, self.span.hi),
                            }));
                            try!(self.bump());
                        }
                        _ => {
                            let e = self.mk_mac_expr(span.lo, span.hi,
                                                     mac.and_then(|m| m.node));
                            let e = try!(self.parse_dot_or_call_expr_with(e));
                            let e = try!(self.parse_assoc_expr_with(0, Some(e)));
                            try!(self.handle_expression_like_statement(
                                e,
                                span,
                                &mut stmts,
                                &mut expr));
                        }
                    }
                }
                StmtMac(m, style) => {
                    // statement macro; might be an expr
                    match self.token {
                        token::Semi => {
                            stmts.push(P(Spanned {
                                node: StmtMac(m, MacStmtWithSemicolon),
                                span: mk_sp(span.lo, self.span.hi),
                            }));
                            try!(self.bump());
                        }
                        token::CloseDelim(token::Brace) => {
                            // if a block ends in `m!(arg)` without
                            // a `;`, it must be an expr
                            expr = Some(self.mk_mac_expr(span.lo, span.hi,
                                                         m.and_then(|x| x.node)));
                        }
                        _ => {
                            stmts.push(P(Spanned {
                                node: StmtMac(m, style),
                                span: span
                            }));
                        }
                    }
                }
                _ => { // all other kinds of statements:
                    let mut hi = span.hi;
                    if classify::stmt_ends_with_semi(&node) {
                        try!(self.commit_stmt_expecting(token::Semi));
                        hi = self.last_span.hi;
                    }

                    stmts.push(P(Spanned {
                        node: node,
                        span: mk_sp(span.lo, hi)
                    }));
                }
            }
        }

        Ok(P(ast::Block {
            stmts: stmts,
            expr: expr,
            id: ast::DUMMY_NODE_ID,
            rules: s,
            span: mk_sp(lo, self.last_span.hi),
        }))
    }

    fn handle_expression_like_statement(
            &mut self,
            e: P<Expr>,
            span: Span,
            stmts: &mut Vec<P<Stmt>>,
            last_block_expr: &mut Option<P<Expr>>) -> PResult<()> {
        // expression without semicolon
        if classify::expr_requires_semi_to_be_stmt(&*e) {
            // Just check for errors and recover; do not eat semicolon yet.
            try!(self.commit_stmt(&[],
                             &[token::Semi, token::CloseDelim(token::Brace)]));
        }

        match self.token {
            token::Semi => {
                try!(self.bump());
                let span_with_semi = Span {
                    lo: span.lo,
                    hi: self.last_span.hi,
                    expn_id: span.expn_id,
                };
                stmts.push(P(Spanned {
                    node: StmtSemi(e, ast::DUMMY_NODE_ID),
                    span: span_with_semi,
                }));
            }
            token::CloseDelim(token::Brace) => *last_block_expr = Some(e),
            _ => {
                stmts.push(P(Spanned {
                    node: StmtExpr(e, ast::DUMMY_NODE_ID),
                    span: span
                }));
            }
        }
        Ok(())
    }

    // Parses a sequence of bounds if a `:` is found,
    // otherwise returns empty list.
    fn parse_colon_then_ty_param_bounds(&mut self,
                                        mode: BoundParsingMode)
                                        -> PResult<OwnedSlice<TyParamBound>>
    {
        if !try!(self.eat(&token::Colon) ){
            Ok(OwnedSlice::empty())
        } else {
            self.parse_ty_param_bounds(mode)
        }
    }

    // matches bounds    = ( boundseq )?
    // where   boundseq  = ( polybound + boundseq ) | polybound
    // and     polybound = ( 'for' '<' 'region '>' )? bound
    // and     bound     = 'region | trait_ref
    fn parse_ty_param_bounds(&mut self,
                             mode: BoundParsingMode)
                             -> PResult<OwnedSlice<TyParamBound>>
    {
        let mut result = vec!();
        loop {
            let question_span = self.span;
            let ate_question = try!(self.eat(&token::Question));
            match self.token {
                token::Lifetime(lifetime) => {
                    if ate_question {
                        self.span_err(question_span,
                                      "`?` may only modify trait bounds, not lifetime bounds");
                    }
                    result.push(RegionTyParamBound(ast::Lifetime {
                        id: ast::DUMMY_NODE_ID,
                        span: self.span,
                        name: lifetime.name
                    }));
                    try!(self.bump());
                }
                token::ModSep | token::Ident(..) => {
                    let poly_trait_ref = try!(self.parse_poly_trait_ref());
                    let modifier = if ate_question {
                        if mode == BoundParsingMode::Modified {
                            TraitBoundModifier::Maybe
                        } else {
                            self.span_err(question_span,
                                          "unexpected `?`");
                            TraitBoundModifier::None
                        }
                    } else {
                        TraitBoundModifier::None
                    };
                    result.push(TraitTyParamBound(poly_trait_ref, modifier))
                }
                _ => break,
            }

            if !try!(self.eat(&token::BinOp(token::Plus)) ){
                break;
            }
        }

        return Ok(OwnedSlice::from_vec(result));
    }

    /// Matches typaram = IDENT (`?` unbound)? optbounds ( EQ ty )?
    fn parse_ty_param(&mut self) -> PResult<TyParam> {
        let span = self.span;
        let ident = try!(self.parse_ident());

        let bounds = try!(self.parse_colon_then_ty_param_bounds(BoundParsingMode::Modified));

        let default = if self.check(&token::Eq) {
            try!(self.bump());
            Some(try!(self.parse_ty_sum()))
        } else {
            None
        };

        Ok(TyParam {
            ident: ident,
            id: ast::DUMMY_NODE_ID,
            bounds: bounds,
            default: default,
            span: span,
        })
    }

    /// Parse a set of optional generic type parameter declarations. Where
    /// clauses are not parsed here, and must be added later via
    /// `parse_where_clause()`.
    ///
    /// matches generics = ( ) | ( < > ) | ( < typaramseq ( , )? > ) | ( < lifetimes ( , )? > )
    ///                  | ( < lifetimes , typaramseq ( , )? > )
    /// where   typaramseq = ( typaram ) | ( typaram , typaramseq )
    pub fn parse_generics(&mut self) -> PResult<ast::Generics> {
        maybe_whole!(self, NtGenerics);

        if try!(self.eat(&token::Lt) ){
            let lifetime_defs = try!(self.parse_lifetime_defs());
            let mut seen_default = false;
            let ty_params = try!(self.parse_seq_to_gt(Some(token::Comma), |p| {
                try!(p.forbid_lifetime());
                let ty_param = try!(p.parse_ty_param());
                if ty_param.default.is_some() {
                    seen_default = true;
                } else if seen_default {
                    let last_span = p.last_span;
                    p.span_err(last_span,
                               "type parameters with a default must be trailing");
                }
                Ok(ty_param)
            }));
            Ok(ast::Generics {
                lifetimes: lifetime_defs,
                ty_params: ty_params,
                where_clause: WhereClause {
                    id: ast::DUMMY_NODE_ID,
                    predicates: Vec::new(),
                }
            })
        } else {
            Ok(ast_util::empty_generics())
        }
    }

    fn parse_generic_values_after_lt(&mut self) -> PResult<(Vec<ast::Lifetime>,
                                                            Vec<P<Ty>>,
                                                            Vec<P<TypeBinding>>)> {
        let span_lo = self.span.lo;
        let lifetimes = try!(self.parse_lifetimes(token::Comma));

        let missing_comma = !lifetimes.is_empty() &&
                            !self.token.is_like_gt() &&
                            self.last_token
                                .as_ref().map_or(true,
                                                 |x| &**x != &token::Comma);

        if missing_comma {

            let msg = format!("expected `,` or `>` after lifetime \
                              name, found `{}`",
                              self.this_token_to_string());
            self.span_err(self.span, &msg);

            let span_hi = self.span.hi;
            let span_hi = if self.parse_ty().is_ok() {
                self.span.hi
            } else {
                span_hi
            };

            let msg = format!("did you mean a single argument type &'a Type, \
                              or did you mean the comma-separated arguments \
                              'a, Type?");
            self.span_note(mk_sp(span_lo, span_hi), &msg);

            self.abort_if_errors()
        }

        // First parse types.
        let (types, returned) = try!(self.parse_seq_to_gt_or_return(
            Some(token::Comma),
            |p| {
                try!(p.forbid_lifetime());
                if p.look_ahead(1, |t| t == &token::Eq) {
                    Ok(None)
                } else {
                    Ok(Some(try!(p.parse_ty_sum())))
                }
            }
        ));

        // If we found the `>`, don't continue.
        if !returned {
            return Ok((lifetimes, types.into_vec(), Vec::new()));
        }

        // Then parse type bindings.
        let bindings = try!(self.parse_seq_to_gt(
            Some(token::Comma),
            |p| {
                try!(p.forbid_lifetime());
                let lo = p.span.lo;
                let ident = try!(p.parse_ident());
                let found_eq = try!(p.eat(&token::Eq));
                if !found_eq {
                    let span = p.span;
                    p.span_warn(span, "whoops, no =?");
                }
                let ty = try!(p.parse_ty());
                let hi = ty.span.hi;
                let span = mk_sp(lo, hi);
                return Ok(P(TypeBinding{id: ast::DUMMY_NODE_ID,
                    ident: ident,
                    ty: ty,
                    span: span,
                }));
            }
        ));
        Ok((lifetimes, types.into_vec(), bindings.into_vec()))
    }

    fn forbid_lifetime(&mut self) -> PResult<()> {
        if self.token.is_lifetime() {
            let span = self.span;
            return Err(self.span_fatal(span, "lifetime parameters must be declared \
                                        prior to type parameters"))
        }
        Ok(())
    }

    /// Parses an optional `where` clause and places it in `generics`.
    ///
    /// ```ignore
    /// where T : Trait<U, V> + 'b, 'a : 'b
    /// ```
    pub fn parse_where_clause(&mut self) -> PResult<ast::WhereClause> {
        maybe_whole!(self, NtWhereClause);

        let mut where_clause = WhereClause {
            id: ast::DUMMY_NODE_ID,
            predicates: Vec::new(),
        };

        if !try!(self.eat_keyword(keywords::Where)) {
            return Ok(where_clause);
        }

        let mut parsed_something = false;
        loop {
            let lo = self.span.lo;
            match self.token {
                token::OpenDelim(token::Brace) => {
                    break
                }

                token::Lifetime(..) => {
                    let bounded_lifetime =
                        try!(self.parse_lifetime());

                    try!(self.eat(&token::Colon));

                    let bounds =
                        try!(self.parse_lifetimes(token::BinOp(token::Plus)));

                    let hi = self.last_span.hi;
                    let span = mk_sp(lo, hi);

                    where_clause.predicates.push(ast::WherePredicate::RegionPredicate(
                        ast::WhereRegionPredicate {
                            span: span,
                            lifetime: bounded_lifetime,
                            bounds: bounds
                        }
                    ));

                    parsed_something = true;
                }

                _ => {
                    let bound_lifetimes = if try!(self.eat_keyword(keywords::For) ){
                        // Higher ranked constraint.
                        try!(self.expect(&token::Lt));
                        let lifetime_defs = try!(self.parse_lifetime_defs());
                        try!(self.expect_gt());
                        lifetime_defs
                    } else {
                        vec![]
                    };

                    let bounded_ty = try!(self.parse_ty());

                    if try!(self.eat(&token::Colon) ){
                        let bounds = try!(self.parse_ty_param_bounds(BoundParsingMode::Bare));
                        let hi = self.last_span.hi;
                        let span = mk_sp(lo, hi);

                        if bounds.is_empty() {
                            self.span_err(span,
                                          "each predicate in a `where` clause must have \
                                           at least one bound in it");
                        }

                        where_clause.predicates.push(ast::WherePredicate::BoundPredicate(
                                ast::WhereBoundPredicate {
                                    span: span,
                                    bound_lifetimes: bound_lifetimes,
                                    bounded_ty: bounded_ty,
                                    bounds: bounds,
                        }));

                        parsed_something = true;
                    } else if try!(self.eat(&token::Eq) ){
                        // let ty = try!(self.parse_ty());
                        let hi = self.last_span.hi;
                        let span = mk_sp(lo, hi);
                        // where_clause.predicates.push(
                        //     ast::WherePredicate::EqPredicate(ast::WhereEqPredicate {
                        //         id: ast::DUMMY_NODE_ID,
                        //         span: span,
                        //         path: panic!("NYI"), //bounded_ty,
                        //         ty: ty,
                        // }));
                        // parsed_something = true;
                        // // FIXME(#18433)
                        self.span_err(span,
                                     "equality constraints are not yet supported \
                                     in where clauses (#20041)");
                    } else {
                        let last_span = self.last_span;
                        self.span_err(last_span,
                              "unexpected token in `where` clause");
                    }
                }
            };

            if !try!(self.eat(&token::Comma) ){
                break
            }
        }

        if !parsed_something {
            let last_span = self.last_span;
            self.span_err(last_span,
                          "a `where` clause must have at least one predicate \
                           in it");
        }

        Ok(where_clause)
    }

    fn parse_fn_args(&mut self, named_args: bool, allow_variadic: bool)
                     -> PResult<(Vec<Arg> , bool)> {
        let sp = self.span;
        let mut args: Vec<Option<Arg>> =
            try!(self.parse_unspanned_seq(
                &token::OpenDelim(token::Paren),
                &token::CloseDelim(token::Paren),
                seq_sep_trailing_allowed(token::Comma),
                |p| {
                    if p.token == token::DotDotDot {
                        try!(p.bump());
                        if allow_variadic {
                            if p.token != token::CloseDelim(token::Paren) {
                                let span = p.span;
                                return Err(p.span_fatal(span,
                                    "`...` must be last in argument list for variadic function"))
                            }
                        } else {
                            let span = p.span;
                            return Err(p.span_fatal(span,
                                         "only foreign functions are allowed to be variadic"))
                        }
                        Ok(None)
                    } else {
                        Ok(Some(try!(p.parse_arg_general(named_args))))
                    }
                }
            ));

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

        let args = args.into_iter().map(|x| x.unwrap()).collect();

        Ok((args, variadic))
    }

    /// Parse the argument list and result type of a function declaration
    pub fn parse_fn_decl(&mut self, allow_variadic: bool) -> PResult<P<FnDecl>> {

        let (args, variadic) = try!(self.parse_fn_args(true, allow_variadic));
        let ret_ty = try!(self.parse_ret_ty());

        Ok(P(FnDecl {
            inputs: args,
            output: ret_ty,
            variadic: variadic
        }))
    }

    fn is_self_ident(&mut self) -> bool {
        match self.token {
          token::Ident(id, token::Plain) => id.name == special_idents::self_.name,
          _ => false
        }
    }

    fn expect_self_ident(&mut self) -> PResult<ast::Ident> {
        match self.token {
            token::Ident(id, token::Plain) if id.name == special_idents::self_.name => {
                try!(self.bump());
                Ok(id)
            },
            _ => {
                let token_str = self.this_token_to_string();
                return Err(self.fatal(&format!("expected `self`, found `{}`",
                                   token_str)))
            }
        }
    }

    fn is_self_type_ident(&mut self) -> bool {
        match self.token {
          token::Ident(id, token::Plain) => id.name == special_idents::type_self.name,
          _ => false
        }
    }

    fn expect_self_type_ident(&mut self) -> PResult<ast::Ident> {
        match self.token {
            token::Ident(id, token::Plain) if id.name == special_idents::type_self.name => {
                try!(self.bump());
                Ok(id)
            },
            _ => {
                let token_str = self.this_token_to_string();
                Err(self.fatal(&format!("expected `Self`, found `{}`",
                                   token_str)))
            }
        }
    }

    /// Parse the argument list and result type of a function
    /// that may have a self type.
    fn parse_fn_decl_with_self<F>(&mut self,
                                  parse_arg_fn: F) -> PResult<(ExplicitSelf, P<FnDecl>)> where
        F: FnMut(&mut Parser) -> PResult<Arg>,
    {
        fn maybe_parse_borrowed_explicit_self(this: &mut Parser)
                                              -> PResult<ast::ExplicitSelf_> {
            // The following things are possible to see here:
            //
            //     fn(&mut self)
            //     fn(&mut self)
            //     fn(&'lt self)
            //     fn(&'lt mut self)
            //
            // We already know that the current token is `&`.

            if this.look_ahead(1, |t| t.is_keyword(keywords::SelfValue)) {
                try!(this.bump());
                Ok(SelfRegion(None, MutImmutable, try!(this.expect_self_ident())))
            } else if this.look_ahead(1, |t| t.is_mutability()) &&
                      this.look_ahead(2, |t| t.is_keyword(keywords::SelfValue)) {
                try!(this.bump());
                let mutability = try!(this.parse_mutability());
                Ok(SelfRegion(None, mutability, try!(this.expect_self_ident())))
            } else if this.look_ahead(1, |t| t.is_lifetime()) &&
                      this.look_ahead(2, |t| t.is_keyword(keywords::SelfValue)) {
                try!(this.bump());
                let lifetime = try!(this.parse_lifetime());
                Ok(SelfRegion(Some(lifetime), MutImmutable, try!(this.expect_self_ident())))
            } else if this.look_ahead(1, |t| t.is_lifetime()) &&
                      this.look_ahead(2, |t| t.is_mutability()) &&
                      this.look_ahead(3, |t| t.is_keyword(keywords::SelfValue)) {
                try!(this.bump());
                let lifetime = try!(this.parse_lifetime());
                let mutability = try!(this.parse_mutability());
                Ok(SelfRegion(Some(lifetime), mutability, try!(this.expect_self_ident())))
            } else {
                Ok(SelfStatic)
            }
        }

        try!(self.expect(&token::OpenDelim(token::Paren)));

        // A bit of complexity and lookahead is needed here in order to be
        // backwards compatible.
        let lo = self.span.lo;
        let mut self_ident_lo = self.span.lo;
        let mut self_ident_hi = self.span.hi;

        let mut mutbl_self = MutImmutable;
        let explicit_self = match self.token {
            token::BinOp(token::And) => {
                let eself = try!(maybe_parse_borrowed_explicit_self(self));
                self_ident_lo = self.last_span.lo;
                self_ident_hi = self.last_span.hi;
                eself
            }
            token::BinOp(token::Star) => {
                // Possibly "*self" or "*mut self" -- not supported. Try to avoid
                // emitting cryptic "unexpected token" errors.
                try!(self.bump());
                let _mutability = if self.token.is_mutability() {
                    try!(self.parse_mutability())
                } else {
                    MutImmutable
                };
                if self.is_self_ident() {
                    let span = self.span;
                    self.span_err(span, "cannot pass self by raw pointer");
                    try!(self.bump());
                }
                // error case, making bogus self ident:
                SelfValue(special_idents::self_)
            }
            token::Ident(..) => {
                if self.is_self_ident() {
                    let self_ident = try!(self.expect_self_ident());

                    // Determine whether this is the fully explicit form, `self:
                    // TYPE`.
                    if try!(self.eat(&token::Colon) ){
                        SelfExplicit(try!(self.parse_ty_sum()), self_ident)
                    } else {
                        SelfValue(self_ident)
                    }
                } else if self.token.is_mutability() &&
                        self.look_ahead(1, |t| t.is_keyword(keywords::SelfValue)) {
                    mutbl_self = try!(self.parse_mutability());
                    let self_ident = try!(self.expect_self_ident());

                    // Determine whether this is the fully explicit form,
                    // `self: TYPE`.
                    if try!(self.eat(&token::Colon) ){
                        SelfExplicit(try!(self.parse_ty_sum()), self_ident)
                    } else {
                        SelfValue(self_ident)
                    }
                } else {
                    SelfStatic
                }
            }
            _ => SelfStatic,
        };

        let explicit_self_sp = mk_sp(self_ident_lo, self_ident_hi);

        // shared fall-through for the three cases below. borrowing prevents simply
        // writing this as a closure
        macro_rules! parse_remaining_arguments {
            ($self_id:ident) =>
            {
            // If we parsed a self type, expect a comma before the argument list.
            match self.token {
                token::Comma => {
                    try!(self.bump());
                    let sep = seq_sep_trailing_allowed(token::Comma);
                    let mut fn_inputs = try!(self.parse_seq_to_before_end(
                        &token::CloseDelim(token::Paren),
                        sep,
                        parse_arg_fn
                    ));
                    fn_inputs.insert(0, Arg::new_self(explicit_self_sp, mutbl_self, $self_id));
                    fn_inputs
                }
                token::CloseDelim(token::Paren) => {
                    vec!(Arg::new_self(explicit_self_sp, mutbl_self, $self_id))
                }
                _ => {
                    let token_str = self.this_token_to_string();
                    return Err(self.fatal(&format!("expected `,` or `)`, found `{}`",
                                       token_str)))
                }
            }
            }
        }

        let fn_inputs = match explicit_self {
            SelfStatic =>  {
                let sep = seq_sep_trailing_allowed(token::Comma);
                try!(self.parse_seq_to_before_end(&token::CloseDelim(token::Paren),
                                                  sep, parse_arg_fn))
            }
            SelfValue(id) => parse_remaining_arguments!(id),
            SelfRegion(_,_,id) => parse_remaining_arguments!(id),
            SelfExplicit(_,id) => parse_remaining_arguments!(id),
        };


        try!(self.expect(&token::CloseDelim(token::Paren)));

        let hi = self.span.hi;

        let ret_ty = try!(self.parse_ret_ty());

        let fn_decl = P(FnDecl {
            inputs: fn_inputs,
            output: ret_ty,
            variadic: false
        });

        Ok((spanned(lo, hi, explicit_self), fn_decl))
    }

    // parse the |arg, arg| header on a lambda
    fn parse_fn_block_decl(&mut self) -> PResult<P<FnDecl>> {
        let inputs_captures = {
            if try!(self.eat(&token::OrOr) ){
                Vec::new()
            } else {
                try!(self.expect(&token::BinOp(token::Or)));
                try!(self.parse_obsolete_closure_kind());
                let args = try!(self.parse_seq_to_before_end(
                    &token::BinOp(token::Or),
                    seq_sep_trailing_allowed(token::Comma),
                    |p| p.parse_fn_block_arg()
                ));
                try!(self.bump());
                args
            }
        };
        let output = try!(self.parse_ret_ty());

        Ok(P(FnDecl {
            inputs: inputs_captures,
            output: output,
            variadic: false
        }))
    }

    /// Parse the name and optional generic types of a function header.
    fn parse_fn_header(&mut self) -> PResult<(Ident, ast::Generics)> {
        let id = try!(self.parse_ident());
        let generics = try!(self.parse_generics());
        Ok((id, generics))
    }

    fn mk_item(&mut self, lo: BytePos, hi: BytePos, ident: Ident,
               node: Item_, vis: Visibility,
               attrs: Vec<Attribute>) -> P<Item> {
        P(Item {
            ident: ident,
            attrs: attrs,
            id: ast::DUMMY_NODE_ID,
            node: node,
            vis: vis,
            span: mk_sp(lo, hi)
        })
    }

    /// Parse an item-position function declaration.
    fn parse_item_fn(&mut self,
                     unsafety: Unsafety,
                     constness: Constness,
                     abi: abi::Abi)
                     -> PResult<ItemInfo> {
        let (ident, mut generics) = try!(self.parse_fn_header());
        let decl = try!(self.parse_fn_decl(false));
        generics.where_clause = try!(self.parse_where_clause());
        let (inner_attrs, body) = try!(self.parse_inner_attrs_and_block());
        Ok((ident, ItemFn(decl, unsafety, constness, abi, generics, body), Some(inner_attrs)))
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
    pub fn parse_fn_front_matter(&mut self) -> PResult<(ast::Constness, ast::Unsafety, abi::Abi)> {
        let is_const_fn = try!(self.eat_keyword(keywords::Const));
        let unsafety = try!(self.parse_unsafety());
        let (constness, unsafety, abi) = if is_const_fn {
            (Constness::Const, unsafety, abi::Rust)
        } else {
            let abi = if try!(self.eat_keyword(keywords::Extern)) {
                try!(self.parse_opt_abi()).unwrap_or(abi::C)
            } else {
                abi::Rust
            };
            (Constness::NotConst, unsafety, abi)
        };
        try!(self.expect_keyword(keywords::Fn));
        Ok((constness, unsafety, abi))
    }

    /// Parse an impl item.
    pub fn parse_impl_item(&mut self) -> PResult<P<ImplItem>> {
        maybe_whole!(no_clone self, NtImplItem);

        let mut attrs = try!(self.parse_outer_attributes());
        let lo = self.span.lo;
        let vis = try!(self.parse_visibility());
        let (name, node) = if try!(self.eat_keyword(keywords::Type)) {
            let name = try!(self.parse_ident());
            try!(self.expect(&token::Eq));
            let typ = try!(self.parse_ty_sum());
            try!(self.expect(&token::Semi));
            (name, TypeImplItem(typ))
        } else if self.is_const_item() {
            try!(self.expect_keyword(keywords::Const));
            let name = try!(self.parse_ident());
            try!(self.expect(&token::Colon));
            let typ = try!(self.parse_ty_sum());
            try!(self.expect(&token::Eq));
            let expr = try!(self.parse_expr());
            try!(self.commit_expr_expecting(&expr, token::Semi));
            (name, ConstImplItem(typ, expr))
        } else {
            let (name, inner_attrs, node) = try!(self.parse_impl_method(vis));
            attrs.extend(inner_attrs);
            (name, node)
        };

        Ok(P(ImplItem {
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, self.last_span.hi),
            ident: name,
            vis: vis,
            attrs: attrs,
            node: node
        }))
    }

    fn complain_if_pub_macro(&mut self, visa: Visibility, span: Span) {
        match visa {
            Public => {
                self.span_err(span, "can't qualify macro invocation with `pub`");
                self.fileline_help(span, "try adjusting the macro to put `pub` inside \
                                      the invocation");
            }
            Inherited => (),
        }
    }

    /// Parse a method or a macro invocation in a trait impl.
    fn parse_impl_method(&mut self, vis: Visibility)
                         -> PResult<(Ident, Vec<ast::Attribute>, ast::ImplItem_)> {
        // code copied from parse_macro_use_or_failure... abstraction!
        if !self.token.is_any_keyword()
            && self.look_ahead(1, |t| *t == token::Not)
            && (self.look_ahead(2, |t| *t == token::OpenDelim(token::Paren))
                || self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace))) {
            // method macro.

            let last_span = self.last_span;
            self.complain_if_pub_macro(vis, last_span);

            let pth = try!(self.parse_path(NoTypesAllowed));
            try!(self.expect(&token::Not));

            // eat a matched-delimiter token tree:
            let delim = try!(self.expect_open_delim());
            let tts = try!(self.parse_seq_to_end(&token::CloseDelim(delim),
                                            seq_sep_none(),
                                            |p| p.parse_token_tree()));
            let m_ = Mac_ { path: pth, tts: tts, ctxt: EMPTY_CTXT };
            let m: ast::Mac = codemap::Spanned { node: m_,
                                                span: mk_sp(self.span.lo,
                                                            self.span.hi) };
            if delim != token::Brace {
                try!(self.expect(&token::Semi))
            }
            Ok((token::special_idents::invalid, vec![], ast::MacImplItem(m)))
        } else {
            let (constness, unsafety, abi) = try!(self.parse_fn_front_matter());
            let ident = try!(self.parse_ident());
            let mut generics = try!(self.parse_generics());
            let (explicit_self, decl) = try!(self.parse_fn_decl_with_self(|p| {
                    p.parse_arg()
                }));
            generics.where_clause = try!(self.parse_where_clause());
            let (inner_attrs, body) = try!(self.parse_inner_attrs_and_block());
            Ok((ident, inner_attrs, MethodImplItem(ast::MethodSig {
                generics: generics,
                abi: abi,
                explicit_self: explicit_self,
                unsafety: unsafety,
                constness: constness,
                decl: decl
             }, body)))
        }
    }

    /// Parse trait Foo { ... }
    fn parse_item_trait(&mut self, unsafety: Unsafety) -> PResult<ItemInfo> {

        let ident = try!(self.parse_ident());
        let mut tps = try!(self.parse_generics());

        // Parse supertrait bounds.
        let bounds = try!(self.parse_colon_then_ty_param_bounds(BoundParsingMode::Bare));

        tps.where_clause = try!(self.parse_where_clause());

        let meths = try!(self.parse_trait_items());
        Ok((ident, ItemTrait(unsafety, tps, bounds, meths), None))
    }

    /// Parses items implementations variants
    ///    impl<T> Foo { ... }
    ///    impl<T> ToString for &'static T { ... }
    ///    impl Send for .. {}
    fn parse_item_impl(&mut self, unsafety: ast::Unsafety) -> PResult<ItemInfo> {
        let impl_span = self.span;

        // First, parse type parameters if necessary.
        let mut generics = try!(self.parse_generics());

        // Special case: if the next identifier that follows is '(', don't
        // allow this to be parsed as a trait.
        let could_be_trait = self.token != token::OpenDelim(token::Paren);

        let neg_span = self.span;
        let polarity = if try!(self.eat(&token::Not) ){
            ast::ImplPolarity::Negative
        } else {
            ast::ImplPolarity::Positive
        };

        // Parse the trait.
        let mut ty = try!(self.parse_ty_sum());

        // Parse traits, if necessary.
        let opt_trait = if could_be_trait && try!(self.eat_keyword(keywords::For) ){
            // New-style trait. Reinterpret the type as a trait.
            match ty.node {
                TyPath(None, ref path) => {
                    Some(TraitRef {
                        path: (*path).clone(),
                        ref_id: ty.id,
                    })
                }
                _ => {
                    self.span_err(ty.span, "not a trait");
                    None
                }
            }
        } else {
            match polarity {
                ast::ImplPolarity::Negative => {
                    // This is a negated type implementation
                    // `impl !MyType {}`, which is not allowed.
                    self.span_err(neg_span, "inherent implementation can't be negated");
                },
                _ => {}
            }
            None
        };

        if opt_trait.is_some() && try!(self.eat(&token::DotDot) ){
            if generics.is_parameterized() {
                self.span_err(impl_span, "default trait implementations are not \
                                          allowed to have generics");
            }

            try!(self.expect(&token::OpenDelim(token::Brace)));
            try!(self.expect(&token::CloseDelim(token::Brace)));
            Ok((ast_util::impl_pretty_name(&opt_trait, None),
             ItemDefaultImpl(unsafety, opt_trait.unwrap()), None))
        } else {
            if opt_trait.is_some() {
                ty = try!(self.parse_ty_sum());
            }
            generics.where_clause = try!(self.parse_where_clause());

            try!(self.expect(&token::OpenDelim(token::Brace)));
            let attrs = try!(self.parse_inner_attributes());

            let mut impl_items = vec![];
            while !try!(self.eat(&token::CloseDelim(token::Brace))) {
                impl_items.push(try!(self.parse_impl_item()));
            }

            Ok((ast_util::impl_pretty_name(&opt_trait, Some(&*ty)),
             ItemImpl(unsafety, polarity, generics, opt_trait, ty, impl_items),
             Some(attrs)))
        }
    }

    /// Parse a::B<String,i32>
    fn parse_trait_ref(&mut self) -> PResult<TraitRef> {
        Ok(ast::TraitRef {
            path: try!(self.parse_path(LifetimeAndTypesWithoutColons)),
            ref_id: ast::DUMMY_NODE_ID,
        })
    }

    fn parse_late_bound_lifetime_defs(&mut self) -> PResult<Vec<ast::LifetimeDef>> {
        if try!(self.eat_keyword(keywords::For) ){
            try!(self.expect(&token::Lt));
            let lifetime_defs = try!(self.parse_lifetime_defs());
            try!(self.expect_gt());
            Ok(lifetime_defs)
        } else {
            Ok(Vec::new())
        }
    }

    /// Parse for<'l> a::B<String,i32>
    fn parse_poly_trait_ref(&mut self) -> PResult<PolyTraitRef> {
        let lo = self.span.lo;
        let lifetime_defs = try!(self.parse_late_bound_lifetime_defs());

        Ok(ast::PolyTraitRef {
            bound_lifetimes: lifetime_defs,
            trait_ref: try!(self.parse_trait_ref()),
            span: mk_sp(lo, self.last_span.hi),
        })
    }

    /// Parse struct Foo { ... }
    fn parse_item_struct(&mut self) -> PResult<ItemInfo> {
        let class_name = try!(self.parse_ident());
        let mut generics = try!(self.parse_generics());

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
            generics.where_clause = try!(self.parse_where_clause());
            if try!(self.eat(&token::Semi)) {
                // If we see a: `struct Foo<T> where T: Copy;` style decl.
                VariantData::Unit(ast::DUMMY_NODE_ID)
            } else {
                // If we see: `struct Foo<T> where T: Copy { ... }`
                VariantData::Struct(try!(self.parse_record_struct_body(ParsePub::Yes)),
                                    ast::DUMMY_NODE_ID)
            }
        // No `where` so: `struct Foo<T>;`
        } else if try!(self.eat(&token::Semi) ){
            VariantData::Unit(ast::DUMMY_NODE_ID)
        // Record-style struct definition
        } else if self.token == token::OpenDelim(token::Brace) {
            VariantData::Struct(try!(self.parse_record_struct_body(ParsePub::Yes)),
                                ast::DUMMY_NODE_ID)
        // Tuple-style struct definition with optional where-clause.
        } else if self.token == token::OpenDelim(token::Paren) {
            let body = VariantData::Tuple(try!(self.parse_tuple_struct_body(ParsePub::Yes)),
                                          ast::DUMMY_NODE_ID);
            generics.where_clause = try!(self.parse_where_clause());
            try!(self.expect(&token::Semi));
            body
        } else {
            let token_str = self.this_token_to_string();
            return Err(self.fatal(&format!("expected `where`, `{{`, `(`, or `;` after struct \
                                            name, found `{}`", token_str)))
        };

        Ok((class_name, ItemStruct(vdata, generics), None))
    }

    pub fn parse_record_struct_body(&mut self, parse_pub: ParsePub) -> PResult<Vec<StructField>> {
        let mut fields = Vec::new();
        if try!(self.eat(&token::OpenDelim(token::Brace)) ){
            while self.token != token::CloseDelim(token::Brace) {
                fields.push(try!(self.parse_struct_decl_field(parse_pub)));
            }

            try!(self.bump());
        } else {
            let token_str = self.this_token_to_string();
            return Err(self.fatal(&format!("expected `where`, or `{{` after struct \
                                name, found `{}`",
                                token_str)));
        }

        Ok(fields)
    }

    pub fn parse_tuple_struct_body(&mut self, parse_pub: ParsePub) -> PResult<Vec<StructField>> {
        // This is the case where we find `struct Foo<T>(T) where T: Copy;`
        // Unit like structs are handled in parse_item_struct function
        let fields = try!(self.parse_unspanned_seq(
            &token::OpenDelim(token::Paren),
            &token::CloseDelim(token::Paren),
            seq_sep_trailing_allowed(token::Comma),
            |p| {
                let attrs = try!(p.parse_outer_attributes());
                let lo = p.span.lo;
                let struct_field_ = ast::StructField_ {
                    kind: UnnamedField (
                        if parse_pub == ParsePub::Yes {
                            try!(p.parse_visibility())
                        } else {
                            Inherited
                        }
                    ),
                    id: ast::DUMMY_NODE_ID,
                    ty: try!(p.parse_ty_sum()),
                    attrs: attrs,
                };
                Ok(spanned(lo, p.span.hi, struct_field_))
            }));

        Ok(fields)
    }

    /// Parse a structure field declaration
    pub fn parse_single_struct_field(&mut self,
                                     vis: Visibility,
                                     attrs: Vec<Attribute> )
                                     -> PResult<StructField> {
        let a_var = try!(self.parse_name_and_ty(vis, attrs));
        match self.token {
            token::Comma => {
                try!(self.bump());
            }
            token::CloseDelim(token::Brace) => {}
            _ => {
                let span = self.span;
                let token_str = self.this_token_to_string();
                return Err(self.span_fatal_help(span,
                                     &format!("expected `,`, or `}}`, found `{}`",
                                             token_str),
                                     "struct fields should be separated by commas"))
            }
        }
        Ok(a_var)
    }

    /// Parse an element of a struct definition
    fn parse_struct_decl_field(&mut self, parse_pub: ParsePub) -> PResult<StructField> {

        let attrs = try!(self.parse_outer_attributes());

        if try!(self.eat_keyword(keywords::Pub) ){
            if parse_pub == ParsePub::No {
                let span = self.last_span;
                self.span_err(span, "`pub` is not allowed here");
            }
            return self.parse_single_struct_field(Public, attrs);
        }

        return self.parse_single_struct_field(Inherited, attrs);
    }

    /// Parse visibility: PUB or nothing
    fn parse_visibility(&mut self) -> PResult<Visibility> {
        if try!(self.eat_keyword(keywords::Pub)) { Ok(Public) }
        else { Ok(Inherited) }
    }

    /// Given a termination token, parse all of the items in a module
    fn parse_mod_items(&mut self, term: &token::Token, inner_lo: BytePos) -> PResult<Mod> {
        let mut items = vec![];
        while let Some(item) = try!(self.parse_item()) {
            items.push(item);
        }

        if !try!(self.eat(term)) {
            let token_str = self.this_token_to_string();
            return Err(self.fatal(&format!("expected item, found `{}`", token_str)));
        }

        let hi = if self.span == codemap::DUMMY_SP {
            inner_lo
        } else {
            self.last_span.hi
        };

        Ok(ast::Mod {
            inner: mk_sp(inner_lo, hi),
            items: items
        })
    }

    fn parse_item_const(&mut self, m: Option<Mutability>) -> PResult<ItemInfo> {
        let id = try!(self.parse_ident());
        try!(self.expect(&token::Colon));
        let ty = try!(self.parse_ty_sum());
        try!(self.expect(&token::Eq));
        let e = try!(self.parse_expr());
        try!(self.commit_expr_expecting(&*e, token::Semi));
        let item = match m {
            Some(m) => ItemStatic(ty, m, e),
            None => ItemConst(ty, e),
        };
        Ok((id, item, None))
    }

    /// Parse a `mod <foo> { ... }` or `mod <foo>;` item
    fn parse_item_mod(&mut self, outer_attrs: &[Attribute]) -> PResult<ItemInfo> {
        let id_span = self.span;
        let id = try!(self.parse_ident());
        if self.check(&token::Semi) {
            try!(self.bump());
            // This mod is in an external file. Let's go get it!
            let (m, attrs) = try!(self.eval_src_mod(id, outer_attrs, id_span));
            Ok((id, m, Some(attrs)))
        } else {
            self.push_mod_path(id, outer_attrs);
            try!(self.expect(&token::OpenDelim(token::Brace)));
            let mod_inner_lo = self.span.lo;
            let old_owns_directory = self.owns_directory;
            self.owns_directory = true;
            let attrs = try!(self.parse_inner_attributes());
            let m = try!(self.parse_mod_items(&token::CloseDelim(token::Brace), mod_inner_lo));
            self.owns_directory = old_owns_directory;
            self.pop_mod_path();
            Ok((id, ItemMod(m), Some(attrs)))
        }
    }

    fn push_mod_path(&mut self, id: Ident, attrs: &[Attribute]) {
        let default_path = self.id_to_interned_str(id);
        let file_path = match attr::first_attr_value_str_by_name(attrs, "path") {
            Some(d) => d,
            None => default_path,
        };
        self.mod_path_stack.push(file_path)
    }

    fn pop_mod_path(&mut self) {
        self.mod_path_stack.pop().unwrap();
    }

    pub fn submod_path_from_attr(attrs: &[ast::Attribute], dir_path: &Path) -> Option<PathBuf> {
        attr::first_attr_value_str_by_name(attrs, "path").map(|d| dir_path.join(&*d))
    }

    /// Returns either a path to a module, or .
    pub fn default_submod_path(id: ast::Ident, dir_path: &Path, codemap: &CodeMap) -> ModulePath
    {
        let mod_name = id.to_string();
        let default_path_str = format!("{}.rs", mod_name);
        let secondary_path_str = format!("{}/mod.rs", mod_name);
        let default_path = dir_path.join(&default_path_str);
        let secondary_path = dir_path.join(&secondary_path_str);
        let default_exists = codemap.file_exists(&default_path);
        let secondary_exists = codemap.file_exists(&secondary_path);

        let result = match (default_exists, secondary_exists) {
            (true, false) => Ok(ModulePathSuccess { path: default_path, owns_directory: false }),
            (false, true) => Ok(ModulePathSuccess { path: secondary_path, owns_directory: true }),
            (false, false) => Err(ModulePathError {
                err_msg: format!("file not found for module `{}`", mod_name),
                help_msg: format!("name the file either {} or {} inside the directory {:?}",
                                  default_path_str,
                                  secondary_path_str,
                                  dir_path.display()),
            }),
            (true, true) => Err(ModulePathError {
                err_msg: format!("file for module `{}` found at both {} and {}",
                                 mod_name,
                                 default_path_str,
                                 secondary_path_str),
                help_msg: "delete or rename one of them to remove the ambiguity".to_owned(),
            }),
        };

        ModulePath {
            name: mod_name,
            path_exists: default_exists || secondary_exists,
            result: result,
        }
    }

    fn submod_path(&mut self,
                   id: ast::Ident,
                   outer_attrs: &[ast::Attribute],
                   id_sp: Span) -> PResult<ModulePathSuccess> {
        let mut prefix = PathBuf::from(&self.sess.codemap().span_to_filename(self.span));
        prefix.pop();
        let mut dir_path = prefix;
        for part in &self.mod_path_stack {
            dir_path.push(&**part);
        }

        if let Some(p) = Parser::submod_path_from_attr(outer_attrs, &dir_path) {
            return Ok(ModulePathSuccess { path: p, owns_directory: true });
        }

        let paths = Parser::default_submod_path(id, &dir_path, self.sess.codemap());

        if !self.owns_directory {
            self.span_err(id_sp, "cannot declare a new module at this location");
            let this_module = match self.mod_path_stack.last() {
                Some(name) => name.to_string(),
                None => self.root_module_name.as_ref().unwrap().clone(),
            };
            self.span_note(id_sp,
                           &format!("maybe move this module `{0}` to its own directory \
                                     via `{0}/mod.rs`",
                                    this_module));
            if paths.path_exists {
                self.span_note(id_sp,
                               &format!("... or maybe `use` the module `{}` instead \
                                         of possibly redeclaring it",
                                        paths.name));
            }
            self.abort_if_errors();
        }

        match paths.result {
            Ok(succ) => Ok(succ),
            Err(err) => Err(self.span_fatal_help(id_sp, &err.err_msg, &err.help_msg)),
        }
    }

    /// Read a module from a source file.
    fn eval_src_mod(&mut self,
                    id: ast::Ident,
                    outer_attrs: &[ast::Attribute],
                    id_sp: Span)
                    -> PResult<(ast::Item_, Vec<ast::Attribute> )> {
        let ModulePathSuccess { path, owns_directory } = try!(self.submod_path(id,
                                                                               outer_attrs,
                                                                               id_sp));

        self.eval_src_mod_from_path(path,
                                    owns_directory,
                                    id.to_string(),
                                    id_sp)
    }

    fn eval_src_mod_from_path(&mut self,
                              path: PathBuf,
                              owns_directory: bool,
                              name: String,
                              id_sp: Span) -> PResult<(ast::Item_, Vec<ast::Attribute> )> {
        let mut included_mod_stack = self.sess.included_mod_stack.borrow_mut();
        match included_mod_stack.iter().position(|p| *p == path) {
            Some(i) => {
                let mut err = String::from("circular modules: ");
                let len = included_mod_stack.len();
                for p in &included_mod_stack[i.. len] {
                    err.push_str(&p.to_string_lossy());
                    err.push_str(" -> ");
                }
                err.push_str(&path.to_string_lossy());
                return Err(self.span_fatal(id_sp, &err[..]));
            }
            None => ()
        }
        included_mod_stack.push(path.clone());
        drop(included_mod_stack);

        let mut p0 = new_sub_parser_from_file(self.sess,
                                              self.cfg.clone(),
                                              &path,
                                              owns_directory,
                                              Some(name),
                                              id_sp);
        let mod_inner_lo = p0.span.lo;
        let mod_attrs = try!(p0.parse_inner_attributes());
        let m0 = try!(p0.parse_mod_items(&token::Eof, mod_inner_lo));
        self.sess.included_mod_stack.borrow_mut().pop();
        Ok((ast::ItemMod(m0), mod_attrs))
    }

    /// Parse a function declaration from a foreign module
    fn parse_item_foreign_fn(&mut self, vis: ast::Visibility, lo: BytePos,
                             attrs: Vec<Attribute>) -> PResult<P<ForeignItem>> {
        try!(self.expect_keyword(keywords::Fn));

        let (ident, mut generics) = try!(self.parse_fn_header());
        let decl = try!(self.parse_fn_decl(true));
        generics.where_clause = try!(self.parse_where_clause());
        let hi = self.span.hi;
        try!(self.expect(&token::Semi));
        Ok(P(ast::ForeignItem {
            ident: ident,
            attrs: attrs,
            node: ForeignItemFn(decl, generics),
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, hi),
            vis: vis
        }))
    }

    /// Parse a static item from a foreign module
    fn parse_item_foreign_static(&mut self, vis: ast::Visibility, lo: BytePos,
                                 attrs: Vec<Attribute>) -> PResult<P<ForeignItem>> {
        try!(self.expect_keyword(keywords::Static));
        let mutbl = try!(self.eat_keyword(keywords::Mut));

        let ident = try!(self.parse_ident());
        try!(self.expect(&token::Colon));
        let ty = try!(self.parse_ty_sum());
        let hi = self.span.hi;
        try!(self.expect(&token::Semi));
        Ok(P(ForeignItem {
            ident: ident,
            attrs: attrs,
            node: ForeignItemStatic(ty, mutbl),
            id: ast::DUMMY_NODE_ID,
            span: mk_sp(lo, hi),
            vis: vis
        }))
    }

    /// Parse extern crate links
    ///
    /// # Examples
    ///
    /// extern crate foo;
    /// extern crate bar as foo;
    fn parse_item_extern_crate(&mut self,
                               lo: BytePos,
                               visibility: Visibility,
                               attrs: Vec<Attribute>)
                                -> PResult<P<Item>> {

        let crate_name = try!(self.parse_ident());
        let (maybe_path, ident) = if let Some(ident) = try!(self.parse_rename()) {
            (Some(crate_name.name), ident)
        } else {
            (None, crate_name)
        };
        try!(self.expect(&token::Semi));

        let last_span = self.last_span;

        if visibility == ast::Public {
            self.span_warn(mk_sp(lo, last_span.hi),
                           "`pub extern crate` does not work as expected and should not be used. \
                            Likely to become an error. Prefer `extern crate` and `pub use`.");
        }

        Ok(self.mk_item(lo,
                        last_span.hi,
                        ident,
                        ItemExternCrate(maybe_path),
                        visibility,
                        attrs))
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
                              mut attrs: Vec<Attribute>)
                              -> PResult<P<Item>> {
        try!(self.expect(&token::OpenDelim(token::Brace)));

        let abi = opt_abi.unwrap_or(abi::C);

        attrs.extend(try!(self.parse_inner_attributes()));

        let mut foreign_items = vec![];
        while let Some(item) = try!(self.parse_foreign_item()) {
            foreign_items.push(item);
        }
        try!(self.expect(&token::CloseDelim(token::Brace)));

        let last_span = self.last_span;
        let m = ast::ForeignMod {
            abi: abi,
            items: foreign_items
        };
        Ok(self.mk_item(lo,
                     last_span.hi,
                     special_idents::invalid,
                     ItemForeignMod(m),
                     visibility,
                     attrs))
    }

    /// Parse type Foo = Bar;
    fn parse_item_type(&mut self) -> PResult<ItemInfo> {
        let ident = try!(self.parse_ident());
        let mut tps = try!(self.parse_generics());
        tps.where_clause = try!(self.parse_where_clause());
        try!(self.expect(&token::Eq));
        let ty = try!(self.parse_ty_sum());
        try!(self.expect(&token::Semi));
        Ok((ident, ItemTy(ty, tps), None))
    }

    /// Parse the part of an "enum" decl following the '{'
    fn parse_enum_def(&mut self, _generics: &ast::Generics) -> PResult<EnumDef> {
        let mut variants = Vec::new();
        let mut all_nullary = true;
        let mut any_disr = None;
        while self.token != token::CloseDelim(token::Brace) {
            let variant_attrs = try!(self.parse_outer_attributes());
            let vlo = self.span.lo;

            let struct_def;
            let mut disr_expr = None;
            let ident = try!(self.parse_ident());
            if self.check(&token::OpenDelim(token::Brace)) {
                // Parse a struct variant.
                all_nullary = false;
                struct_def = VariantData::Struct(try!(self.parse_record_struct_body(ParsePub::No)),
                                                 ast::DUMMY_NODE_ID);
            } else if self.check(&token::OpenDelim(token::Paren)) {
                all_nullary = false;
                struct_def = VariantData::Tuple(try!(self.parse_tuple_struct_body(ParsePub::No)),
                                                ast::DUMMY_NODE_ID);
            } else if try!(self.eat(&token::Eq) ){
                disr_expr = Some(try!(self.parse_expr()));
                any_disr = disr_expr.as_ref().map(|expr| expr.span);
                struct_def = VariantData::Unit(ast::DUMMY_NODE_ID);
            } else {
                struct_def = VariantData::Unit(ast::DUMMY_NODE_ID);
            }

            let vr = ast::Variant_ {
                name: ident,
                attrs: variant_attrs,
                data: struct_def,
                disr_expr: disr_expr,
            };
            variants.push(P(spanned(vlo, self.last_span.hi, vr)));

            if !try!(self.eat(&token::Comma)) { break; }
        }
        try!(self.expect(&token::CloseDelim(token::Brace)));
        match any_disr {
            Some(disr_span) if !all_nullary =>
                self.span_err(disr_span,
                    "discriminator values can only be used with a c-like enum"),
            _ => ()
        }

        Ok(ast::EnumDef { variants: variants })
    }

    /// Parse an "enum" declaration
    fn parse_item_enum(&mut self) -> PResult<ItemInfo> {
        let id = try!(self.parse_ident());
        let mut generics = try!(self.parse_generics());
        generics.where_clause = try!(self.parse_where_clause());
        try!(self.expect(&token::OpenDelim(token::Brace)));

        let enum_definition = try!(self.parse_enum_def(&generics));
        Ok((id, ItemEnum(enum_definition, generics), None))
    }

    /// Parses a string as an ABI spec on an extern type or module. Consumes
    /// the `extern` keyword, if one is found.
    fn parse_opt_abi(&mut self) -> PResult<Option<abi::Abi>> {
        match self.token {
            token::Literal(token::Str_(s), suf) | token::Literal(token::StrRaw(s, _), suf) => {
                let sp = self.span;
                self.expect_no_suffix(sp, "ABI spec", suf);
                try!(self.bump());
                match abi::lookup(&s.as_str()) {
                    Some(abi) => Ok(Some(abi)),
                    None => {
                        let last_span = self.last_span;
                        self.span_err(
                            last_span,
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

    /// Parse one of the items allowed by the flags.
    /// NB: this function no longer parses the items inside an
    /// extern crate.
    fn parse_item_(&mut self, attrs: Vec<Attribute>,
                   macros_allowed: bool) -> PResult<Option<P<Item>>> {
        let nt_item = match self.token {
            token::Interpolated(token::NtItem(ref item)) => {
                Some((**item).clone())
            }
            _ => None
        };
        match nt_item {
            Some(mut item) => {
                try!(self.bump());
                let mut attrs = attrs;
                mem::swap(&mut item.attrs, &mut attrs);
                item.attrs.extend(attrs);
                return Ok(Some(P(item)));
            }
            None => {}
        }

        let lo = self.span.lo;

        let visibility = try!(self.parse_visibility());

        if try!(self.eat_keyword(keywords::Use) ){
            // USE ITEM
            let item_ = ItemUse(try!(self.parse_view_path()));
            try!(self.expect(&token::Semi));

            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    token::special_idents::invalid,
                                    item_,
                                    visibility,
                                    attrs);
            return Ok(Some(item));
        }

        if try!(self.eat_keyword(keywords::Extern)) {
            if try!(self.eat_keyword(keywords::Crate)) {
                return Ok(Some(try!(self.parse_item_extern_crate(lo, visibility, attrs))));
            }

            let opt_abi = try!(self.parse_opt_abi());

            if try!(self.eat_keyword(keywords::Fn) ){
                // EXTERN FUNCTION ITEM
                let abi = opt_abi.unwrap_or(abi::C);
                let (ident, item_, extra_attrs) =
                    try!(self.parse_item_fn(Unsafety::Normal, Constness::NotConst, abi));
                let last_span = self.last_span;
                let item = self.mk_item(lo,
                                        last_span.hi,
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return Ok(Some(item));
            } else if self.check(&token::OpenDelim(token::Brace)) {
                return Ok(Some(try!(self.parse_item_foreign_mod(lo, opt_abi, visibility, attrs))));
            }

            try!(self.expect_one_of(&[], &[]));
        }

        if try!(self.eat_keyword(keywords::Static) ){
            // STATIC ITEM
            let m = if try!(self.eat_keyword(keywords::Mut)) {MutMutable} else {MutImmutable};
            let (ident, item_, extra_attrs) = try!(self.parse_item_const(Some(m)));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Const) ){
            if self.check_keyword(keywords::Fn)
                || (self.check_keyword(keywords::Unsafe)
                    && self.look_ahead(1, |t| t.is_keyword(keywords::Fn))) {
                // CONST FUNCTION ITEM
                let unsafety = if try!(self.eat_keyword(keywords::Unsafe) ){
                    Unsafety::Unsafe
                } else {
                    Unsafety::Normal
                };
                try!(self.bump());
                let (ident, item_, extra_attrs) =
                    try!(self.parse_item_fn(unsafety, Constness::Const, abi::Rust));
                let last_span = self.last_span;
                let item = self.mk_item(lo,
                                        last_span.hi,
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return Ok(Some(item));
            }

            // CONST ITEM
            if try!(self.eat_keyword(keywords::Mut) ){
                let last_span = self.last_span;
                self.span_err(last_span, "const globals cannot be mutable");
                self.fileline_help(last_span, "did you mean to declare a static?");
            }
            let (ident, item_, extra_attrs) = try!(self.parse_item_const(None));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(keywords::Unsafe) &&
            self.look_ahead(1, |t| t.is_keyword(keywords::Trait))
        {
            // UNSAFE TRAIT ITEM
            try!(self.expect_keyword(keywords::Unsafe));
            try!(self.expect_keyword(keywords::Trait));
            let (ident, item_, extra_attrs) =
                try!(self.parse_item_trait(ast::Unsafety::Unsafe));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(keywords::Unsafe) &&
            self.look_ahead(1, |t| t.is_keyword(keywords::Impl))
        {
            // IMPL ITEM
            try!(self.expect_keyword(keywords::Unsafe));
            try!(self.expect_keyword(keywords::Impl));
            let (ident, item_, extra_attrs) = try!(self.parse_item_impl(ast::Unsafety::Unsafe));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(keywords::Fn) {
            // FUNCTION ITEM
            try!(self.bump());
            let (ident, item_, extra_attrs) =
                try!(self.parse_item_fn(Unsafety::Normal, Constness::NotConst, abi::Rust));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(keywords::Unsafe)
            && self.look_ahead(1, |t| *t != token::OpenDelim(token::Brace)) {
            // UNSAFE FUNCTION ITEM
            try!(self.bump());
            let abi = if try!(self.eat_keyword(keywords::Extern) ){
                try!(self.parse_opt_abi()).unwrap_or(abi::C)
            } else {
                abi::Rust
            };
            try!(self.expect_keyword(keywords::Fn));
            let (ident, item_, extra_attrs) =
                try!(self.parse_item_fn(Unsafety::Unsafe, Constness::NotConst, abi));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Mod) ){
            // MODULE ITEM
            let (ident, item_, extra_attrs) =
                try!(self.parse_item_mod(&attrs[..]));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Type) ){
            // TYPE ITEM
            let (ident, item_, extra_attrs) = try!(self.parse_item_type());
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Enum) ){
            // ENUM ITEM
            let (ident, item_, extra_attrs) = try!(self.parse_item_enum());
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Trait) ){
            // TRAIT ITEM
            let (ident, item_, extra_attrs) =
                try!(self.parse_item_trait(ast::Unsafety::Normal));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Impl) ){
            // IMPL ITEM
            let (ident, item_, extra_attrs) = try!(self.parse_item_impl(ast::Unsafety::Normal));
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if try!(self.eat_keyword(keywords::Struct) ){
            // STRUCT ITEM
            let (ident, item_, extra_attrs) = try!(self.parse_item_struct());
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        self.parse_macro_use_or_failure(attrs,macros_allowed,lo,visibility)
    }

    /// Parse a foreign item.
    fn parse_foreign_item(&mut self) -> PResult<Option<P<ForeignItem>>> {
        let attrs = try!(self.parse_outer_attributes());
        let lo = self.span.lo;
        let visibility = try!(self.parse_visibility());

        if self.check_keyword(keywords::Static) {
            // FOREIGN STATIC ITEM
            return Ok(Some(try!(self.parse_item_foreign_static(visibility, lo, attrs))));
        }
        if self.check_keyword(keywords::Fn) || self.check_keyword(keywords::Unsafe) {
            // FOREIGN FUNCTION ITEM
            return Ok(Some(try!(self.parse_item_foreign_fn(visibility, lo, attrs))));
        }

        // FIXME #5668: this will occur for a macro invocation:
        match try!(self.parse_macro_use_or_failure(attrs, true, lo, visibility)) {
            Some(item) => {
                return Err(self.span_fatal(item.span, "macros cannot expand to foreign items"));
            }
            None => Ok(None)
        }
    }

    /// This is the fall-through for parsing items.
    fn parse_macro_use_or_failure(
        &mut self,
        attrs: Vec<Attribute> ,
        macros_allowed: bool,
        lo: BytePos,
        visibility: Visibility
    ) -> PResult<Option<P<Item>>> {
        if macros_allowed && !self.token.is_any_keyword()
                && self.look_ahead(1, |t| *t == token::Not)
                && (self.look_ahead(2, |t| t.is_plain_ident())
                    || self.look_ahead(2, |t| *t == token::OpenDelim(token::Paren))
                    || self.look_ahead(2, |t| *t == token::OpenDelim(token::Brace))) {
            // MACRO INVOCATION ITEM

            let last_span = self.last_span;
            self.complain_if_pub_macro(visibility, last_span);

            // item macro.
            let pth = try!(self.parse_path(NoTypesAllowed));
            try!(self.expect(&token::Not));

            // a 'special' identifier (like what `macro_rules!` uses)
            // is optional. We should eventually unify invoc syntax
            // and remove this.
            let id = if self.token.is_plain_ident() {
                try!(self.parse_ident())
            } else {
                token::special_idents::invalid // no special identifier
            };
            // eat a matched-delimiter token tree:
            let delim = try!(self.expect_open_delim());
            let tts = try!(self.parse_seq_to_end(&token::CloseDelim(delim),
                                            seq_sep_none(),
                                            |p| p.parse_token_tree()));
            // single-variant-enum... :
            let m = Mac_ { path: pth, tts: tts, ctxt: EMPTY_CTXT };
            let m: ast::Mac = codemap::Spanned { node: m,
                                             span: mk_sp(self.span.lo,
                                                         self.span.hi) };

            if delim != token::Brace {
                if !try!(self.eat(&token::Semi) ){
                    let last_span = self.last_span;
                    self.span_err(last_span,
                                  "macros that expand to items must either \
                                   be surrounded with braces or followed by \
                                   a semicolon");
                }
            }

            let item_ = ItemMac(m);
            let last_span = self.last_span;
            let item = self.mk_item(lo,
                                    last_span.hi,
                                    id,
                                    item_,
                                    visibility,
                                    attrs);
            return Ok(Some(item));
        }

        // FAILURE TO PARSE ITEM
        match visibility {
            Inherited => {}
            Public => {
                let last_span = self.last_span;
                return Err(self.span_fatal(last_span, "unmatched visibility `pub`"));
            }
        }

        if !attrs.is_empty() {
            self.expected_item_err(&attrs);
        }
        Ok(None)
    }

    pub fn parse_item(&mut self) -> PResult<Option<P<Item>>> {
        let attrs = try!(self.parse_outer_attributes());
        self.parse_item_(attrs, true)
    }


    /// Matches view_path : MOD? non_global_path as IDENT
    /// | MOD? non_global_path MOD_SEP LBRACE RBRACE
    /// | MOD? non_global_path MOD_SEP LBRACE ident_seq RBRACE
    /// | MOD? non_global_path MOD_SEP STAR
    /// | MOD? non_global_path
    fn parse_view_path(&mut self) -> PResult<P<ViewPath>> {
        let lo = self.span.lo;

        // Allow a leading :: because the paths are absolute either way.
        // This occurs with "use $crate::..." in macros.
        try!(self.eat(&token::ModSep));

        if self.check(&token::OpenDelim(token::Brace)) {
            // use {foo,bar}
            let idents = try!(self.parse_unspanned_seq(
                &token::OpenDelim(token::Brace),
                &token::CloseDelim(token::Brace),
                seq_sep_trailing_allowed(token::Comma),
                |p| p.parse_path_list_item()));
            let path = ast::Path {
                span: mk_sp(lo, self.span.hi),
                global: false,
                segments: Vec::new()
            };
            return Ok(P(spanned(lo, self.span.hi, ViewPathList(path, idents))));
        }

        let first_ident = try!(self.parse_ident());
        let mut path = vec!(first_ident);
        if let token::ModSep = self.token {
            // foo::bar or foo::{a,b,c} or foo::*
            while self.check(&token::ModSep) {
                try!(self.bump());

                match self.token {
                  token::Ident(..) => {
                    let ident = try!(self.parse_ident());
                    path.push(ident);
                  }

                  // foo::bar::{a,b,c}
                  token::OpenDelim(token::Brace) => {
                    let idents = try!(self.parse_unspanned_seq(
                        &token::OpenDelim(token::Brace),
                        &token::CloseDelim(token::Brace),
                        seq_sep_trailing_allowed(token::Comma),
                        |p| p.parse_path_list_item()
                    ));
                    let path = ast::Path {
                        span: mk_sp(lo, self.span.hi),
                        global: false,
                        segments: path.into_iter().map(|identifier| {
                            ast::PathSegment {
                                identifier: identifier,
                                parameters: ast::PathParameters::none(),
                            }
                        }).collect()
                    };
                    return Ok(P(spanned(lo, self.span.hi, ViewPathList(path, idents))));
                  }

                  // foo::bar::*
                  token::BinOp(token::Star) => {
                    try!(self.bump());
                    let path = ast::Path {
                        span: mk_sp(lo, self.span.hi),
                        global: false,
                        segments: path.into_iter().map(|identifier| {
                            ast::PathSegment {
                                identifier: identifier,
                                parameters: ast::PathParameters::none(),
                            }
                        }).collect()
                    };
                    return Ok(P(spanned(lo, self.span.hi, ViewPathGlob(path))));
                  }

                  // fall-through for case foo::bar::;
                  token::Semi => {
                    self.span_err(self.span, "expected identifier or `{` or `*`, found `;`");
                  }

                  _ => break
                }
            }
        }
        let mut rename_to = path[path.len() - 1];
        let path = ast::Path {
            span: mk_sp(lo, self.last_span.hi),
            global: false,
            segments: path.into_iter().map(|identifier| {
                ast::PathSegment {
                    identifier: identifier,
                    parameters: ast::PathParameters::none(),
                }
            }).collect()
        };
        rename_to = try!(self.parse_rename()).unwrap_or(rename_to);
        Ok(P(spanned(lo, self.last_span.hi, ViewPathSimple(rename_to, path))))
    }

    fn parse_rename(&mut self) -> PResult<Option<Ident>> {
        if try!(self.eat_keyword(keywords::As)) {
            self.parse_ident().map(Some)
        } else {
            Ok(None)
        }
    }

    /// Parses a source module as a crate. This is the main
    /// entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<Crate> {
        let lo = self.span.lo;
        Ok(ast::Crate {
            attrs: try!(self.parse_inner_attributes()),
            module: try!(self.parse_mod_items(&token::Eof, lo)),
            config: self.cfg.clone(),
            span: mk_sp(lo, self.span.lo),
            exported_macros: Vec::new(),
        })
    }

    pub fn parse_optional_str(&mut self)
                              -> PResult<Option<(InternedString,
                                                 ast::StrStyle,
                                                 Option<ast::Name>)>> {
        let ret = match self.token {
            token::Literal(token::Str_(s), suf) => {
                (self.id_to_interned_str(ast::Ident::with_empty_ctxt(s)), ast::CookedStr, suf)
            }
            token::Literal(token::StrRaw(s, n), suf) => {
                (self.id_to_interned_str(ast::Ident::with_empty_ctxt(s)), ast::RawStr(n), suf)
            }
            _ => return Ok(None)
        };
        try!(self.bump());
        Ok(Some(ret))
    }

    pub fn parse_str(&mut self) -> PResult<(InternedString, StrStyle)> {
        match try!(self.parse_optional_str()) {
            Some((s, style, suf)) => {
                let sp = self.last_span;
                self.expect_no_suffix(sp, "string literal", suf);
                Ok((s, style))
            }
            _ =>  Err(self.fatal("expected string literal"))
        }
    }

    /// Parse attributes that appear before an item
    pub fn parse_outer_attributes(&mut self) -> PResult<Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = Vec::new();
        loop {
            debug!("parse_outer_attributes: self.token={:?}",
                   self.token);
            match self.token {
              token::Pound => {
                attrs.push(try!(self.parse_attribute(false)));
              }
              token::DocComment(s) => {
                let attr = attr::mk_sugared_doc_attr(
                    attr::mk_attr_id(),
                    self.id_to_interned_str(ast::Ident::with_empty_ctxt(s)),
                    self.span.lo,
                    self.span.hi
                );
                if attr.node.style != ast::AttrStyle::Outer {
                  return Err(self.fatal("expected outer comment"));
                }
                attrs.push(attr);
                try!(self.bump());
              }
              _ => break
            }
        }
        return Ok(attrs);
    }

    /// Matches `attribute = # ! [ meta_item ]`
    ///
    /// If permit_inner is true, then a leading `!` indicates an inner
    /// attribute
    pub fn parse_attribute(&mut self, permit_inner: bool) -> PResult<ast::Attribute> {
        debug!("parse_attributes: permit_inner={:?} self.token={:?}",
               permit_inner, self.token);
        let (span, value, mut style) = match self.token {
            token::Pound => {
                let lo = self.span.lo;
                try!(self.bump());

                if permit_inner { self.expected_tokens.push(TokenType::Token(token::Not)); }
                let style = if self.token == token::Not {
                    try!(self.bump());
                    if !permit_inner {
                        let span = self.span;
                        self.span_err(span,
                                      "an inner attribute is not permitted in \
                                       this context");
                        self.fileline_help(span,
                                       "place inner attribute at the top of the module or block");
                    }
                    ast::AttrStyle::Inner
                } else {
                    ast::AttrStyle::Outer
                };

                try!(self.expect(&token::OpenDelim(token::Bracket)));
                let meta_item = try!(self.parse_meta_item());
                let hi = self.span.hi;
                try!(self.expect(&token::CloseDelim(token::Bracket)));

                (mk_sp(lo, hi), meta_item, style)
            }
            _ => {
                let token_str = self.this_token_to_string();
                return Err(self.fatal(&format!("expected `#`, found `{}`", token_str)));
            }
        };

        if permit_inner && self.token == token::Semi {
            try!(self.bump());
            self.span_warn(span, "this inner attribute syntax is deprecated. \
                           The new syntax is `#![foo]`, with a bang and no semicolon");
            style = ast::AttrStyle::Inner;
        }

        Ok(Spanned {
            span: span,
            node: ast::Attribute_ {
                id: attr::mk_attr_id(),
                style: style,
                value: value,
                is_sugared_doc: false
            }
        })
    }

    /// Parse attributes that appear after the opening of an item. These should
    /// be preceded by an exclamation mark, but we accept and warn about one
    /// terminated by a semicolon.

    /// matches inner_attrs*
    pub fn parse_inner_attributes(&mut self) -> PResult<Vec<ast::Attribute>> {
        let mut attrs: Vec<ast::Attribute> = vec![];
        loop {
            match self.token {
                token::Pound => {
                    // Don't even try to parse if it's not an inner attribute.
                    if !self.look_ahead(1, |t| t == &token::Not) {
                        break;
                    }

                    let attr = try!(self.parse_attribute(true));
                    assert!(attr.node.style == ast::AttrStyle::Inner);
                    attrs.push(attr);
                }
                token::DocComment(s) => {
                    // we need to get the position of this token before we bump.
                    let Span { lo, hi, .. } = self.span;
                    let str = self.id_to_interned_str(ast::Ident::with_empty_ctxt(s));
                    let attr = attr::mk_sugared_doc_attr(attr::mk_attr_id(), str, lo, hi);
                    if attr.node.style == ast::AttrStyle::Inner {
                        attrs.push(attr);
                        try!(self.bump());
                    } else {
                        break;
                    }
                }
                _ => break
            }
        }
        Ok(attrs)
    }

    /// matches meta_item = IDENT
    /// | IDENT = lit
    /// | IDENT meta_seq
    pub fn parse_meta_item(&mut self) -> PResult<P<ast::MetaItem>> {
        let nt_meta = match self.token {
            token::Interpolated(token::NtMeta(ref e)) => {
                Some(e.clone())
            }
            _ => None
        };

        match nt_meta {
            Some(meta) => {
                try!(self.bump());
                return Ok(meta);
            }
            None => {}
        }

        let lo = self.span.lo;
        let ident = try!(self.parse_ident());
        let name = self.id_to_interned_str(ident);
        match self.token {
            token::Eq => {
                try!(self.bump());
                let lit = try!(self.parse_lit());
                // FIXME #623 Non-string meta items are not serialized correctly;
                // just forbid them for now
                match lit.node {
                    ast::LitStr(..) => {}
                    _ => {
                        self.span_err(
                            lit.span,
                            "non-string literals are not allowed in meta-items");
                    }
                }
                let hi = self.span.hi;
                Ok(P(spanned(lo, hi, ast::MetaNameValue(name, lit))))
            }
            token::OpenDelim(token::Paren) => {
                let inner_items = try!(self.parse_meta_seq());
                let hi = self.span.hi;
                Ok(P(spanned(lo, hi, ast::MetaList(name, inner_items))))
            }
            _ => {
                let hi = self.last_span.hi;
                Ok(P(spanned(lo, hi, ast::MetaWord(name))))
            }
        }
    }

    /// matches meta_seq = ( COMMASEP(meta_item) )
    fn parse_meta_seq(&mut self) -> PResult<Vec<P<ast::MetaItem>>> {
        self.parse_unspanned_seq(&token::OpenDelim(token::Paren),
                                 &token::CloseDelim(token::Paren),
                                 seq_sep_trailing_allowed(token::Comma),
                                 |p| p.parse_meta_item())
    }
}
