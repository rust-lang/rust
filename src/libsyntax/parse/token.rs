// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{P, Ident, Name, Mrk};
use ast_util;
use ext::mtwt;
use parse::token;
use util::interner::{RcStr, StrInterner};
use util::interner;

use serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fmt;
use std::path::BytesContainer;
use std::mem;
use std::rc::Rc;
use std::string::String;

#[allow(non_camel_case_types)]
#[deriving(Clone, Encodable, Decodable, Eq, TotalEq, Hash, Show)]
pub enum BinOp {
    PLUS,
    MINUS,
    STAR,
    SLASH,
    PERCENT,
    CARET,
    AND,
    OR,
    SHL,
    SHR,
}

#[allow(non_camel_case_types)]
#[deriving(Clone, Encodable, Decodable, Eq, TotalEq, Hash, Show)]
pub enum Token {
    /* Expression-operator symbols. */
    EQ,
    LT,
    LE,
    EQEQ,
    NE,
    GE,
    GT,
    ANDAND,
    OROR,
    NOT,
    TILDE,
    BINOP(BinOp),
    BINOPEQ(BinOp),

    /* Structural symbols */
    AT,
    DOT,
    DOTDOT,
    DOTDOTDOT,
    COMMA,
    SEMI,
    COLON,
    MOD_SEP,
    RARROW,
    LARROW,
    FAT_ARROW,
    LPAREN,
    RPAREN,
    LBRACKET,
    RBRACKET,
    LBRACE,
    RBRACE,
    POUND,
    DOLLAR,

    /* Literals */
    LIT_CHAR(char),
    LIT_INT(i64, ast::IntTy),
    LIT_UINT(u64, ast::UintTy),
    LIT_INT_UNSUFFIXED(i64),
    LIT_FLOAT(ast::Ident, ast::FloatTy),
    LIT_FLOAT_UNSUFFIXED(ast::Ident),
    LIT_STR(ast::Ident),
    LIT_STR_RAW(ast::Ident, uint), /* raw str delimited by n hash symbols */

    /* Name components */
    // an identifier contains an "is_mod_name" boolean,
    // indicating whether :: follows this token with no
    // whitespace in between.
    IDENT(ast::Ident, bool),
    UNDERSCORE,
    LIFETIME(ast::Ident),

    /* For interpolation */
    INTERPOLATED(Nonterminal),

    DOC_COMMENT(ast::Ident),
    EOF,
}

#[deriving(Clone, Encodable, Decodable, Eq, TotalEq, Hash)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem(@ast::Item),
    NtBlock(P<ast::Block>),
    NtStmt(@ast::Stmt),
    NtPat( @ast::Pat),
    NtExpr(@ast::Expr),
    NtTy(  P<ast::Ty>),
    NtIdent(Box<ast::Ident>, bool),
    NtMeta(@ast::MetaItem), // stuff inside brackets for attributes
    NtPath(Box<ast::Path>),
    NtTT(  @ast::TokenTree), // needs @ed to break a circularity
    NtMatchers(Vec<ast::Matcher> )
}

impl fmt::Show for Nonterminal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            NtItem(..) => f.pad("NtItem(..)"),
            NtBlock(..) => f.pad("NtBlock(..)"),
            NtStmt(..) => f.pad("NtStmt(..)"),
            NtPat(..) => f.pad("NtPat(..)"),
            NtExpr(..) => f.pad("NtExpr(..)"),
            NtTy(..) => f.pad("NtTy(..)"),
            NtIdent(..) => f.pad("NtIdent(..)"),
            NtMeta(..) => f.pad("NtMeta(..)"),
            NtPath(..) => f.pad("NtPath(..)"),
            NtTT(..) => f.pad("NtTT(..)"),
            NtMatchers(..) => f.pad("NtMatchers(..)"),
        }
    }
}

pub fn binop_to_str(o: BinOp) -> String {
    match o {
      PLUS => "+".to_string(),
      MINUS => "-".to_string(),
      STAR => "*".to_string(),
      SLASH => "/".to_string(),
      PERCENT => "%".to_string(),
      CARET => "^".to_string(),
      AND => "&".to_string(),
      OR => "|".to_string(),
      SHL => "<<".to_string(),
      SHR => ">>".to_string()
    }
}

pub fn to_str(t: &Token) -> String {
    match *t {
      EQ => "=".to_string(),
      LT => "<".to_string(),
      LE => "<=".to_string(),
      EQEQ => "==".to_string(),
      NE => "!=".to_string(),
      GE => ">=".to_string(),
      GT => ">".to_string(),
      NOT => "!".to_string(),
      TILDE => "~".to_string(),
      OROR => "||".to_string(),
      ANDAND => "&&".to_string(),
      BINOP(op) => binop_to_str(op),
      BINOPEQ(op) => {
          let mut s = binop_to_str(op);
          s.push_str("=");
          s
      }

      /* Structural symbols */
      AT => "@".to_string(),
      DOT => ".".to_string(),
      DOTDOT => "..".to_string(),
      DOTDOTDOT => "...".to_string(),
      COMMA => ",".to_string(),
      SEMI => ";".to_string(),
      COLON => ":".to_string(),
      MOD_SEP => "::".to_string(),
      RARROW => "->".to_string(),
      LARROW => "<-".to_string(),
      FAT_ARROW => "=>".to_string(),
      LPAREN => "(".to_string(),
      RPAREN => ")".to_string(),
      LBRACKET => "[".to_string(),
      RBRACKET => "]".to_string(),
      LBRACE => "{".to_string(),
      RBRACE => "}".to_string(),
      POUND => "#".to_string(),
      DOLLAR => "$".to_string(),

      /* Literals */
      LIT_CHAR(c) => {
          let mut res = String::from_str("'");
          c.escape_default(|c| {
              res.push_char(c);
          });
          res.push_char('\'');
          res
      }
      LIT_INT(i, t) => ast_util::int_ty_to_str(t, Some(i),
                                               ast_util::ForceSuffix),
      LIT_UINT(u, t) => ast_util::uint_ty_to_str(t, Some(u),
                                                 ast_util::ForceSuffix),
      LIT_INT_UNSUFFIXED(i) => { (i as u64).to_str().to_string() }
      LIT_FLOAT(s, t) => {
        let mut body = String::from_str(get_ident(s).get());
        if body.as_slice().ends_with(".") {
            body.push_char('0');  // `10.f` is not a float literal
        }
        body.push_str(ast_util::float_ty_to_str(t).as_slice());
        body
      }
      LIT_FLOAT_UNSUFFIXED(s) => {
        let mut body = String::from_str(get_ident(s).get());
        if body.as_slice().ends_with(".") {
            body.push_char('0');  // `10.f` is not a float literal
        }
        body
      }
      LIT_STR(s) => {
          (format!("\"{}\"", get_ident(s).get().escape_default())).to_string()
      }
      LIT_STR_RAW(s, n) => {
          (format!("r{delim}\"{string}\"{delim}",
                  delim="#".repeat(n), string=get_ident(s))).to_string()
      }

      /* Name components */
      IDENT(s, _) => get_ident(s).get().to_string(),
      LIFETIME(s) => {
          (format!("'{}", get_ident(s))).to_string()
      }
      UNDERSCORE => "_".to_string(),

      /* Other */
      DOC_COMMENT(s) => get_ident(s).get().to_string(),
      EOF => "<eof>".to_string(),
      INTERPOLATED(ref nt) => {
        match nt {
            &NtExpr(e) => ::print::pprust::expr_to_str(e),
            &NtMeta(e) => ::print::pprust::meta_item_to_str(e),
            _ => {
                let mut s = "an interpolated ".to_string();
                match *nt {
                    NtItem(..) => s.push_str("item"),
                    NtBlock(..) => s.push_str("block"),
                    NtStmt(..) => s.push_str("statement"),
                    NtPat(..) => s.push_str("pattern"),
                    NtMeta(..) => fail!("should have been handled"),
                    NtExpr(..) => fail!("should have been handled above"),
                    NtTy(..) => s.push_str("type"),
                    NtIdent(..) => s.push_str("identifier"),
                    NtPath(..) => s.push_str("path"),
                    NtTT(..) => s.push_str("tt"),
                    NtMatchers(..) => s.push_str("matcher sequence")
                };
                s
            }
        }
      }
    }
}

pub fn can_begin_expr(t: &Token) -> bool {
    match *t {
      LPAREN => true,
      LBRACE => true,
      LBRACKET => true,
      IDENT(_, _) => true,
      UNDERSCORE => true,
      TILDE => true,
      LIT_CHAR(_) => true,
      LIT_INT(_, _) => true,
      LIT_UINT(_, _) => true,
      LIT_INT_UNSUFFIXED(_) => true,
      LIT_FLOAT(_, _) => true,
      LIT_FLOAT_UNSUFFIXED(_) => true,
      LIT_STR(_) => true,
      LIT_STR_RAW(_, _) => true,
      POUND => true,
      AT => true,
      NOT => true,
      BINOP(MINUS) => true,
      BINOP(STAR) => true,
      BINOP(AND) => true,
      BINOP(OR) => true, // in lambda syntax
      OROR => true, // in lambda syntax
      MOD_SEP => true,
      INTERPOLATED(NtExpr(..))
      | INTERPOLATED(NtIdent(..))
      | INTERPOLATED(NtBlock(..))
      | INTERPOLATED(NtPath(..)) => true,
      _ => false
    }
}

/// Returns the matching close delimiter if this is an open delimiter,
/// otherwise `None`.
pub fn close_delimiter_for(t: &Token) -> Option<Token> {
    match *t {
        LPAREN   => Some(RPAREN),
        LBRACE   => Some(RBRACE),
        LBRACKET => Some(RBRACKET),
        _        => None
    }
}

pub fn is_lit(t: &Token) -> bool {
    match *t {
      LIT_CHAR(_) => true,
      LIT_INT(_, _) => true,
      LIT_UINT(_, _) => true,
      LIT_INT_UNSUFFIXED(_) => true,
      LIT_FLOAT(_, _) => true,
      LIT_FLOAT_UNSUFFIXED(_) => true,
      LIT_STR(_) => true,
      LIT_STR_RAW(_, _) => true,
      _ => false
    }
}

pub fn is_ident(t: &Token) -> bool {
    match *t { IDENT(_, _) => true, _ => false }
}

pub fn is_ident_or_path(t: &Token) -> bool {
    match *t {
      IDENT(_, _) | INTERPOLATED(NtPath(..)) => true,
      _ => false
    }
}

pub fn is_plain_ident(t: &Token) -> bool {
    match *t { IDENT(_, false) => true, _ => false }
}

pub fn is_bar(t: &Token) -> bool {
    match *t { BINOP(OR) | OROR => true, _ => false }
}

// Get the first "argument"
macro_rules! first {
    ( $first:expr, $( $remainder:expr, )* ) => ( $first )
}

// Get the last "argument" (has to be done recursively to avoid phoney local ambiguity error)
macro_rules! last {
    ( $first:expr, $( $remainder:expr, )+ ) => ( last!( $( $remainder, )+ ) );
    ( $first:expr, ) => ( $first )
}

// In this macro, there is the requirement that the name (the number) must be monotonically
// increasing by one in the special identifiers, starting at 0; the same holds for the keywords,
// except starting from the next number instead of zero, and with the additional exception that
// special identifiers are *also* allowed (they are deduplicated in the important place, the
// interner), an exception which is demonstrated by "static" and "self".
macro_rules! declare_special_idents_and_keywords {(
    // So now, in these rules, why is each definition parenthesised?
    // Answer: otherwise we get a spurious local ambiguity bug on the "}"
    pub mod special_idents {
        $( ($si_name:expr, $si_static:ident, $si_str:expr); )*
    }

    pub mod keywords {
        'strict:
        $( ($sk_name:expr, $sk_variant:ident, $sk_str:expr); )*
        'reserved:
        $( ($rk_name:expr, $rk_variant:ident, $rk_str:expr); )*
    }
) => {
    static STRICT_KEYWORD_START: Name = first!($( $sk_name, )*);
    static STRICT_KEYWORD_FINAL: Name = last!($( $sk_name, )*);
    static RESERVED_KEYWORD_START: Name = first!($( $rk_name, )*);
    static RESERVED_KEYWORD_FINAL: Name = last!($( $rk_name, )*);

    pub mod special_idents {
        use ast::Ident;
        $( pub static $si_static: Ident = Ident { name: $si_name, ctxt: 0 }; )*
    }

    /**
     * All the valid words that have meaning in the Rust language.
     *
     * Rust keywords are either 'strict' or 'reserved'.  Strict keywords may not
     * appear as identifiers at all. Reserved keywords are not used anywhere in
     * the language and may not appear as identifiers.
     */
    pub mod keywords {
        use ast::Ident;

        pub enum Keyword {
            $( $sk_variant, )*
            $( $rk_variant, )*
        }

        impl Keyword {
            pub fn to_ident(&self) -> Ident {
                match *self {
                    $( $sk_variant => Ident { name: $sk_name, ctxt: 0 }, )*
                    $( $rk_variant => Ident { name: $rk_name, ctxt: 0 }, )*
                }
            }
        }
    }

    fn mk_fresh_ident_interner() -> IdentInterner {
        // The indices here must correspond to the numbers in
        // special_idents, in Keyword to_ident(), and in static
        // constants below.
        let mut init_vec = Vec::new();
        $(init_vec.push($si_str);)*
        $(init_vec.push($sk_str);)*
        $(init_vec.push($rk_str);)*
        interner::StrInterner::prefill(init_vec.as_slice())
    }
}}

// If the special idents get renumbered, remember to modify these two as appropriate
static SELF_KEYWORD_NAME: Name = 1;
static STATIC_KEYWORD_NAME: Name = 2;

declare_special_idents_and_keywords! {
    pub mod special_idents {
        // These ones are statics
        (0,                          invalid,                "");
        (super::SELF_KEYWORD_NAME,   self_,                  "self");
        (super::STATIC_KEYWORD_NAME, statik,                 "static");

        // for matcher NTs
        (3,                          tt,                     "tt");
        (4,                          matchers,               "matchers");

        // outside of libsyntax
        (5,                          clownshoe_abi,          "__rust_abi");
        (6,                          opaque,                 "<opaque>");
        (7,                          unnamed_field,          "<unnamed_field>");
        (8,                          type_self,              "Self");
    }

    pub mod keywords {
        // These ones are variants of the Keyword enum

        'strict:
        (9,                          As,         "as");
        (10,                         Break,      "break");
        (11,                         Crate,      "crate");
        (12,                         Else,       "else");
        (13,                         Enum,       "enum");
        (14,                         Extern,     "extern");
        (15,                         False,      "false");
        (16,                         Fn,         "fn");
        (17,                         For,        "for");
        (18,                         If,         "if");
        (19,                         Impl,       "impl");
        (20,                         In,         "in");
        (21,                         Let,        "let");
        (22,                         Loop,       "loop");
        (23,                         Match,      "match");
        (24,                         Mod,        "mod");
        (25,                         Mut,        "mut");
        (26,                         Once,       "once");
        (27,                         Pub,        "pub");
        (28,                         Ref,        "ref");
        (29,                         Return,     "return");
        // Static and Self are also special idents (prefill de-dupes)
        (super::STATIC_KEYWORD_NAME, Static,     "static");
        (super::SELF_KEYWORD_NAME,   Self,       "self");
        (30,                         Struct,     "struct");
        (31,                         Super,      "super");
        (32,                         True,       "true");
        (33,                         Trait,      "trait");
        (34,                         Type,       "type");
        (35,                         Unsafe,     "unsafe");
        (36,                         Use,        "use");
        (37,                         Virtual,    "virtual");
        (38,                         While,      "while");
        (39,                         Continue,   "continue");
        (40,                         Proc,       "proc");
        (41,                         Box,        "box");

        'reserved:
        (42,                         Alignof,    "alignof");
        (43,                         Be,         "be");
        (44,                         Const,      "const");
        (45,                         Offsetof,   "offsetof");
        (46,                         Priv,       "priv");
        (47,                         Pure,       "pure");
        (48,                         Sizeof,     "sizeof");
        (49,                         Typeof,     "typeof");
        (50,                         Unsized,    "unsized");
        (51,                         Yield,      "yield");
        (52,                         Do,         "do");
    }
}

/**
 * Maps a token to a record specifying the corresponding binary
 * operator
 */
pub fn token_to_binop(tok: &Token) -> Option<ast::BinOp> {
  match *tok {
      BINOP(STAR)    => Some(ast::BiMul),
      BINOP(SLASH)   => Some(ast::BiDiv),
      BINOP(PERCENT) => Some(ast::BiRem),
      BINOP(PLUS)    => Some(ast::BiAdd),
      BINOP(MINUS)   => Some(ast::BiSub),
      BINOP(SHL)     => Some(ast::BiShl),
      BINOP(SHR)     => Some(ast::BiShr),
      BINOP(AND)     => Some(ast::BiBitAnd),
      BINOP(CARET)   => Some(ast::BiBitXor),
      BINOP(OR)      => Some(ast::BiBitOr),
      LT             => Some(ast::BiLt),
      LE             => Some(ast::BiLe),
      GE             => Some(ast::BiGe),
      GT             => Some(ast::BiGt),
      EQEQ           => Some(ast::BiEq),
      NE             => Some(ast::BiNe),
      ANDAND         => Some(ast::BiAnd),
      OROR           => Some(ast::BiOr),
      _              => None
  }
}

// looks like we can get rid of this completely...
pub type IdentInterner = StrInterner;

// if an interner exists in TLS, return it. Otherwise, prepare a
// fresh one.
// FIXME(eddyb) #8726 This should probably use a task-local reference.
pub fn get_ident_interner() -> Rc<IdentInterner> {
    local_data_key!(key: Rc<::parse::token::IdentInterner>)
    match key.get() {
        Some(interner) => interner.clone(),
        None => {
            let interner = Rc::new(mk_fresh_ident_interner());
            key.replace(Some(interner.clone()));
            interner
        }
    }
}

/// Represents a string stored in the task-local interner. Because the
/// interner lives for the life of the task, this can be safely treated as an
/// immortal string, as long as it never crosses between tasks.
///
/// FIXME(pcwalton): You must be careful about what you do in the destructors
/// of objects stored in TLS, because they may run after the interner is
/// destroyed. In particular, they must not access string contents. This can
/// be fixed in the future by just leaking all strings until task death
/// somehow.
#[deriving(Clone, Eq, Hash, Ord, TotalEq, TotalOrd)]
pub struct InternedString {
    string: RcStr,
}

impl InternedString {
    #[inline]
    pub fn new(string: &'static str) -> InternedString {
        InternedString {
            string: RcStr::new(string),
        }
    }

    #[inline]
    fn new_from_rc_str(string: RcStr) -> InternedString {
        InternedString {
            string: string,
        }
    }

    #[inline]
    pub fn get<'a>(&'a self) -> &'a str {
        self.string.as_slice()
    }
}

impl BytesContainer for InternedString {
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] {
        // FIXME(pcwalton): This is a workaround for the incorrect signature
        // of `BytesContainer`, which is itself a workaround for the lack of
        // DST.
        unsafe {
            let this = self.get();
            mem::transmute(this.container_as_bytes())
        }
    }
}

impl fmt::Show for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.string.as_slice())
    }
}

impl<'a> Equiv<&'a str> for InternedString {
    fn equiv(&self, other: & &'a str) -> bool {
        (*other) == self.string.as_slice()
    }
}

impl<D:Decoder<E>, E> Decodable<D, E> for InternedString {
    fn decode(d: &mut D) -> Result<InternedString, E> {
        Ok(get_name(get_ident_interner().intern(
                    try!(d.read_str()).as_slice())))
    }
}

impl<S:Encoder<E>, E> Encodable<S, E> for InternedString {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        s.emit_str(self.string.as_slice())
    }
}

/// Returns the string contents of a name, using the task-local interner.
#[inline]
pub fn get_name(name: Name) -> InternedString {
    let interner = get_ident_interner();
    InternedString::new_from_rc_str(interner.get(name))
}

/// Returns the string contents of an identifier, using the task-local
/// interner.
#[inline]
pub fn get_ident(ident: Ident) -> InternedString {
    get_name(ident.name)
}

/// Interns and returns the string contents of an identifier, using the
/// task-local interner.
#[inline]
pub fn intern_and_get_ident(s: &str) -> InternedString {
    get_name(intern(s))
}

/// Maps a string to its interned representation.
#[inline]
pub fn intern(s: &str) -> Name {
    get_ident_interner().intern(s)
}

/// gensym's a new uint, using the current interner.
#[inline]
pub fn gensym(s: &str) -> Name {
    get_ident_interner().gensym(s)
}

/// Maps a string to an identifier with an empty syntax context.
#[inline]
pub fn str_to_ident(s: &str) -> ast::Ident {
    ast::Ident::new(intern(s))
}

/// Maps a string to a gensym'ed identifier.
#[inline]
pub fn gensym_ident(s: &str) -> ast::Ident {
    ast::Ident::new(gensym(s))
}

// create a fresh name that maps to the same string as the old one.
// note that this guarantees that str_ptr_eq(ident_to_str(src),interner_get(fresh_name(src)));
// that is, that the new name and the old one are connected to ptr_eq strings.
pub fn fresh_name(src: &ast::Ident) -> Name {
    let interner = get_ident_interner();
    interner.gensym_copy(src.name)
    // following: debug version. Could work in final except that it's incompatible with
    // good error messages and uses of struct names in ambiguous could-be-binding
    // locations. Also definitely destroys the guarantee given above about ptr_eq.
    /*let num = rand::task_rng().gen_uint_range(0,0xffff);
    gensym(format!("{}_{}",ident_to_str(src),num))*/
}

// create a fresh mark.
pub fn fresh_mark() -> Mrk {
    gensym("mark")
}

// See the macro above about the types of keywords

pub fn is_keyword(kw: keywords::Keyword, tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => { kw.to_ident().name == sid.name }
        _ => { false }
    }
}

pub fn is_any_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.name {
            SELF_KEYWORD_NAME | STATIC_KEYWORD_NAME |
            STRICT_KEYWORD_START .. RESERVED_KEYWORD_FINAL => true,
            _ => false,
        },
        _ => false
    }
}

pub fn is_strict_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.name {
            SELF_KEYWORD_NAME | STATIC_KEYWORD_NAME |
            STRICT_KEYWORD_START .. STRICT_KEYWORD_FINAL => true,
            _ => false,
        },
        _ => false,
    }
}

pub fn is_reserved_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.name {
            RESERVED_KEYWORD_START .. RESERVED_KEYWORD_FINAL => true,
            _ => false,
        },
        _ => false,
    }
}

pub fn mtwt_token_eq(t1 : &Token, t2 : &Token) -> bool {
    match (t1,t2) {
        (&IDENT(id1,_),&IDENT(id2,_)) | (&LIFETIME(id1),&LIFETIME(id2)) =>
            mtwt::resolve(id1) == mtwt::resolve(id2),
        _ => *t1 == *t2
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ast;
    use ext::mtwt;

    fn mark_ident(id : ast::Ident, m : ast::Mrk) -> ast::Ident {
        ast::Ident{name:id.name,ctxt:mtwt::new_mark(m,id.ctxt)}
    }

    #[test] fn mtwt_token_eq_test() {
        assert!(mtwt_token_eq(&GT,&GT));
        let a = str_to_ident("bac");
        let a1 = mark_ident(a,92);
        assert!(mtwt_token_eq(&IDENT(a,true),&IDENT(a1,false)));
    }
}
