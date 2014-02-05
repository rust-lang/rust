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
use ast::{P, Name, Mrk};
use ast_util;
use parse::token;
use util::interner::{RcStr, StrInterner};
use util::interner;

use serialize::{Decodable, Decoder, Encodable, Encoder};
use std::cast;
use std::char;
use std::fmt;
use std::local_data;
use std::path::BytesContainer;

#[allow(non_camel_case_types)]
#[deriving(Clone, Encodable, Decodable, Eq, IterBytes)]
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
#[deriving(Clone, Encodable, Decodable, Eq, IterBytes)]
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
    DARROW,
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
    LIT_CHAR(u32),
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

#[deriving(Clone, Encodable, Decodable, Eq, IterBytes)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem(@ast::Item),
    NtBlock(P<ast::Block>),
    NtStmt(@ast::Stmt),
    NtPat( @ast::Pat),
    NtExpr(@ast::Expr),
    NtTy(  P<ast::Ty>),
    NtIdent(~ast::Ident, bool),
    NtAttr(@ast::Attribute), // #[foo]
    NtPath(~ast::Path),
    NtTT(  @ast::TokenTree), // needs @ed to break a circularity
    NtMatchers(~[ast::Matcher])
}

pub fn binop_to_str(o: BinOp) -> ~str {
    match o {
      PLUS => ~"+",
      MINUS => ~"-",
      STAR => ~"*",
      SLASH => ~"/",
      PERCENT => ~"%",
      CARET => ~"^",
      AND => ~"&",
      OR => ~"|",
      SHL => ~"<<",
      SHR => ~">>"
    }
}

pub fn to_str(input: @IdentInterner, t: &Token) -> ~str {
    match *t {
      EQ => ~"=",
      LT => ~"<",
      LE => ~"<=",
      EQEQ => ~"==",
      NE => ~"!=",
      GE => ~">=",
      GT => ~">",
      NOT => ~"!",
      TILDE => ~"~",
      OROR => ~"||",
      ANDAND => ~"&&",
      BINOP(op) => binop_to_str(op),
      BINOPEQ(op) => binop_to_str(op) + "=",

      /* Structural symbols */
      AT => ~"@",
      DOT => ~".",
      DOTDOT => ~"..",
      DOTDOTDOT => ~"...",
      COMMA => ~",",
      SEMI => ~";",
      COLON => ~":",
      MOD_SEP => ~"::",
      RARROW => ~"->",
      LARROW => ~"<-",
      DARROW => ~"<->",
      FAT_ARROW => ~"=>",
      LPAREN => ~"(",
      RPAREN => ~")",
      LBRACKET => ~"[",
      RBRACKET => ~"]",
      LBRACE => ~"{",
      RBRACE => ~"}",
      POUND => ~"#",
      DOLLAR => ~"$",

      /* Literals */
      LIT_CHAR(c) => {
          let mut res = ~"'";
          char::from_u32(c).unwrap().escape_default(|c| {
              res.push_char(c);
          });
          res.push_char('\'');
          res
      }
      LIT_INT(i, t) => {
          i.to_str() + ast_util::int_ty_to_str(t)
      }
      LIT_UINT(u, t) => {
          u.to_str() + ast_util::uint_ty_to_str(t)
      }
      LIT_INT_UNSUFFIXED(i) => { i.to_str() }
      LIT_FLOAT(ref s, t) => {
        let body_string = get_ident(s.name);
        let mut body = body_string.get().to_str();
        if body.ends_with(".") {
            body.push_char('0');  // `10.f` is not a float literal
        }
        body + ast_util::float_ty_to_str(t)
      }
      LIT_FLOAT_UNSUFFIXED(ref s) => {
        let body_string = get_ident(s.name);
        let mut body = body_string.get().to_owned();
        if body.ends_with(".") {
            body.push_char('0');  // `10.f` is not a float literal
        }
        body
      }
      LIT_STR(ref s) => {
          let literal_string = get_ident(s.name);
          format!("\"{}\"", literal_string.get().escape_default())
      }
      LIT_STR_RAW(ref s, n) => {
          let literal_string = get_ident(s.name);
          format!("r{delim}\"{string}\"{delim}",
                  delim="#".repeat(n), string=literal_string.get())
      }

      /* Name components */
      IDENT(s, _) => input.get(s.name).into_owned(),
      LIFETIME(s) => {
          let name = input.get(s.name);
          format!("'{}", name.as_slice())
      }
      UNDERSCORE => ~"_",

      /* Other */
      DOC_COMMENT(ref s) => {
          let comment_string = get_ident(s.name);
          comment_string.get().to_str()
      }
      EOF => ~"<eof>",
      INTERPOLATED(ref nt) => {
        match nt {
            &NtExpr(e) => ::print::pprust::expr_to_str(e, input),
            &NtAttr(e) => ::print::pprust::attribute_to_str(e, input),
            _ => {
                ~"an interpolated " +
                    match *nt {
                        NtItem(..) => ~"item",
                        NtBlock(..) => ~"block",
                        NtStmt(..) => ~"statement",
                        NtPat(..) => ~"pattern",
                        NtAttr(..) => fail!("should have been handled"),
                        NtExpr(..) => fail!("should have been handled above"),
                        NtTy(..) => ~"type",
                        NtIdent(..) => ~"identifier",
                        NtPath(..) => ~"path",
                        NtTT(..) => ~"tt",
                        NtMatchers(..) => ~"matcher sequence"
                    }
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

/// what's the opposite delimiter?
pub fn flip_delimiter(t: &token::Token) -> token::Token {
    match *t {
      LPAREN => RPAREN,
      LBRACE => RBRACE,
      LBRACKET => RBRACKET,
      RPAREN => LPAREN,
      RBRACE => LBRACE,
      RBRACKET => LBRACKET,
      _ => fail!()
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

    fn mk_fresh_ident_interner() -> @IdentInterner {
        // The indices here must correspond to the numbers in
        // special_idents, in Keyword to_ident(), and in static
        // constants below.
        let init_vec = ~[
            $( $si_str, )*
            $( $sk_str, )*
            $( $rk_str, )*
        ];

        @interner::StrInterner::prefill(init_vec)
    }
}}

// If the special idents get renumbered, remember to modify these two as appropriate
static SELF_KEYWORD_NAME: Name = 3;
static STATIC_KEYWORD_NAME: Name = 10;

declare_special_idents_and_keywords! {
    pub mod special_idents {
        // These ones are statics

        (0,                          anon,                   "anon");
        (1,                          invalid,                "");       // ''
        (2,                          clownshoes_extensions,  "__extensions__");

        (super::SELF_KEYWORD_NAME,   self_,                  "self"); // 'self'

        // for matcher NTs
        (4,                          tt,                     "tt");
        (5,                          matchers,               "matchers");

        // outside of libsyntax
        (6,                          arg,                    "arg");
        (7,                          clownshoe_abi,          "__rust_abi");
        (8,                          main,                   "main");
        (9,                          opaque,                 "<opaque>");
        (super::STATIC_KEYWORD_NAME, statik,                 "static");
        (11,                         clownshoes_foreign_mod, "__foreign_mod__");
        (12,                         unnamed_field,          "<unnamed_field>");
        (13,                         type_self,              "Self"); // `Self`
    }

    pub mod keywords {
        // These ones are variants of the Keyword enum

        'strict:
        (14,                         As,         "as");
        (15,                         Break,      "break");
        (16,                         Const,      "const");
        (17,                         Else,       "else");
        (18,                         Enum,       "enum");
        (19,                         Extern,     "extern");
        (20,                         False,      "false");
        (21,                         Fn,         "fn");
        (22,                         For,        "for");
        (23,                         If,         "if");
        (24,                         Impl,       "impl");
        (25,                         In,         "in");
        (26,                         Let,        "let");
        (27,                         __LogLevel, "__log_level");
        (28,                         Loop,       "loop");
        (29,                         Match,      "match");
        (30,                         Mod,        "mod");
        (31,                         Mut,        "mut");
        (32,                         Once,       "once");
        (33,                         Priv,       "priv");
        (34,                         Pub,        "pub");
        (35,                         Ref,        "ref");
        (36,                         Return,     "return");
        // Static and Self are also special idents (prefill de-dupes)
        (super::STATIC_KEYWORD_NAME, Static,     "static");
        (super::SELF_KEYWORD_NAME,   Self,       "self");
        (37,                         Struct,     "struct");
        (38,                         Super,      "super");
        (39,                         True,       "true");
        (40,                         Trait,      "trait");
        (41,                         Type,       "type");
        (42,                         Unsafe,     "unsafe");
        (43,                         Use,        "use");
        (44,                         While,      "while");
        (45,                         Continue,   "continue");
        (46,                         Proc,       "proc");
        (47,                         Box,        "box");

        'reserved:
        (48,                         Alignof,    "alignof");
        (49,                         Be,         "be");
        (50,                         Offsetof,   "offsetof");
        (51,                         Pure,       "pure");
        (52,                         Sizeof,     "sizeof");
        (53,                         Typeof,     "typeof");
        (54,                         Unsized,    "unsized");
        (55,                         Yield,      "yield");
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
pub fn get_ident_interner() -> @IdentInterner {
    local_data_key!(key: @@::parse::token::IdentInterner)
    match local_data::get(key, |k| k.map(|k| *k)) {
        Some(interner) => *interner,
        None => {
            let interner = mk_fresh_ident_interner();
            local_data::set(key, @interner);
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
#[deriving(Clone, Eq, IterBytes, Ord, TotalEq, TotalOrd)]
pub struct InternedString {
    priv string: RcStr,
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
            cast::transmute(this.container_as_bytes())
        }
    }
}

impl fmt::Show for InternedString {
    fn fmt(obj: &InternedString, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f.buf, "{}", obj.string.as_slice())
    }
}

impl<'a> Equiv<&'a str> for InternedString {
    fn equiv(&self, other: & &'a str) -> bool {
        (*other) == self.string.as_slice()
    }
}

impl<D:Decoder> Decodable<D> for InternedString {
    fn decode(d: &mut D) -> InternedString {
        let interner = get_ident_interner();
        get_ident(interner.intern(d.read_str()))
    }
}

impl<E:Encoder> Encodable<E> for InternedString {
    fn encode(&self, e: &mut E) {
        e.emit_str(self.string.as_slice())
    }
}

/// Returns the string contents of an identifier, using the task-local
/// interner.
#[inline]
pub fn get_ident(idx: Name) -> InternedString {
    let interner = get_ident_interner();
    InternedString::new_from_rc_str(interner.get(idx))
}

/// Interns and returns the string contents of an identifier, using the
/// task-local interner.
#[inline]
pub fn intern_and_get_ident(s: &str) -> InternedString {
    get_ident(intern(s))
}

/* for when we don't care about the contents; doesn't interact with TLD or
   serialization */
pub fn mk_fake_ident_interner() -> @IdentInterner {
    @interner::StrInterner::new()
}

// maps a string to its interned representation
#[inline]
pub fn intern(str : &str) -> Name {
    let interner = get_ident_interner();
    interner.intern(str)
}

// gensyms a new uint, using the current interner
pub fn gensym(str : &str) -> Name {
    let interner = get_ident_interner();
    interner.gensym(str)
}

// maps a string to an identifier with an empty syntax context
pub fn str_to_ident(str : &str) -> ast::Ident {
    ast::Ident::new(intern(str))
}

// maps a string to a gensym'ed identifier
pub fn gensym_ident(str : &str) -> ast::Ident {
    ast::Ident::new(gensym(str))
}

// create a fresh name that maps to the same string as the old one.
// note that this guarantees that str_ptr_eq(ident_to_str(src),interner_get(fresh_name(src)));
// that is, that the new name and the old one are connected to ptr_eq strings.
pub fn fresh_name(src : &ast::Ident) -> Name {
    let interner = get_ident_interner();
    interner.gensym_copy(src.name)
    // following: debug version. Could work in final except that it's incompatible with
    // good error messages and uses of struct names in ambiguous could-be-binding
    // locations. Also definitely destroys the guarantee given above about ptr_eq.
    /*let num = rand::rng().gen_uint_range(0,0xffff);
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
        (&IDENT(id1,_),&IDENT(id2,_)) =>
        ast_util::mtwt_resolve(id1) == ast_util::mtwt_resolve(id2),
        _ => *t1 == *t2
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use ast;
    use ast_util;

    fn mark_ident(id : ast::Ident, m : ast::Mrk) -> ast::Ident {
        ast::Ident{name:id.name,ctxt:ast_util::new_mark(m,id.ctxt)}
    }

    #[test] fn mtwt_token_eq_test() {
        assert!(mtwt_token_eq(&GT,&GT));
        let a = str_to_ident("bac");
        let a1 = mark_ident(a,92);
        assert!(mtwt_token_eq(&IDENT(a,true),&IDENT(a1,false)));
    }
}
