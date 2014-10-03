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
use ast::{Ident, Name, Mrk};
use ext::mtwt;
use parse::token;
use ptr::P;
use util::interner::{RcStr, StrInterner};
use util::interner;

use serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fmt;
use std::mem;
use std::path::BytesContainer;
use std::rc::Rc;

#[allow(non_camel_case_types)]
#[deriving(Clone, Encodable, Decodable, PartialEq, Eq, Hash, Show)]
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
#[deriving(Clone, Encodable, Decodable, PartialEq, Eq, Hash, Show)]
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
    QUESTION,

    /* Literals */
    LIT_BYTE(Name),
    LIT_CHAR(Name),
    LIT_INTEGER(Name),
    LIT_FLOAT(Name),
    LIT_STR(Name),
    LIT_STR_RAW(Name, uint), /* raw str delimited by n hash symbols */
    LIT_BINARY(Name),
    LIT_BINARY_RAW(Name, uint), /* raw binary str delimited by n hash symbols */

    /* Name components */
    /// An identifier contains an "is_mod_name" boolean,
    /// indicating whether :: follows this token with no
    /// whitespace in between.
    IDENT(Ident, bool),
    UNDERSCORE,
    LIFETIME(Ident),

    /* For interpolation */
    INTERPOLATED(Nonterminal),
    DOC_COMMENT(Name),

    // Junk. These carry no data because we don't really care about the data
    // they *would* carry, and don't really want to allocate a new ident for
    // them. Instead, users could extract that from the associated span.

    /// Whitespace
    WS,
    /// Comment
    COMMENT,
    SHEBANG(Name),

    EOF,
}

#[deriving(Clone, Encodable, Decodable, PartialEq, Eq, Hash)]
/// For interpolation during macro expansion.
pub enum Nonterminal {
    NtItem( P<ast::Item>),
    NtBlock(P<ast::Block>),
    NtStmt( P<ast::Stmt>),
    NtPat(  P<ast::Pat>),
    NtExpr( P<ast::Expr>),
    NtTy(   P<ast::Ty>),
    /// See IDENT, above, for meaning of bool in NtIdent:
    NtIdent(Box<Ident>, bool),
    /// Stuff inside brackets for attributes
    NtMeta( P<ast::MetaItem>),
    NtPath(Box<ast::Path>),
    NtTT(   P<ast::TokenTree>), // needs P'ed to break a circularity
    NtMatchers(Vec<ast::Matcher>)
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

pub fn binop_to_string(o: BinOp) -> &'static str {
    match o {
      PLUS => "+",
      MINUS => "-",
      STAR => "*",
      SLASH => "/",
      PERCENT => "%",
      CARET => "^",
      AND => "&",
      OR => "|",
      SHL => "<<",
      SHR => ">>"
    }
}

pub fn to_string(t: &Token) -> String {
    match *t {
      EQ => "=".into_string(),
      LT => "<".into_string(),
      LE => "<=".into_string(),
      EQEQ => "==".into_string(),
      NE => "!=".into_string(),
      GE => ">=".into_string(),
      GT => ">".into_string(),
      NOT => "!".into_string(),
      TILDE => "~".into_string(),
      OROR => "||".into_string(),
      ANDAND => "&&".into_string(),
      BINOP(op) => binop_to_string(op).into_string(),
      BINOPEQ(op) => {
          let mut s = binop_to_string(op).into_string();
          s.push_str("=");
          s
      }

      /* Structural symbols */
      AT => "@".into_string(),
      DOT => ".".into_string(),
      DOTDOT => "..".into_string(),
      DOTDOTDOT => "...".into_string(),
      COMMA => ",".into_string(),
      SEMI => ";".into_string(),
      COLON => ":".into_string(),
      MOD_SEP => "::".into_string(),
      RARROW => "->".into_string(),
      LARROW => "<-".into_string(),
      FAT_ARROW => "=>".into_string(),
      LPAREN => "(".into_string(),
      RPAREN => ")".into_string(),
      LBRACKET => "[".into_string(),
      RBRACKET => "]".into_string(),
      LBRACE => "{".into_string(),
      RBRACE => "}".into_string(),
      POUND => "#".into_string(),
      DOLLAR => "$".into_string(),
      QUESTION => "?".into_string(),

      /* Literals */
      LIT_BYTE(b) => {
          format!("b'{}'", b.as_str())
      }
      LIT_CHAR(c) => {
          format!("'{}'", c.as_str())
      }
      LIT_INTEGER(c) | LIT_FLOAT(c) => {
          c.as_str().into_string()
      }

      LIT_STR(s) => {
          format!("\"{}\"", s.as_str())
      }
      LIT_STR_RAW(s, n) => {
        format!("r{delim}\"{string}\"{delim}",
                 delim="#".repeat(n), string=s.as_str())
      }
      LIT_BINARY(v) => {
          format!("b\"{}\"", v.as_str())
      }
      LIT_BINARY_RAW(s, n) => {
        format!("br{delim}\"{string}\"{delim}",
                 delim="#".repeat(n), string=s.as_str())
      }

      /* Name components */
      IDENT(s, _) => get_ident(s).get().into_string(),
      LIFETIME(s) => {
          format!("{}", get_ident(s))
      }
      UNDERSCORE => "_".into_string(),

      /* Other */
      DOC_COMMENT(s) => s.as_str().into_string(),
      EOF => "<eof>".into_string(),
      WS => " ".into_string(),
      COMMENT => "/* */".into_string(),
      SHEBANG(s) => format!("/* shebang: {}*/", s.as_str()),

      INTERPOLATED(ref nt) => {
        match nt {
            &NtExpr(ref e) => ::print::pprust::expr_to_string(&**e),
            &NtMeta(ref e) => ::print::pprust::meta_item_to_string(&**e),
            &NtTy(ref e) => ::print::pprust::ty_to_string(&**e),
            &NtPath(ref e) => ::print::pprust::path_to_string(&**e),
            _ => {
                let mut s = "an interpolated ".into_string();
                match *nt {
                    NtItem(..) => s.push_str("item"),
                    NtBlock(..) => s.push_str("block"),
                    NtStmt(..) => s.push_str("statement"),
                    NtPat(..) => s.push_str("pattern"),
                    NtMeta(..) => fail!("should have been handled"),
                    NtExpr(..) => fail!("should have been handled"),
                    NtTy(..) => fail!("should have been handled"),
                    NtIdent(..) => s.push_str("identifier"),
                    NtPath(..) => fail!("should have been handled"),
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
      LIT_BYTE(_) => true,
      LIT_CHAR(_) => true,
      LIT_INTEGER(_) => true,
      LIT_FLOAT(_) => true,
      LIT_STR(_) => true,
      LIT_STR_RAW(_, _) => true,
      LIT_BINARY(_) => true,
      LIT_BINARY_RAW(_, _) => true,
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
      LIT_BYTE(_) => true,
      LIT_CHAR(_) => true,
      LIT_INTEGER(_) => true,
      LIT_FLOAT(_) => true,
      LIT_STR(_) => true,
      LIT_STR_RAW(_, _) => true,
      LIT_BINARY(_) => true,
      LIT_BINARY_RAW(_, _) => true,
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
    static STRICT_KEYWORD_START: Name = first!($( Name($sk_name), )*);
    static STRICT_KEYWORD_FINAL: Name = last!($( Name($sk_name), )*);
    static RESERVED_KEYWORD_START: Name = first!($( Name($rk_name), )*);
    static RESERVED_KEYWORD_FINAL: Name = last!($( Name($rk_name), )*);

    pub mod special_idents {
        use ast::{Ident, Name};
        $(
            #[allow(non_uppercase_statics)]
            pub static $si_static: Ident = Ident { name: Name($si_name), ctxt: 0 };
         )*
    }

    pub mod special_names {
        use ast::Name;
        $( #[allow(non_uppercase_statics)] pub static $si_static: Name =  Name($si_name); )*
    }

    /**
     * All the valid words that have meaning in the Rust language.
     *
     * Rust keywords are either 'strict' or 'reserved'.  Strict keywords may not
     * appear as identifiers at all. Reserved keywords are not used anywhere in
     * the language and may not appear as identifiers.
     */
    pub mod keywords {
        use ast::Name;

        pub enum Keyword {
            $( $sk_variant, )*
            $( $rk_variant, )*
        }

        impl Keyword {
            pub fn to_name(&self) -> Name {
                match *self {
                    $( $sk_variant => Name($sk_name), )*
                    $( $rk_variant => Name($rk_name), )*
                }
            }
        }
    }

    fn mk_fresh_ident_interner() -> IdentInterner {
        // The indices here must correspond to the numbers in
        // special_idents, in Keyword to_name(), and in static
        // constants below.
        let mut init_vec = Vec::new();
        $(init_vec.push($si_str);)*
        $(init_vec.push($sk_str);)*
        $(init_vec.push($rk_str);)*
        interner::StrInterner::prefill(init_vec.as_slice())
    }
}}

// If the special idents get renumbered, remember to modify these two as appropriate
pub static SELF_KEYWORD_NAME: Name = Name(SELF_KEYWORD_NAME_NUM);
static STATIC_KEYWORD_NAME: Name = Name(STATIC_KEYWORD_NAME_NUM);
static SUPER_KEYWORD_NAME: Name = Name(SUPER_KEYWORD_NAME_NUM);

pub static SELF_KEYWORD_NAME_NUM: u32 = 1;
static STATIC_KEYWORD_NAME_NUM: u32 = 2;
static SUPER_KEYWORD_NAME_NUM: u32 = 3;

// NB: leaving holes in the ident table is bad! a different ident will get
// interned with the id from the hole, but it will be between the min and max
// of the reserved words, and thus tagged as "reserved".

declare_special_idents_and_keywords! {
    pub mod special_idents {
        // These ones are statics
        (0,                          invalid,                "");
        (super::SELF_KEYWORD_NAME_NUM,   self_,              "self");
        (super::STATIC_KEYWORD_NAME_NUM, statik,             "static");
        (super::SUPER_KEYWORD_NAME_NUM, super_,              "super");
        (4,                          static_lifetime,        "'static");

        // for matcher NTs
        (5,                          tt,                     "tt");
        (6,                          matchers,               "matchers");

        // outside of libsyntax
        (7,                          clownshoe_abi,          "__rust_abi");
        (8,                          opaque,                 "<opaque>");
        (9,                          unnamed_field,          "<unnamed_field>");
        (10,                         type_self,              "Self");
        (11,                         prelude_import,         "prelude_import");
    }

    pub mod keywords {
        // These ones are variants of the Keyword enum

        'strict:
        (12,                         As,         "as");
        (13,                         Break,      "break");
        (14,                         Crate,      "crate");
        (15,                         Else,       "else");
        (16,                         Enum,       "enum");
        (17,                         Extern,     "extern");
        (18,                         False,      "false");
        (19,                         Fn,         "fn");
        (20,                         For,        "for");
        (21,                         If,         "if");
        (22,                         Impl,       "impl");
        (23,                         In,         "in");
        (24,                         Let,        "let");
        (25,                         Loop,       "loop");
        (26,                         Match,      "match");
        (27,                         Mod,        "mod");
        (28,                         Move,       "move");
        (29,                         Mut,        "mut");
        (30,                         Once,       "once");
        (31,                         Pub,        "pub");
        (32,                         Ref,        "ref");
        (33,                         Return,     "return");
        // Static and Self are also special idents (prefill de-dupes)
        (super::STATIC_KEYWORD_NAME_NUM, Static, "static");
        (super::SELF_KEYWORD_NAME_NUM,   Self,   "self");
        (34,                         Struct,     "struct");
        (super::SUPER_KEYWORD_NAME_NUM, Super,   "super");
        (35,                         True,       "true");
        (36,                         Trait,      "trait");
        (37,                         Type,       "type");
        (38,                         Unsafe,     "unsafe");
        (39,                         Use,        "use");
        (40,                         Virtual,    "virtual");
        (41,                         While,      "while");
        (42,                         Continue,   "continue");
        (43,                         Proc,       "proc");
        (44,                         Box,        "box");
        (45,                         Const,      "const");
        (46,                         Where,      "where");

        'reserved:
        (47,                         Alignof,    "alignof");
        (48,                         Be,         "be");
        (49,                         Offsetof,   "offsetof");
        (50,                         Priv,       "priv");
        (51,                         Pure,       "pure");
        (52,                         Sizeof,     "sizeof");
        (53,                         Typeof,     "typeof");
        (54,                         Unsized,    "unsized");
        (55,                         Yield,      "yield");
        (56,                         Do,         "do");
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
#[deriving(Clone, PartialEq, Hash, PartialOrd, Eq, Ord)]
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
pub fn str_to_ident(s: &str) -> Ident {
    Ident::new(intern(s))
}

/// Maps a string to a gensym'ed identifier.
#[inline]
pub fn gensym_ident(s: &str) -> Ident {
    Ident::new(gensym(s))
}

// create a fresh name that maps to the same string as the old one.
// note that this guarantees that str_ptr_eq(ident_to_string(src),interner_get(fresh_name(src)));
// that is, that the new name and the old one are connected to ptr_eq strings.
pub fn fresh_name(src: &Ident) -> Name {
    let interner = get_ident_interner();
    interner.gensym_copy(src.name)
    // following: debug version. Could work in final except that it's incompatible with
    // good error messages and uses of struct names in ambiguous could-be-binding
    // locations. Also definitely destroys the guarantee given above about ptr_eq.
    /*let num = rand::task_rng().gen_uint_range(0,0xffff);
    gensym(format!("{}_{}",ident_to_string(src),num))*/
}

// create a fresh mark.
pub fn fresh_mark() -> Mrk {
    gensym("mark").uint() as u32
}

// See the macro above about the types of keywords

pub fn is_keyword(kw: keywords::Keyword, tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => { kw.to_name() == sid.name }
        _ => { false }
    }
}

pub fn is_any_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => {
            let n = sid.name;

               n == SELF_KEYWORD_NAME
            || n == STATIC_KEYWORD_NAME
            || n == SUPER_KEYWORD_NAME
            || STRICT_KEYWORD_START <= n
            && n <= RESERVED_KEYWORD_FINAL
        },
        _ => false
    }
}

pub fn is_strict_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => {
            let n = sid.name;

               n == SELF_KEYWORD_NAME
            || n == STATIC_KEYWORD_NAME
            || n == SUPER_KEYWORD_NAME
            || STRICT_KEYWORD_START <= n
            && n <= STRICT_KEYWORD_FINAL
        },
        token::IDENT(sid, true) => {
            let n = sid.name;

               n != SELF_KEYWORD_NAME
            && n != SUPER_KEYWORD_NAME
            && STRICT_KEYWORD_START <= n
            && n <= STRICT_KEYWORD_FINAL
        }
        _ => false,
    }
}

pub fn is_reserved_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => {
            let n = sid.name;

               RESERVED_KEYWORD_START <= n
            && n <= RESERVED_KEYWORD_FINAL
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
        ast::Ident { name: id.name, ctxt:mtwt::apply_mark(m, id.ctxt) }
    }

    #[test] fn mtwt_token_eq_test() {
        assert!(mtwt_token_eq(&GT,&GT));
        let a = str_to_ident("bac");
        let a1 = mark_ident(a,92);
        assert!(mtwt_token_eq(&IDENT(a,true),&IDENT(a1,false)));
    }
}
