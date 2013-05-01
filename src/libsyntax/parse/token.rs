// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast_util;
use parse::token;
use util::interner::Interner;
use util::interner;

use core::cmp::Equiv;
use core::hashmap::HashSet;
use core::to_bytes;

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
pub enum binop {
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

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
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
    BINOP(binop),
    BINOPEQ(binop),

    /* Structural symbols */
    AT,
    DOT,
    DOTDOT,
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
    LIT_INT(i64, ast::int_ty),
    LIT_UINT(u64, ast::uint_ty),
    LIT_INT_UNSUFFIXED(i64),
    LIT_FLOAT(ast::ident, ast::float_ty),
    LIT_FLOAT_UNSUFFIXED(ast::ident),
    LIT_STR(ast::ident),

    /* Name components */
    // an identifier contains an "is_mod_name" boolean,
    // indicating whether :: follows this token with no
    // whitespace in between.
    IDENT(ast::ident, bool),
    UNDERSCORE,
    LIFETIME(ast::ident),

    /* For interpolation */
    INTERPOLATED(nonterminal),

    DOC_COMMENT(ast::ident),
    EOF,
}

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
/// For interpolation during macro expansion.
pub enum nonterminal {
    nt_item(@ast::item),
    nt_block(ast::blk),
    nt_stmt(@ast::stmt),
    nt_pat( @ast::pat),
    nt_expr(@ast::expr),
    nt_ty(  @ast::Ty),
    nt_ident(ast::ident, bool),
    nt_path(@ast::Path),
    nt_tt(  @ast::token_tree), //needs @ed to break a circularity
    nt_matchers(~[ast::matcher])
}

pub fn binop_to_str(o: binop) -> ~str {
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

pub fn to_str(in: @ident_interner, t: &Token) -> ~str {
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
      BINOPEQ(op) => binop_to_str(op) + ~"=",

      /* Structural symbols */
      AT => ~"@",
      DOT => ~".",
      DOTDOT => ~"..",
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
      LIT_INT(c, ast::ty_char) => {
        ~"'" + char::escape_default(c as char) + ~"'"
      }
      LIT_INT(i, t) => {
          i.to_str() + ast_util::int_ty_to_str(t)
      }
      LIT_UINT(u, t) => {
          u.to_str() + ast_util::uint_ty_to_str(t)
      }
      LIT_INT_UNSUFFIXED(i) => { i.to_str() }
      LIT_FLOAT(s, t) => {
        let mut body = copy *in.get(s);
        if body.ends_with(~".") {
            body = body + ~"0";  // `10.f` is not a float literal
        }
        body + ast_util::float_ty_to_str(t)
      }
      LIT_FLOAT_UNSUFFIXED(s) => {
        let mut body = copy *in.get(s);
        if body.ends_with(~".") {
            body = body + ~"0";  // `10.f` is not a float literal
        }
        body
      }
      LIT_STR(s) => { ~"\"" + str::escape_default(*in.get(s)) + ~"\"" }

      /* Name components */
      IDENT(s, _) => copy *in.get(s),
      LIFETIME(s) => fmt!("'%s", *in.get(s)),
      UNDERSCORE => ~"_",

      /* Other */
      DOC_COMMENT(s) => copy *in.get(s),
      EOF => ~"<eof>",
      INTERPOLATED(ref nt) => {
        match nt {
            &nt_expr(e) => ::print::pprust::expr_to_str(e, in),
            _ => {
                ~"an interpolated " +
                    match (*nt) {
                      nt_item(*) => ~"item",
                      nt_block(*) => ~"block",
                      nt_stmt(*) => ~"statement",
                      nt_pat(*) => ~"pattern",
                      nt_expr(*) => fail!(~"should have been handled above"),
                      nt_ty(*) => ~"type",
                      nt_ident(*) => ~"identifier",
                      nt_path(*) => ~"path",
                      nt_tt(*) => ~"tt",
                      nt_matchers(*) => ~"matcher sequence"
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
      LIT_INT(_, _) => true,
      LIT_UINT(_, _) => true,
      LIT_INT_UNSUFFIXED(_) => true,
      LIT_FLOAT(_, _) => true,
      LIT_FLOAT_UNSUFFIXED(_) => true,
      LIT_STR(_) => true,
      POUND => true,
      AT => true,
      NOT => true,
      BINOP(MINUS) => true,
      BINOP(STAR) => true,
      BINOP(AND) => true,
      BINOP(OR) => true, // in lambda syntax
      OROR => true, // in lambda syntax
      MOD_SEP => true,
      INTERPOLATED(nt_expr(*))
      | INTERPOLATED(nt_ident(*))
      | INTERPOLATED(nt_block(*))
      | INTERPOLATED(nt_path(*)) => true,
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
      LIT_INT(_, _) => true,
      LIT_UINT(_, _) => true,
      LIT_INT_UNSUFFIXED(_) => true,
      LIT_FLOAT(_, _) => true,
      LIT_FLOAT_UNSUFFIXED(_) => true,
      LIT_STR(_) => true,
      _ => false
    }
}

pub fn is_ident(t: &Token) -> bool {
    match *t { IDENT(_, _) => true, _ => false }
}

pub fn is_ident_or_path(t: &Token) -> bool {
    match *t {
      IDENT(_, _) | INTERPOLATED(nt_path(*)) => true,
      _ => false
    }
}

pub fn is_plain_ident(t: &Token) -> bool {
    match *t { IDENT(_, false) => true, _ => false }
}

pub fn is_bar(t: &Token) -> bool {
    match *t { BINOP(OR) | OROR => true, _ => false }
}


pub mod special_idents {
    use ast::ident;

    pub static underscore : ident = ident { repr: 0u, ctxt: 0};
    pub static anon : ident = ident { repr: 1u, ctxt: 0};
    pub static dtor : ident = ident { repr: 2u, ctxt: 0}; // 'drop', but that's
                                                 // reserved
    pub static invalid : ident = ident { repr: 3u, ctxt: 0}; // ''
    pub static unary : ident = ident { repr: 4u, ctxt: 0};
    pub static not_fn : ident = ident { repr: 5u, ctxt: 0};
    pub static idx_fn : ident = ident { repr: 6u, ctxt: 0};
    pub static unary_minus_fn : ident = ident { repr: 7u, ctxt: 0};
    pub static clownshoes_extensions : ident = ident { repr: 8u, ctxt: 0};

    pub static self_ : ident = ident { repr: 9u, ctxt: 0}; // 'self'

    /* for matcher NTs */
    pub static item : ident = ident { repr: 10u, ctxt: 0};
    pub static block : ident = ident { repr: 11u, ctxt: 0};
    pub static stmt : ident = ident { repr: 12u, ctxt: 0};
    pub static pat : ident = ident { repr: 13u, ctxt: 0};
    pub static expr : ident = ident { repr: 14u, ctxt: 0};
    pub static ty : ident = ident { repr: 15u, ctxt: 0};
    pub static ident : ident = ident { repr: 16u, ctxt: 0};
    pub static path : ident = ident { repr: 17u, ctxt: 0};
    pub static tt : ident = ident { repr: 18u, ctxt: 0};
    pub static matchers : ident = ident { repr: 19u, ctxt: 0};

    pub static str : ident = ident { repr: 20u, ctxt: 0}; // for the type

    /* outside of libsyntax */
    pub static ty_visitor : ident = ident { repr: 21u, ctxt: 0};
    pub static arg : ident = ident { repr: 22u, ctxt: 0};
    pub static descrim : ident = ident { repr: 23u, ctxt: 0};
    pub static clownshoe_abi : ident = ident { repr: 24u, ctxt: 0};
    pub static clownshoe_stack_shim : ident = ident { repr: 25u, ctxt: 0};
    pub static tydesc : ident = ident { repr: 26u, ctxt: 0};
    pub static literally_dtor : ident = ident { repr: 27u, ctxt: 0};
    pub static main : ident = ident { repr: 28u, ctxt: 0};
    pub static opaque : ident = ident { repr: 29u, ctxt: 0};
    pub static blk : ident = ident { repr: 30u, ctxt: 0};
    pub static static : ident = ident { repr: 31u, ctxt: 0};
    pub static intrinsic : ident = ident { repr: 32u, ctxt: 0};
    pub static clownshoes_foreign_mod: ident = ident { repr: 33u, ctxt: 0};
    pub static unnamed_field: ident = ident { repr: 34u, ctxt: 0};
    pub static c_abi: ident = ident { repr: 35u, ctxt: 0};
    pub static type_self: ident = ident { repr: 36u, ctxt: 0};    // `Self`
}

pub struct StringRef<'self>(&'self str);

impl<'self> Equiv<@~str> for StringRef<'self> {
    #[inline(always)]
    fn equiv(&self, other: &@~str) -> bool { str::eq_slice(**self, **other) }
}

impl<'self> to_bytes::IterBytes for StringRef<'self> {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}

/**
 * Maps a token to a record specifying the corresponding binary
 * operator
 */
pub fn token_to_binop(tok: Token) -> Option<ast::binop> {
  match tok {
      BINOP(STAR)    => Some(ast::mul),
      BINOP(SLASH)   => Some(ast::div),
      BINOP(PERCENT) => Some(ast::rem),
      BINOP(PLUS)    => Some(ast::add),
      BINOP(MINUS)   => Some(ast::subtract),
      BINOP(SHL)     => Some(ast::shl),
      BINOP(SHR)     => Some(ast::shr),
      BINOP(AND)     => Some(ast::bitand),
      BINOP(CARET)   => Some(ast::bitxor),
      BINOP(OR)      => Some(ast::bitor),
      LT             => Some(ast::lt),
      LE             => Some(ast::le),
      GE             => Some(ast::ge),
      GT             => Some(ast::gt),
      EQEQ           => Some(ast::eq),
      NE             => Some(ast::ne),
      ANDAND         => Some(ast::and),
      OROR           => Some(ast::or),
      _              => None
  }
}

pub struct ident_interner {
    priv interner: Interner<@~str>,
}

pub impl ident_interner {
    fn intern(&self, val: @~str) -> ast::ident {
        ast::ident { repr: self.interner.intern(val), ctxt: 0 }
    }
    fn gensym(&self, val: @~str) -> ast::ident {
        ast::ident { repr: self.interner.gensym(val), ctxt: 0 }
    }
    fn get(&self, idx: ast::ident) -> @~str {
        self.interner.get(idx.repr)
    }
    fn len(&self) -> uint {
        self.interner.len()
    }
    fn find_equiv<Q:Hash + IterBytes + Equiv<@~str>>(&self, val: &Q)
                                                     -> Option<ast::ident> {
        match self.interner.find_equiv(val) {
            Some(v) => Some(ast::ident { repr: v, ctxt: 0 }),
            None => None,
        }
    }
}

// return a fresh interner, preloaded with special identifiers.
// EFFECT: stores this interner in TLS
pub fn mk_fresh_ident_interner() -> @ident_interner {
    // the indices here must correspond to the numbers in
    // special_idents.
    let init_vec = ~[
        @~"_",                  // 0
        @~"anon",               // 1
        @~"drop",               // 2
        @~"",                   // 3
        @~"unary",              // 4
        @~"!",                  // 5
        @~"[]",                 // 6
        @~"unary-",             // 7
        @~"__extensions__",     // 8
        @~"self",               // 9
        @~"item",               // 10
        @~"block",              // 11
        @~"stmt",               // 12
        @~"pat",                // 13
        @~"expr",               // 14
        @~"ty",                 // 15
        @~"ident",              // 16
        @~"path",               // 17
        @~"tt",                 // 18
        @~"matchers",           // 19
        @~"str",                // 20
        @~"TyVisitor",          // 21
        @~"arg",                // 22
        @~"descrim",            // 23
        @~"__rust_abi",         // 24
        @~"__rust_stack_shim",  // 25
        @~"TyDesc",             // 26
        @~"dtor",               // 27
        @~"main",               // 28
        @~"<opaque>",           // 29
        @~"blk",                // 30
        @~"static",             // 31
        @~"intrinsic",          // 32
        @~"__foreign_mod__",    // 33
        @~"__field__",          // 34
        @~"C",                  // 35
        @~"Self",               // 36
    ];

    let rv = @ident_interner {
        interner: interner::Interner::prefill(init_vec)
    };
    unsafe {
        task::local_data::local_data_set(interner_key!(), @rv);
    }
    rv
}

// if an interner exists in TLS, return it. Otherwise, prepare a
// fresh one.
pub fn mk_ident_interner() -> @ident_interner {
    unsafe {
        match task::local_data::local_data_get(interner_key!()) {
            Some(interner) => *interner,
            None => {
                mk_fresh_ident_interner()
            }
        }
    }
}

/* for when we don't care about the contents; doesn't interact with TLD or
   serialization */
pub fn mk_fake_ident_interner() -> @ident_interner {
    @ident_interner { interner: interner::Interner::new() }
}

/**
 * All the valid words that have meaning in the Rust language.
 *
 * Rust keywords are either 'temporary', 'strict' or 'reserved'.  Temporary
 * keywords are contextual and may be used as identifiers anywhere.  They are
 * expected to disappear from the grammar soon.  Strict keywords may not
 * appear as identifiers at all. Reserved keywords are not used anywhere in
 * the language and may not appear as identifiers.
 */
pub fn keyword_table() -> HashSet<~str> {
    let mut keywords = HashSet::new();
    let mut tmp = temporary_keyword_table();
    let mut strict = strict_keyword_table();
    let mut reserved = reserved_keyword_table();

    do tmp.consume |word|      { keywords.insert(word); }
    do strict.consume |word|   { keywords.insert(word); }
    do reserved.consume |word| { keywords.insert(word); }
    return keywords;
}

/// Keywords that may be used as identifiers
pub fn temporary_keyword_table() -> HashSet<~str> {
    let mut words = HashSet::new();
    let keys = ~[
        ~"self", ~"static",
    ];
    do vec::consume(keys) |_, s| {
        words.insert(s);
    }
    return words;
}

/// Full keywords. May not appear anywhere else.
pub fn strict_keyword_table() -> HashSet<~str> {
    let mut words = HashSet::new();
    let keys = ~[
        ~"as",
        ~"break",
        ~"const", ~"copy",
        ~"do", ~"drop",
        ~"else", ~"enum", ~"extern",
        ~"false", ~"fn", ~"for",
        ~"if", ~"impl",
        ~"let", ~"__log", ~"loop",
        ~"match", ~"mod", ~"mut",
        ~"once",
        ~"priv", ~"pub", ~"pure",
        ~"ref", ~"return",
        ~"struct", ~"super",
        ~"true", ~"trait", ~"type",
        ~"unsafe", ~"use",
        ~"while"
    ];
    do vec::consume(keys) |_, w| {
        words.insert(w);
    }
    return words;
}

pub fn reserved_keyword_table() -> HashSet<~str> {
    let mut words = HashSet::new();
    let keys = ~[
        ~"be"
    ];
    do vec::consume(keys) |_, s| {
        words.insert(s);
    }
    return words;
}


// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
