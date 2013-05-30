// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast;
use ast_util;
use parse::token;
use util::interner::StrInterner;
use util::interner;

use core::cmp::Equiv;
use core::to_bytes;

#[deriving(Encodable, Decodable, Eq)]
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

#[deriving(Encodable, Decodable, Eq)]
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

#[deriving(Encodable, Decodable, Eq)]
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
      BINOPEQ(op) => binop_to_str(op) + "=",

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
        ~"'" + char::escape_default(c as char) + "'"
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
        if body.ends_with(".") {
            body += "0";  // `10.f` is not a float literal
        }
        body + ast_util::float_ty_to_str(t)
      }
      LIT_FLOAT_UNSUFFIXED(s) => {
        let mut body = copy *in.get(s);
        if body.ends_with(".") {
            body += "0";  // `10.f` is not a float literal
        }
        body
      }
      LIT_STR(s) => { ~"\"" + str::escape_default(*in.get(s)) + "\"" }

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
                      nt_expr(*) => fail!("should have been handled above"),
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

    pub static underscore : ident = ident { repr: 0, ctxt: 0};
    pub static anon : ident = ident { repr: 1, ctxt: 0};
    pub static invalid : ident = ident { repr: 2, ctxt: 0}; // ''
    pub static unary : ident = ident { repr: 3, ctxt: 0};
    pub static not_fn : ident = ident { repr: 4, ctxt: 0};
    pub static idx_fn : ident = ident { repr: 5, ctxt: 0};
    pub static unary_minus_fn : ident = ident { repr: 6, ctxt: 0};
    pub static clownshoes_extensions : ident = ident { repr: 7, ctxt: 0};

    pub static self_ : ident = ident { repr: 8, ctxt: 0}; // 'self'

    /* for matcher NTs */
    pub static item : ident = ident { repr: 9, ctxt: 0};
    pub static block : ident = ident { repr: 10, ctxt: 0};
    pub static stmt : ident = ident { repr: 11, ctxt: 0};
    pub static pat : ident = ident { repr: 12, ctxt: 0};
    pub static expr : ident = ident { repr: 13, ctxt: 0};
    pub static ty : ident = ident { repr: 14, ctxt: 0};
    pub static ident : ident = ident { repr: 15, ctxt: 0};
    pub static path : ident = ident { repr: 16, ctxt: 0};
    pub static tt : ident = ident { repr: 17, ctxt: 0};
    pub static matchers : ident = ident { repr: 18, ctxt: 0};

    pub static str : ident = ident { repr: 19, ctxt: 0}; // for the type

    /* outside of libsyntax */
    pub static ty_visitor : ident = ident { repr: 20, ctxt: 0};
    pub static arg : ident = ident { repr: 21, ctxt: 0};
    pub static descrim : ident = ident { repr: 22, ctxt: 0};
    pub static clownshoe_abi : ident = ident { repr: 23, ctxt: 0};
    pub static clownshoe_stack_shim : ident = ident { repr: 24, ctxt: 0};
    pub static tydesc : ident = ident { repr: 25, ctxt: 0};
    pub static main : ident = ident { repr: 26, ctxt: 0};
    pub static opaque : ident = ident { repr: 27, ctxt: 0};
    pub static blk : ident = ident { repr: 28, ctxt: 0};
    pub static statik : ident = ident { repr: 29, ctxt: 0};
    pub static intrinsic : ident = ident { repr: 30, ctxt: 0};
    pub static clownshoes_foreign_mod: ident = ident { repr: 31, ctxt: 0};
    pub static unnamed_field: ident = ident { repr: 32, ctxt: 0};
    pub static c_abi: ident = ident { repr: 33, ctxt: 0};
    pub static type_self: ident = ident { repr: 34, ctxt: 0};    // `Self`
}

pub struct StringRef<'self>(&'self str);

impl<'self> Equiv<@~str> for StringRef<'self> {
    #[inline(always)]
    fn equiv(&self, other: &@~str) -> bool { str::eq_slice(**self, **other) }
}

impl<'self> to_bytes::IterBytes for StringRef<'self> {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        (**self).iter_bytes(lsb0, f)
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
    priv interner: StrInterner,
}

pub impl ident_interner {
    fn intern(&self, val: &str) -> ast::ident {
        ast::ident { repr: self.interner.intern(val), ctxt: 0 }
    }
    fn gensym(&self, val: &str) -> ast::ident {
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
fn mk_fresh_ident_interner() -> @ident_interner {
    // the indices here must correspond to the numbers in
    // special_idents.
    let init_vec = ~[
        "_",                  // 0
        "anon",               // 1
        "",                   // 2
        "unary",              // 3
        "!",                  // 4
        "[]",                 // 5
        "unary-",             // 6
        "__extensions__",     // 7
        "self",               // 8
        "item",               // 9
        "block",              // 10
        "stmt",               // 11
        "pat",                // 12
        "expr",               // 13
        "ty",                 // 14
        "ident",              // 15
        "path",               // 16
        "tt",                 // 17
        "matchers",           // 18
        "str",                // 19
        "TyVisitor",          // 20
        "arg",                // 21
        "descrim",            // 22
        "__rust_abi",         // 23
        "__rust_stack_shim",  // 24
        "TyDesc",             // 25
        "main",               // 26
        "<opaque>",           // 27
        "blk",                // 28
        "static",             // 29
        "intrinsic",          // 30
        "__foreign_mod__",    // 31
        "__field__",          // 32
        "C",                  // 33
        "Self",               // 34

        "as",                 // 35
        "break",              // 36
        "const",              // 37
        "copy",               // 38
        "do",                 // 39
        "drop",               // 40
        "else",               // 41
        "enum",               // 42
        "extern",             // 43
        "false",              // 44
        "fn",                 // 45
        "for",                // 46
        "if",                 // 47
        "impl",               // 48
        "let",                // 49
        "__log",              // 50
        "loop",               // 51
        "match",              // 52
        "mod",                // 53
        "mut",                // 54
        "once",               // 55
        "priv",               // 56
        "pub",                // 57
        "pure",               // 58
        "ref",                // 59
        "return",             // 60
        "static",             // 29 -- also a special ident
        "self",               //  8 -- also a special ident
        "struct",             // 61
        "super",              // 62
        "true",               // 63
        "trait",              // 64
        "type",               // 65
        "unsafe",             // 66
        "use",                // 67
        "while",              // 68

        "be",                 // 69
    ];

    @ident_interner {
        interner: interner::StrInterner::prefill(init_vec)
    }
}

// if an interner exists in TLS, return it. Otherwise, prepare a
// fresh one.
pub fn get_ident_interner() -> @ident_interner {
    unsafe {
        let key =
            (cast::transmute::<(uint, uint),
             &fn(v: @@::parse::token::ident_interner)>(
                 (-3 as uint, 0u)));
        match local_data::local_data_get(key) {
            Some(interner) => *interner,
            None => {
                let interner = mk_fresh_ident_interner();
                unsafe {
                    local_data::local_data_set(key, @interner);
                }
                interner
            }
        }
    }
}

/* for when we don't care about the contents; doesn't interact with TLD or
   serialization */
pub fn mk_fake_ident_interner() -> @ident_interner {
    @ident_interner { interner: interner::StrInterner::new() }
}

// maps a string to its interned representation
pub fn intern(str : &str) -> ast::ident {
    let interner = get_ident_interner();
    interner.intern(str)
}

/**
 * All the valid words that have meaning in the Rust language.
 *
 * Rust keywords are either 'strict' or 'reserved'.  Strict keywords may not
 * appear as identifiers at all. Reserved keywords are not used anywhere in
 * the language and may not appear as identifiers.
 */
pub mod keywords {
    use ast::ident;

    pub enum Keyword {
        // Strict keywords
        As,
        Break,
        Const,
        Copy,
        Do,
        Drop,
        Else,
        Enum,
        Extern,
        False,
        Fn,
        For,
        If,
        Impl,
        Let,
        __Log,
        Loop,
        Match,
        Mod,
        Mut,
        Once,
        Priv,
        Pub,
        Pure,
        Ref,
        Return,
        Static,
        Self,
        Struct,
        Super,
        True,
        Trait,
        Type,
        Unsafe,
        Use,
        While,

        // Reserved keywords
        Be,
    }

    pub impl Keyword {
        fn to_ident(&self) -> ident {
            match *self {
                As => ident { repr: 35, ctxt: 0 },
                   Break => ident { repr: 36, ctxt: 0 },
                   Const => ident { repr: 37, ctxt: 0 },
                   Copy => ident { repr: 38, ctxt: 0 },
                   Do => ident { repr: 39, ctxt: 0 },
                   Drop => ident { repr: 40, ctxt: 0 },
                   Else => ident { repr: 41, ctxt: 0 },
                   Enum => ident { repr: 42, ctxt: 0 },
                   Extern => ident { repr: 43, ctxt: 0 },
                   False => ident { repr: 44, ctxt: 0 },
                   Fn => ident { repr: 45, ctxt: 0 },
                   For => ident { repr: 46, ctxt: 0 },
                   If => ident { repr: 47, ctxt: 0 },
                   Impl => ident { repr: 48, ctxt: 0 },
                   Let => ident { repr: 49, ctxt: 0 },
                   __Log => ident { repr: 50, ctxt: 0 },
                   Loop => ident { repr: 51, ctxt: 0 },
                   Match => ident { repr: 52, ctxt: 0 },
                   Mod => ident { repr: 53, ctxt: 0 },
                   Mut => ident { repr: 54, ctxt: 0 },
                   Once => ident { repr: 55, ctxt: 0 },
                   Priv => ident { repr: 56, ctxt: 0 },
                   Pub => ident { repr: 57, ctxt: 0 },
                   Pure => ident { repr: 58, ctxt: 0 },
                   Ref => ident { repr: 59, ctxt: 0 },
                   Return => ident { repr: 60, ctxt: 0 },
                   Static => ident { repr: 29, ctxt: 0 },
                   Self => ident { repr: 8, ctxt: 0 },
                   Struct => ident { repr: 61, ctxt: 0 },
                   Super => ident { repr: 62, ctxt: 0 },
                   True => ident { repr: 63, ctxt: 0 },
                   Trait => ident { repr: 64, ctxt: 0 },
                   Type => ident { repr: 65, ctxt: 0 },
                   Unsafe => ident { repr: 66, ctxt: 0 },
                   Use => ident { repr: 67, ctxt: 0 },
                   While => ident { repr: 68, ctxt: 0 },
                   Be => ident { repr: 69, ctxt: 0 },
            }
        }
    }
}

pub fn is_keyword(kw: keywords::Keyword, tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => { kw.to_ident().repr == sid.repr }
        _ => { false }
    }
}

pub fn is_any_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.repr {
            8 | 29 | 35 .. 69 => true,
            _ => false,
        },
        _ => false
    }
}

pub fn is_strict_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.repr {
            8 | 29 | 35 .. 68 => true,
            _ => false,
        },
        _ => false,
    }
}

pub fn is_reserved_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.repr {
            69 => true,
            _ => false,
        },
        _ => false,
    }
}
