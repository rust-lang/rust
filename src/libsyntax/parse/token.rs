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
use ast::Name;
use ast_util;
use parse::token;
use util::interner::StrInterner;
use util::interner;

use std::cmp::Equiv;
use std::local_data;
use std::rand;
use std::rand::RngUtil;

#[deriving(Clone, Encodable, Decodable, Eq, IterBytes)]
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

#[deriving(Clone, Encodable, Decodable, Eq, IterBytes)]
/// For interpolation during macro expansion.
pub enum nonterminal {
    nt_item(@ast::item),
    nt_block(ast::Block),
    nt_stmt(@ast::stmt),
    nt_pat( @ast::pat),
    nt_expr(@ast::expr),
    nt_ty(   ast::Ty),
    nt_ident(ast::ident, bool),
    nt_path( ast::Path),
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
          let mut res = ~"'";
          do (c as char).escape_default |c| {
              res.push_char(c);
          }
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
        let mut body = ident_to_str(s).to_owned();
        if body.ends_with(".") {
            body.push_char('0');  // `10.f` is not a float literal
        }
        body + ast_util::float_ty_to_str(t)
      }
      LIT_FLOAT_UNSUFFIXED(ref s) => {
        let mut body = ident_to_str(s).to_owned();
        if body.ends_with(".") {
            body.push_char('0');  // `10.f` is not a float literal
        }
        body
      }
      LIT_STR(ref s) => { fmt!("\"%s\"", ident_to_str(s).escape_default()) }

      /* Name components */
      IDENT(s, _) => in.get(s.name).to_owned(),
      LIFETIME(s) => fmt!("'%s", in.get(s.name)),
      UNDERSCORE => ~"_",

      /* Other */
      DOC_COMMENT(ref s) => ident_to_str(s).to_owned(),
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

    pub static underscore : ident = ident { name: 0, ctxt: 0};
    pub static anon : ident = ident { name: 1, ctxt: 0};
    pub static invalid : ident = ident { name: 2, ctxt: 0}; // ''
    pub static unary : ident = ident { name: 3, ctxt: 0};
    pub static not_fn : ident = ident { name: 4, ctxt: 0};
    pub static idx_fn : ident = ident { name: 5, ctxt: 0};
    pub static unary_minus_fn : ident = ident { name: 6, ctxt: 0};
    pub static clownshoes_extensions : ident = ident { name: 7, ctxt: 0};

    pub static self_ : ident = ident { name: 8, ctxt: 0}; // 'self'

    /* for matcher NTs */
    pub static item : ident = ident { name: 9, ctxt: 0};
    pub static block : ident = ident { name: 10, ctxt: 0};
    pub static stmt : ident = ident { name: 11, ctxt: 0};
    pub static pat : ident = ident { name: 12, ctxt: 0};
    pub static expr : ident = ident { name: 13, ctxt: 0};
    pub static ty : ident = ident { name: 14, ctxt: 0};
    pub static ident : ident = ident { name: 15, ctxt: 0};
    pub static path : ident = ident { name: 16, ctxt: 0};
    pub static tt : ident = ident { name: 17, ctxt: 0};
    pub static matchers : ident = ident { name: 18, ctxt: 0};

    pub static str : ident = ident { name: 19, ctxt: 0}; // for the type

    /* outside of libsyntax */
    pub static arg : ident = ident { name: 20, ctxt: 0};
    pub static descrim : ident = ident { name: 21, ctxt: 0};
    pub static clownshoe_abi : ident = ident { name: 22, ctxt: 0};
    pub static clownshoe_stack_shim : ident = ident { name: 23, ctxt: 0};
    pub static main : ident = ident { name: 24, ctxt: 0};
    pub static opaque : ident = ident { name: 25, ctxt: 0};
    pub static blk : ident = ident { name: 26, ctxt: 0};
    pub static statik : ident = ident { name: 27, ctxt: 0};
    pub static clownshoes_foreign_mod: ident = ident { name: 28, ctxt: 0};
    pub static unnamed_field: ident = ident { name: 29, ctxt: 0};
    pub static c_abi: ident = ident { name: 30, ctxt: 0};
    pub static type_self: ident = ident { name: 31, ctxt: 0};    // `Self`
}

/**
 * Maps a token to a record specifying the corresponding binary
 * operator
 */
pub fn token_to_binop(tok: &Token) -> Option<ast::binop> {
  match *tok {
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

impl ident_interner {
    pub fn intern(&self, val: &str) -> Name {
        self.interner.intern(val)
    }
    pub fn gensym(&self, val: &str) -> Name {
        self.interner.gensym(val)
    }
    pub fn get(&self, idx: Name) -> @str {
        self.interner.get(idx)
    }
    // is this really something that should be exposed?
    pub fn len(&self) -> uint {
        self.interner.len()
    }
    pub fn find_equiv<Q:Hash + IterBytes + Equiv<@str>>(&self, val: &Q)
                                                     -> Option<Name> {
        self.interner.find_equiv(val)
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
        "arg",                // 20
        "descrim",            // 21
        "__rust_abi",         // 22
        "__rust_stack_shim",  // 23
        "main",               // 24
        "<opaque>",           // 25
        "blk",                // 26
        "static",             // 27
        "__foreign_mod__",    // 28
        "__field__",          // 29
        "C",                  // 30
        "Self",               // 31

        "as",                 // 32
        "break",              // 33
        "const",              // 34
        "copy",               // 35
        "do",                 // 36
        "else",               // 37
        "enum",               // 38
        "extern",             // 39
        "false",              // 40
        "fn",                 // 41
        "for",                // 42
        "if",                 // 43
        "impl",               // 44
        "let",                // 45
        "__log",              // 46
        "loop",               // 47
        "match",              // 48
        "mod",                // 49
        "mut",                // 50
        "once",               // 51
        "priv",               // 52
        "pub",                // 53
        "pure",               // 54
        "ref",                // 55
        "return",             // 56
        "static",             // 27 -- also a special ident
        "self",               //  8 -- also a special ident
        "struct",             // 57
        "super",              // 58
        "true",               // 59
        "trait",              // 60
        "type",               // 61
        "unsafe",             // 62
        "use",                // 63
        "while",              // 64

        "be",                 // 65
        "in",                 // 66
        "foreach",            // 67
    ];

    @ident_interner {
        interner: interner::StrInterner::prefill(init_vec)
    }
}

// if an interner exists in TLS, return it. Otherwise, prepare a
// fresh one.
pub fn get_ident_interner() -> @ident_interner {
    static key: local_data::Key<@@::parse::token::ident_interner> =
        &local_data::Key;
    match local_data::get(key, |k| k.map(|&k| *k)) {
        Some(interner) => *interner,
        None => {
            let interner = mk_fresh_ident_interner();
            local_data::set(key, @interner);
            interner
        }
    }
}

/* for when we don't care about the contents; doesn't interact with TLD or
   serialization */
pub fn mk_fake_ident_interner() -> @ident_interner {
    @ident_interner { interner: interner::StrInterner::new() }
}

// maps a string to its interned representation
pub fn intern(str : &str) -> Name {
    let interner = get_ident_interner();
    interner.intern(str)
}

// gensyms a new uint, using the current interner
pub fn gensym(str : &str) -> Name {
    let interner = get_ident_interner();
    interner.gensym(str)
}

// map an interned representation back to a string
pub fn interner_get(name : Name) -> @str {
    get_ident_interner().get(name)
}

// maps an identifier to the string that it corresponds to
pub fn ident_to_str(id : &ast::ident) -> @str {
    interner_get(id.name)
}

// maps a string to an identifier with an empty syntax context
pub fn str_to_ident(str : &str) -> ast::ident {
    ast::new_ident(intern(str))
}

// maps a string to a gensym'ed identifier
pub fn gensym_ident(str : &str) -> ast::ident {
    ast::new_ident(gensym(str))
}


// create a fresh name. In principle, this is just a
// gensym, but for debugging purposes, you'd like the
// resulting name to have a suggestive stringify, without
// paying the cost of guaranteeing that the name is
// truly unique.  I'm going to try to strike a balance
// by using a gensym with a name that has a random number
// at the end. So, the gensym guarantees the uniqueness,
// and the int helps to avoid confusion.
pub fn fresh_name(src_name : &str) -> Name {
    let num = rand::rng().gen_uint_range(0,0xffff);
   gensym(fmt!("%s_%u",src_name,num))
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
        Do,
        Else,
        Enum,
        Extern,
        False,
        Fn,
        For,
        ForEach,
        If,
        Impl,
        In,
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

    impl Keyword {
        pub fn to_ident(&self) -> ident {
            match *self {
                As => ident { name: 32, ctxt: 0 },
                Break => ident { name: 33, ctxt: 0 },
                Const => ident { name: 34, ctxt: 0 },
                Do => ident { name: 36, ctxt: 0 },
                Else => ident { name: 37, ctxt: 0 },
                Enum => ident { name: 38, ctxt: 0 },
                Extern => ident { name: 39, ctxt: 0 },
                False => ident { name: 40, ctxt: 0 },
                Fn => ident { name: 41, ctxt: 0 },
                For => ident { name: 42, ctxt: 0 },
                ForEach => ident { name: 67, ctxt: 0 },
                If => ident { name: 43, ctxt: 0 },
                Impl => ident { name: 44, ctxt: 0 },
                In => ident { name: 66, ctxt: 0 },
                Let => ident { name: 45, ctxt: 0 },
                __Log => ident { name: 46, ctxt: 0 },
                Loop => ident { name: 47, ctxt: 0 },
                Match => ident { name: 48, ctxt: 0 },
                Mod => ident { name: 49, ctxt: 0 },
                Mut => ident { name: 50, ctxt: 0 },
                Once => ident { name: 51, ctxt: 0 },
                Priv => ident { name: 52, ctxt: 0 },
                Pub => ident { name: 53, ctxt: 0 },
                Pure => ident { name: 54, ctxt: 0 },
                Ref => ident { name: 55, ctxt: 0 },
                Return => ident { name: 56, ctxt: 0 },
                Static => ident { name: 27, ctxt: 0 },
                Self => ident { name: 8, ctxt: 0 },
                Struct => ident { name: 57, ctxt: 0 },
                Super => ident { name: 58, ctxt: 0 },
                True => ident { name: 59, ctxt: 0 },
                Trait => ident { name: 60, ctxt: 0 },
                Type => ident { name: 61, ctxt: 0 },
                Unsafe => ident { name: 62, ctxt: 0 },
                Use => ident { name: 63, ctxt: 0 },
                While => ident { name: 64, ctxt: 0 },
                Be => ident { name: 65, ctxt: 0 },
            }
        }
    }
}

pub fn is_keyword(kw: keywords::Keyword, tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => { kw.to_ident().name == sid.name }
        _ => { false }
    }
}

pub fn is_any_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.name {
            8 | 27 | 32 .. 65 => true,
            _ => false,
        },
        _ => false
    }
}

pub fn is_strict_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.name {
            8 | 27 | 32 .. 64 => true,
            _ => false,
        },
        _ => false,
    }
}

pub fn is_reserved_keyword(tok: &Token) -> bool {
    match *tok {
        token::IDENT(sid, false) => match sid.name {
            65 => true,
            _ => false,
        },
        _ => false,
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use std::io;
    #[test] fn t1() {
        let a = fresh_name("ghi");
        printfln!("interned name: %u,\ntextual name: %s\n",
                  a, interner_get(a));
    }
}
