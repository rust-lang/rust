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
use ast::{Name, Mrk};
use ast_util;
use parse::token;
use util::interner::StrInterner;
use util::interner;

use std::cast;
use std::char;
use std::local_data;

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
    LIT_INT(i64, ast::int_ty),
    LIT_UINT(u64, ast::uint_ty),
    LIT_INT_UNSUFFIXED(i64),
    LIT_FLOAT(ast::Ident, ast::float_ty),
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
    INTERPOLATED(nonterminal),

    DOC_COMMENT(ast::Ident),
    EOF,
}

#[deriving(Clone, Encodable, Decodable, Eq, IterBytes)]
/// For interpolation during macro expansion.
pub enum nonterminal {
    nt_item(@ast::item),
    nt_block(~ast::Block),
    nt_stmt(@ast::Stmt),
    nt_pat( @ast::Pat),
    nt_expr(@ast::Expr),
    nt_ty(  ~ast::Ty),
    nt_ident(~ast::Ident, bool),
    nt_attr(@ast::Attribute),   // #[foo]
    nt_path(~ast::Path),
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

pub fn to_str(input: @ident_interner, t: &Token) -> ~str {
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
      LIT_STR(ref s) => { format!("\"{}\"", ident_to_str(s).escape_default()) }
      LIT_STR_RAW(ref s, n) => {
          format!("r{delim}\"{string}\"{delim}",
                  delim="#".repeat(n), string=ident_to_str(s))
      }

      /* Name components */
      IDENT(s, _) => input.get(s.name).to_owned(),
      LIFETIME(s) => format!("'{}", input.get(s.name)),
      UNDERSCORE => ~"_",

      /* Other */
      DOC_COMMENT(ref s) => ident_to_str(s).to_owned(),
      EOF => ~"<eof>",
      INTERPOLATED(ref nt) => {
        match nt {
            &nt_expr(e) => ::print::pprust::expr_to_str(e, input),
            &nt_attr(e) => ::print::pprust::attribute_to_str(e, input),
            _ => {
                ~"an interpolated " +
                    match (*nt) {
                      nt_item(*) => ~"item",
                      nt_block(*) => ~"block",
                      nt_stmt(*) => ~"statement",
                      nt_pat(*) => ~"pattern",
                      nt_attr(*) => fail!("should have been handled"),
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
    use ast::Ident;

    pub static underscore : Ident = Ident { name: 0, ctxt: 0}; // apparently unused?
    pub static anon : Ident = Ident { name: 1, ctxt: 0};
    pub static invalid : Ident = Ident { name: 2, ctxt: 0}; // ''
    pub static unary : Ident = Ident { name: 3, ctxt: 0}; // apparently unused?
    pub static not_fn : Ident = Ident { name: 4, ctxt: 0}; // apparently unused?
    pub static idx_fn : Ident = Ident { name: 5, ctxt: 0}; // apparently unused?
    pub static unary_minus_fn : Ident = Ident { name: 6, ctxt: 0}; // apparently unused?
    pub static clownshoes_extensions : Ident = Ident { name: 7, ctxt: 0};

    pub static self_ : Ident = Ident { name: super::SELF_KEYWORD_NAME, ctxt: 0}; // 'self'

    /* for matcher NTs */
    // none of these appear to be used, but perhaps references to
    // these are artificially fabricated by the macro system....
    pub static item : Ident = Ident { name: 9, ctxt: 0};
    pub static block : Ident = Ident { name: 10, ctxt: 0};
    pub static stmt : Ident = Ident { name: 11, ctxt: 0};
    pub static pat : Ident = Ident { name: 12, ctxt: 0};
    pub static expr : Ident = Ident { name: 13, ctxt: 0};
    pub static ty : Ident = Ident { name: 14, ctxt: 0};
    pub static ident : Ident = Ident { name: 15, ctxt: 0};
    pub static path : Ident = Ident { name: 16, ctxt: 0};
    pub static tt : Ident = Ident { name: 17, ctxt: 0};
    pub static matchers : Ident = Ident { name: 18, ctxt: 0};

    pub static str : Ident = Ident { name: 19, ctxt: 0}; // for the type // apparently unused?

    /* outside of libsyntax */
    pub static arg : Ident = Ident { name: 20, ctxt: 0};
    pub static descrim : Ident = Ident { name: 21, ctxt: 0};
    pub static clownshoe_abi : Ident = Ident { name: 22, ctxt: 0};
    pub static clownshoe_stack_shim : Ident = Ident { name: 23, ctxt: 0};
    pub static main : Ident = Ident { name: 24, ctxt: 0};
    pub static opaque : Ident = Ident { name: 25, ctxt: 0};
    pub static blk : Ident = Ident { name: 26, ctxt: 0};
    pub static statik : Ident = Ident { name: super::STATIC_KEYWORD_NAME, ctxt: 0};
    pub static clownshoes_foreign_mod: Ident = Ident { name: 28, ctxt: 0};
    pub static unnamed_field: Ident = Ident { name: 29, ctxt: 0};
    pub static c_abi: Ident = Ident { name: 30, ctxt: 0}; // apparently unused?
    pub static type_self: Ident = Ident { name: 31, ctxt: 0};    // `Self`
}

// here are the ones that actually occur in the source. Maybe the rest
// should be removed?
/*
special_idents::anon
special_idents::arg
special_idents::blk
special_idents::clownshoe_abi
special_idents::clownshoe_stack_shim
special_idents::clownshoes_extensions
special_idents::clownshoes_foreign_mod
special_idents::descrim
special_idents::invalid
special_idents::main
special_idents::matchers
special_idents::opaque
special_idents::self_
special_idents::statik
special_idents::tt
special_idents::type_self
special_idents::unnamed_field
*/

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
pub type ident_interner = StrInterner;

// return a fresh interner, preloaded with special identifiers.
fn mk_fresh_ident_interner() -> @ident_interner {
    // The indices here must correspond to the numbers in
    // special_idents, in Keyword to_ident(), and in static
    // constants below.
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
        "<unnamed_field>",    // 29
        "C",                  // 30
        "Self",               // 31

        "as",                 // 32
        "break",              // 33
        "const",              // 34
        "do",                 // 35
        "else",               // 36
        "enum",               // 37
        "extern",             // 38
        "false",              // 39
        "fn",                 // 40
        "for",                // 41
        "if",                 // 42
        "impl",               // 43
        "let",                // 44
        "__log_level",        // 45
        "loop",               // 46
        "match",              // 47
        "mod",                // 48
        "mut",                // 49
        "once",               // 50
        "priv",               // 51
        "pub",                // 52
        "ref",                // 53
        "return",             // 54
        "static",             // 27 -- also a special ident (prefill de-dupes)
        "self",               //  8 -- also a special ident (prefill de-dupes)
        "struct",             // 55
        "super",              // 56
        "true",               // 57
        "trait",              // 58
        "type",               // 59
        "unsafe",             // 60
        "use",                // 61
        "while",              // 62
        "in",                 // 63
        "continue",           // 64
        "proc",               // 65

        "be",                 // 66
        "pure",               // 67
        "yield",              // 68
        "typeof",             // 69
        "alignof",            // 70
        "offsetof",           // 71
        "sizeof",             // 72
    ];

    @interner::StrInterner::prefill(init_vec)
}

static SELF_KEYWORD_NAME: Name = 8;
static STATIC_KEYWORD_NAME: Name = 27;
static STRICT_KEYWORD_START: Name = 32;
static STRICT_KEYWORD_FINAL: Name = 65;
static RESERVED_KEYWORD_START: Name = 66;
static RESERVED_KEYWORD_FINAL: Name = 72;

// if an interner exists in TLS, return it. Otherwise, prepare a
// fresh one.
pub fn get_ident_interner() -> @ident_interner {
    local_data_key!(key: @@::parse::token::ident_interner)
    match local_data::get(key, |k| k.map(|k| *k)) {
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
    @interner::StrInterner::new()
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
pub fn ident_to_str(id : &ast::Ident) -> @str {
    interner_get(id.name)
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

// it looks like there oughta be a str_ptr_eq fn, but no one bothered to implement it?

// determine whether two @str values are pointer-equal
pub fn str_ptr_eq(a : @str, b : @str) -> bool {
    unsafe {
        let p : uint = cast::transmute(a);
        let q : uint = cast::transmute(b);
        let result = p == q;
        // got to transmute them back, to make sure the ref count is correct:
        let _junk1 : @str = cast::transmute(p);
        let _junk2 : @str = cast::transmute(q);
        result
    }
}

// return true when two identifiers refer (through the intern table) to the same ptr_eq
// string. This is used to compare identifiers in places where hygienic comparison is
// not wanted (i.e. not lexical vars).
pub fn ident_spelling_eq(a : &ast::Ident, b : &ast::Ident) -> bool {
    str_ptr_eq(interner_get(a.name),interner_get(b.name))
}

// create a fresh mark.
pub fn fresh_mark() -> Mrk {
    gensym("mark")
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
        If,
        Impl,
        In,
        Let,
        __LogLevel,
        Loop,
        Match,
        Mod,
        Mut,
        Once,
        Priv,
        Pub,
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
        Continue,
        Proc,

        // Reserved keywords
        Alignof,
        Be,
        Offsetof,
        Pure,
        Sizeof,
        Typeof,
        Yield,
    }

    impl Keyword {
        pub fn to_ident(&self) -> Ident {
            match *self {
                As => Ident { name: 32, ctxt: 0 },
                Break => Ident { name: 33, ctxt: 0 },
                Const => Ident { name: 34, ctxt: 0 },
                Do => Ident { name: 35, ctxt: 0 },
                Else => Ident { name: 36, ctxt: 0 },
                Enum => Ident { name: 37, ctxt: 0 },
                Extern => Ident { name: 38, ctxt: 0 },
                False => Ident { name: 39, ctxt: 0 },
                Fn => Ident { name: 40, ctxt: 0 },
                For => Ident { name: 41, ctxt: 0 },
                If => Ident { name: 42, ctxt: 0 },
                Impl => Ident { name: 43, ctxt: 0 },
                In => Ident { name: 63, ctxt: 0 },
                Let => Ident { name: 44, ctxt: 0 },
                __LogLevel => Ident { name: 45, ctxt: 0 },
                Loop => Ident { name: 46, ctxt: 0 },
                Match => Ident { name: 47, ctxt: 0 },
                Mod => Ident { name: 48, ctxt: 0 },
                Mut => Ident { name: 49, ctxt: 0 },
                Once => Ident { name: 50, ctxt: 0 },
                Priv => Ident { name: 51, ctxt: 0 },
                Pub => Ident { name: 52, ctxt: 0 },
                Ref => Ident { name: 53, ctxt: 0 },
                Return => Ident { name: 54, ctxt: 0 },
                Static => Ident { name: super::STATIC_KEYWORD_NAME, ctxt: 0 },
                Self => Ident { name: super::SELF_KEYWORD_NAME, ctxt: 0 },
                Struct => Ident { name: 55, ctxt: 0 },
                Super => Ident { name: 56, ctxt: 0 },
                True => Ident { name: 57, ctxt: 0 },
                Trait => Ident { name: 58, ctxt: 0 },
                Type => Ident { name: 59, ctxt: 0 },
                Unsafe => Ident { name: 60, ctxt: 0 },
                Use => Ident { name: 61, ctxt: 0 },
                While => Ident { name: 62, ctxt: 0 },
                Continue => Ident { name: 64, ctxt: 0 },
                Proc => Ident { name: 65, ctxt: 0 },

                Alignof => Ident { name: 70, ctxt: 0 },
                Be => Ident { name: 66, ctxt: 0 },
                Offsetof => Ident { name: 71, ctxt: 0 },
                Pure => Ident { name: 67, ctxt: 0 },
                Sizeof => Ident { name: 72, ctxt: 0 },
                Typeof => Ident { name: 69, ctxt: 0 },
                Yield => Ident { name: 68, ctxt: 0 },
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


    #[test] fn str_ptr_eq_tests(){
        let a = @"abc";
        let b = @"abc";
        let c = a;
        assert!(str_ptr_eq(a,c));
        assert!(!str_ptr_eq(a,b));
    }

    #[test] fn fresh_name_pointer_sharing() {
        let ghi = str_to_ident("ghi");
        assert_eq!(ident_to_str(&ghi),@"ghi");
        assert!(str_ptr_eq(ident_to_str(&ghi),ident_to_str(&ghi)))
        let fresh = ast::Ident::new(fresh_name(&ghi));
        assert_eq!(ident_to_str(&fresh),@"ghi");
        assert!(str_ptr_eq(ident_to_str(&ghi),ident_to_str(&fresh)));
    }

}
