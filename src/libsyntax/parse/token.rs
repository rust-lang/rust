use util::interner;
use util::interner::interner;
use std::map::{hashmap, str_hash};
use std::serialization::{serializer,
                            deserializer,
                            serialize_uint,
                            deserialize_uint,
                            serialize_i64,
                            deserialize_i64,
                            serialize_u64,
                            deserialize_u64,
                            serialize_bool,
                            deserialize_bool};

#[auto_serialize]
type str_num = uint;

#[auto_serialize]
enum binop {
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

#[auto_serialize]
enum token {
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
    ELLIPSIS,
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
    LIT_FLOAT(str_num, ast::float_ty),
    LIT_STR(str_num),

    /* Name components */
    IDENT(str_num, bool),
    UNDERSCORE,

    /* For interpolation */
    INTERPOLATED(nonterminal),

    DOC_COMMENT(str_num),
    EOF,
}

#[auto_serialize]
/// For interpolation during macro expansion.
enum nonterminal {
    nt_item(@ast::item),
    nt_block(ast::blk),
    nt_stmt(@ast::stmt),
    nt_pat( @ast::pat),
    nt_expr(@ast::expr),
    nt_ty(  @ast::ty),
    nt_ident(str_num, bool),
    nt_path(@ast::path),
    nt_tt(  @ast::token_tree), //needs @ed to break a circularity
    nt_matchers(~[ast::matcher])
}

fn binop_to_str(o: binop) -> ~str {
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

fn to_str(in: interner<@~str>, t: token) -> ~str {
    match t {
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
      ELLIPSIS => ~"...",
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
        int::to_str(i as int, 10u) + ast_util::int_ty_to_str(t)
      }
      LIT_UINT(u, t) => {
        uint::to_str(u as uint, 10u) + ast_util::uint_ty_to_str(t)
      }
      LIT_INT_UNSUFFIXED(i) => {
        int::to_str(i as int, 10u)
      }
      LIT_FLOAT(s, t) => {
        let mut body = *in.get(s);
        if body.ends_with(~".") {
            body = body + ~"0";  // `10.f` is not a float literal
        }
        body + ast_util::float_ty_to_str(t)
      }
      LIT_STR(s) => { ~"\"" + str::escape_default( *in.get(s)) + ~"\"" }

      /* Name components */
      IDENT(s, _) => *in.get(s),

      UNDERSCORE => ~"_",

      /* Other */
      DOC_COMMENT(s) => *in.get(s),
      EOF => ~"<eof>",
      INTERPOLATED(nt) => {
        ~"an interpolated " +
            match nt {
              nt_item(*) => ~"item",
              nt_block(*) => ~"block",
              nt_stmt(*) => ~"statement",
              nt_pat(*) => ~"pattern",
              nt_expr(*) => ~"expression",
              nt_ty(*) => ~"type",
              nt_ident(*) => ~"identifier",
              nt_path(*) => ~"path",
              nt_tt(*) => ~"tt",
              nt_matchers(*) => ~"matcher sequence"
            }
      }
    }
}

pure fn can_begin_expr(t: token) -> bool {
    match t {
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
fn flip_delimiter(&t: token::token) -> token::token {
    match t {
      token::LPAREN => token::RPAREN,
      token::LBRACE => token::RBRACE,
      token::LBRACKET => token::RBRACKET,
      token::RPAREN => token::LPAREN,
      token::RBRACE => token::LBRACE,
      token::RBRACKET => token::LBRACKET,
      _ => fail
    }
}



fn is_lit(t: token) -> bool {
    match t {
      LIT_INT(_, _) => true,
      LIT_UINT(_, _) => true,
      LIT_INT_UNSUFFIXED(_) => true,
      LIT_FLOAT(_, _) => true,
      LIT_STR(_) => true,
      _ => false
    }
}

pure fn is_ident(t: token) -> bool {
    match t { IDENT(_, _) => true, _ => false }
}

pure fn is_ident_or_path(t: token) -> bool {
    match t {
      IDENT(_, _) | INTERPOLATED(nt_path(*)) => true,
      _ => false
    }
}

pure fn is_plain_ident(t: token) -> bool {
    match t { IDENT(_, false) => true, _ => false }
}

pure fn is_bar(t: token) -> bool {
    match t { BINOP(OR) | OROR => true, _ => false }
}


mod special_idents {
    import ast::ident;
    const underscore : ident = 0u;
    const anon : ident = 1u;
    const dtor : ident = 2u; // 'drop', but that's reserved
    const invalid : ident = 3u; // ''
    const unary : ident = 4u;
    const not_fn : ident = 5u;
    const idx_fn : ident = 6u;
    const unary_minus_fn : ident = 7u;
    const clownshoes_extensions : ident = 8u;

    const self_ : ident = 9u; // 'self'

    /* for matcher NTs */
    const item : ident = 10u;
    const block : ident = 11u;
    const stmt : ident = 12u;
    const pat : ident = 13u;
    const expr : ident = 14u;
    const ty : ident = 15u;
    const ident : ident = 16u;
    const path : ident = 17u;
    const tt : ident = 18u;
    const matchers : ident = 19u;

    const str : ident = 20u; // for the type

    /* outside of libsyntax */
    const ty_visitor : ident = 21u;
    const arg : ident = 22u;
    const descrim : ident = 23u;
    const clownshoe_abi : ident = 24u;
    const clownshoe_stack_shim : ident = 25u;
    const tydesc : ident = 26u;
    const literally_dtor : ident = 27u;
    const main : ident = 28u;
    const opaque : ident = 29u;
    const blk : ident = 30u;
    const static : ident = 31u;
    const intrinsic : ident = 32u;
    const clownshoes_foreign_mod: ident = 33;
}

type ident_interner = util::interner::interner<@~str>;

/** Key for thread-local data for sneaking interner information to the
 * serializer/deserializer. It sounds like a hack because it is one.
 * Bonus ultra-hack: functions as keys don't work across crates,
 * so we have to use a unique number. See taskgroup_key! in task.rs
 * for another case of this. */
macro_rules! interner_key (
    () => (unsafe::transmute::<(uint, uint), &fn(+@@token::ident_interner)>(
        (-3 as uint, 0u)))
)

fn mk_ident_interner() -> ident_interner {
    /* the indices here must correspond to the numbers in special_idents */
    let init_vec = ~[@~"_", @~"anon", @~"drop", @~"", @~"unary", @~"!",
                     @~"[]", @~"unary-", @~"__extensions__", @~"self",
                     @~"item", @~"block", @~"stmt", @~"pat", @~"expr",
                     @~"ty", @~"ident", @~"path", @~"tt", @~"matchers",
                     @~"str", @~"TyVisitor", @~"arg", @~"descrim",
                     @~"__rust_abi", @~"__rust_stack_shim", @~"TyDesc",
                     @~"dtor", @~"main", @~"<opaque>", @~"blk", @~"static",
                     @~"intrinsic", @~"__foreign_mod__"];

    let rv = interner::mk_prefill::<@~str>(|x| str::hash(*x),
                                           |x,y| str::eq(*x, *y), init_vec);

    /* having multiple interners will just confuse the serializer */
    unsafe{ assert task::local_data_get(interner_key!()).is_none() };
    unsafe{ task::local_data_set(interner_key!(), @rv) };
    rv
}

/* for when we don't care about the contents; doesn't interact with TLD or
   serialization */
fn mk_fake_ident_interner() -> ident_interner {
    interner::mk::<@~str>(|x| str::hash(*x), |x,y| str::eq(*x, *y))
}

/**
 * All the valid words that have meaning in the Rust language.
 *
 * Rust keywords are either 'contextual' or 'restricted'. Contextual
 * keywords may be used as identifiers because their appearance in
 * the grammar is unambiguous. Restricted keywords may not appear
 * in positions that might otherwise contain _value identifiers_.
 */
fn keyword_table() -> hashmap<~str, ()> {
    let keywords = str_hash();
    for contextual_keyword_table().each_key |word| {
        keywords.insert(word, ());
    }
    for restricted_keyword_table().each_key |word| {
        keywords.insert(word, ());
    }
    keywords
}

/// Keywords that may be used as identifiers
fn contextual_keyword_table() -> hashmap<~str, ()> {
    let words = str_hash();
    let keys = ~[
        ~"as",
        ~"else",
        ~"move",
        ~"priv", ~"pub",
        ~"self", ~"send", ~"static",
        ~"use"
    ];
    for keys.each |word| {
        words.insert(word, ());
    }
    words
}

/**
 * Keywords that may not appear in any position that might otherwise contain a
 * _value identifier_. Restricted keywords may still be used as other types of
 * identifiers.
 *
 * Reasons:
 *
 * * For some (most?), if used at the start of a line, they will cause the
 *   line to be interpreted as a specific kind of statement, which would be
 *   confusing.
 *
 * * `true` or `false` as identifiers would always be shadowed by
 *   the boolean constants
 */
fn restricted_keyword_table() -> hashmap<~str, ()> {
    let words = str_hash();
    let keys = ~[
        ~"again", ~"assert",
        ~"break",
        ~"const", ~"copy",
        ~"do", ~"drop",
        ~"else", ~"enum", ~"export", ~"extern",
        ~"fail", ~"false", ~"fn", ~"for",
        ~"if", ~"impl", ~"import",
        ~"let", ~"log", ~"loop",
        ~"match", ~"mod", ~"module", ~"move", ~"mut",
        ~"new",
        ~"owned",
        ~"pure",
        ~"ref", ~"return",
        ~"struct",
        ~"true", ~"trait", ~"type",
        ~"unchecked", ~"unsafe",
        ~"while"
    ];
    for keys.each |word| {
        words.insert(word, ());
    }
    words
}

impl binop : cmp::Eq {
    pure fn eq(&&other: binop) -> bool {
        (self as uint) == (other as uint)
    }
}

impl token : cmp::Eq {
    pure fn eq(&&other: token) -> bool {
        match self {
            EQ => {
                match other {
                    EQ => true,
                    _ => false
                }
            }
            LT => {
                match other {
                    LT => true,
                    _ => false
                }
            }
            LE => {
                match other {
                    LE => true,
                    _ => false
                }
            }
            EQEQ => {
                match other {
                    EQEQ => true,
                    _ => false
                }
            }
            NE => {
                match other {
                    NE => true,
                    _ => false
                }
            }
            GE => {
                match other {
                    GE => true,
                    _ => false
                }
            }
            GT => {
                match other {
                    GT => true,
                    _ => false
                }
            }
            ANDAND => {
                match other {
                    ANDAND => true,
                    _ => false
                }
            }
            OROR => {
                match other {
                    OROR => true,
                    _ => false
                }
            }
            NOT => {
                match other {
                    NOT => true,
                    _ => false
                }
            }
            TILDE => {
                match other {
                    TILDE => true,
                    _ => false
                }
            }
            BINOP(e0a) => {
                match other {
                    BINOP(e0b) => e0a == e0b,
                    _ => false
                }
            }
            BINOPEQ(e0a) => {
                match other {
                    BINOPEQ(e0b) => e0a == e0b,
                    _ => false
                }
            }
            AT => {
                match other {
                    AT => true,
                    _ => false
                }
            }
            DOT => {
                match other {
                    DOT => true,
                    _ => false
                }
            }
            DOTDOT => {
                match other {
                    DOTDOT => true,
                    _ => false
                }
            }
            ELLIPSIS => {
                match other {
                    ELLIPSIS => true,
                    _ => false
                }
            }
            COMMA => {
                match other {
                    COMMA => true,
                    _ => false
                }
            }
            SEMI => {
                match other {
                    SEMI => true,
                    _ => false
                }
            }
            COLON => {
                match other {
                    COLON => true,
                    _ => false
                }
            }
            MOD_SEP => {
                match other {
                    MOD_SEP => true,
                    _ => false
                }
            }
            RARROW => {
                match other {
                    RARROW => true,
                    _ => false
                }
            }
            LARROW => {
                match other {
                    LARROW => true,
                    _ => false
                }
            }
            DARROW => {
                match other {
                    DARROW => true,
                    _ => false
                }
            }
            FAT_ARROW => {
                match other {
                    FAT_ARROW => true,
                    _ => false
                }
            }
            LPAREN => {
                match other {
                    LPAREN => true,
                    _ => false
                }
            }
            RPAREN => {
                match other {
                    RPAREN => true,
                    _ => false
                }
            }
            LBRACKET => {
                match other {
                    LBRACKET => true,
                    _ => false
                }
            }
            RBRACKET => {
                match other {
                    RBRACKET => true,
                    _ => false
                }
            }
            LBRACE => {
                match other {
                    LBRACE => true,
                    _ => false
                }
            }
            RBRACE => {
                match other {
                    RBRACE => true,
                    _ => false
                }
            }
            POUND => {
                match other {
                    POUND => true,
                    _ => false
                }
            }
            DOLLAR => {
                match other {
                    DOLLAR => true,
                    _ => false
                }
            }
            LIT_INT(e0a, e1a) => {
                match other {
                    LIT_INT(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            LIT_UINT(e0a, e1a) => {
                match other {
                    LIT_UINT(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            LIT_INT_UNSUFFIXED(e0a) => {
                match other {
                    LIT_INT_UNSUFFIXED(e0b) => e0a == e0b,
                    _ => false
                }
            }
            LIT_FLOAT(e0a, e1a) => {
                match other {
                    LIT_FLOAT(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            LIT_STR(e0a) => {
                match other {
                    LIT_STR(e0b) => e0a == e0b,
                    _ => false
                }
            }
            IDENT(e0a, e1a) => {
                match other {
                    IDENT(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            UNDERSCORE => {
                match other {
                    UNDERSCORE => true,
                    _ => false
                }
            }
            INTERPOLATED(_) => {
                match other {
                    INTERPOLATED(_) => true,
                    _ => false
                }
            }
            DOC_COMMENT(e0a) => {
                match other {
                    DOC_COMMENT(e0b) => e0a == e0b,
                    _ => false
                }
            }
            EOF => {
                match other {
                    EOF => true,
                    _ => false
                }
            }
        }
    }
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
