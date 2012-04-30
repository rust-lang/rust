
import util::interner;
import util::interner::interner;
import std::map::{hashmap, str_hash};

type str_num = uint;

enum binop {
    PLUS,
    MINUS,
    STAR,
    SLASH,
    PERCENT,
    CARET,
    AND,
    OR,
    LSL,
    LSR,
    ASR,
}

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
    ELLIPSIS,
    COMMA,
    SEMI,
    COLON,
    MOD_SEP,
    RARROW,
    LARROW,
    DARROW,
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
    LIT_FLOAT(str_num, ast::float_ty),
    LIT_STR(str_num),

    /* Name components */
    IDENT(str_num, bool),
    UNDERSCORE,
    EOF,

}

fn binop_to_str(o: binop) -> str {
    alt o {
      PLUS { ret "+"; }
      MINUS { ret "-"; }
      STAR { ret "*"; }
      SLASH { ret "/"; }
      PERCENT { ret "%"; }
      CARET { ret "^"; }
      AND { ret "&"; }
      OR { ret "|"; }
      LSL { ret "<<"; }
      LSR { ret ">>"; }
      ASR { ret ">>>"; }
    }
}

fn to_str(in: interner<str>, t: token) -> str {
    alt t {
      EQ { ret "="; }
      LT { ret "<"; }
      LE { ret "<="; }
      EQEQ { ret "=="; }
      NE { ret "!="; }
      GE { ret ">="; }
      GT { ret ">"; }
      NOT { ret "!"; }
      TILDE { ret "~"; }
      OROR { ret "||"; }
      ANDAND { ret "&&"; }
      BINOP(op) { ret binop_to_str(op); }
      BINOPEQ(op) { ret binop_to_str(op) + "="; }

      /* Structural symbols */
      AT {
        ret "@";
      }
      DOT { ret "."; }
      ELLIPSIS { ret "..."; }
      COMMA { ret ","; }
      SEMI { ret ";"; }
      COLON { ret ":"; }
      MOD_SEP { ret "::"; }
      RARROW { ret "->"; }
      LARROW { ret "<-"; }
      DARROW { ret "<->"; }
      LPAREN { ret "("; }
      RPAREN { ret ")"; }
      LBRACKET { ret "["; }
      RBRACKET { ret "]"; }
      LBRACE { ret "{"; }
      RBRACE { ret "}"; }
      POUND { ret "#"; }
      DOLLAR { ret "$"; }

      /* Literals */
      LIT_INT(c, ast::ty_char) {
        // FIXME: escape.
        let mut tmp = "'";
        str::push_char(tmp, c as char);
        str::push_char(tmp, '\'');
        ret tmp;
      }
      LIT_INT(i, t) {
        ret int::to_str(i as int, 10u) + ast_util::int_ty_to_str(t);
      }
      LIT_UINT(u, t) {
        ret uint::to_str(u as uint, 10u) + ast_util::uint_ty_to_str(t);
      }
      LIT_FLOAT(s, t) {
        ret interner::get::<str>(in, s) +
            ast_util::float_ty_to_str(t);
      }
      LIT_STR(s) { // FIXME: escape.
        ret "\"" + interner::get::<str>(in, s) + "\"";
      }

      /* Name components */
      IDENT(s, _) {
        ret interner::get::<str>(in, s);
      }
      UNDERSCORE { ret "_"; }
      EOF { ret "<eof>"; }
    }
}


pure fn can_begin_expr(t: token) -> bool {
    alt t {
      LPAREN { true }
      LBRACE { true }
      LBRACKET { true }
      IDENT(_, _) { true }
      UNDERSCORE { true }
      TILDE { true }
      LIT_INT(_, _) { true }
      LIT_UINT(_, _) { true }
      LIT_FLOAT(_, _) { true }
      LIT_STR(_) { true }
      POUND { true }
      AT { true }
      NOT { true }
      BINOP(MINUS) { true }
      BINOP(STAR) { true }
      BINOP(AND) { true }
      MOD_SEP { true }
      _ { false }
    }
}

fn is_lit(t: token::token) -> bool {
    ret alt t {
          token::LIT_INT(_, _) { true }
          token::LIT_UINT(_, _) { true }
          token::LIT_FLOAT(_, _) { true }
          token::LIT_STR(_) { true }
          _ { false }
        }
}

fn is_ident(t: token::token) -> bool {
    alt t { token::IDENT(_, _) { ret true; } _ { } }
    ret false;
}

fn is_plain_ident(t: token::token) -> bool {
    ret alt t { token::IDENT(_, false) { true } _ { false } };
}

fn is_bar(t: token::token) -> bool {
    alt t { token::BINOP(token::OR) | token::OROR { true } _ { false } }
}

#[doc = "
All the valid words that have meaning in the Rust language.

Rust keywords are either 'contextual' or 'restricted'. Contextual
keywords may be used as identifiers because their appearance in
the grammar is unambiguous. Restricted keywords may not appear
in positions that might otherwise contain _value identifiers_.
"]
fn keyword_table() -> hashmap<str, ()> {
    let keywords = str_hash();
    for contextual_keyword_table().each_key {|word|
        keywords.insert(word, ());
    }
    for restricted_keyword_table().each_key {|word|
        keywords.insert(word, ());
    }
    ret keywords;
}

#[doc = "Keywords that may be used as identifiers"]
fn contextual_keyword_table() -> hashmap<str, ()> {
    let words = str_hash();
    let keys = [
        "as",
        "bind",
        "else",
        "implements",
        "move",
        "of",
        "priv",
        "self", "send", "static",
        "to",
        "use",
        "with"
    ];
    for keys.each {|word|
        words.insert(word, ());
    }
    words
}

#[doc = "
Keywords that may not appear in any position that might otherwise contain a
_value identifier_. Restricted keywords may still be used as other types of
identifiers.

Reasons:

* For some (most?), if used at the start of a line, they will cause the line
  to be interpreted as a specific kind of statement, which would be confusing.

* `true` or `false` as identifiers would always be shadowed by
  the boolean constants
"]
fn restricted_keyword_table() -> hashmap<str, ()> {
    let words = str_hash();
    let keys = [
        "alt",
        "assert",
        "be", "break",
        "check", "claim", "class", "const", "cont", "copy", "crust",
        "do",
        "else", "enum", "export",
        "fail", "false", "fn", "for",
        "if", "iface", "impl", "import",
        "let", "log", "loop",
        "mod", "mut",
        "native", "new",
        "pure",
        "resource", "ret",
        "true", "trait", "type",
        "unchecked", "unsafe",
        "while"
    ];
    for keys.each {|word|
        words.insert(word, ());
    }
    words
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
