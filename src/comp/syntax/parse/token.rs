
import ast::ty_mach;
import ast::ty_mach_to_str;
import std::map::new_str_hash;
import util::interner;
import std::int;
import std::uint;
import std::str;

type str_num = uint;

tag binop {
    PLUS;
    MINUS;
    STAR;
    SLASH;
    PERCENT;
    CARET;
    AND;
    OR;
    LSL;
    LSR;
    ASR;
}

tag token {


    /* Expression-operator symbols. */
    EQ;
    LT;
    LE;
    EQEQ;
    NE;
    GE;
    GT;
    ANDAND;
    OROR;
    NOT;
    TILDE;
    BINOP(binop);
    BINOPEQ(binop);


    /* Structural symbols */
    AT;
    DOT;
    ELLIPSIS;
    COMMA;
    SEMI;
    COLON;
    MOD_SEP;
    QUES;
    RARROW;
    SEND;
    RECV;
    LARROW;
    DARROW;
    LPAREN;
    RPAREN;
    LBRACKET;
    RBRACKET;
    LBRACE;
    RBRACE;
    POUND;
    POUND_LBRACE;
    POUND_LT;


    /* Literals */
    LIT_INT(int);
    LIT_UINT(uint);
    LIT_MACH_INT(ty_mach, int);
    LIT_FLOAT(str_num);
    LIT_MACH_FLOAT(ty_mach, str_num);
    LIT_STR(str_num);
    LIT_CHAR(char);
    LIT_BOOL(bool);


    /* Name components */
    IDENT(str_num, bool);
    IDX(int);
    UNDERSCORE;
    BRACEQUOTE(str_num);
    EOF;
}

fn binop_to_str(o: binop) -> str {
    alt o {
      PLUS. { ret "+"; }
      MINUS. { ret "-"; }
      STAR. { ret "*"; }
      SLASH. { ret "/"; }
      PERCENT. { ret "%"; }
      CARET. { ret "^"; }
      AND. { ret "&"; }
      OR. { ret "|"; }
      LSL. { ret "<<"; }
      LSR. { ret ">>"; }
      ASR. { ret ">>>"; }
    }
}

fn to_str(r: lexer::reader, t: token) -> str {
    alt t {
      EQ. { ret "="; }
      LT. { ret "<"; }
      LE. { ret "<="; }
      EQEQ. { ret "=="; }
      NE. { ret "!="; }
      GE. { ret ">="; }
      GT. { ret ">"; }
      NOT. { ret "!"; }
      TILDE. { ret "~"; }
      OROR. { ret "||"; }
      ANDAND. { ret "&&"; }
      BINOP(op) { ret binop_to_str(op); }
      BINOPEQ(op) { ret binop_to_str(op) + "="; }

      /* Structural symbols */
      AT. {
        ret "@";
      }
      DOT. { ret "."; }
      ELLIPSIS. { ret "..."; }
      COMMA. { ret ","; }
      SEMI. { ret ";"; }
      COLON. { ret ":"; }
      MOD_SEP. { ret "::"; }
      QUES. { ret "?"; }
      RARROW. { ret "->"; }
      SEND. { ret "<|"; }
      RECV. { ret "|>"; }
      LARROW. { ret "<-"; }
      DARROW. { ret "<->"; }
      LPAREN. { ret "("; }
      RPAREN. { ret ")"; }
      LBRACKET. { ret "["; }
      RBRACKET. { ret "]"; }
      LBRACE. { ret "{"; }
      RBRACE. { ret "}"; }
      POUND. { ret "#"; }
      POUND_LBRACE. { ret "#{"; }
      POUND_LT. { ret "#<"; }

      /* Literals */
      LIT_INT(i) {
        ret int::to_str(i, 10u);
      }
      LIT_UINT(u) { ret uint::to_str(u, 10u); }
      LIT_MACH_INT(tm, i) {
        ret int::to_str(i, 10u) + "_" + ty_mach_to_str(tm);
      }
      LIT_MACH_FLOAT(tm, s) {
        ret interner::get[str](*r.get_interner(), s) + "_" +
                ty_mach_to_str(tm);
      }
      LIT_FLOAT(s) { ret interner::get[str](*r.get_interner(), s); }
      LIT_STR(s) { // FIXME: escape.
        ret "\"" + interner::get[str](*r.get_interner(), s) + "\"";
      }
      LIT_CHAR(c) {
        // FIXME: escape.
        let tmp = "'";
        str::push_char(tmp, c);
        str::push_byte(tmp, '\'' as u8);
        ret tmp;
      }
      LIT_BOOL(b) { if b { ret "true"; } else { ret "false"; } }

      /* Name components */
      IDENT(s, _) {
        ret interner::get[str](*r.get_interner(), s);
      }
      IDX(i) { ret "_" + int::to_str(i, 10u); }
      UNDERSCORE. { ret "_"; }
      BRACEQUOTE(_) { ret "<bracequote>"; }
      EOF. { ret "<eof>"; }
    }
}


pred can_begin_expr(t: token) -> bool {
    alt t {
      LPAREN. { true }
      LBRACE. { true }
      LBRACKET. { true }
      IDENT(_, _) { true }
      UNDERSCORE. { true }
      TILDE. { true }
      LIT_INT(_) { true }
      LIT_UINT(_) { true }
      LIT_MACH_INT(_, _) { true }
      LIT_FLOAT(_) { true }
      LIT_MACH_FLOAT(_, _) { true }
      LIT_STR(_) { true }
      LIT_CHAR(_) { true }
      POUND. { true }
      AT. { true }
      NOT. { true }
      BINOP(MINUS.) { true }
      BINOP(STAR.) { true }
      _ { false }
    }
}
// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
