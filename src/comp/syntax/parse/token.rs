
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

fn binop_to_str(binop o) -> str {
    alt (o) {
        case (PLUS) { ret "+"; }
        case (MINUS) { ret "-"; }
        case (STAR) { ret "*"; }
        case (SLASH) { ret "/"; }
        case (PERCENT) { ret "%"; }
        case (CARET) { ret "^"; }
        case (AND) { ret "&"; }
        case (OR) { ret "|"; }
        case (LSL) { ret "<<"; }
        case (LSR) { ret ">>"; }
        case (ASR) { ret ">>>"; }
    }
}

fn to_str(lexer::reader r, token t) -> str {
    alt (t) {
        case (EQ) { ret "="; }
        case (LT) { ret "<"; }
        case (LE) { ret "<="; }
        case (EQEQ) { ret "=="; }
        case (NE) { ret "!="; }
        case (GE) { ret ">="; }
        case (GT) { ret ">"; }
        case (NOT) { ret "!"; }
        case (TILDE) { ret "~"; }
        case (OROR) { ret "||"; }
        case (ANDAND) { ret "&&"; }
        case (BINOP(?op)) { ret binop_to_str(op); }
        case (BINOPEQ(?op)) { ret binop_to_str(op) + "="; }
        case (
             /* Structural symbols */
             AT) {
            ret "@";
        }
        case (DOT) { ret "."; }
        case (COMMA) { ret ","; }
        case (SEMI) { ret ";"; }
        case (COLON) { ret ":"; }
        case (MOD_SEP) { ret "::"; }
        case (QUES) { ret "?"; }
        case (RARROW) { ret "->"; }
        case (SEND) { ret "<|"; }
        case (RECV) { ret "|>"; }
        case (LARROW) { ret "<-"; }
        case (DARROW) { ret "<->"; }
        case (LPAREN) { ret "("; }
        case (RPAREN) { ret ")"; }
        case (LBRACKET) { ret "["; }
        case (RBRACKET) { ret "]"; }
        case (LBRACE) { ret "{"; }
        case (RBRACE) { ret "}"; }
        case (POUND) { ret "#"; }
        case (
             /* Literals */
             LIT_INT(?i)) {
            ret int::to_str(i, 10u);
        }
        case (LIT_UINT(?u)) { ret uint::to_str(u, 10u); }
        case (LIT_MACH_INT(?tm, ?i)) {
            ret int::to_str(i, 10u) + "_" + ty_mach_to_str(tm);
        }
        case (LIT_MACH_FLOAT(?tm, ?s)) {
            ret interner::get[str](*r.get_interner(), s) + "_" +
                    ty_mach_to_str(tm);
        }
        case (LIT_FLOAT(?s)) { ret interner::get[str](*r.get_interner(), s); }
        case (LIT_STR(?s)) {
            // FIXME: escape.

            ret "\"" + interner::get[str](*r.get_interner(), s) + "\"";
        }
        case (LIT_CHAR(?c)) {
            // FIXME: escape.

            auto tmp = "'";
            str::push_char(tmp, c);
            str::push_byte(tmp, '\'' as u8);
            ret tmp;
        }
        case (LIT_BOOL(?b)) { if (b) { ret "true"; } else { ret "false"; } }
        case (
             /* Name components */
             IDENT(?s, _)) {
            ret interner::get[str](*r.get_interner(), s);
        }
        case (IDX(?i)) { ret "_" + int::to_str(i, 10u); }
        case (UNDERSCORE) { ret "_"; }
        case (BRACEQUOTE(_)) { ret "<bracequote>"; }
        case (EOF) { ret "<eof>"; }
    }
}


pred can_begin_expr(token t) -> bool {
    alt (t) {
        case (LPAREN) { true }
        case (LBRACE) { true }
        case (IDENT(_,_)) { true }
        case (UNDERSCORE) { true }
        case (TILDE) { true }
        case (LIT_INT(_)) { true }
        case (LIT_UINT(_)) { true }
        case (LIT_MACH_INT(_,_)) { true }
        case (LIT_FLOAT(_)) { true }
        case (LIT_MACH_FLOAT(_,_)) { true }
        case (LIT_STR(_)) { true }
        case (LIT_CHAR(_)) { true }
        case (POUND) { true }
        case (AT) { true }
        case (_) { false }
    }
}
// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
