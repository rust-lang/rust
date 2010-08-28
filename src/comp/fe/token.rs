import util.common.ty_mach;
import util.common.ty_mach_to_str;
import std._int;
import std._uint;

type op = tag
    (PLUS(),
     MINUS(),
     STAR(),
     SLASH(),
     PERCENT(),
     EQ(),
     LT(),
     LE(),
     EQEQ(),
     NE(),
     GE(),
     GT(),
     NOT(),
     TILDE(),
     CARET(),
     AND(),
     ANDAND(),
     OR(),
     OROR(),
     LSL(),
     LSR(),
     ASR());

type token = tag
    (OP(op),
     OPEQ(op),
     AS(),
     WITH(),

     /* Structural symbols */
     AT(),
     DOT(),
     COMMA(),
     SEMI(),
     COLON(),
     RARROW(),
     SEND(),
     LARROW(),
     LPAREN(),
     RPAREN(),
     LBRACKET(),
     RBRACKET(),
     LBRACE(),
     RBRACE(),

     /* Module and crate keywords */
     MOD(),
     USE(),
     AUTH(),
     META(),

     /* Metaprogramming keywords */
     SYNTAX(),
     POUND(),

     /* Statement keywords */
     IF(),
     ELSE(),
     DO(),
     WHILE(),
     ALT(),
     CASE(),

     FAIL(),
     DROP(),

     IN(),
     FOR(),
     EACH(),
     PUT(),
     RET(),
     BE(),

     /* Type and type-state keywords */
     TYPE(),
     CHECK(),
     CLAIM(),
     PROVE(),

     /* Effect keywords */
     IO(),
     STATE(),
     UNSAFE(),

     /* Type qualifiers */
     NATIVE(),
     AUTO(),
     MUTABLE(),

     /* Name management */
     IMPORT(),
     EXPORT(),

     /* Value / stmt declarators */
     LET(),

     /* Magic runtime services */
     LOG(),
     SPAWN(),
     BIND(),
     THREAD(),
     YIELD(),
     JOIN(),

     /* Literals */
     LIT_INT(int),
     LIT_UINT(uint),
     LIT_MACH_INT(ty_mach, int),
     LIT_STR(str),
     LIT_CHAR(char),
     LIT_BOOL(bool),

     /* Name components */
     IDENT(str),
     IDX(int),
     UNDERSCORE(),

     /* Reserved type names */
     BOOL(),
     INT(),
     UINT(),
     FLOAT(),
     CHAR(),
     STR(),
     MACH(ty_mach),

     /* Algebraic type constructors */
     REC(),
     TUP(),
     TAG(),
     VEC(),
     ANY(),

     /* Callable type constructors */
     FN(),
     ITER(),

     /* Object type */
     OBJ(),

     /* Comm and task types */
     CHAN(),
     PORT(),
     TASK(),

     BRACEQUOTE(str),
     EOF());

fn op_to_str(op o) -> str {
    alt (o) {
         case (PLUS()) { ret "+"; }
         case (MINUS()) { ret "-"; }
         case (STAR()) { ret "*"; }
         case (SLASH()) { ret "/"; }
         case (PERCENT()) { ret "%"; }
         case (EQ()) { ret "="; }
         case (LT()) { ret "<"; }
         case (LE()) { ret "<="; }
         case (EQEQ()) { ret "=="; }
         case (NE()) { ret "!="; }
         case (GE()) { ret ">="; }
         case (GT()) { ret ">"; }
         case (NOT()) { ret "!"; }
         case (TILDE()) { ret "~"; }
         case (CARET()) { ret "^"; }
         case (AND()) { ret "&"; }
         case (ANDAND()) { ret "&&"; }
         case (OR()) { ret "|"; }
         case (OROR()) { ret "||"; }
         case (LSL()) { ret "<<"; }
         case (LSR()) { ret ">>"; }
         case (ASR()) { ret ">>>"; }
    }
}

fn to_str(token t) -> str {
    alt (t) {
        case (OP(op)) { ret op_to_str(op); }
        case (OPEQ(op)) { ret op_to_str(op) + "="; }
        case (AS()) { ret "as"; }
        case (WITH()) { ret "with"; }

        /* Structural symbols */
        case (AT()) { ret "@"; }
        case (DOT()) { ret "."; }
        case (COMMA()) { ret ","; }
        case (SEMI()) { ret ";"; }
        case (COLON()) { ret ":"; }
        case (RARROW()) { ret "->"; }
        case (SEND()) { ret "<|"; }
        case (LARROW()) { ret "<-"; }
        case (LPAREN()) { ret "("; }
        case (RPAREN()) { ret ")"; }
        case (LBRACKET()) { ret "["; }
        case (RBRACKET()) { ret "]"; }
        case (LBRACE()) { ret "{"; }
        case (RBRACE()) { ret "}"; }

        /* Module and crate keywords */
        case (MOD()) { ret "mod"; }
        case (USE()) { ret "use"; }
        case (AUTH()) { ret "auth"; }
        case (META()) { ret "meta"; }

        /* Metaprogramming keywords */
        case (SYNTAX()) { ret "syntax"; }
        case (POUND()) { ret "#"; }

        /* Statement keywords */
        case (IF()) { ret "if"; }
        case (ELSE()) { ret "else"; }
        case (DO()) { ret "do"; }
        case (WHILE()) { ret "while"; }
        case (ALT()) { ret "alt"; }
        case (CASE()) { ret "case"; }

        case (FAIL()) { ret "fail"; }
        case (DROP()) { ret "drop"; }

        case (IN()) { ret "in"; }
        case (FOR()) { ret "for"; }
        case (EACH()) { ret "each"; }
        case (PUT()) { ret "put"; }
        case (RET()) { ret "ret"; }
        case (BE()) { ret "be"; }

        /* Type and type-state keywords */
        case (TYPE()) { ret "type"; }
        case (CHECK()) { ret "check"; }
        case (CLAIM()) { ret "claim"; }
        case (PROVE()) { ret "prove"; }

        /* Effect keywords */
        case (IO()) { ret "io"; }
        case (STATE()) { ret "state"; }
        case (UNSAFE()) { ret "unsafe"; }

        /* Type qualifiers */
        case (NATIVE()) { ret "native"; }
        case (AUTO()) { ret "auto"; }
        case (MUTABLE()) { ret "mutable"; }

        /* Name management */
        case (IMPORT()) { ret "import"; }
        case (EXPORT()) { ret "export"; }

        /* Value / stmt declarators */
        case (LET()) { ret "let"; }

        /* Magic runtime services */
        case (LOG()) { ret "log"; }
        case (SPAWN()) { ret "spawn"; }
        case (BIND()) { ret "bind"; }
        case (THREAD()) { ret "thread"; }
        case (YIELD()) { ret "yield"; }
        case (JOIN()) { ret "join"; }

        /* Literals */
        case (LIT_INT(i)) { ret _int.to_str(i, 10u); }
        case (LIT_UINT(u)) { ret _uint.to_str(u, 10u); }
        case (LIT_MACH_INT(tm, i)) {
            ret  _int.to_str(i, 10u)
                + "_" + ty_mach_to_str(tm);
        }

        case (LIT_STR(s)) {
            // FIXME: escape.
            ret "\"" + s + "\"";
        }
        case (LIT_CHAR(c)) {
            // FIXME: escape and encode.
            auto tmp = "";
            tmp += (c as u8);
            ret tmp;
        }

        case (LIT_BOOL(b)) {
            if (b) { ret "true"; } else { ret "false"; }
        }

        /* Name components */
        case (IDENT(s)) { ret s; }
        case (IDX(i)) { ret "_" + _int.to_str(i, 10u); }
        case (UNDERSCORE()) { ret "_"; }

        /* Reserved type names */
        case (BOOL()) { ret "bool"; }
        case (INT()) { ret "int"; }
        case (UINT()) { ret "uint"; }
        case (FLOAT()) { ret "float"; }
        case (CHAR()) { ret "char"; }
        case (STR()) { ret "str"; }
        case (MACH(tm)) { ret ty_mach_to_str(tm); }

        /* Algebraic type constructors */
        case (REC()) { ret "rec"; }
        case (TUP()) { ret "tup"; }
        case (TAG()) { ret "tag"; }
        case (VEC()) { ret "vec"; }
        case (ANY()) { ret "any"; }

        /* Callable type constructors */
        case (FN()) { ret "fn"; }
        case (ITER()) { ret "iter"; }

        /* Object type */
        case (OBJ()) { ret "obj"; }

        /* Comm and task types */
        case (CHAN()) { ret "chan"; }
        case (PORT()) { ret "port"; }
        case (TASK()) { ret "task"; }

        case (BRACEQUOTE(_)) { ret "<bracequote>"; }
        case (EOF()) { ret "<eof>"; }
    }
}



// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
