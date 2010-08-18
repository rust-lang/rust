import util.common.ty_mach;

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
     LIT_UINT(int),
     LIT_MACH_INT(ty_mach, int),
     LIT_STR(str),
     LIT_CHAR(int),
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



// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
