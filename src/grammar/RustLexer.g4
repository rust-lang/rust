lexer grammar RustLexer;

tokens {
    EQ, LT, LE, EQEQ, NE, GE, GT, ANDAND, OROR, NOT, TILDE, PLUT,
    MINUS, STAR, SLASH, PERCENT, CARET, AND, OR, SHL, SHR, BINOP,
    BINOPEQ, AT, DOT, DOTDOT, DOTDOTDOT, COMMA, SEMI, COLON,
    MOD_SEP, RARROW, FAT_ARROW, LPAREN, RPAREN, LBRACKET, RBRACKET,
    LBRACE, RBRACE, POUND, DOLLAR, UNDERSCORE, LIT_CHAR,
    LIT_INTEGER, LIT_FLOAT, LIT_STR, LIT_STR_RAW, LIT_BINARY,
    LIT_BINARY_RAW, IDENT, LIFETIME, WHITESPACE, DOC_COMMENT,
    COMMENT
}

/* Note: due to antlr limitations, we can't represent XID_start and
 * XID_continue properly. ASCII-only substitute. */

fragment XID_start : [_a-zA-Z] ;
fragment XID_continue : [_a-zA-Z0-9] ;


/* Expression-operator symbols */

EQ      : '=' ;
LT      : '<' ;
LE      : '<=' ;
EQEQ    : '==' ;
NE      : '!=' ;
GE      : '>=' ;
GT      : '>' ;
ANDAND  : '&&' ;
OROR    : '||' ;
NOT     : '!' ;
TILDE   : '~' ;
PLUS    : '+' ;
MINUS   : '-' ;
STAR    : '*' ;
SLASH   : '/' ;
PERCENT : '%' ;
CARET   : '^' ;
AND     : '&' ;
OR      : '|' ;
SHL     : '<<' ;
SHR     : '>>' ;

BINOP
    : PLUS
    | SLASH
    | MINUS
    | STAR
    | PERCENT
    | CARET
    | AND
    | OR
    | SHL
    | SHR
    ;

BINOPEQ : BINOP EQ ;

/* "Structural symbols" */

AT         : '@' ;
DOT        : '.' ;
DOTDOT     : '..' ;
DOTDOTDOT  : '...' ;
COMMA      : ',' ;
SEMI       : ';' ;
COLON      : ':' ;
MOD_SEP    : '::' ;
RARROW     : '->' ;
FAT_ARROW  : '=>' ;
LPAREN     : '(' ;
RPAREN     : ')' ;
LBRACKET   : '[' ;
RBRACKET   : ']' ;
LBRACE     : '{' ;
RBRACE     : '}' ;
POUND      : '#';
DOLLAR     : '$' ;
UNDERSCORE : '_' ;

// Literals

fragment HEXIT
  : [0-9a-fA-F]
  ;

fragment CHAR_ESCAPE
  : [nrt\\'"0]
  | [xX] HEXIT HEXIT
  | 'u' HEXIT HEXIT HEXIT HEXIT
  | 'U' HEXIT HEXIT HEXIT HEXIT HEXIT HEXIT HEXIT HEXIT
  ;

fragment SUFFIX
  : IDENT
  ;

LIT_CHAR
  : '\'' ( '\\' CHAR_ESCAPE | ~[\\'\n\t\r] ) '\'' SUFFIX?
  ;

LIT_BYTE
  : 'b\'' ( '\\' ( [xX] HEXIT HEXIT | [nrt\\'"0] ) | ~[\\'\n\t\r] ) '\'' SUFFIX?
  ;

LIT_INTEGER
  : [0-9][0-9_]* SUFFIX?
  | '0b' [01][01_]* SUFFIX?
  | '0o' [0-7][0-7_]* SUFFIX?
  | '0x' [0-9a-fA-F][0-9a-fA-F_]* SUFFIX?
  ;

LIT_FLOAT
  : [0-9][0-9_]* ('.' {
        /* dot followed by another dot is a range, no float */
        _input.LA(1) != '.' &&
        /* dot followed by an identifier is an integer with a function call, no float */
        _input.LA(1) != '_' &&
        _input.LA(1) != 'a' &&
        _input.LA(1) != 'b' &&
        _input.LA(1) != 'c' &&
        _input.LA(1) != 'd' &&
        _input.LA(1) != 'e' &&
        _input.LA(1) != 'f' &&
        _input.LA(1) != 'g' &&
        _input.LA(1) != 'h' &&
        _input.LA(1) != 'i' &&
        _input.LA(1) != 'j' &&
        _input.LA(1) != 'k' &&
        _input.LA(1) != 'l' &&
        _input.LA(1) != 'm' &&
        _input.LA(1) != 'n' &&
        _input.LA(1) != 'o' &&
        _input.LA(1) != 'p' &&
        _input.LA(1) != 'q' &&
        _input.LA(1) != 'r' &&
        _input.LA(1) != 's' &&
        _input.LA(1) != 't' &&
        _input.LA(1) != 'u' &&
        _input.LA(1) != 'v' &&
        _input.LA(1) != 'w' &&
        _input.LA(1) != 'x' &&
        _input.LA(1) != 'y' &&
        _input.LA(1) != 'z' &&
        _input.LA(1) != 'A' &&
        _input.LA(1) != 'B' &&
        _input.LA(1) != 'C' &&
        _input.LA(1) != 'D' &&
        _input.LA(1) != 'E' &&
        _input.LA(1) != 'F' &&
        _input.LA(1) != 'G' &&
        _input.LA(1) != 'H' &&
        _input.LA(1) != 'I' &&
        _input.LA(1) != 'J' &&
        _input.LA(1) != 'K' &&
        _input.LA(1) != 'L' &&
        _input.LA(1) != 'M' &&
        _input.LA(1) != 'N' &&
        _input.LA(1) != 'O' &&
        _input.LA(1) != 'P' &&
        _input.LA(1) != 'Q' &&
        _input.LA(1) != 'R' &&
        _input.LA(1) != 'S' &&
        _input.LA(1) != 'T' &&
        _input.LA(1) != 'U' &&
        _input.LA(1) != 'V' &&
        _input.LA(1) != 'W' &&
        _input.LA(1) != 'X' &&
        _input.LA(1) != 'Y' &&
        _input.LA(1) != 'Z'
  }? | ('.' [0-9][0-9_]*)? ([eE] [-+]? [0-9][0-9_]*)? SUFFIX?)
  ;

LIT_STR
  : '"' ('\\\n' | '\\\r\n' | '\\' CHAR_ESCAPE | .)*? '"' SUFFIX?
  ;

LIT_BINARY : 'b' LIT_STR SUFFIX?;
LIT_BINARY_RAW : 'rb' LIT_STR_RAW SUFFIX?;

/* this is a bit messy */

fragment LIT_STR_RAW_INNER
  : '"' .*? '"'
  | LIT_STR_RAW_INNER2
  ;

fragment LIT_STR_RAW_INNER2
  : POUND LIT_STR_RAW_INNER POUND
  ;

LIT_STR_RAW
  : 'r' LIT_STR_RAW_INNER SUFFIX?
  ;


QUESTION : '?';

IDENT : XID_start XID_continue* ;

fragment QUESTION_IDENTIFIER : QUESTION? IDENT;

LIFETIME : '\'' IDENT ;

WHITESPACE : [ \r\n\t]+ ;

UNDOC_COMMENT     : '////' ~[\r\n]* -> type(COMMENT) ;
YESDOC_COMMENT    : '///' ~[\r\n]* -> type(DOC_COMMENT) ;
OUTER_DOC_COMMENT : '//!' ~[\r\n]* -> type(DOC_COMMENT) ;
LINE_COMMENT      : '//' ~[\r\n]* -> type(COMMENT) ;

DOC_BLOCK_COMMENT
  : ('/**' ~[*] | '/*!') (DOC_BLOCK_COMMENT | .)*? '*/' -> type(DOC_COMMENT)
  ;

BLOCK_COMMENT : '/*' (BLOCK_COMMENT | .)*? '*/' -> type(COMMENT) ;
