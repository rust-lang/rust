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

LIT_CHAR
  : '\'' ( '\\' CHAR_ESCAPE | ~[\\'\n\t\r] ) '\''
  ;

LIT_BYTE
  : 'b\'' ( '\\' ( [xX] HEXIT HEXIT | [nrt\\'"0] ) | ~[\\'\n\t\r] ) '\''
  ;

fragment INT_SUFFIX
  : 'i'
  | 'i8'
  | 'i16'
  | 'i32'
  | 'i64'
  | 'u'
  | 'u8'
  | 'u16'
  | 'u32'
  | 'u64'
  ;

LIT_INTEGER
  : [0-9][0-9_]* INT_SUFFIX?
  | '0b' [01][01_]* INT_SUFFIX?
  | '0o' [0-7][0-7_]* INT_SUFFIX?
  | '0x' [0-9a-fA-F][0-9a-fA-F_]* INT_SUFFIX?
  ;

fragment FLOAT_SUFFIX
  : 'f32'
  | 'f64'
  ;

LIT_FLOAT
  : [0-9][0-9_]* ('.' | ('.' [0-9][0-9_]*)? ([eE] [-+]? [0-9][0-9_]*)? FLOAT_SUFFIX?)
  ;

LIT_STR
  : '"' ('\\\n' | '\\\r\n' | '\\' CHAR_ESCAPE | .)*? '"'
  ;

LIT_BINARY : 'b' LIT_STR ;
LIT_BINARY_RAW : 'rb' LIT_STR_RAW ;

/* this is a bit messy */

fragment LIT_STR_RAW_INNER
  : '"' .*? '"'
  | LIT_STR_RAW_INNER2
  ;

fragment LIT_STR_RAW_INNER2
  : POUND LIT_STR_RAW_INNER POUND
  ;

LIT_STR_RAW
  : 'r' LIT_STR_RAW_INNER
  ;

IDENT : XID_start XID_continue* ;

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
