- Start Date: 2014-05-23
- RFC PR: [rust-lang/rfcs#90](https://github.com/rust-lang/rfcs/pull/90)
- Rust Issue: [rust-lang/rust#14504](https://github.com/rust-lang/rust/issues/14504)

# Summary

Simplify Rust's lexical syntax to make tooling easier to use and easier to
define.

# Motivation

Rust's lexer does a lot of work. It un-escapes escape sequences in string and
character literals, and parses numeric literals of 4 different bases. It also
strips comments, which is sensible, but can be undesirable for pretty printing
or syntax highlighting without hacks. Since many characters are allowed in
strings both escaped and raw (tabs, newlines, and unicode characters come to
mind), after lexing it is impossible to tell if a given character was escaped
or unescaped in the source, making the lexer difficult to test against a
model.

# Detailed design

The following (antlr4) grammar completely describes the proposed lexical
syntax:

    lexer grammar RustLexer;

    /* import Xidstart, Xidcont; */

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
    LARROW     : '->' ;
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

    KEYWORD : STRICT_KEYWORD | RESERVED_KEYWORD ;

    fragment STRICT_KEYWORD
      : 'as'
      | 'box'
      | 'break'
      | 'continue'
      | 'crate'
      | 'else'
      | 'enum'
      | 'extern'
      | 'fn'
      | 'for'
      | 'if'
      | 'impl'
      | 'in'
      | 'let'
      | 'loop'
      | 'match'
      | 'mod'
      | 'mut'
      | 'once'
      | 'proc'
      | 'pub'
      | 'ref'
      | 'return'
      | 'self'
      | 'static'
      | 'struct'
      | 'super'
      | 'trait'
      | 'true'
      | 'type'
      | 'unsafe'
      | 'use'
      | 'virtual'
      | 'while'
      ;

    fragment RESERVED_KEYWORD
      : 'alignof'
      | 'be'
      | 'const'
      | 'do'
      | 'offsetof'
      | 'priv'
      | 'pure'
      | 'sizeof'
      | 'typeof'
      | 'unsized'
      | 'yield'
      ;

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

    INT_SUFFIX
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

    FLOAT_SUFFIX
      : 'f32'
      | 'f64'
      | 'f128'
      ;

    LIT_FLOAT
      : [0-9][0-9_]* ('.' | ('.' [0-9][0-9_]*)? ([eE] [-+]? [0-9][0-9_]*)? FLOAT_SUFFIX?)
      ;

    LIT_STR
      : '"' ('\\\n' | '\\\r\n' | '\\' CHAR_ESCAPE | .)*? '"'
      ;

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

    fragment BLOCK_COMMENT
      : '/*' (BLOCK_COMMENT | .)*? '*/'
      ;

    COMMENT
      : '//' ~[\r\n]*
      | BLOCK_COMMENT
      ;

    IDENT : XID_start XID_continue* ;

    LIFETIME : '\'' IDENT ;

    WHITESPACE : [ \r\n\t]+ ;


There are a few notable changes from today's lexical syntax:

- Non-doc comments are not stripped. To compensate, when encountering a
  COMMENT token the parser can check itself whether or not it's a doc comment.
  This can be done with a simple regex: `(//(/[^/]|!)|/\*(\*[^*]|!))`.
- Numeric literals are not differentiated based on presence of type suffix,
  nor are they converted from binary/octal/hexadecimal to decimal, nor are
  underscores stripped. This can be done trivially in the parser.
- Character escapes are not unescaped. That is, if you write '\x20', this
  lexer will give you `LIT_CHAR('\x20')` rather than `LIT_CHAR(' ')`. The same
  applies to string literals.

The output of the lexer then becomes annotated spans -- which part of the
document corresponds to which token type. Even whitespace is categorized.

# Drawbacks

Including comments and whitespace in the token stream is very non-traditional
and not strictly necessary.
