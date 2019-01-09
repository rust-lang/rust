%{
#include <stdio.h>
#include <ctype.h>

static int num_hashes;
static int end_hashes;
static int saw_non_hash;

%}

%option stack
%option yylineno

%x str
%x rawstr
%x rawstr_esc_begin
%x rawstr_esc_body
%x rawstr_esc_end
%x byte
%x bytestr
%x rawbytestr
%x rawbytestr_nohash
%x pound
%x shebang_or_attr
%x ltorchar
%x linecomment
%x doc_line
%x blockcomment
%x doc_block
%x suffix

ident [a-zA-Z\x80-\xff_][a-zA-Z0-9\x80-\xff_]*

%%

<suffix>{ident}            { BEGIN(INITIAL); }
<suffix>(.|\n)  { yyless(0); BEGIN(INITIAL); }

[ \n\t\r]             { }

\xef\xbb\xbf {
  // UTF-8 byte order mark (BOM), ignore if in line 1, error otherwise
  if (yyget_lineno() != 1) {
    return -1;
  }
}

\/\/(\/|\!)           { BEGIN(doc_line); yymore(); }
<doc_line>\n          { BEGIN(INITIAL);
                        yyleng--;
                        yytext[yyleng] = 0;
                        return ((yytext[2] == '!') ? INNER_DOC_COMMENT : OUTER_DOC_COMMENT);
                      }
<doc_line>[^\n]*      { yymore(); }

\/\/|\/\/\/\/         { BEGIN(linecomment); }
<linecomment>\n       { BEGIN(INITIAL); }
<linecomment>[^\n]*   { }

\/\*(\*|\!)[^*]       { yy_push_state(INITIAL); yy_push_state(doc_block); yymore(); }
<doc_block>\/\*       { yy_push_state(doc_block); yymore(); }
<doc_block>\*\/       {
    yy_pop_state();
    if (yy_top_state() == doc_block) {
        yymore();
    } else {
        return ((yytext[2] == '!') ? INNER_DOC_COMMENT : OUTER_DOC_COMMENT);
    }
}
<doc_block>(.|\n)     { yymore(); }

\/\*                  { yy_push_state(blockcomment); }
<blockcomment>\/\*    { yy_push_state(blockcomment); }
<blockcomment>\*\/    { yy_pop_state(); }
<blockcomment>(.|\n)   { }

_        { return UNDERSCORE; }
abstract { return ABSTRACT; }
alignof  { return ALIGNOF; }
as       { return AS; }
become   { return BECOME; }
box      { return BOX; }
break    { return BREAK; }
catch    { return CATCH; }
const    { return CONST; }
continue { return CONTINUE; }
crate    { return CRATE; }
default  { return DEFAULT; }
do       { return DO; }
else     { return ELSE; }
enum     { return ENUM; }
extern   { return EXTERN; }
false    { return FALSE; }
final    { return FINAL; }
fn       { return FN; }
for      { return FOR; }
if       { return IF; }
impl     { return IMPL; }
in       { return IN; }
let      { return LET; }
loop     { return LOOP; }
macro    { return MACRO; }
match    { return MATCH; }
mod      { return MOD; }
move     { return MOVE; }
mut      { return MUT; }
offsetof { return OFFSETOF; }
override { return OVERRIDE; }
priv     { return PRIV; }
proc     { return PROC; }
pure     { return PURE; }
pub      { return PUB; }
ref      { return REF; }
return   { return RETURN; }
self     { return SELF; }
sizeof   { return SIZEOF; }
static   { return STATIC; }
struct   { return STRUCT; }
super    { return SUPER; }
trait    { return TRAIT; }
true     { return TRUE; }
type     { return TYPE; }
typeof   { return TYPEOF; }
union    { return UNION; }
unsafe   { return UNSAFE; }
unsized  { return UNSIZED; }
use      { return USE; }
virtual  { return VIRTUAL; }
where    { return WHERE; }
while    { return WHILE; }
yield    { return YIELD; }

{ident}  { return IDENT; }

0x[0-9a-fA-F_]+                                    { BEGIN(suffix); return LIT_INTEGER; }
0o[0-7_]+                                          { BEGIN(suffix); return LIT_INTEGER; }
0b[01_]+                                           { BEGIN(suffix); return LIT_INTEGER; }
[0-9][0-9_]*                                       { BEGIN(suffix); return LIT_INTEGER; }
[0-9][0-9_]*\.(\.|[a-zA-Z])    { yyless(yyleng - 2); BEGIN(suffix); return LIT_INTEGER; }

[0-9][0-9_]*\.[0-9_]*([eE][-\+]?[0-9_]+)?          { BEGIN(suffix); return LIT_FLOAT; }
[0-9][0-9_]*(\.[0-9_]*)?[eE][-\+]?[0-9_]+          { BEGIN(suffix); return LIT_FLOAT; }

;      { return ';'; }
,      { return ','; }
\.\.\. { return DOTDOTDOT; }
\.\.   { return DOTDOT; }
\.     { return '.'; }
\(     { return '('; }
\)     { return ')'; }
\{     { return '{'; }
\}     { return '}'; }
\[     { return '['; }
\]     { return ']'; }
@      { return '@'; }
#      { BEGIN(pound); yymore(); }
<pound>\! { BEGIN(shebang_or_attr); yymore(); }
<shebang_or_attr>\[ {
  BEGIN(INITIAL);
  yyless(2);
  return SHEBANG;
}
<shebang_or_attr>[^\[\n]*\n {
  // Since the \n was eaten as part of the token, yylineno will have
  // been incremented to the value 2 if the shebang was on the first
  // line. This yyless undoes that, setting yylineno back to 1.
  yyless(yyleng - 1);
  if (yyget_lineno() == 1) {
    BEGIN(INITIAL);
    return SHEBANG_LINE;
  } else {
    BEGIN(INITIAL);
    yyless(2);
    return SHEBANG;
  }
}
<pound>. { BEGIN(INITIAL); yyless(1); return '#'; }

\~     { return '~'; }
::     { return MOD_SEP; }
:      { return ':'; }
\$     { return '$'; }
\?     { return '?'; }

==    { return EQEQ; }
=>    { return FAT_ARROW; }
=     { return '='; }
\!=   { return NE; }
\!    { return '!'; }
\<=   { return LE; }
\<\<  { return SHL; }
\<\<= { return SHLEQ; }
\<    { return '<'; }
\>=   { return GE; }
\>\>  { return SHR; }
\>\>= { return SHREQ; }
\>    { return '>'; }

\x27                                      { BEGIN(ltorchar); yymore(); }
<ltorchar>static                          { BEGIN(INITIAL); return STATIC_LIFETIME; }
<ltorchar>{ident}                         { BEGIN(INITIAL); return LIFETIME; }
<ltorchar>\\[nrt\\\x27\x220]\x27          { BEGIN(suffix); return LIT_CHAR; }
<ltorchar>\\x[0-9a-fA-F]{2}\x27           { BEGIN(suffix); return LIT_CHAR; }
<ltorchar>\\u\{([0-9a-fA-F]_*){1,6}\}\x27 { BEGIN(suffix); return LIT_CHAR; }
<ltorchar>.\x27                           { BEGIN(suffix); return LIT_CHAR; }
<ltorchar>[\x80-\xff]{2,4}\x27            { BEGIN(suffix); return LIT_CHAR; }
<ltorchar><<EOF>>                         { BEGIN(INITIAL); return -1; }

b\x22              { BEGIN(bytestr); yymore(); }
<bytestr>\x22      { BEGIN(suffix); return LIT_BYTE_STR; }

<bytestr><<EOF>>                     { return -1; }
<bytestr>\\[n\nrt\\\x27\x220]        { yymore(); }
<bytestr>\\x[0-9a-fA-F]{2}           { yymore(); }
<bytestr>\\u\{([0-9a-fA-F]_*){1,6}\} { yymore(); }
<bytestr>\\[^n\nrt\\\x27\x220]       { return -1; }
<bytestr>(.|\n)                      { yymore(); }

br\x22                      { BEGIN(rawbytestr_nohash); yymore(); }
<rawbytestr_nohash>\x22     { BEGIN(suffix); return LIT_BYTE_STR_RAW; }
<rawbytestr_nohash>(.|\n)   { yymore(); }
<rawbytestr_nohash><<EOF>>  { return -1; }

br/# {
    BEGIN(rawbytestr);
    yymore();
    num_hashes = 0;
    saw_non_hash = 0;
    end_hashes = 0;
}
<rawbytestr># {
    if (!saw_non_hash) {
        num_hashes++;
    } else if (end_hashes != 0) {
        end_hashes++;
        if (end_hashes == num_hashes) {
            BEGIN(INITIAL);
            return LIT_BYTE_STR_RAW;
        }
    }
    yymore();
}
<rawbytestr>\x22# {
    end_hashes = 1;
    if (end_hashes == num_hashes) {
        BEGIN(INITIAL);
        return LIT_BYTE_STR_RAW;
    }
    yymore();
}
<rawbytestr>(.|\n) {
    if (!saw_non_hash) {
        saw_non_hash = 1;
    }
    if (end_hashes != 0) {
        end_hashes = 0;
    }
    yymore();
}
<rawbytestr><<EOF>> { return -1; }

b\x27                           { BEGIN(byte); yymore(); }
<byte>\\[nrt\\\x27\x220]\x27    { BEGIN(INITIAL); return LIT_BYTE; }
<byte>\\x[0-9a-fA-F]{2}\x27     { BEGIN(INITIAL); return LIT_BYTE; }
<byte>\\u([0-9a-fA-F]_*){4}\x27 { BEGIN(INITIAL); return LIT_BYTE; }
<byte>\\U([0-9a-fA-F]_*){8}\x27 { BEGIN(INITIAL); return LIT_BYTE; }
<byte>.\x27                     { BEGIN(INITIAL); return LIT_BYTE; }
<byte><<EOF>>                   { BEGIN(INITIAL); return -1; }

r\x22           { BEGIN(rawstr); yymore(); }
<rawstr>\x22    { BEGIN(suffix); return LIT_STR_RAW; }
<rawstr>(.|\n)  { yymore(); }
<rawstr><<EOF>> { return -1; }

r/#             {
    BEGIN(rawstr_esc_begin);
    yymore();
    num_hashes = 0;
    saw_non_hash = 0;
    end_hashes = 0;
}

<rawstr_esc_begin># {
    num_hashes++;
    yymore();
}
<rawstr_esc_begin>\x22 {
    BEGIN(rawstr_esc_body);
    yymore();
}
<rawstr_esc_begin>(.|\n) { return -1; }

<rawstr_esc_body>\x22/# {
  BEGIN(rawstr_esc_end);
  yymore();
 }
<rawstr_esc_body>(.|\n) {
  yymore();
 }

<rawstr_esc_end># {
  end_hashes++;
  if (end_hashes == num_hashes) {
    BEGIN(INITIAL);
    return LIT_STR_RAW;
  }
  yymore();
 }
<rawstr_esc_end>[^#] {
  end_hashes = 0;
  BEGIN(rawstr_esc_body);
  yymore();
 }

<rawstr_esc_begin,rawstr_esc_body,rawstr_esc_end><<EOF>> { return -1; }

\x22                     { BEGIN(str); yymore(); }
<str>\x22                { BEGIN(suffix); return LIT_STR; }

<str><<EOF>>                     { return -1; }
<str>\\[n\nr\rt\\\x27\x220]      { yymore(); }
<str>\\x[0-9a-fA-F]{2}           { yymore(); }
<str>\\u\{([0-9a-fA-F]_*){1,6}\} { yymore(); }
<str>\\[^n\nrt\\\x27\x220]       { return -1; }
<str>(.|\n)                      { yymore(); }

\<-  { return LARROW; }
-\>  { return RARROW; }
-    { return '-'; }
-=   { return MINUSEQ; }
&&   { return ANDAND; }
&    { return '&'; }
&=   { return ANDEQ; }
\|\| { return OROR; }
\|   { return '|'; }
\|=  { return OREQ; }
\+   { return '+'; }
\+=  { return PLUSEQ; }
\*   { return '*'; }
\*=  { return STAREQ; }
\/   { return '/'; }
\/=  { return SLASHEQ; }
\^   { return '^'; }
\^=  { return CARETEQ; }
%    { return '%'; }
%=   { return PERCENTEQ; }

<<EOF>> { return 0; }

%%
