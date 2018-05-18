% Grammar

# Introduction

This document is the primary reference for the Rust programming language grammar. It
provides only one kind of material:

  - Chapters that formally define the language grammar.

This document does not serve as an introduction to the language. Background
familiarity with the language is assumed. A separate [guide] is available to
help acquire such background.

This document also does not serve as a reference to the [standard] library
included in the language distribution. Those libraries are documented
separately by extracting documentation attributes from their source code. Many
of the features that one might expect to be language features are library
features in Rust, so what you're looking for may be there, not here.

[guide]: guide.html
[standard]: std/index.html

# Notation

Rust's grammar is defined over Unicode codepoints, each conventionally denoted
`U+XXXX`, for 4 or more hexadecimal digits `X`. _Most_ of Rust's grammar is
confined to the ASCII range of Unicode, and is described in this document by a
dialect of Extended Backus-Naur Form (EBNF), specifically a dialect of EBNF
supported by common automated LL(k) parsing tools such as `llgen`, rather than
the dialect given in ISO 14977. The dialect can be defined self-referentially
as follows:

```antlr
grammar : rule + ;
rule    : nonterminal ':' productionrule ';' ;
productionrule : production [ '|' production ] * ;
production : term * ;
term : element repeats ;
element : LITERAL | IDENTIFIER | '[' productionrule ']' ;
repeats : [ '*' | '+' ] NUMBER ? | NUMBER ? | '?' ;
```

Where:

- Whitespace in the grammar is ignored.
- Square brackets are used to group rules.
- `LITERAL` is a single printable ASCII character, or an escaped hexadecimal
  ASCII code of the form `\xQQ`, in single quotes, denoting the corresponding
  Unicode codepoint `U+00QQ`.
- `IDENTIFIER` is a nonempty string of ASCII letters and underscores.
- The `repeat` forms apply to the adjacent `element`, and are as follows:
  - `?` means zero or one repetition
  - `*` means zero or more repetitions
  - `+` means one or more repetitions
  - NUMBER trailing a repeat symbol gives a maximum repetition count
  - NUMBER on its own gives an exact repetition count

This EBNF dialect should hopefully be familiar to many readers.

## Unicode productions

A few productions in Rust's grammar permit Unicode codepoints outside the ASCII
range. We define these productions in terms of character properties specified
in the Unicode standard, rather than in terms of ASCII-range codepoints. The
section [Special Unicode Productions](#special-unicode-productions) lists these
productions.

## String table productions

Some rules in the grammar &mdash; notably [unary
operators](#unary-operator-expressions), [binary
operators](#binary-operator-expressions), and [keywords](#keywords) &mdash; are
given in a simplified form: as a listing of a table of unquoted, printable
whitespace-separated strings. These cases form a subset of the rules regarding
the [token](#tokens) rule, and are assumed to be the result of a
lexical-analysis phase feeding the parser, driven by a DFA, operating over the
disjunction of all such string table entries.

When such a string enclosed in double-quotes (`"`) occurs inside the grammar,
it is an implicit reference to a single member of such a string table
production. See [tokens](#tokens) for more information.

# Lexical structure

## Input format

Rust input is interpreted as a sequence of Unicode codepoints encoded in UTF-8.
Most Rust grammar rules are defined in terms of printable ASCII-range
codepoints, but a small number are defined in terms of Unicode properties or
explicit codepoint lists. [^inputformat]

[^inputformat]: Substitute definitions for the special Unicode productions are
  provided to the grammar verifier, restricted to ASCII range, when verifying the
  grammar in this document.

## Special Unicode Productions

The following productions in the Rust grammar are defined in terms of Unicode
properties: `ident`, `non_null`, `non_eol`, `non_single_quote` and
`non_double_quote`.

### Identifiers

The `ident` production is any nonempty Unicode string of
the following form:

- The first character is in one of the following ranges `U+0041` to `U+005A`
("A" to "Z"), `U+0061` to `U+007A` ("a" to "z"), or `U+005F` ("\_").
- The remaining characters are in the range `U+0030` to `U+0039` ("0" to "9"),
or any of the prior valid initial characters.

as long as the identifier does _not_ occur in the set of [keywords](#keywords).

### Delimiter-restricted productions

Some productions are defined by exclusion of particular Unicode characters:

- `non_null` is any single Unicode character aside from `U+0000` (null)
- `non_eol` is any single Unicode character aside from `U+000A` (`'\n'`)
- `non_single_quote` is any single Unicode character aside from `U+0027`  (`'`)
- `non_double_quote` is any single Unicode character aside from `U+0022` (`"`)

## Comments

```antlr
comment : block_comment | line_comment ;
block_comment : "/*" block_comment_body * "*/" ;
block_comment_body : [block_comment | character] * ;
line_comment : "//" non_eol * ;
```

**FIXME:** add doc grammar?

## Whitespace

```antlr
whitespace_char : '\x20' | '\x09' | '\x0a' | '\x0d' ;
whitespace : [ whitespace_char | comment ] + ;
```

## Tokens

```antlr
simple_token : keyword | unop | binop ;
token : simple_token | ident | literal | symbol | whitespace token ;
```

### Keywords

<p id="keyword-table-marker"></p>

|          |          |          |          |          |
|----------|----------|----------|----------|----------|
| _        | abstract | alignof  | as       | become   |
| box      | break    | const    | continue | crate    |
| do       | else     | enum     | extern   | false    |
| final    | fn       | for      | if       | impl     |
| in       | let      | loop     | macro    | match    |
| mod      | move     | mut      | offsetof | override |
| priv     | proc     | pub      | pure     | ref      |
| return   | Self     | self     | sizeof   | static   |
| struct   | super    | trait    | true     | type     |
| typeof   | unsafe   | unsized  | use      | virtual  |
| where    | while    | yield    |          |          |


Each of these keywords has special meaning in its grammar, and all of them are
excluded from the `ident` rule.

Not all of these keywords are used by the language. Some of them were used
before Rust 1.0, and were left reserved once their implementations were
removed. Some of them were reserved before 1.0 to make space for possible
future features.

### Literals

```antlr
lit_suffix : ident;
literal : [ string_lit | char_lit | byte_string_lit | byte_lit | num_lit | bool_lit ] lit_suffix ?;
```

The optional `lit_suffix` production is only used for certain numeric literals,
but is reserved for future extension. That is, the above gives the lexical
grammar, but a Rust parser will reject everything but the 12 special cases
mentioned in [Number literals](reference/tokens.html#number-literals) in the
reference.

#### Character and string literals

```antlr
char_lit : '\x27' char_body '\x27' ;
string_lit : '"' string_body * '"' | 'r' raw_string ;

char_body : non_single_quote
          | '\x5c' [ '\x27' | common_escape | unicode_escape ] ;

string_body : non_double_quote
            | '\x5c' [ '\x22' | common_escape | unicode_escape ] ;
raw_string : '"' raw_string_body '"' | '#' raw_string '#' ;

common_escape : '\x5c'
              | 'n' | 'r' | 't' | '0'
              | 'x' hex_digit 2
unicode_escape : 'u' '{' hex_digit+ 6 '}';

hex_digit : 'a' | 'b' | 'c' | 'd' | 'e' | 'f'
          | 'A' | 'B' | 'C' | 'D' | 'E' | 'F'
          | dec_digit ;
oct_digit : '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' ;
dec_digit : '0' | nonzero_dec ;
nonzero_dec: '1' | '2' | '3' | '4'
           | '5' | '6' | '7' | '8' | '9' ;
```

#### Byte and byte string literals

```antlr
byte_lit : "b\x27" byte_body '\x27' ;
byte_string_lit : "b\x22" string_body * '\x22' | "br" raw_byte_string ;

byte_body : ascii_non_single_quote
          | '\x5c' [ '\x27' | common_escape ] ;

byte_string_body : ascii_non_double_quote
            | '\x5c' [ '\x22' | common_escape ] ;
raw_byte_string : '"' raw_byte_string_body '"' | '#' raw_byte_string '#' ;

```

#### Number literals

```antlr
num_lit : nonzero_dec [ dec_digit | '_' ] * float_suffix ?
        | '0' [       [ dec_digit | '_' ] * float_suffix ?
              | 'b'   [ '1' | '0' | '_' ] +
              | 'o'   [ oct_digit | '_' ] +
              | 'x'   [ hex_digit | '_' ] +  ] ;

float_suffix : [ exponent | '.' dec_lit exponent ? ] ? ;

exponent : ['E' | 'e'] ['-' | '+' ] ? dec_lit ;
dec_lit : [ dec_digit | '_' ] + ;
```

#### Boolean literals

```antlr
bool_lit : [ "true" | "false" ] ;
```

The two values of the boolean type are written `true` and `false`.

### Symbols

```antlr
symbol : "::" | "->"
       | '#' | '[' | ']' | '(' | ')' | '{' | '}'
       | ',' | ';' ;
```

Symbols are a general class of printable [tokens](#tokens) that play structural
roles in a variety of grammar productions. They are cataloged here for
completeness as the set of remaining miscellaneous printable tokens that do not
otherwise appear as [unary operators](#unary-operator-expressions), [binary
operators](#binary-operator-expressions), or [keywords](#keywords).

## Paths

```antlr
expr_path : [ "::" ] ident [ "::" expr_path_tail ] + ;
expr_path_tail : '<' type_expr [ ',' type_expr ] + '>'
               | expr_path ;

type_path : ident [ type_path_tail ] + ;
type_path_tail : '<' type_expr [ ',' type_expr ] + '>'
               | "::" type_path ;
```

# Syntax extensions

## Macros

```antlr
expr_macro_rules : "macro_rules" '!' ident '(' macro_rule * ')' ';'
                 | "macro_rules" '!' ident '{' macro_rule * '}' ;
macro_rule : '(' matcher * ')' "=>" '(' transcriber * ')' ';' ;
matcher : '(' matcher * ')' | '[' matcher * ']'
        | '{' matcher * '}' | '$' ident ':' ident
        | '$' '(' matcher * ')' sep_token? [ '*' | '+' ]
        | non_special_token ;
transcriber : '(' transcriber * ')' | '[' transcriber * ']'
            | '{' transcriber * '}' | '$' ident
            | '$' '(' transcriber * ')' sep_token? [ '*' | '+' ]
            | non_special_token ;
```

# Crates and source files

**FIXME:** grammar? What production covers #![crate_id = "foo"] ?

# Items and attributes

**FIXME:** grammar?

## Items

```antlr
item : vis ? mod_item | fn_item | type_item | struct_item | enum_item
     | const_item | static_item | trait_item | impl_item | extern_block_item ;
```

### Type Parameters

**FIXME:** grammar?

### Modules

```antlr
mod_item : "mod" ident ( ';' | '{' mod '}' );
mod : [ view_item | item ] * ;
```

#### View items

```antlr
view_item : extern_crate_decl | use_decl ';' ;
```

##### Extern crate declarations

```antlr
extern_crate_decl : "extern" "crate" crate_name
crate_name: ident | ( ident "as" ident )
```

##### Use declarations

```antlr
use_decl : vis ? "use" [ path "as" ident
                        | path_glob ] ;

path_glob : ident [ "::" [ path_glob
                          | '*' ] ] ?
          | '{' path_item [ ',' path_item ] * '}' ;

path_item : ident | "self" ;
```

### Functions

**FIXME:** grammar?

#### Generic functions

**FIXME:** grammar?

#### Unsafety

**FIXME:** grammar?

##### Unsafe functions

**FIXME:** grammar?

##### Unsafe blocks

**FIXME:** grammar?

#### Diverging functions

**FIXME:** grammar?

### Type definitions

**FIXME:** grammar?

### Structures

**FIXME:** grammar?

### Enumerations

**FIXME:** grammar?

### Constant items

```antlr
const_item : "const" ident ':' type '=' expr ';' ;
```

### Static items

```antlr
static_item : "static" ident ':' type '=' expr ';' ;
```

#### Mutable statics

**FIXME:** grammar?

### Traits

**FIXME:** grammar?

### Implementations

**FIXME:** grammar?

### External blocks

```antlr
extern_block_item : "extern" '{' extern_block '}' ;
extern_block : [ foreign_fn ] * ;
```

## Visibility and Privacy

```antlr
vis : "pub" ;
```
### Re-exporting and Visibility

See [Use declarations](#use-declarations).

## Attributes

```antlr
attribute : '#' '!' ? '[' meta_item ']' ;
meta_item : ident [ '=' literal
                  | '(' meta_seq ')' ] ? ;
meta_seq : meta_item [ ',' meta_seq ] ? ;
```

# Statements and expressions

## Statements

```antlr
stmt : decl_stmt | expr_stmt | ';' ;
```

### Declaration statements

```antlr
decl_stmt : item | let_decl ;
```

#### Item declarations

See [Items](#items).

#### Variable declarations

```antlr
let_decl : "let" pat [':' type ] ? [ init ] ? ';' ;
init : [ '=' ] expr ;
```

### Expression statements

```antlr
expr_stmt : expr ';' ;
```

## Expressions

```antlr
expr : literal | path | tuple_expr | unit_expr | struct_expr
     | block_expr | method_call_expr | field_expr | array_expr
     | idx_expr | range_expr | unop_expr | binop_expr
     | paren_expr | call_expr | lambda_expr | while_expr
     | loop_expr | break_expr | continue_expr | for_expr
     | if_expr | match_expr | if_let_expr | while_let_expr
     | return_expr ;
```

#### Lvalues, rvalues and temporaries

**FIXME:** grammar?

#### Moved and copied types

**FIXME:** Do we want to capture this in the grammar as different productions?

### Literal expressions

See [Literals](#literals).

### Path expressions

See [Paths](#paths).

### Tuple expressions

```antlr
tuple_expr : '(' [ expr [ ',' expr ] * | expr ',' ] ? ')' ;
```

### Unit expressions

```antlr
unit_expr : "()" ;
```

### Structure expressions

```antlr
struct_expr_field_init : ident | ident ':' expr ;
struct_expr : expr_path '{' struct_expr_field_init
                      [ ',' struct_expr_field_init ] *
                      [ ".." expr ] '}' |
              expr_path '(' expr
                      [ ',' expr ] * ')' |
              expr_path ;
```

### Block expressions

```antlr
block_expr : '{' [ stmt | item ] *
                 [ expr ] '}' ;
```

### Method-call expressions

```antlr
method_call_expr : expr '.' ident paren_expr_list ;
```

### Field expressions

```antlr
field_expr : expr '.' ident ;
```

### Array expressions

```antlr
array_expr : '[' "mut" ? array_elems? ']' ;

array_elems : [expr [',' expr]*] | [expr ';' expr] ;
```

### Index expressions

```antlr
idx_expr : expr '[' expr ']' ;
```

### Range expressions

```antlr
range_expr : expr ".." expr |
             expr ".." |
             ".." expr |
             ".." ;
```

### Unary operator expressions

```antlr
unop_expr : unop expr ;
unop : '-' | '*' | '!' ;
```

### Binary operator expressions

```antlr
binop_expr : expr binop expr | type_cast_expr
           | assignment_expr | compound_assignment_expr ;
binop : arith_op | bitwise_op | lazy_bool_op | comp_op
```

#### Arithmetic operators

```antlr
arith_op : '+' | '-' | '*' | '/' | '%' ;
```

#### Bitwise operators

```antlr
bitwise_op : '&' | '|' | '^' | "<<" | ">>" ;
```

#### Lazy boolean operators

```antlr
lazy_bool_op : "&&" | "||" ;
```

#### Comparison operators

```antlr
comp_op : "==" | "!=" | '<' | '>' | "<=" | ">=" ;
```

#### Type cast expressions

```antlr
type_cast_expr : value "as" type ;
```

#### Assignment expressions

```antlr
assignment_expr : expr '=' expr ;
```

#### Compound assignment expressions

```antlr
compound_assignment_expr : expr [ arith_op | bitwise_op ] '=' expr ;
```

### Grouped expressions

```antlr
paren_expr : '(' expr ')' ;
```

### Call expressions

```antlr
expr_list : [ expr [ ',' expr ]* ] ? ;
paren_expr_list : '(' expr_list ')' ;
call_expr : expr paren_expr_list ;
```

### Lambda expressions

```antlr
ident_list : [ ident [ ',' ident ]* ] ? ;
lambda_expr : '|' ident_list '|' expr ;
```

### While loops

```antlr
while_expr : [ lifetime ':' ] ? "while" no_struct_literal_expr '{' block '}' ;
```

### Infinite loops

```antlr
loop_expr : [ lifetime ':' ] ? "loop" '{' block '}';
```

### Break expressions

```antlr
break_expr : "break" [ lifetime ] ?;
```

### Continue expressions

```antlr
continue_expr : "continue" [ lifetime ] ?;
```

### For expressions

```antlr
for_expr : [ lifetime ':' ] ? "for" pat "in" no_struct_literal_expr '{' block '}' ;
```

### If expressions

```antlr
if_expr : "if" no_struct_literal_expr '{' block '}'
          else_tail ? ;

else_tail : "else" [ if_expr | if_let_expr
                   | '{' block '}' ] ;
```

### Match expressions

```antlr
match_expr : "match" no_struct_literal_expr '{' match_arm * '}' ;

match_arm : attribute * match_pat "=>" [ expr "," | '{' block '}' ] ;

match_pat : pat [ '|' pat ] * [ "if" expr ] ? ;
```

### If let expressions

```antlr
if_let_expr : "if" "let" pat '=' expr '{' block '}'
               else_tail ? ;
```

### While let loops

```antlr
while_let_expr : [ lifetime ':' ] ? "while" "let" pat '=' expr '{' block '}' ;
```

### Return expressions

```antlr
return_expr : "return" expr ? ;
```

# Type system

**FIXME:** is this entire chapter relevant here? Or should it all have been covered by some production already?

## Types

### Primitive types

**FIXME:** grammar?

#### Machine types

**FIXME:** grammar?

#### Machine-dependent integer types

**FIXME:** grammar?

### Textual types

**FIXME:** grammar?

### Tuple types

**FIXME:** grammar?

### Array, and Slice types

**FIXME:** grammar?

### Structure types

**FIXME:** grammar?

### Enumerated types

**FIXME:** grammar?

### Pointer types

**FIXME:** grammar?

### Function types

**FIXME:** grammar?

### Closure types

```antlr
closure_type := [ 'unsafe' ] [ '<' lifetime-list '>' ] '|' arg-list '|'
                [ ':' bound-list ] [ '->' type ]
lifetime-list := lifetime | lifetime ',' lifetime-list
arg-list := ident ':' type | ident ':' type ',' arg-list
```

### Never type
An empty type

```antlr
never_type : "!" ;
```

### Object types

**FIXME:** grammar?

### Type parameters

**FIXME:** grammar?

### Type parameter bounds

```antlr
bound-list := bound | bound '+' bound-list '+' ?
bound := ty_bound | lt_bound
lt_bound := lifetime
ty_bound := ty_bound_noparen | (ty_bound_noparen)
ty_bound_noparen := [?] [ for<lt_param_defs> ] simple_path
```

### Self types

**FIXME:** grammar?

## Type kinds

**FIXME:** this is probably not relevant to the grammar...

# Memory and concurrency models

**FIXME:** is this entire chapter relevant here? Or should it all have been covered by some production already?

## Memory model

### Memory allocation and lifetime

### Memory ownership

### Variables

### Boxes

## Threads

### Communication between threads

### Thread lifecycle
