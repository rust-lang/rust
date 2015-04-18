% Grammar

# Introduction

This document is the primary reference for the Rust programming language grammar. It
provides only one kind of material:

  - Chapters that formally define the language grammar.

This document does not serve as an introduction to the language. Background
familiarity with the language is assumed. A separate [guide] is available to
help acquire such background familiarity.

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
properties: `ident`, `non_null`, `non_star`, `non_eol`, `non_slash_or_star`,
`non_single_quote` and `non_double_quote`.

### Identifiers

The `ident` production is any nonempty Unicode string of the following form:

- The first character has property `XID_start`
- The remaining characters have property `XID_continue`

that does _not_ occur in the set of [keywords](#keywords).

> **Note**: `XID_start` and `XID_continue` as character properties cover the
> character ranges used to form the more familiar C and Java language-family
> identifiers.

### Delimiter-restricted productions

Some productions are defined by exclusion of particular Unicode characters:

- `non_null` is any single Unicode character aside from `U+0000` (null)
- `non_eol` is `non_null` restricted to exclude `U+000A` (`'\n'`)
- `non_star` is `non_null` restricted to exclude `U+002A` (`*`)
- `non_slash_or_star` is `non_null` restricted to exclude `U+002F` (`/`) and `U+002A` (`*`)
- `non_single_quote` is `non_null` restricted to exclude `U+0027`  (`'`)
- `non_double_quote` is `non_null` restricted to exclude `U+0022` (`"`)

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

|          |          |          |          |        |
|----------|----------|----------|----------|--------|
| abstract | alignof  | as       | become   | box    |
| break    | const    | continue | crate    | do     |
| else     | enum     | extern   | false    | final  |
| fn       | for      | if       | impl     | in     |
| let      | loop     | match    | mod      | move   |
| mut      | offsetof | once     | override | priv   |
| proc     | pub      | pure     | ref      | return |
| sizeof   | static   | self     | struct   | super  |
| true     | trait    | type     | typeof   | unsafe |
| unsized  | use      | virtual  | where    | while  |
| yield    |          |          |          |        |


Each of these keywords has special meaning in its grammar, and all of them are
excluded from the `ident` rule.

### Literals

```antlr
lit_suffix : ident;
literal : [ string_lit | char_lit | byte_string_lit | byte_lit | num_lit ] lit_suffix ?;
```

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

**FIXME:** write grammar

The two values of the boolean type are written `true` and `false`.

### Symbols

```antlr
symbol : "::" "->"
       | '#' | '[' | ']' | '(' | ')' | '{' | '}'
       | ',' | ';' ;
```

Symbols are a general class of printable [token](#tokens) that play structural
roles in a variety of grammar productions. They are catalogued here for
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
expr_macro_rules : "macro_rules" '!' ident '(' macro_rule * ')' ;
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
item : mod_item | fn_item | type_item | struct_item | enum_item
     | static_item | trait_item | impl_item | extern_block ;
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
view_item : extern_crate_decl | use_decl ;
```

##### Extern crate declarations

```antlr
extern_crate_decl : "extern" "crate" crate_name
crate_name: ident | ( string_lit as ident )
```

##### Use declarations

```antlr
use_decl : "pub" ? "use" [ path "as" ident
                          | path_glob ] ;

path_glob : ident [ "::" [ path_glob
                          | '*' ] ] ?
          | '{' path_item [ ',' path_item ] * '}' ;

path_item : ident | "mod" ;
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

**FIXME:** grammar?

### Re-exporting and Visibility

**FIXME:** grammar?

## Attributes

```antlr
attribute : "#!" ? '[' meta_item ']' ;
meta_item : ident [ '=' literal
                  | '(' meta_seq ')' ] ? ;
meta_seq : meta_item [ ',' meta_seq ] ? ;
```

# Statements and expressions

## Statements

**FIXME:** grammar?

### Declaration statements

**FIXME:** grammar?

A _declaration statement_ is one that introduces one or more *names* into the
enclosing statement block. The declared names may denote new slots or new
items.

#### Item declarations

**FIXME:** grammar?

An _item declaration statement_ has a syntactic form identical to an
[item](#items) declaration within a module. Declaring an item &mdash; a
function, enumeration, structure, type, static, trait, implementation or module
&mdash; locally within a statement block is simply a way of restricting its
scope to a narrow region containing all of its uses; it is otherwise identical
in meaning to declaring the item outside the statement block.

#### Slot declarations

```antlr
let_decl : "let" pat [':' type ] ? [ init ] ? ';' ;
init : [ '=' ] expr ;
```

### Expression statements

**FIXME:** grammar?

## Expressions

**FIXME:** grammar?

#### Lvalues, rvalues and temporaries

**FIXME:** grammar?

#### Moved and copied types

**FIXME:** Do we want to capture this in the grammar as different productions?

### Literal expressions

**FIXME:** grammar?

### Path expressions

**FIXME:** grammar?

### Tuple expressions

**FIXME:** grammar?

### Unit expressions

**FIXME:** grammar?

### Structure expressions

```antlr
struct_expr : expr_path '{' ident ':' expr
                      [ ',' ident ':' expr ] *
                      [ ".." expr ] '}' |
              expr_path '(' expr
                      [ ',' expr ] * ')' |
              expr_path ;
```

### Block expressions

```antlr
block_expr : '{' [ view_item ] *
                 [ stmt ';' | item ] *
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

array_elems : [expr [',' expr]*] | [expr ',' ".." expr] ;
```

### Index expressions

```antlr
idx_expr : expr '[' expr ']' ;
```

### Unary operator expressions

**FIXME:** grammar?

### Binary operator expressions

```antlr
binop_expr : expr binop expr ;
```

#### Arithmetic operators

**FIXME:** grammar?

#### Bitwise operators

**FIXME:** grammar?

#### Lazy boolean operators

**FIXME:** grammar?

#### Comparison operators

**FIXME:** grammar?

#### Type cast expressions

**FIXME:** grammar?

#### Assignment expressions

**FIXME:** grammar?

#### Compound assignment expressions

**FIXME:** grammar?

#### Operator precedence

The precedence of Rust binary operators is ordered as follows, going from
strong to weak:

```text
* / %
as
+ -
<< >>
&
^
|
< > <= >=
== !=
&&
||
=
```

Operators at the same precedence level are evaluated left-to-right. [Unary
operators](#unary-operator-expressions) have the same precedence level and it
is stronger than any of the binary operators'.

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
while_expr : "while" no_struct_literal_expr '{' block '}' ;
```

### Infinite loops

```antlr
loop_expr : [ lifetime ':' ] "loop" '{' block '}';
```

### Break expressions

```antlr
break_expr : "break" [ lifetime ];
```

### Continue expressions

```antlr
continue_expr : "continue" [ lifetime ];
```

### For expressions

```antlr
for_expr : "for" pat "in" no_struct_literal_expr '{' block '}' ;
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
else_tail : "else" [ if_expr | if_let_expr | '{' block '}' ] ;
```

### While let loops

```antlr
while_let_expr : "while" "let" pat '=' expr '{' block '}' ;
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
procedure_type := 'proc' [ '<' lifetime-list '>' ] '(' arg-list ')'
                  [ ':' bound-list ] [ '->' type ]
lifetime-list := lifetime | lifetime ',' lifetime-list
arg-list := ident ':' type | ident ':' type ',' arg-list
bound-list := bound | bound '+' bound-list
bound := path | lifetime
```

### Object types

**FIXME:** grammar?

### Type parameters

**FIXME:** grammar?

### Self types

**FIXME:** grammar?

## Type kinds

**FIXME:** this this probably not relevant to the grammar...

# Memory and concurrency models

**FIXME:** is this entire chapter relevant here? Or should it all have been covered by some production already?

## Memory model

### Memory allocation and lifetime

### Memory ownership

### Memory slots

### Boxes

## Tasks

### Communication between tasks

### Task lifecycle
