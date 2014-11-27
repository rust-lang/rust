# **This is a work in progress**

% The Rust Grammar

# Introduction

This document is the primary reference for the Rust programming language grammar. It
provides only one kind of material:

  - Chapters that formally define the language grammar and, for each
    construct.

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
| abstract | alignof  | as       | be       | box    |
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
unicode_escape : 'u' hex_digit 4
               | 'U' hex_digit 8 ;

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
array_expr : '[' "mut" ? vec_elems? ']' ;

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

```
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

## Types

Every slot, item and value in a Rust program has a type. The _type_ of a
*value* defines the interpretation of the memory holding it.

Built-in types and type-constructors are tightly integrated into the language,
in nontrivial ways that are not possible to emulate in user-defined types.
User-defined types have limited capabilities.

### Primitive types

The primitive types are the following:

* The "unit" type `()`, having the single "unit" value `()` (occasionally called
  "nil"). [^unittype]
* The boolean type `bool` with values `true` and `false`.
* The machine types.
* The machine-dependent integer and floating-point types.

[^unittype]: The "unit" value `()` is *not* a sentinel "null pointer" value for
    reference slots; the "unit" type is the implicit return type from functions
    otherwise lacking a return type, and can be used in other contexts (such as
    message-sending or type-parametric code) as a zero-size type.]

#### Machine types

The machine types are the following:

* The unsigned word types `u8`, `u16`, `u32` and `u64`, with values drawn from
  the integer intervals [0, 2^8 - 1], [0, 2^16 - 1], [0, 2^32 - 1] and
  [0, 2^64 - 1] respectively.

* The signed two's complement word types `i8`, `i16`, `i32` and `i64`, with
  values drawn from the integer intervals [-(2^(7)), 2^7 - 1],
  [-(2^(15)), 2^15 - 1], [-(2^(31)), 2^31 - 1], [-(2^(63)), 2^63 - 1]
  respectively.

* The IEEE 754-2008 `binary32` and `binary64` floating-point types: `f32` and
  `f64`, respectively.

#### Machine-dependent integer types

The `uint` type is an unsigned integer type with the same number of bits as the
platform's pointer type. It can represent every memory address in the process.

The `int` type is a signed integer type with the same number of bits as the
platform's pointer type. The theoretical upper bound on object and array size
is the maximum `int` value. This ensures that `int` can be used to calculate
differences between pointers into an object or array and can address every byte
within an object along with one byte past the end.

### Textual types

The types `char` and `str` hold textual data.

A value of type `char` is a [Unicode scalar value](
http://www.unicode.org/glossary/#unicode_scalar_value) (ie. a code point that
is not a surrogate), represented as a 32-bit unsigned word in the 0x0000 to
0xD7FF or 0xE000 to 0x10FFFF range. A `[char]` array is effectively an UCS-4 /
UTF-32 string.

A value of type `str` is a Unicode string, represented as an array of 8-bit
unsigned bytes holding a sequence of UTF-8 codepoints. Since `str` is of
unknown size, it is not a _first class_ type, but can only be instantiated
through a pointer type, such as `&str` or `String`.

### Tuple types

A tuple *type* is a heterogeneous product of other types, called the *elements*
of the tuple. It has no nominal name and is instead structurally typed.

Tuple types and values are denoted by listing the types or values of their
elements, respectively, in a parenthesized, comma-separated list.

Because tuple elements don't have a name, they can only be accessed by
pattern-matching.

The members of a tuple are laid out in memory contiguously, in order specified
by the tuple type.

An example of a tuple type and its use:

```
type Pair<'a> = (int, &'a str);
let p: Pair<'static> = (10, "hello");
let (a, b) = p;
assert!(b != "world");
```

### Array, and Slice types

Rust has two different types for a list of items:

* `[T ..N]`, an 'array'
* `&[T]`, a 'slice'.

An array has a fixed size, and can be allocated on either the stack or the
heap.

A slice is a 'view' into an array. It doesn't own the data it points
to, it borrows it.

An example of each kind:

```{rust}
let vec: Vec<int>  = vec![1, 2, 3];
let arr: [int, ..3] = [1, 2, 3];
let s: &[int]      = vec.as_slice();
```

As you can see, the `vec!` macro allows you to create a `Vec<T>` easily. The
`vec!` macro is also part of the standard library, rather than the language.

All in-bounds elements of arrays, and slices are always initialized, and access
to an array or slice is always bounds-checked.

### Structure types

A `struct` *type* is a heterogeneous product of other types, called the
*fields* of the type.[^structtype]

[^structtype]: `struct` types are analogous `struct` types in C,
    the *record* types of the ML family,
    or the *structure* types of the Lisp family.

New instances of a `struct` can be constructed with a [struct
expression](#structure-expressions).

The memory layout of a `struct` is undefined by default to allow for compiler
optimizations like field reordering, but it can be fixed with the
`#[repr(...)]` attribute. In either case, fields may be given in any order in
a corresponding struct *expression*; the resulting `struct` value will always
have the same memory layout.

The fields of a `struct` may be qualified by [visibility
modifiers](#re-exporting-and-visibility), to allow access to data in a
structure outside a module.

A _tuple struct_ type is just like a structure type, except that the fields are
anonymous.

A _unit-like struct_ type is like a structure type, except that it has no
fields. The one value constructed by the associated [structure
expression](#structure-expressions) is the only value that inhabits such a
type.

### Enumerated types

An *enumerated type* is a nominal, heterogeneous disjoint union type, denoted
by the name of an [`enum` item](#enumerations). [^enumtype]

[^enumtype]: The `enum` type is analogous to a `data` constructor declaration in
             ML, or a *pick ADT* in Limbo.

An [`enum` item](#enumerations) declares both the type and a number of *variant
constructors*, each of which is independently named and takes an optional tuple
of arguments.

New instances of an `enum` can be constructed by calling one of the variant
constructors, in a [call expression](#call-expressions).

Any `enum` value consumes as much memory as the largest variant constructor for
its corresponding `enum` type.

Enum types cannot be denoted *structurally* as types, but must be denoted by
named reference to an [`enum` item](#enumerations).

### Recursive types

Nominal types &mdash; [enumerations](#enumerated-types) and
[structures](#structure-types) &mdash; may be recursive. That is, each `enum`
constructor or `struct` field may refer, directly or indirectly, to the
enclosing `enum` or `struct` type itself. Such recursion has restrictions:

* Recursive types must include a nominal type in the recursion
  (not mere [type definitions](#type-definitions),
   or other structural types such as [arrays](#array,-and-slice-types) or [tuples](#tuple-types)).
* A recursive `enum` item must have at least one non-recursive constructor
  (in order to give the recursion a basis case).
* The size of a recursive type must be finite;
  in other words the recursive fields of the type must be [pointer types](#pointer-types).
* Recursive type definitions can cross module boundaries, but not module *visibility* boundaries,
  or crate boundaries (in order to simplify the module system and type checker).

An example of a *recursive* type and its use:

```
enum List<T> {
  Nil,
  Cons(T, Box<List<T>>)
}

let a: List<int> = List::Cons(7, box List::Cons(13, box List::Nil));
```

### Pointer types

All pointers in Rust are explicit first-class values. They can be copied,
stored into data structures, and returned from functions. There are two
varieties of pointer in Rust:

* References (`&`)
  : These point to memory _owned by some other value_.
    A reference type is written `&type` for some lifetime-variable `f`,
    or just `&'a type` when you need an explicit lifetime.
    Copying a reference is a "shallow" operation:
    it involves only copying the pointer itself.
    Releasing a reference typically has no effect on the value it points to,
    with the exception of temporary values, which are released when the last
    reference to them is released.

* Raw pointers (`*`)
  : Raw pointers are pointers without safety or liveness guarantees.
    Raw pointers are written as `*const T` or `*mut T`,
    for example `*const int` means a raw pointer to an integer.
    Copying or dropping a raw pointer has no effect on the lifecycle of any
    other value. Dereferencing a raw pointer or converting it to any other
    pointer type is an [`unsafe` operation](#unsafe-functions).
    Raw pointers are generally discouraged in Rust code;
    they exist to support interoperability with foreign code,
    and writing performance-critical or low-level functions.

The standard library contains additional 'smart pointer' types beyond references
and raw pointers.

### Function types

The function type constructor `fn` forms new function types. A function type
consists of a possibly-empty set of function-type modifiers (such as `unsafe`
or `extern`), a sequence of input types and an output type.

An example of a `fn` type:

```
fn add(x: int, y: int) -> int {
  return x + y;
}

let mut x = add(5,7);

type Binop<'a> = |int,int|: 'a -> int;
let bo: Binop = add;
x = bo(5,7);
```

### Closure types

```{.ebnf .notation}
closure_type := [ 'unsafe' ] [ '<' lifetime-list '>' ] '|' arg-list '|'
                [ ':' bound-list ] [ '->' type ]
procedure_type := 'proc' [ '<' lifetime-list '>' ] '(' arg-list ')'
                  [ ':' bound-list ] [ '->' type ]
lifetime-list := lifetime | lifetime ',' lifetime-list
arg-list := ident ':' type | ident ':' type ',' arg-list
bound-list := bound | bound '+' bound-list
bound := path | lifetime
```

The type of a closure mapping an input of type `A` to an output of type `B` is
`|A| -> B`. A closure with no arguments or return values has type `||`.
Similarly, a procedure mapping `A` to `B` is `proc(A) -> B` and a no-argument
and no-return value closure has type `proc()`.

An example of creating and calling a closure:

```rust
let captured_var = 10i;

let closure_no_args = || println!("captured_var={}", captured_var);

let closure_args = |arg: int| -> int {
  println!("captured_var={}, arg={}", captured_var, arg);
  arg // Note lack of semicolon after 'arg'
};

fn call_closure(c1: ||, c2: |int| -> int) {
  c1();
  c2(2);
}

call_closure(closure_no_args, closure_args);

```

Unlike closures, procedures may only be invoked once, but own their
environment, and are allowed to move out of their environment. Procedures are
allocated on the heap (unlike closures). An example of creating and calling a
procedure:

```rust
let string = "Hello".to_string();

// Creates a new procedure, passing it to the `spawn` function.
spawn(proc() {
  println!("{} world!", string);
});

// the variable `string` has been moved into the previous procedure, so it is
// no longer usable.


// Create an invoke a procedure. Note that the procedure is *moved* when
// invoked, so it cannot be invoked again.
let f = proc(n: int) { n + 22 };
println!("answer: {}", f(20));

```

### Object types

Every trait item (see [traits](#traits)) defines a type with the same name as
the trait. This type is called the _object type_ of the trait. Object types
permit "late binding" of methods, dispatched using _virtual method tables_
("vtables"). Whereas most calls to trait methods are "early bound" (statically
resolved) to specific implementations at compile time, a call to a method on an
object type is only resolved to a vtable entry at compile time. The actual
implementation for each vtable entry can vary on an object-by-object basis.

Given a pointer-typed expression `E` of type `&T` or `Box<T>`, where `T`
implements trait `R`, casting `E` to the corresponding pointer type `&R` or
`Box<R>` results in a value of the _object type_ `R`. This result is
represented as a pair of pointers: the vtable pointer for the `T`
implementation of `R`, and the pointer value of `E`.

An example of an object type:

```
trait Printable {
  fn stringify(&self) -> String;
}

impl Printable for int {
  fn stringify(&self) -> String { self.to_string() }
}

fn print(a: Box<Printable>) {
   println!("{}", a.stringify());
}

fn main() {
   print(box 10i as Box<Printable>);
}
```

In this example, the trait `Printable` occurs as an object type in both the
type signature of `print`, and the cast expression in `main`.

### Type parameters

Within the body of an item that has type parameter declarations, the names of
its type parameters are types:

```ignore
fn map<A: Clone, B: Clone>(f: |A| -> B, xs: &[A]) -> Vec<B> {
    if xs.len() == 0 {
       return vec![];
    }
    let first: B = f(xs[0].clone());
    let mut rest: Vec<B> = map(f, xs.slice(1, xs.len()));
    rest.insert(0, first);
    return rest;
}
```

Here, `first` has type `B`, referring to `map`'s `B` type parameter; and `rest`
has type `Vec<B>`, a vector type with element type `B`.

### Self types

The special type `self` has a meaning within methods inside an impl item. It
refers to the type of the implicit `self` argument. For example, in:

```
trait Printable {
  fn make_string(&self) -> String;
}

impl Printable for String {
    fn make_string(&self) -> String {
        (*self).clone()
    }
}
```

`self` refers to the value of type `String` that is the receiver for a call to
the method `make_string`.

## Type kinds

Types in Rust are categorized into kinds, based on various properties of the
components of the type. The kinds are:

* `Send`
  : Types of this kind can be safely sent between tasks.
    This kind includes scalars, boxes, procs, and
    structural types containing only other owned types.
    All `Send` types are `'static`.
* `Copy`
  : Types of this kind consist of "Plain Old Data"
    which can be copied by simply moving bits.
    All values of this kind can be implicitly copied.
    This kind includes scalars and immutable references,
    as well as structural types containing other `Copy` types.
* `'static`
  : Types of this kind do not contain any references (except for
    references with the `static` lifetime, which are allowed).
    This can be a useful guarantee for code
    that breaks borrowing assumptions
    using [`unsafe` operations](#unsafe-functions).
* `Drop`
  : This is not strictly a kind,
    but its presence interacts with kinds:
    the `Drop` trait provides a single method `drop`
    that takes no parameters,
    and is run when values of the type are dropped.
    Such a method is called a "destructor",
    and are always executed in "top-down" order:
    a value is completely destroyed
    before any of the values it owns run their destructors.
    Only `Send` types can implement `Drop`.

* _Default_
  : Types with destructors, closure environments,
    and various other _non-first-class_ types,
    are not copyable at all.
    Such types can usually only be accessed through pointers,
    or in some cases, moved between mutable locations.

Kinds can be supplied as _bounds_ on type parameters, like traits, in which
case the parameter is constrained to types satisfying that kind.

By default, type parameters do not carry any assumed kind-bounds at all. When
instantiating a type parameter, the kind bounds on the parameter are checked to
be the same or narrower than the kind of the type that it is instantiated with.

Sending operations are not part of the Rust language, but are implemented in
the library. Generic functions that send values bound the kind of these values
to sendable.

# Memory and concurrency models

Rust has a memory model centered around concurrently-executing _tasks_. Thus
its memory model and its concurrency model are best discussed simultaneously,
as parts of each only make sense when considered from the perspective of the
other.

When reading about the memory model, keep in mind that it is partitioned in
order to support tasks; and when reading about tasks, keep in mind that their
isolation and communication mechanisms are only possible due to the ownership
and lifetime semantics of the memory model.

## Memory model

A Rust program's memory consists of a static set of *items*, a set of
[tasks](#tasks) each with its own *stack*, and a *heap*. Immutable portions of
the heap may be shared between tasks, mutable portions may not.

Allocations in the stack consist of *slots*, and allocations in the heap
consist of *boxes*.

### Memory allocation and lifetime

The _items_ of a program are those functions, modules and types that have their
value calculated at compile-time and stored uniquely in the memory image of the
rust process. Items are neither dynamically allocated nor freed.

A task's _stack_ consists of activation frames automatically allocated on entry
to each function as the task executes. A stack allocation is reclaimed when
control leaves the frame containing it.

The _heap_ is a general term that describes boxes.  The lifetime of an
allocation in the heap depends on the lifetime of the box values pointing to
it. Since box values may themselves be passed in and out of frames, or stored
in the heap, heap allocations may outlive the frame they are allocated within.

### Memory ownership

A task owns all memory it can *safely* reach through local variables, as well
as boxes and references.

When a task sends a value that has the `Send` trait to another task, it loses
ownership of the value sent and can no longer refer to it. This is statically
guaranteed by the combined use of "move semantics", and the compiler-checked
_meaning_ of the `Send` trait: it is only instantiated for (transitively)
sendable kinds of data constructor and pointers, never including references.

When a stack frame is exited, its local allocations are all released, and its
references to boxes are dropped.

When a task finishes, its stack is necessarily empty and it therefore has no
references to any boxes; the remainder of its heap is immediately freed.

### Memory slots

A task's stack contains slots.

A _slot_ is a component of a stack frame, either a function parameter, a
[temporary](#lvalues,-rvalues-and-temporaries), or a local variable.

A _local variable_ (or *stack-local* allocation) holds a value directly,
allocated within the stack's memory. The value is a part of the stack frame.

Local variables are immutable unless declared otherwise like: `let mut x = ...`.

Function parameters are immutable unless declared with `mut`. The `mut` keyword
applies only to the following parameter (so `|mut x, y|` and `fn f(mut x:
Box<int>, y: Box<int>)` declare one mutable variable `x` and one immutable
variable `y`).

Methods that take either `self` or `Box<Self>` can optionally place them in a
mutable slot by prefixing them with `mut` (similar to regular arguments):

```
trait Changer {
    fn change(mut self) -> Self;
    fn modify(mut self: Box<Self>) -> Box<Self>;
}
```

Local variables are not initialized when allocated; the entire frame worth of
local variables are allocated at once, on frame-entry, in an uninitialized
state. Subsequent statements within a function may or may not initialize the
local variables. Local variables can be used only after they have been
initialized; this is enforced by the compiler.

### Boxes

A _box_ is a reference to a heap allocation holding another value, which is
constructed by the prefix operator `box`. When the standard library is in use,
the type of a box is `std::owned::Box<T>`.

An example of a box type and value:

```
let x: Box<int> = box 10;
```

Box values exist in 1:1 correspondence with their heap allocation, copying a
box value makes a shallow copy of the pointer. Rust will consider a shallow
copy of a box to move ownership of the value. After a value has been moved,
the source location cannot be used unless it is reinitialized.

```
let x: Box<int> = box 10;
let y = x;
// attempting to use `x` will result in an error here
```

## Tasks

An executing Rust program consists of a tree of tasks. A Rust _task_ consists
of an entry function, a stack, a set of outgoing communication channels and
incoming communication ports, and ownership of some portion of the heap of a
single operating-system process.

### Communication between tasks

Rust tasks are isolated and generally unable to interfere with one another's
memory directly, except through [`unsafe` code](#unsafe-functions).  All
contact between tasks is mediated by safe forms of ownership transfer, and data
races on memory are prohibited by the type system.

When you wish to send data between tasks, the values are restricted to the
[`Send` type-kind](#type-kinds). Restricting communication interfaces to this
kind ensures that no references move between tasks. Thus access to an entire
data structure can be mediated through its owning "root" value; no further
locking or copying is required to avoid data races within the substructure of
such a value.

### Task lifecycle

The _lifecycle_ of a task consists of a finite set of states and events that
cause transitions between the states. The lifecycle states of a task are:

* running
* blocked
* panicked
* dead

A task begins its lifecycle &mdash; once it has been spawned &mdash; in the
*running* state. In this state it executes the statements of its entry
function, and any functions called by the entry function.

A task may transition from the *running* state to the *blocked* state any time
it makes a blocking communication call. When the call can be completed &mdash;
when a message arrives at a sender, or a buffer opens to receive a message
&mdash; then the blocked task will unblock and transition back to *running*.

A task may transition to the *panicked* state at any time, due being killed by
some external event or internally, from the evaluation of a `panic!()` macro.
Once *panicking*, a task unwinds its stack and transitions to the *dead* state.
Unwinding the stack of a task is done by the task itself, on its own control
stack. If a value with a destructor is freed during unwinding, the code for the
destructor is run, also on the task's control stack. Running the destructor
code causes a temporary transition to a *running* state, and allows the
destructor code to cause any subsequent state transitions. The original task
of unwinding and panicking thereby may suspend temporarily, and may involve
(recursive) unwinding of the stack of a failed destructor. Nonetheless, the
outermost unwinding activity will continue until the stack is unwound and the
task transitions to the *dead* state. There is no way to "recover" from task
panics. Once a task has temporarily suspended its unwinding in the *panicking*
state, a panic occurring from within this destructor results in *hard* panic.
A hard panic currently results in the process aborting.

A task in the *dead* state cannot transition to other states; it exists only to
have its termination status inspected by other tasks, and/or to await
reclamation when the last reference to it drops.

# Runtime services, linkage and debugging

The Rust _runtime_ is a relatively compact collection of Rust code that
provides fundamental services and datatypes to all Rust tasks at run-time. It
is smaller and simpler than many modern language runtimes. It is tightly
integrated into the language's execution model of memory, tasks, communication
and logging.

### Memory allocation

The runtime memory-management system is based on a _service-provider
interface_, through which the runtime requests blocks of memory from its
environment and releases them back to its environment when they are no longer
needed. The default implementation of the service-provider interface consists
of the C runtime functions `malloc` and `free`.

The runtime memory-management system, in turn, supplies Rust tasks with
facilities for allocating releasing stacks, as well as allocating and freeing
heap data.

### Built in types

The runtime provides C and Rust code to assist with various built-in types,
such as arrays, strings, and the low level communication system (ports,
channels, tasks).

Support for other built-in types such as simple types, tuples and enums is
open-coded by the Rust compiler.

### Task scheduling and communication

The runtime provides code to manage inter-task communication. This includes
the system of task-lifecycle state transitions depending on the contents of
queues, as well as code to copy values between queues and their recipients and
to serialize values for transmission over operating-system inter-process
communication facilities.

### Linkage

The Rust compiler supports various methods to link crates together both
statically and dynamically. This section will explore the various methods to
link Rust crates together, and more information about native libraries can be
found in the [ffi guide][ffi].

In one session of compilation, the compiler can generate multiple artifacts
through the usage of either command line flags or the `crate_type` attribute.
If one or more command line flag is specified, all `crate_type` attributes will
be ignored in favor of only building the artifacts specified by command line.

* `--crate-type=bin`, `#[crate_type = "bin"]` - A runnable executable will be
  produced. This requires that there is a `main` function in the crate which
  will be run when the program begins executing. This will link in all Rust and
  native dependencies, producing a distributable binary.

* `--crate-type=lib`, `#[crate_type = "lib"]` - A Rust library will be produced.
  This is an ambiguous concept as to what exactly is produced because a library
  can manifest itself in several forms. The purpose of this generic `lib` option
  is to generate the "compiler recommended" style of library. The output library
  will always be usable by rustc, but the actual type of library may change from
  time-to-time. The remaining output types are all different flavors of
  libraries, and the `lib` type can be seen as an alias for one of them (but the
  actual one is compiler-defined).

* `--crate-type=dylib`, `#[crate_type = "dylib"]` - A dynamic Rust library will
  be produced. This is different from the `lib` output type in that this forces
  dynamic library generation. The resulting dynamic library can be used as a
  dependency for other libraries and/or executables. This output type will
  create `*.so` files on linux, `*.dylib` files on osx, and `*.dll` files on
  windows.

* `--crate-type=staticlib`, `#[crate_type = "staticlib"]` - A static system
  library will be produced. This is different from other library outputs in that
  the Rust compiler will never attempt to link to `staticlib` outputs. The
  purpose of this output type is to create a static library containing all of
  the local crate's code along with all upstream dependencies. The static
  library is actually a `*.a` archive on linux and osx and a `*.lib` file on
  windows. This format is recommended for use in situations such as linking
  Rust code into an existing non-Rust application because it will not have
  dynamic dependencies on other Rust code.

* `--crate-type=rlib`, `#[crate_type = "rlib"]` - A "Rust library" file will be
  produced. This is used as an intermediate artifact and can be thought of as a
  "static Rust library". These `rlib` files, unlike `staticlib` files, are
  interpreted by the Rust compiler in future linkage. This essentially means
  that `rustc` will look for metadata in `rlib` files like it looks for metadata
  in dynamic libraries. This form of output is used to produce statically linked
  executables as well as `staticlib` outputs.

Note that these outputs are stackable in the sense that if multiple are
specified, then the compiler will produce each form of output at once without
having to recompile. However, this only applies for outputs specified by the
same method. If only `crate_type` attributes are specified, then they will all
be built, but if one or more `--crate-type` command line flag is specified,
then only those outputs will be built.

With all these different kinds of outputs, if crate A depends on crate B, then
the compiler could find B in various different forms throughout the system. The
only forms looked for by the compiler, however, are the `rlib` format and the
dynamic library format. With these two options for a dependent library, the
compiler must at some point make a choice between these two formats. With this
in mind, the compiler follows these rules when determining what format of
dependencies will be used:

1. If a static library is being produced, all upstream dependencies are
   required to be available in `rlib` formats. This requirement stems from the
   reason that a dynamic library cannot be converted into a static format.

   Note that it is impossible to link in native dynamic dependencies to a static
   library, and in this case warnings will be printed about all unlinked native
   dynamic dependencies.

2. If an `rlib` file is being produced, then there are no restrictions on what
   format the upstream dependencies are available in. It is simply required that
   all upstream dependencies be available for reading metadata from.

   The reason for this is that `rlib` files do not contain any of their upstream
   dependencies. It wouldn't be very efficient for all `rlib` files to contain a
   copy of `libstd.rlib`!

3. If an executable is being produced and the `-C prefer-dynamic` flag is not
   specified, then dependencies are first attempted to be found in the `rlib`
   format. If some dependencies are not available in an rlib format, then
   dynamic linking is attempted (see below).

4. If a dynamic library or an executable that is being dynamically linked is
   being produced, then the compiler will attempt to reconcile the available
   dependencies in either the rlib or dylib format to create a final product.

   A major goal of the compiler is to ensure that a library never appears more
   than once in any artifact. For example, if dynamic libraries B and C were
   each statically linked to library A, then a crate could not link to B and C
   together because there would be two copies of A. The compiler allows mixing
   the rlib and dylib formats, but this restriction must be satisfied.

   The compiler currently implements no method of hinting what format a library
   should be linked with. When dynamically linking, the compiler will attempt to
   maximize dynamic dependencies while still allowing some dependencies to be
   linked in via an rlib.

   For most situations, having all libraries available as a dylib is recommended
   if dynamically linking. For other situations, the compiler will emit a
   warning if it is unable to determine which formats to link each library with.

In general, `--crate-type=bin` or `--crate-type=lib` should be sufficient for
all compilation needs, and the other options are just available if more
fine-grained control is desired over the output format of a Rust crate.

# Appendix: Rationales and design tradeoffs

*TODO*.

# Appendix: Influences and further references

## Influences

>  The essential problem that must be solved in making a fault-tolerant
>  software system is therefore that of fault-isolation. Different programmers
>  will write different modules, some modules will be correct, others will have
>  errors. We do not want the errors in one module to adversely affect the
>  behaviour of a module which does not have any errors.
>
>  &mdash; Joe Armstrong

>  In our approach, all data is private to some process, and processes can
>  only communicate through communications channels. *Security*, as used
>  in this paper, is the property which guarantees that processes in a system
>  cannot affect each other except by explicit communication.
>
>  When security is absent, nothing which can be proven about a single module
>  in isolation can be guaranteed to hold when that module is embedded in a
>  system [...]
>
>  &mdash; Robert Strom and Shaula Yemini

>  Concurrent and applicative programming complement each other. The
>  ability to send messages on channels provides I/O without side effects,
>  while the avoidance of shared data helps keep concurrent processes from
>  colliding.
>
>  &mdash; Rob Pike

Rust is not a particularly original language. It may however appear unusual by
contemporary standards, as its design elements are drawn from a number of
"historical" languages that have, with a few exceptions, fallen out of favour.
Five prominent lineages contribute the most, though their influences have come
and gone during the course of Rust's development:

* The NIL (1981) and Hermes (1990) family. These languages were developed by
  Robert Strom, Shaula Yemini, David Bacon and others in their group at IBM
  Watson Research Center (Yorktown Heights, NY, USA).

* The Erlang (1987) language, developed by Joe Armstrong, Robert Virding, Claes
  Wikstr&ouml;m, Mike Williams and others in their group at the Ericsson Computer
  Science Laboratory (&Auml;lvsj&ouml;, Stockholm, Sweden) .

* The Sather (1990) language, developed by Stephen Omohundro, Chu-Cheow Lim,
  Heinz Schmidt and others in their group at The International Computer
  Science Institute of the University of California, Berkeley (Berkeley, CA,
  USA).

* The Newsqueak (1988), Alef (1995), and Limbo (1996) family. These
  languages were developed by Rob Pike, Phil Winterbottom, Sean Dorward and
  others in their group at Bell Labs Computing Sciences Research Center
  (Murray Hill, NJ, USA).

* The Napier (1985) and Napier88 (1988) family. These languages were
  developed by Malcolm Atkinson, Ron Morrison and others in their group at
  the University of St. Andrews (St. Andrews, Fife, UK).

Additional specific influences can be seen from the following languages:

* The structural algebraic types and compilation manager of SML.
* The attribute and assembly systems of C#.
* The references and deterministic destructor system of C++.
* The memory region systems of the ML Kit and Cyclone.
* The typeclass system of Haskell.
* The lexical identifier rule of Python.
* The block syntax of Ruby.

[ffi]: guide-ffi.html
[plugin]: guide-plugin.html
