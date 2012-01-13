% Rust Reference Manual
% January 2012

# Introduction

This document is the reference manual for the Rust programming language. It
provides three kinds of material:

  - Chapters that formally define the language grammar and, for each
    construct, informally describe its semantics and give examples of its
    use.
  - Chapters that informally describe the memory model, concurrency model,
    runtime services, linkage model and debugging facilities.
  - Appendix chapters providing rationale and references to languages that
    influenced the design.

This document does not serve as a tutorial introduction to the
language. Background familiarity with the language is assumed. A separate
tutorial document is available at <http://www.rust-lang.org/doc/tutorial>
to help acquire such background familiarity.

This document also does not serve as a reference to the core or standard
libraries included in the language distribution. Those libraries are
documented separately by extracting documentation attributes from their
source code. Formatted documentation can be found at the following
locations:

  - Core library: <http://doc.rust-lang.org/doc/core>
  - Standard library: <http://doc.rust-lang.org/doc/std>

## Disclaimer

Rust is a work in progress. The language continues to evolve as the design
shifts and is fleshed out in working code. Certain parts work, certain parts
do not, certain parts will be removed or changed.

This manual is a snapshot written in the present tense. All features
described exist in working code, but some are quite primitive or remain to
be further modified by planned work. Some may be temporary. It is a
*draft*, and we ask that you not take anything you read here as final.

If you have suggestions to make, please try to focus them on *reductions* to
the language: possible features that can be combined or omitted. We aim to
keep the size and complexity of the language under control.

# Notation

Rust's grammar is defined over Unicode codepoints, each conventionally
denoted `U+XXXX`, for 4 or more hexadecimal digits `X`. _Most_ of Rust's
grammar is confined to the ASCII range of Unicode, and is described in this
document by a dialect of Extended Backus-Naur Form (EBNF), specifically a
dialect of EBNF supported by common automated LL(k) parsing tools such as
`llgen`, rather than the dialect given in ISO 14977. The dialect can be
defined self-referentially as follows:

~~~~~~~~ {.ebnf .notation}

grammar : rule + ;
rule    : nonterminal ':' productionrule ';' ;
productionrule : production [ '|' production ] * ;
production : term * ;
term : element repeats ;
element : LITERAL | IDENTIFIER | '[' productionrule ']' ;
repeats : [ '*' | '+' ] NUMBER ? | NUMBER ? | '?' ;

~~~~~~~~

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

The grammar for Rust given in this document is extracted and verified as
LL(1) by an automated grammar-analysis tool, and further tested against the
Rust sources. The generated parser is currently *not* the one used by the
Rust compiler itself, but in the future we hope to relate the two together
more precisely. As of this writing they are only related by testing against
existing source code.

## Unicode productions

A small number of productions in Rust's grammar permit Unicode codepoints
ouside the ASCII range; these productions are defined in terms of character
properties given by the Unicode standard, rather than ASCII-range
codepoints. These are given in the section [Special Unicode
Productions](#special-unicode-productions).

## String table productions

Some rules in the grammar -- notably [operators](#operators),
[keywords](#keywords) and [reserved words](#reserved-words) -- are given in a
simplified form: as a listing of a table of unquoted, printable
whitespace-separated strings. These cases form a subset of the rules regarding
the [token](#tokens) rule, and are assumed to be the result of a
lexical-analysis phase feeding the parser, driven by a DFA, operating over the
disjunction of all such string table entries.

When such a string enclosed in double-quotes (`"`) occurs inside the
grammar, it is an implicit reference to a single member of such a string table
production. See [tokens](#tokens) for more information.


# Lexical structure

## Input format

Rust input is interpreted in as a sequence of Unicode codepoints encoded in
UTF-8. No normalization is performed during input processing. Most Rust
grammar rules are defined in terms of printable ASCII-range codepoints, but
a small number are defined in terms of Unicode properties or explicit
codepoint lists. ^[Surrogate definitions for the special Unicode productions
are provided to the grammar verifier, restricted to ASCII range, when
verifying the grammar in this document.]

## Special Unicode Productions

The following productions in the Rust grammar are defined in terms of
Unicode properties: `ident`, `non_null`, `non_star`, `non_eol`, `non_slash`,
`non_single_quote` and `non_double_quote`.

### Identifier

The `ident` production is any nonempty Unicode string of the following form:

   - The first character has property `XID_start`
   - The remaining characters have property `XID_continue`

that does _not_ occur in the set of [keywords](#keywords) or [reserved
words](#reserved-words).

Note: `XID_start` and `XID_continue` as character properties cover the
character ranges used to form the more familiar C and Java language-family
identifiers.

### Delimiter-restricted productions

Some productions are defined by exclusion of particular Unicode characters:

  - `non_null` is any single Unicode character aside from `U+0000` (null)
  - `non_eol` is `non_null` restricted to exclude `U+000A` (`'\n'`)
  - `non_star` is `non_null` restricted to exclude `U+002A` (`*`)
  - `non_slash` is `non_null` restricted to exclude `U+002F` (`/`)
  - `non_single_quote` is `non_null` restricted to exclude `U+0027`  (`'`)
  - `non_double_quote` is `non_null` restricted to exclude `U+0022` (`"`)

## Comments

~~~~~~~~ {.ebnf .gram}
comment : block_comment | line_comment ;
block_comment : "/*" block_comment_body * "*/" ;
block_comment_body : block_comment | non_star * | '*' non_slash ;
line_comment : "//" non_eol * ;
~~~~~~~~

Comments in Rust code follow the general C++ style of line and block-comment
forms, with proper nesting of block-comment delimiters. Comments are
interpreted as a form of whitespace.

## Whitespace

~~~~~~~~ {.ebnf .gram}
whitespace_char : '\x20' | '\x09' | '\x0a' | '\x0d' ;
whitespace : [ whitespace_char | comment ] + ;
~~~~~~~~

The `whitespace_char` production is any nonempty Unicode string consisting of any
of the following Unicode characters: `U+0020` (space, `' '`), `U+0009` (tab,
`'\t'`), `U+000A` (LF, `'\n'`), `U+000D` (CR, `'\r'`).

Rust is a "free-form" language, meaning that all forms of whitespace serve
only to separate _tokens_ in the grammar, and have no semantic meaning.

A Rust program has identical meaning if each whitespace element is replaced
with any other legal whitespace element, such as a single space character.

## Tokens

~~~~~~~~ {.ebnf .gram}
simple_token : keyword | reserved | unop | binop ; 
token : simple_token | ident | immediate | symbol | whitespace token ;
~~~~~~~~

Tokens are primitive productions in the grammar defined by regular
(non-recursive) languages. "Simple" tokens are given in [string table
production](#string-table-productions) form, and occur in the rest of the
grammar as double-quoted strings. Other tokens have exact rules given.

### Keywords

The keywords in [crate files](#crate-files) are the following strings:

~~~~~~~~ {.keyword}
import export use mod dir
~~~~~~~~

The keywords in [source files](#source-files) are the following strings:

~~~~~~~~ {.keyword}
alt any as assert
be bind block bool break
char check claim const cont
do
else export
f32 f64 fail false float fn for
i16 i32 i64 i8 if import in int
let log
mod mutable
native note
obj  
prove pure
resource ret
self str syntax
tag true type
u16 u32 u64 u8 uint unchecked unsafe use
vec
while with
~~~~~~~~

Any of these have special meaning in their respective grammars, and are
excluded from the `ident` rule.

### Reserved words

The reserved words are the following strings:

~~~~~~~~ {.reserved}
m32 m64 m128
f80 f16 f128
class trait
~~~~~~~~

Any of these may have special meaning in future versions of the language, do
are excluded from the `ident` rule.

### Immediates

Immediates are a subset of all possible literals: those that are defined as
single tokens, rather than sequences of tokens.

An immediate is a form of [constant expression](#constant-expression), so is
evaluated (primarily) at compile time.

~~~~~~~~ {.ebnf .gram}
immediate : string_lit | char_lit | num_lit ;
~~~~~~~~

#### Character and string literals

~~~~~~~~ {.ebnf .gram}
char_lit : '\x27' char_body '\x27' ;
string_lit : '"' string_body * '"' ;

char_body : non_single_quote
          | '\x5c' [ '\x27' | common_escape ] ;

string_body : non_double_quote
            | '\x5c' [ '\x22' | common_escape ] ;

common_escape : '\x5c'
              | 'n' | 'r' | 't'
              | 'x' hex_digit 2
              | 'u' hex_digit 4
              | 'U' hex_digit 8 ;

hex_digit : 'a' | 'b' | 'c' | 'd' | 'e' | 'f'
          | 'A' | 'B' | 'C' | 'D' | 'E' | 'F'
          | dec_digit ;
dec_digit : '0' | nonzero_dec ;
nonzero_dec: '1' | '2' | '3' | '4'
           | '5' | '6' | '7' | '8' | '9' ;
~~~~~~~~

A _character literal_ is a single Unicode character enclosed within two
`U+0027` (single-quote) characters, with the exception of `U+0027` itself,
which must be _escaped_ by a preceding U+005C character (`\`).

A _string literal_ is a sequence of any Unicode characters enclosed within
two `U+0022` (double-quote) characters, with the exception of `U+0022`
itself, which must be _escaped_ by a preceding `U+005C` character (`\`).

Some additional _escapes_ are available in either character or string
literals. An escape starts with a `U+005C` (`\`) and continues with one of
the following forms:

  * An _8-bit codepoint escape_ escape starts with `U+0078` (`x`) and is
    followed by exactly two _hex digits_. It denotes the Unicode codepoint
    equal to the provided hex value.
  * A _16-bit codepoint escape_ starts with `U+0075` (`u`) and is followed
    by exactly four _hex digits_. It denotes the Unicode codepoint equal to
    the provided hex value.
  * A _32-bit codepoint escape_ starts with `U+0055` (`U`) and is followed
    by exactly eight _hex digits_. It denotes the Unicode codepoint equal to
    the provided hex value.
  * A _whitespace escape_ is one of the characters `U+006E` (`n`), `U+0072`
    (`r`), or `U+0074` (`t`), denoting the unicode values `U+000A` (LF),
    `U+000D` (CR) or `U+0009` (HT) respectively.
  * The _backslash escape_ is the character U+005C (`\`) which must be
    escaped in order to denote *itself*.

#### Number literals

~~~~~~~~ {.ebnf .gram}

num_lit : nonzero_dec [ dec_digit | '_' ] * num_suffix ?
        | '0' [       [ dec_digit | '_' ] + num_suffix ?
              | 'b'   [ '1' | '0' | '_' ] + int_suffix ?
              | 'x'   [ hex_digit | '-' ] + int_suffix ? ] ;

num_suffix : int_suffix | float_suffix ;

int_suffix : 'u' int_suffix_size ?
           | 'i' int_suffix_size ;
int_suffix_size : [ '8' | '1' '6' | '3' '2' | '6' '4' ] ;

float_suffix : [ exponent | '.' dec_lit exponent ? ] float_suffix_ty ? ;
float_suffix_ty : 'f' [ '3' '2' | '6' '4' ] ;
exponent : ['E' | 'e'] ['-' | '+' ] ? dec_lit ;
dec_lit : [ dec_digit | '_' ] + ;
~~~~~~~~

A _number literal_ is either an _integer literal_ or a _floating-point
literal_. The grammar for recognizing the two kinds of literals is mixed
as they are differentiated by suffixes.

##### Integer literals

An _integer literal_ has one of three forms:

  * A _decimal literal_ starts with a *decimal digit* and continues with any
    mixture of *decimal digits* and _underscores_.
  * A _hex literal_ starts with the character sequence `U+0030` `U+0078`
    (`0x`) and continues as any mixture hex digits and underscores.
  * A _binary literal_ starts with the character sequence `U+0030` `U+0062`
    (`0b`) and continues as any mixture binary digits and underscores.

By default, an integer literal is of type `int`. An integer literal may be
followed (immediately, without any spaces) by an _integer suffix_, which
changes the type of the literal. There are two kinds of integer literal
suffix:

  * The `u` suffix gives the literal type `uint`.
  * Each of the signed and unsigned machine types `u8`, `i8`,
    `u16`, `i16`, `u32`, `i32`, `u64` and `i64`
    give the literal the corresponding machine type.


Examples of integer literals of various forms:

~~~~
123;                               // type int
123u;                              // type uint
123_u;                             // type uint
0xff00;                            // type int
0xff_u8;                           // type u8
0b1111_1111_1001_0000_i32;         // type i32
~~~~

##### Floating-point literals

A _floating-point literal_ has one of two forms:

* Two _decimal literals_ separated by a period
  character `U+002E` (`.`), with an optional _exponent_ trailing after the
  second decimal literal.
* A single _decimal literal_ followed by an _exponent_.

By default, a floating-point literal is of type `float`. A floating-point
literal may be followed (immediately, without any spaces) by a
_floating-point suffix_, which changes the type of the literal. There are
only two floating-point suffixes: `f32` and `f64`. Each of these gives the
floating point literal the associated type, rather than `float`.

A set of suffixes are also reserved to accommodate literal support for
types corresponding to reserved tokens. The reserved suffixes are `f16`,
`f80`, `f128`, `m`, `m32`, `m64` and `m128`.

Examples of floating-point literals of various forms:

~~~~
123.0;                             // type float
0.1;                               // type float
0.1f32;                            // type f32
12E+99_f64;                        // type f64
~~~~

### Symbols

~~~~~~~~ {.ebnf .gram}
symbol : "::" "->"
       | '#' | '[' | ']' | '(' | ')' | '{' | '}'
       | ',' | ';' ;
~~~~~~~~

Symbols are a general class of printable [token](#tokens) that play structural
roles in a variety of grammar productions. They are catalogued here for
completeness as the set of remaining miscellaneous printable token that do not
otherwise appear as [operators](#operators), [keywords](#keywords) or [reserved
words](#reserved-words).


## Paths

~~~~~~~~ {.ebnf .gram}

expr_path : ident [ "::" expr_path_tail ] + ;
expr_path_tail : '<' type_expr [ ',' type_expr ] + '>'
               | expr_path ;

type_path : ident [ type_path_tail ] + ;
type_path_tail : '<' type_expr [ ',' type_expr ] + '>'
               | "::" type_path ;

~~~~~~~~

A _path_ is a sequence of one or more path components _logically_ separated by
a namespace qualifier (`::`). If a path consists of only one component, it
may refer to either an [item](#items) or a (variable)[#variables) in a local
control scope. If a path has multiple components, it refers to an item.

Every item has a _canonical path_ within its [crate](#crates), but the path
naming an item is only meaningful within a given crate. There is no global
namespace across crates; an item's canonical path merely identifies it within
the crate.

Two examples of simple paths consisting of only identifier components:

~~~~
x;
x::y::z;
~~~~

Path components are usually [identifiers](#identifiers), but the trailing
component of a path may be an angle-bracket enclosed list of [type
arguments](type-arguments). In [expression](#expressions) context, the type
argument list is given after a final (`::`) namespace qualifier in order to
disambiguate it from a relational expression involving the less-than symbol
(`<`). In [type expression](#type-expressions) context, the final namespace
qualifier is omitted.

Two examples of paths with type arguments:

~~~~
type t = map::hashtbl<int,str>;  // Type arguments used in a type expression
let x = id::<int>(10);           // Type arguments used in a call expression
~~~~


# Crates and source files

Rust is a *compiled* language. Its semantics are divided along a
*phase distinction* between compile-time and run-time. Those semantic
rules that have a *static interpretation* govern the success or failure
of compilation. A program that fails to compile due to violation of a
compile-time rule has no defined semantics at run-time; the compiler should
halt with an error report, and produce no executable artifact.

The compilation model centres on artifacts called _crates_. Each compilation
is directed towards a single crate in source form, and if successful
produces a single crate in binary form, either an executable or a library.

A _crate_ is a unit of compilation and linking, as well as versioning,
distribution and runtime loading. A crate contains a _tree_ of nested
[module](#modules) scopes. The top-level of this tree is a module that is
anonymous -- from the point of view of paths within the module -- and any item
within a crate has a canonical [module path](#paths) denoting its location
within the crate's module tree.

Crates are provided to the Rust compiler through two kinds of file:

  - _crate files_, that end in `.rc` and each define a `crate`.
  - _source files_, that end in `.rs` and each define a `module`.

The Rust compiler is always invoked with a single input file, and always
produces a single output crate.

When the Rust compiler is invoked with a crate file, it reads the _explicit_
definition of the crate it's compiling from that file, and populates the
crate with modules derived from all the source files referenced by the
crate, reading and processing all the referenced modules at once.

When the Rust compiler is invoked with a source file, it creates an
_implicit_ crate and treats the source file and though it was referenced as
the sole module populating this implicit crate. The module name is derived
from the source file name, with the `.rs` extension removed.

## Crate files

~~~~~~~~ {.ebnf .gram}
crate : attribute [ ';' | attribute* directive ] 
      | directive ;
directive : view_item | dir_directive | source_directive ;
~~~~~~~~

A crate file contains a crate definition, for which the production above
defines the grammar. It is a declarative grammar that guides the compiler in
assembling a crate from component source files.^[A crate is somewhat
analogous to an *assembly* in the ECMA-335 CLI model, a *library* in the
SML/NJ Compilation Manager, a *unit* in the Owens and Flatt module system,
or a *configuration* in Mesa.] A crate file describes:

* [Attributes](#attributes) about the crate, such as author, name, version,
  and copyright. These are used for linking, versioning and distributing
  crates.
* The source-file and directory modules that make up the crate.
* Any `use`, `import` or `export` [view items](#view-items) that apply to the
  anonymous module at the top-level of the crate's module tree.

An example of a crate file:

~~~~~~~~
// Linkage attributes
#[ link(name = "projx"
        vers = "2.5",
        uuid = "9cccc5d5-aceb-4af5-8285-811211826b82") ];

// Additional metadata attributes
#[ desc = "Project X",
   license = "BSD" ];
   author = "Jane Doe" ];

// Import a module.
use std (ver = "1.0");

// Define some modules.
#[path = "foo.rs"]
mod foo;
mod bar {
    #[path =  "quux.rs"]
    mod quux;
}
~~~~~~~~

### Dir directives

A `dir_directive` forms a module in the module tree making up the crate, as
well as implicitly relating that module to a directory in the filesystem
containing source files and/or further subdirectories. The filesystem
directory associated with a `dir_directive` module can either be explicit,
or if omitted, is implicitly the same name as the module.

A `source_directive` references a source file, either explicitly or
implicitly by combining the module name with the file extension `.rs`.  The
module contained in that source file is bound to the module path formed by
the `dir_directive` modules containing the `source_directive`.

## Source file

A source file contains a `module`, that is, a sequence of zero-or-more
`item` definitions. Each source file is an implicit module, the name and
location of which -- in the module tree of the current crate -- is defined
from outside the source file: either by an explicit `source_directive` in
a referencing crate file, or by the filename of the source file itself.


# Items and attributes


### Attributes

~~~~~~~~{.ebnf .gram}
attribute : '#' '[' attr_list ']' ;
attr_list : attr [ ',' attr_list ]*
attr : ident [ '=' literal
             | '(' attr_list ')' ] ? ;
~~~~~~~~

Static entities in Rust -- crates, modules and items -- may have _attributes_
applied to them. ^[Attributes in Rust are modeled on Attributes in ECMA-335,
C#] An attribute is a general, free-form piece of metadata that is interpreted
according to name, convention, and language and compiler version.  Attributes
may appear as any of:

* A single identifier, the attribute name
* An identifier followed by the equals sign '=' and a literal, providing a key/value pair
* An identifier followed by a parenthesized list of sub-attribute arguments

Attributes are applied to an entity by placing them within a hash-list
(`#[...]`) as either a prefix to the entity or as a semicolon-delimited
declaration within the entity body.

An example of attributes:

~~~~~~~~
// A function marked as a unit test
#[test]
fn test_foo() {
  ...
}

// General metadata applied to the enclosing module or crate.
#[license = "BSD"];

// A conditionally-compiled module
#[cfg(target_os="linux")]
mod bar {
  ...
}

// A documentation attribute
#[doc = "Add two numbers together."
fn add(x: int, y: int) { x + y }
~~~~~~~~

In future versions of Rust, user-provided extensions to the compiler will be
able to interpret attributes. When this facility is provided, a distinction
will be made between language-reserved and user-available attributes.

At present, only the Rust compiler interprets attributes, so all attribute
names are effectively reserved. Some significant attributes include:

* The `doc` attribute, for documenting code where it's written.
* The `cfg` attribute, for conditional-compilation by build-configuration.
* The `link` attribute, for describing linkage metadata for a crate.
* The `test` attribute, for marking functions as unit tests.

Other attributes may be added or removed during development of the language.


# Statements and expressions

## Call expressions

~~~~~~~~ {.abnf .gram}
expr_list : [ expr [ ',' expr ]* ] ? ;
paren_expr_list : '(' expr_list ')' ;
call_expr : expr paren_expr_list ;
~~~~~~~~

## Operators

### Unary operators

~~~~~~~~ {.unop}
+ - * ! @ ~
~~~~~~~~

### Binary operators

~~~~~~~~ {.binop}
.
+ - * / %
& | ^
|| &&
< <= == >= >
<< >> >>>
<- <-> = += -= *= /= %= &= |= ^= <<= >>= >>>=
~~~~~~~~


## Syntax extensions

~~~~~~~~ {.abnf .gram}
syntax_ext_expr : '#' ident paren_expr_list ? brace_match ? ;
~~~~~~~~

Rust provides a notation for _syntax extension_. The notation for invoking
a syntax extension is a marked syntactic form that can appear as an expression
in the body of a Rust program.

After parsing, a syntax-extension invocation is expanded into a Rust
expression. The name of the extension determines the translation performed. In
future versions of Rust, user-provided syntax extensions aside from macros
will be provided via external crates.

At present, only a set of built-in syntax extensions, as well as macros
introduced inline in source code using the `macro` extension, may be used. The
current built-in syntax extensions are:


* `fmt` expands into code to produce a formatted string, similar to 
      `printf` from C.
* `env` expands into a string literal containing the value of that
      environment variable at compile-time.
* `concat_idents` expands into an identifier which is the 
      concatenation of its arguments.
* `ident_to_str` expands into a string literal containing the name of
      its argument (which must be a literal).
* `log_syntax` causes the compiler to pretty-print its arguments.


Finally, `macro` is used to define a new macro. A macro can abstract over
second-class Rust concepts that are present in syntax. The arguments to
`macro` are pairs (two-element vectors). The pairs consist of an invocation
and the syntax to expand into. An example:

~~~~~~~~
#macro([#apply[fn, [args, ...]], fn(args, ...)]);
~~~~~~~~

In this case, the invocation `#apply[sum, 5, 8, 6]` expands to
`sum(5,8,6)`. If `...` follows an expression (which need not be as
simple as a single identifier) in the input syntax, the matcher will expect an
arbitrary number of occurrences of the thing preceding it, and bind syntax to
the identifiers it contains. If it follows an expression in the output syntax,
it will transcribe that expression repeatedly, according to the identifiers
(bound to syntax) that it contains.

The behaviour of `...` is known as Macro By Example. It allows you to
write a macro with arbitrary repetition by specifying only one case of that
repetition, and following it by `...`, both where the repeated input is
matched, and where the repeated output must be transcribed. A more
sophisticated example:


~~~~~~~~
#macro([#zip_literals[[x, ...], [y, ...]), [[x, y], ...]]);
#macro([#unzip_literals[[x, y], ...], [[x, ...], [y, ...]]]);
~~~~~~~~

In this case, `#zip_literals[[1,2,3], [1,2,3]]` expands to
`[[1,1],[2,2],[3,3]]`, and `#unzip_literals[[1,1], [2,2], [3,3]]`
expands to `[[1,2,3],[1,2,3]]`.

Macro expansion takes place outside-in: that is,
`#unzip_literals[#zip_literals[[1,2,3],[1,2,3]]]` will fail because
`unzip_literals` expects a list, not a macro invocation, as an argument.

The macro system currently has some limitations. It's not possible to
destructure anything other than vector literals (therefore, the arguments to
complicated macros will tend to be an ocean of square brackets). Macro
invocations and `...` can only appear in expression positions. Finally,
macro expansion is currently unhygienic. That is, name collisions between
macro-generated and user-written code can cause unintentional capture.

Future versions of Rust will address these issues.

# Memory and concurrency model

## Memory model

A Rust task's memory consists of a static set of *items*, a set of tasks
each with its own *stack*, and a *heap*. Immutable portions of the
heap may be shared between tasks, mutable portions may not.

Allocations in the stack consist of *slots*, and allocations in the heap
consist of *boxes*.


### Memory allocation and lifetime

The _items_ of a program are those functions, objects, modules and types
that have their value calculated at compile-time and stored uniquely in the
memory image of the rust process. Items are neither dynamically allocated nor
freed.

A task's _stack_ consists of activation frames automatically allocated on
entry to each function as the task executes. A stack allocation is reclaimed
when control leaves the frame containing it.

The _heap_ is a general term that describes two separate sets of boxes:
shared boxes -- which may be subject to garbage collection -- and unique
boxes.  The lifetime of an allocation in the heap depends on the lifetime of
the box values pointing to it. Since box values may themselves be passed in
and out of frames, or stored in the heap, heap allocations may outlive the
frame they are allocated within.


### Memory ownership

A task owns all memory it can *safely* reach through local variables,
shared or unique boxes, and/or references. Sharing memory between tasks can
only be accomplished using *unsafe* constructs, such as raw pointer
operations or calling C code.

When a task sends a value of *unique* kind over a channel, it loses
ownership of the value sent and can no longer refer to it. This is statically
guaranteed by the combined use of "move semantics" and unique kinds, within
the communication system.

When a stack frame is exited, its local allocations are all released, and its
references to boxes (both shared and owned) are dropped.

A shared box may (in the case of a recursive, mutable shared type) be cyclic;
in this case the release of memory inside the shared structure may be deferred
until task-local garbage collection can reclaim it. Code can ensure no such
delayed deallocation occurs by restricting itself to unique boxes and similar
unshared kinds of data.

When a task finishes, its stack is necessarily empty and it therefore has no
references to any boxes; the remainder of its heap is immediately freed.


### Memory slots

A task's stack contains slots.

A _slot_ is a component of a stack frame. A slot is either *local* or
a *reference*.

A _local_ slot (or *stack-local* allocation) holds a value directly,
allocated within the stack's memory. The value is a part of the stack frame.

A _reference_ references a value outside the frame. It may refer to a
value allocated in another frame *or* a boxed value in the heap. The
reference-formation rules ensure that the referent will outlive the reference.

Local slots are always implicitly mutable.

Local slots are not initialized when allocated; the entire frame worth of
local slots are allocated at once, on frame-entry, in an uninitialized
state. Subsequent statements within a function may or may not initialize the
local slots. Local slots can be used only after they have been initialized;
this condition is guaranteed by the typestate system.

References are created for function arguments. If the compiler can not prove
that the referred-to value will outlive the reference, it will try to set
aside a copy of that value to refer to. If this is not semantically safe (for
example, if the referred-to value contains mutable fields), it will reject the
program. If the compiler deems copying the value expensive, it will warn.

A function can be declared to take an argument by mutable reference. This
allows the function to write to the slot that the reference refers to.

An example function that accepts an value by mutable reference:

~~~~~~~~
fn incr(&i: int) {
    i = i + 1;
}
~~~~~~~~


### Memory boxes

A _box_ is a reference to a heap allocation holding another value. There
are two kinds of boxes: *shared boxes* and *unique boxes*.

A _shared box_ type or value is constructed by the prefix *at* sigil `@`.

A _unique box_ type or value is constructed by the prefix *tilde* sigil `~`.

Multiple shared box values can point to the same heap allocation; copying a
shared box value makes a shallow copy of the pointer (optionally incrementing
a reference count, if the shared box is implemented through
reference-counting).

Unique box values exist in 1:1 correspondence with their heap allocation;
copying a unique box value makes a deep copy of the heap allocation and
produces a pointer to the new allocation.

An example of constructing one shared box type and value, and one unique box
type and value:

~~~~~~~~
let x: @int = @10;
let x: ~int = ~10;
~~~~~~~~

Some operations implicitly dereference boxes. Examples of such @dfn{implicit
dereference} operations are:

* arithmetic operators (`x + y - z`)
* field selection (`x.y.z`)


An example of an implicit-dereference operation performed on box values:

~~~~~~~~
let x: @int = @10;
let y: @int = @12;
assert (x + y == 22);
~~~~~~~~

Other operations act on box values as single-word-sized address values. For
these operations, to access the value held in the box requires an explicit
dereference of the box value. Explicitly dereferencing a box is indicated with
the unary *star* operator `*`. Examples of such @dfn{explicit
dereference} operations are:

* copying box values (`x = y`)
* passing box values to functions (`f(x,y)`)


An example of an explicit-dereference operation performed on box values:

~~~~~~~~
fn takes_boxed(b: @int) {
}

fn takes_unboxed(b: int) {
}

fn main() {
    let x: @int = @10;
    takes_boxed(x);
    takes_unboxed(*x);
}
~~~~~~~~


# Runtime services, linkage and debugging

# Appendix: Rationales and design tradeoffs

_TBD_.

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
>  &mdash;  Robert Strom and Shaula Yemini


>  Concurrent and applicative programming complement each other. The
>  ability to send messages on channels provides I/O without side effects,
>  while the avoidance of shared data helps keep concurrent processes from
>  colliding.
>
>  &mdash; Rob Pike


Rust is not a particularly original language. It may however appear unusual
by contemporary standards, as its design elements are drawn from a number of
"historical" languages that have, with a few exceptions, fallen out of
favour. Five prominent lineages contribute the most, though their influences
have come and gone during the course of Rust's development:

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
  others in their group at Bell labs Computing Sciences Reserch Center
  (Murray Hill, NJ, USA).

* The Napier (1985) and Napier88 (1988) family. These languages were
  developed by Malcolm Atkinson, Ron Morrison and others in their group at
  the University of St. Andrews (St. Andrews, Fife, UK).

Additional specific influences can be seen from the following languages:

* The stack-growth implementation of Go.
* The structural algebraic types and compilation manager of SML.
* The attribute and assembly systems of C#.
* The deterministic destructor system of C++.
* The typeclass system of Haskell.
* The lexical identifier rule of Python.
* The block syntax of Ruby.


LocalWords:  codepoint
