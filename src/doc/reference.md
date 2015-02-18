% The Rust Reference

# Introduction

This document is the primary reference for the Rust programming language. It
provides three kinds of material:

  - Chapters that informally describe each language construct and their use.
  - Chapters that informally describe the memory model, concurrency model,
    runtime services, linkage model and debugging facilities.
  - Appendix chapters providing rationale and references to languages that
    influenced the design.

This document does not serve as an introduction to the language. Background
familiarity with the language is assumed. A separate [book] is available to
help acquire such background familiarity.

This document also does not serve as a reference to the [standard] library
included in the language distribution. Those libraries are documented
separately by extracting documentation attributes from their source code. Many
of the features that one might expect to be language features are library
features in Rust, so what you're looking for may be there, not here.

You may also be interested in the [grammar].

[book]: book/index.html
[standard]: std/index.html
[grammar]: grammar.html

# Notation

Rust's grammar is defined over Unicode codepoints, each conventionally denoted
`U+XXXX`, for 4 or more hexadecimal digits `X`. _Most_ of Rust's grammar is
confined to the ASCII range of Unicode, and is described in this document by a
dialect of Extended Backus-Naur Form (EBNF), specifically a dialect of EBNF
supported by common automated LL(k) parsing tools such as `llgen`, rather than
the dialect given in ISO 14977. The dialect can be defined self-referentially
as follows:

```{.ebnf .notation}
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

```{.ebnf .gram}
comment : block_comment | line_comment ;
block_comment : "/*" block_comment_body * "*/" ;
block_comment_body : [block_comment | character] * ;
line_comment : "//" non_eol * ;
```

Comments in Rust code follow the general C++ style of line and block-comment
forms. Nested block comments are supported.

Line comments beginning with exactly _three_ slashes (`///`), and block
comments beginning with exactly one repeated asterisk in the block-open
sequence (`/**`), are interpreted as a special syntax for `doc`
[attributes](#attributes). That is, they are equivalent to writing
`#[doc="..."]` around the body of the comment (this includes the comment
characters themselves, ie `/// Foo` turns into `#[doc="/// Foo"]`).

`//!` comments apply to the parent of the comment, rather than the item that
follows. `//!` comments are usually used to display information on the crate
index page.

Non-doc comments are interpreted as a form of whitespace.

## Whitespace

```{.ebnf .gram}
whitespace_char : '\x20' | '\x09' | '\x0a' | '\x0d' ;
whitespace : [ whitespace_char | comment ] + ;
```

The `whitespace_char` production is any nonempty Unicode string consisting of
any of the following Unicode characters: `U+0020` (space, `' '`), `U+0009`
(tab, `'\t'`), `U+000A` (LF, `'\n'`), `U+000D` (CR, `'\r'`).

Rust is a "free-form" language, meaning that all forms of whitespace serve only
to separate _tokens_ in the grammar, and have no semantic significance.

A Rust program has identical meaning if each whitespace element is replaced
with any other legal whitespace element, such as a single space character.

## Tokens

```{.ebnf .gram}
simple_token : keyword | unop | binop ;
token : simple_token | ident | literal | symbol | whitespace token ;
```

Tokens are primitive productions in the grammar defined by regular
(non-recursive) languages. "Simple" tokens are given in [string table
production](#string-table-productions) form, and occur in the rest of the
grammar as double-quoted strings. Other tokens have exact rules given.

### Keywords

<p id="keyword-table-marker"></p>

|          |          |          |          |         |
|----------|----------|----------|----------|---------|
| abstract | alignof  | as       | become   | box     |
| break    | const    | continue | crate    | do      |
| else     | enum     | extern   | false    | final   |
| fn       | for      | if       | impl     | in      |
| let      | loop     | macro    | match    | mod     |
| move     | mut      | offsetof | override | priv    |
| pub      | pure     | ref      | return   | sizeof  |
| static   | self     | struct   | super    | true    |
| trait    | type     | typeof   | unsafe   | unsized |
| use      | virtual  | where    | while    | yield   |


Each of these keywords has special meaning in its grammar, and all of them are
excluded from the `ident` rule.

Note that some of these keywords are reserved, and do not currently do
anything.

### Literals

A literal is an expression consisting of a single token, rather than a sequence
of tokens, that immediately and directly denotes the value it evaluates to,
rather than referring to it by name or some other evaluation rule. A literal is
a form of constant expression, so is evaluated (primarily) at compile time.

```{.ebnf .gram}
lit_suffix : ident;
literal : [ string_lit | char_lit | byte_string_lit | byte_lit | num_lit ] lit_suffix ?;
```

The optional suffix is only used for certain numeric literals, but is
reserved for future extension, that is, the above gives the lexical
grammar, but a Rust parser will reject everything but the 12 special
cases mentioned in [Number literals](#number-literals) below.

#### Examples

##### Characters and strings

|   | Example | Number of `#` pairs allowed | Available characters | Escapes | Equivalent to |
|---|---------|-----------------------------|----------------------|---------|---------------|
| [Character](#character-literals) | `'H'` | `N/A` | All unicode | `\'` & [Byte escapes](#byte-escapes) & [Unicode escapes](#unicode-escapes) | `N/A` |
| [String](#string-literals) | `"hello"` | `N/A` | All unicode | `\"` & [Byte escapes](#byte-escapes) & [Unicode escapes](#unicode-escapes) | `N/A` |
| [Raw](#raw-string-literals) | `r##"hello"##`  | `0...` | All unicode | `N/A` | `N/A` |
| [Byte](#byte-literals) | `b'H'` | `N/A` | All ASCII | `\'` & [Byte escapes](#byte-escapes) | `u8` |
| [Byte string](#byte-string-literals) | `b"hello"` | `N/A`  | All ASCII | `\"` & [Byte escapes](#byte-escapes) | `&'static [u8]` |
| [Raw byte string](#raw-byte-string-literals) | `br##"hello"##` | `0...` | All ASCII | `N/A` | `&'static [u8]` (unsure...not stated) |

##### Byte escapes

|   | Name |
|---|------|
| `\x7F` | 8-bit character code (exactly 2 digits) |
| `\n` | Newline |
| `\r` | Carriage return |
| `\t` | Tab |
| `\\` | Backslash |

##### Unicode escapes
|   | Name |
|---|------|
| `\u{7FFF}` | 24-bit Unicode character code (up to 6 digits) |

##### Numbers

| [Number literals](#number-literals)`*` | Example | Exponentiation | Suffixes |
|----------------------------------------|---------|----------------|----------|
| Decimal integer | `98_222` | `N/A` | Integer suffixes |
| Hex integer | `0xff` | `N/A` | Integer suffixes |
| Octal integer | `0o77` | `N/A` | Integer suffixes |
| Binary integer | `0b1111_0000` | `N/A` | Integer suffixes |
| Floating-point | `123.0E+77` | `Optional` | Floating-point suffixes |

`*` All number literals allow `_` as a visual separator: `1_234.0E+18f64`

##### Suffixes
| Integer | Floating-point |
|---------|----------------|
| `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `is` (`isize`), `us` (`usize`) | `f32`, `f64` |

#### Character and string literals

```{.ebnf .gram}
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

##### Character literals

A _character literal_ is a single Unicode character enclosed within two
`U+0027` (single-quote) characters, with the exception of `U+0027` itself,
which must be _escaped_ by a preceding U+005C character (`\`).

##### String literals

A _string literal_ is a sequence of any Unicode characters enclosed within two
`U+0022` (double-quote) characters, with the exception of `U+0022` itself,
which must be _escaped_ by a preceding `U+005C` character (`\`), or a _raw
string literal_.

##### Character escapes

Some additional _escapes_ are available in either character or non-raw string
literals. An escape starts with a `U+005C` (`\`) and continues with one of the
following forms:

* An _8-bit codepoint escape_ escape starts with `U+0078` (`x`) and is
  followed by exactly two _hex digits_. It denotes the Unicode codepoint
  equal to the provided hex value.
* A _24-bit codepoint escape_ starts with `U+0075` (`u`) and is followed
  by up to six _hex digits_ surrounded by braces `U+007B` (`{`) and `U+007D`
  (`}`). It denotes the Unicode codepoint equal to the provided hex value.
* A _whitespace escape_ is one of the characters `U+006E` (`n`), `U+0072`
  (`r`), or `U+0074` (`t`), denoting the unicode values `U+000A` (LF),
  `U+000D` (CR) or `U+0009` (HT) respectively.
* The _backslash escape_ is the character `U+005C` (`\`) which must be
  escaped in order to denote *itself*.

##### Raw string literals

Raw string literals do not process any escapes. They start with the character
`U+0072` (`r`), followed by zero or more of the character `U+0023` (`#`) and a
`U+0022` (double-quote) character. The _raw string body_ is not defined in the
EBNF grammar above: it can contain any sequence of Unicode characters and is
terminated only by another `U+0022` (double-quote) character, followed by the
same number of `U+0023` (`#`) characters that preceded the opening `U+0022`
(double-quote) character.

All Unicode characters contained in the raw string body represent themselves,
the characters `U+0022` (double-quote) (except when followed by at least as
many `U+0023` (`#`) characters as were used to start the raw string literal) or
`U+005C` (`\`) do not have any special meaning.

Examples for string literals:

```
"foo"; r"foo";                     // foo
"\"foo\""; r#""foo""#;             // "foo"

"foo #\"# bar";
r##"foo #"# bar"##;                // foo #"# bar

"\x52"; "R"; r"R";                 // R
"\\x52"; r"\x52";                  // \x52
```

#### Byte and byte string literals

```{.ebnf .gram}
byte_lit : "b\x27" byte_body '\x27' ;
byte_string_lit : "b\x22" string_body * '\x22' | "br" raw_byte_string ;

byte_body : ascii_non_single_quote
          | '\x5c' [ '\x27' | common_escape ] ;

byte_string_body : ascii_non_double_quote
            | '\x5c' [ '\x22' | common_escape ] ;
raw_byte_string : '"' raw_byte_string_body '"' | '#' raw_byte_string '#' ;

```

##### Byte literals

A _byte literal_ is a single ASCII character (in the `U+0000` to `U+007F`
range) enclosed within two `U+0027` (single-quote) characters, with the
exception of `U+0027` itself, which must be _escaped_ by a preceding U+005C
character (`\`), or a single _escape_. It is equivalent to a `u8` unsigned
8-bit integer _number literal_.

##### Byte string literals

A non-raw _byte string literal_ is a sequence of ASCII characters and _escapes_,
preceded by the characters `U+0062` (`b`) and `U+0022` (double-quote), and
followed by the character `U+0022`. If the character `U+0022` is present within
the literal, it must be _escaped_ by a preceding `U+005C` (`\`) character.
Alternatively, a byte string literal can be a _raw byte string literal_, defined
below. A byte string literal is equivalent to a `&'static [u8]` borrowed array
of unsigned 8-bit integers.

Some additional _escapes_ are available in either byte or non-raw byte string
literals. An escape starts with a `U+005C` (`\`) and continues with one of the
following forms:

* An _byte escape_ escape starts with `U+0078` (`x`) and is
  followed by exactly two _hex digits_. It denotes the byte
  equal to the provided hex value.
* A _whitespace escape_ is one of the characters `U+006E` (`n`), `U+0072`
  (`r`), or `U+0074` (`t`), denoting the bytes values `0x0A` (ASCII LF),
  `0x0D` (ASCII CR) or `0x09` (ASCII HT) respectively.
* The _backslash escape_ is the character `U+005C` (`\`) which must be
  escaped in order to denote its ASCII encoding `0x5C`.

##### Raw byte string literals

Raw byte string literals do not process any escapes. They start with the
character `U+0062` (`b`), followed by `U+0072` (`r`), followed by zero or more
of the character `U+0023` (`#`), and a `U+0022` (double-quote) character. The
_raw string body_ is not defined in the EBNF grammar above: it can contain any
sequence of ASCII characters and is terminated only by another `U+0022`
(double-quote) character, followed by the same number of `U+0023` (`#`)
characters that preceded the opening `U+0022` (double-quote) character. A raw
byte string literal can not contain any non-ASCII byte.

All characters contained in the raw string body represent their ASCII encoding,
the characters `U+0022` (double-quote) (except when followed by at least as
many `U+0023` (`#`) characters as were used to start the raw string literal) or
`U+005C` (`\`) do not have any special meaning.

Examples for byte string literals:

```
b"foo"; br"foo";                     // foo
b"\"foo\""; br#""foo""#;             // "foo"

b"foo #\"# bar";
br##"foo #"# bar"##;                 // foo #"# bar

b"\x52"; b"R"; br"R";                // R
b"\\x52"; br"\x52";                  // \x52
```

#### Number literals

```{.ebnf .gram}
num_lit : nonzero_dec [ dec_digit | '_' ] * float_suffix ?
        | '0' [       [ dec_digit | '_' ] * float_suffix ?
              | 'b'   [ '1' | '0' | '_' ] +
              | 'o'   [ oct_digit | '_' ] +
              | 'x'   [ hex_digit | '_' ] +  ] ;

float_suffix : [ exponent | '.' dec_lit exponent ? ] ? ;

exponent : ['E' | 'e'] ['-' | '+' ] ? dec_lit ;
dec_lit : [ dec_digit | '_' ] + ;
```

A _number literal_ is either an _integer literal_ or a _floating-point
literal_. The grammar for recognizing the two kinds of literals is mixed.

##### Integer literals

An _integer literal_ has one of four forms:

* A _decimal literal_ starts with a *decimal digit* and continues with any
  mixture of *decimal digits* and _underscores_.
* A _hex literal_ starts with the character sequence `U+0030` `U+0078`
  (`0x`) and continues as any mixture of hex digits and underscores.
* An _octal literal_ starts with the character sequence `U+0030` `U+006F`
  (`0o`) and continues as any mixture of octal digits and underscores.
* A _binary literal_ starts with the character sequence `U+0030` `U+0062`
  (`0b`) and continues as any mixture of binary digits and underscores.

Like any literal, an integer literal may be followed (immediately,
without any spaces) by an _integer suffix_, which forcibly sets the
type of the literal. The integer suffix must be the name of one of the
integral types: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
`isize`, or `usize`.

The type of an _unsuffixed_ integer literal is determined by type inference.
If an integer type can be _uniquely_ determined from the surrounding program
context, the unsuffixed integer literal has that type. If the program context
underconstrains the type, it defaults to the signed 32-bit integer `i32`; if
the program context overconstrains the type, it is considered a static type
error.

Examples of integer literals of various forms:

```
123i32;                            // type i32
123u32;                            // type u32
123_u32;                           // type u32
0xff_u8;                           // type u8
0o70_i16;                          // type i16
0b1111_1111_1001_0000_i32;         // type i32
0usize;                            // type usize
```

##### Floating-point literals

A _floating-point literal_ has one of two forms:

* A _decimal literal_ followed by a period character `U+002E` (`.`). This is
  optionally followed by another decimal literal, with an optional _exponent_.
* A single _decimal literal_ followed by an _exponent_.

By default, a floating-point literal has a generic type, and, like integer
literals, the type must be uniquely determined from the context. There are two valid
_floating-point suffixes_, `f32` and `f64` (the 32-bit and 64-bit floating point
types), which explicitly determine the type of the literal.

Examples of floating-point literals of various forms:

```
123.0f64;        // type f64
0.1f64;          // type f64
0.1f32;          // type f32
12E+99_f64;      // type f64
let x: f64 = 2.; // type f64
```

This last example is different because it is not possible to use the suffix
syntax with a floating point literal ending in a period. `2.f64` would attempt
to call a method named `f64` on `2`.

#### Boolean literals

The two values of the boolean type are written `true` and `false`.

### Symbols

```{.ebnf .gram}
symbol : "::" | "->"
       | '#' | '[' | ']' | '(' | ')' | '{' | '}'
       | ',' | ';' ;
```

Symbols are a general class of printable [token](#tokens) that play structural
roles in a variety of grammar productions. They are catalogued here for
completeness as the set of remaining miscellaneous printable tokens that do not
otherwise appear as [unary operators](#unary-operator-expressions), [binary
operators](#binary-operator-expressions), or [keywords](#keywords).


## Paths

```{.ebnf .gram}
expr_path : [ "::" ] ident [ "::" expr_path_tail ] + ;
expr_path_tail : '<' type_expr [ ',' type_expr ] + '>'
               | expr_path ;

type_path : ident [ type_path_tail ] + ;
type_path_tail : '<' type_expr [ ',' type_expr ] + '>'
               | "::" type_path ;
```

A _path_ is a sequence of one or more path components _logically_ separated by
a namespace qualifier (`::`). If a path consists of only one component, it may
refer to either an [item](#items) or a [slot](#memory-slots) in a local control
scope. If a path has multiple components, it refers to an item.

Every item has a _canonical path_ within its crate, but the path naming an item
is only meaningful within a given crate. There is no global namespace across
crates; an item's canonical path merely identifies it within the crate.

Two examples of simple paths consisting of only identifier components:

```{.ignore}
x;
x::y::z;
```

Path components are usually [identifiers](#identifiers), but the trailing
component of a path may be an angle-bracket-enclosed list of type arguments. In
[expression](#expressions) context, the type argument list is given after a
final (`::`) namespace qualifier in order to disambiguate it from a relational
expression involving the less-than symbol (`<`). In type expression context,
the final namespace qualifier is omitted.

Two examples of paths with type arguments:

```
# struct HashMap<K, V>(K,V);
# fn f() {
# fn id<T>(t: T) -> T { t }
type T = HashMap<i32,String>; // Type arguments used in a type expression
let x  = id::<i32>(10);       // Type arguments used in a call expression
# }
```

Paths can be denoted with various leading qualifiers to change the meaning of
how it is resolved:

* Paths starting with `::` are considered to be global paths where the
  components of the path start being resolved from the crate root. Each
  identifier in the path must resolve to an item.

```rust
mod a {
    pub fn foo() {}
}
mod b {
    pub fn foo() {
        ::a::foo(); // call a's foo function
    }
}
# fn main() {}
```

* Paths starting with the keyword `super` begin resolution relative to the
  parent module. Each further identifier must resolve to an item.

```rust
mod a {
    pub fn foo() {}
}
mod b {
    pub fn foo() {
        super::a::foo(); // call a's foo function
    }
}
# fn main() {}
```

* Paths starting with the keyword `self` begin resolution relative to the
  current module. Each further identifier must resolve to an item.

```rust
fn foo() {}
fn bar() {
    self::foo();
}
# fn main() {}
```

# Syntax extensions

A number of minor features of Rust are not central enough to have their own
syntax, and yet are not implementable as functions. Instead, they are given
names, and invoked through a consistent syntax: `name!(...)`. Examples include:

* `format!` : format data into a string
* `env!` : look up an environment variable's value at compile time
* `file!`: return the path to the file being compiled
* `stringify!` : pretty-print the Rust expression given as an argument
* `include!` : include the Rust expression in the given file
* `include_str!` : include the contents of the given file as a string
* `include_bytes!` : include the contents of the given file as a binary blob
* `error!`, `warn!`, `info!`, `debug!` : provide diagnostic information.

All of the above extensions are expressions with values.

Users of `rustc` can define new syntax extensions in two ways:

* [Compiler plugins][plugin] can include arbitrary
  Rust code that manipulates syntax trees at compile time.

* [Macros](book/macros.html) define new syntax in a higher-level,
  declarative way.

## Macros

```{.ebnf .gram}
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

`macro_rules` allows users to define syntax extension in a declarative way.  We
call such extensions "macros by example" or simply "macros" — to be distinguished
from the "procedural macros" defined in [compiler plugins][plugin].

Currently, macros can expand to expressions, statements, items, or patterns.

(A `sep_token` is any token other than `*` and `+`. A `non_special_token` is
any token other than a delimiter or `$`.)

The macro expander looks up macro invocations by name, and tries each macro
rule in turn. It transcribes the first successful match. Matching and
transcription are closely related to each other, and we will describe them
together.

### Macro By Example

The macro expander matches and transcribes every token that does not begin with
a `$` literally, including delimiters. For parsing reasons, delimiters must be
balanced, but they are otherwise not special.

In the matcher, `$` _name_ `:` _designator_ matches the nonterminal in the Rust
syntax named by _designator_. Valid designators are `item`, `block`, `stmt`,
`pat`, `expr`, `ty` (type), `ident`, `path`, `tt` (either side of the `=>`
in macro rules). In the transcriber, the designator is already known, and so
only the name of a matched nonterminal comes after the dollar sign.

In both the matcher and transcriber, the Kleene star-like operator indicates
repetition. The Kleene star operator consists of `$` and parens, optionally
followed by a separator token, followed by `*` or `+`. `*` means zero or more
repetitions, `+` means at least one repetition. The parens are not matched or
transcribed. On the matcher side, a name is bound to _all_ of the names it
matches, in a structure that mimics the structure of the repetition encountered
on a successful match. The job of the transcriber is to sort that structure
out.

The rules for transcription of these repetitions are called "Macro By Example".
Essentially, one "layer" of repetition is discharged at a time, and all of them
must be discharged by the time a name is transcribed. Therefore, `( $( $i:ident
),* ) => ( $i )` is an invalid macro, but `( $( $i:ident ),* ) => ( $( $i:ident
),*  )` is acceptable (if trivial).

When Macro By Example encounters a repetition, it examines all of the `$`
_name_ s that occur in its body. At the "current layer", they all must repeat
the same number of times, so ` ( $( $i:ident ),* ; $( $j:ident ),* ) => ( $(
($i,$j) ),* )` is valid if given the argument `(a,b,c ; d,e,f)`, but not
`(a,b,c ; d,e)`. The repetition walks through the choices at that layer in
lockstep, so the former input transcribes to `( (a,d), (b,e), (c,f) )`.

Nested repetitions are allowed.

### Parsing limitations

The parser used by the macro system is reasonably powerful, but the parsing of
Rust syntax is restricted in two ways:

1. The parser will always parse as much as possible. If it attempts to match
   `$i:expr [ , ]` against `8 [ , ]`, it will attempt to parse `i` as an array
   index operation and fail. Adding a separator can solve this problem.
2. The parser must have eliminated all ambiguity by the time it reaches a `$`
   _name_ `:` _designator_. This requirement most often affects name-designator
   pairs when they occur at the beginning of, or immediately after, a `$(...)*`;
   requiring a distinctive token in front can solve the problem.

## Syntax extensions useful for the macro author

* `log_syntax!` : print out the arguments at compile time
* `trace_macros!` : supply `true` or `false` to enable or disable macro expansion logging
* `stringify!` : turn the identifier argument into a string literal
* `concat!` : concatenates a comma-separated list of literals
* `concat_idents!` : create a new identifier by concatenating the arguments

The following attributes are used for quasiquoting in procedural macros:

* `quote_expr!`
* `quote_item!`
* `quote_pat!`
* `quote_stmt!`
* `quote_tokens!`
* `quote_ty!`

# Crates and source files

Rust is a *compiled* language. Its semantics obey a *phase distinction*
between compile-time and run-time. Those semantic rules that have a *static
interpretation* govern the success or failure of compilation. We refer to
these rules as "static semantics". Semantic rules called "dynamic semantics"
govern the behavior of programs at run-time. A program that fails to compile
due to violation of a compile-time rule has no defined dynamic semantics; the
compiler should halt with an error report, and produce no executable artifact.

The compilation model centers on artifacts called _crates_. Each compilation
processes a single crate in source form, and if successful, produces a single
crate in binary form: either an executable or a library.[^cratesourcefile]

[^cratesourcefile]: A crate is somewhat analogous to an *assembly* in the
    ECMA-335 CLI model, a *library* in the SML/NJ Compilation Manager, a *unit*
    in the Owens and Flatt module system, or a *configuration* in Mesa.

A _crate_ is a unit of compilation and linking, as well as versioning,
distribution and runtime loading. A crate contains a _tree_ of nested
[module](#modules) scopes. The top level of this tree is a module that is
anonymous (from the point of view of paths within the module) and any item
within a crate has a canonical [module path](#paths) denoting its location
within the crate's module tree.

The Rust compiler is always invoked with a single source file as input, and
always produces a single output crate. The processing of that source file may
result in other source files being loaded as modules. Source files have the
extension `.rs`.

A Rust source file describes a module, the name and location of which &mdash;
in the module tree of the current crate &mdash; are defined from outside the
source file: either by an explicit `mod_item` in a referencing source file, or
by the name of the crate itself.

Each source file contains a sequence of zero or more `item` definitions, and
may optionally begin with any number of `attributes` that apply to the
containing module. Attributes on the anonymous crate module define important
metadata that influences the behavior of the compiler.

```{.rust}
# #![allow(unused_attribute)]
// Crate name
#![crate_name = "projx"]

// Specify the output type
#![crate_type = "lib"]

// Turn on a warning
#![warn(non_camel_case_types)]
```

A crate that contains a `main` function can be compiled to an executable. If a
`main` function is present, its return type must be [`unit`](#primitive-types)
and it must take no arguments.

# Items and attributes

Crates contain [items](#items), each of which may have some number of
[attributes](#attributes) attached to it.

## Items

```{.ebnf .gram}
item : extern_crate_decl | use_decl | mod_item | fn_item | type_item
     | struct_item | enum_item | static_item | trait_item | impl_item
     | extern_block ;
```

An _item_ is a component of a crate. Items are organized within a crate by a
nested set of [modules](#modules). Every crate has a single "outermost"
anonymous module; all further items within the crate have [paths](#paths)
within the module tree of the crate.

Items are entirely determined at compile-time, generally remain fixed during
execution, and may reside in read-only memory.

There are several kinds of item:

* [`extern crate` declarations](#extern-crate-declarations)
* [`use` declarations](#use-declarations)
* [modules](#modules)
* [functions](#functions)
* [type definitions](#type-definitions)
* [structures](#structures)
* [enumerations](#enumerations)
* [static items](#static-items)
* [traits](#traits)
* [implementations](#implementations)

Some items form an implicit scope for the declaration of sub-items. In other
words, within a function or module, declarations of items can (in many cases)
be mixed with the statements, control blocks, and similar artifacts that
otherwise compose the item body. The meaning of these scoped items is the same
as if the item was declared outside the scope &mdash; it is still a static item
&mdash; except that the item's *path name* within the module namespace is
qualified by the name of the enclosing item, or is private to the enclosing
item (in the case of functions). The grammar specifies the exact locations in
which sub-item declarations may appear.

### Type Parameters

All items except modules may be *parameterized* by type. Type parameters are
given as a comma-separated list of identifiers enclosed in angle brackets
(`<...>`), after the name of the item and before its definition. The type
parameters of an item are considered "part of the name", not part of the type
of the item. A referencing [path](#paths) must (in principle) provide type
arguments as a list of comma-separated types enclosed within angle brackets, in
order to refer to the type-parameterized item. In practice, the type-inference
system can usually infer such argument types from context. There are no
general type-parametric types, only type-parametric items. That is, Rust has
no notion of type abstraction: there are no first-class "forall" types.

### Modules

```{.ebnf .gram}
mod_item : "mod" ident ( ';' | '{' mod '}' );
mod : item * ;
```

A module is a container for zero or more [items](#items).

A _module item_ is a module, surrounded in braces, named, and prefixed with the
keyword `mod`. A module item introduces a new, named module into the tree of
modules making up a crate. Modules can nest arbitrarily.

An example of a module:

```
mod math {
    type Complex = (f64, f64);
    fn sin(f: f64) -> f64 {
        /* ... */
# panic!();
    }
    fn cos(f: f64) -> f64 {
        /* ... */
# panic!();
    }
    fn tan(f: f64) -> f64 {
        /* ... */
# panic!();
    }
}
```

Modules and types share the same namespace. Declaring a named type with
the same name as a module in scope is forbidden: that is, a type definition,
trait, struct, enumeration, or type parameter can't shadow the name of a module
in scope, or vice versa.

A module without a body is loaded from an external file, by default with the
same name as the module, plus the `.rs` extension. When a nested submodule is
loaded from an external file, it is loaded from a subdirectory path that
mirrors the module hierarchy.

```{.ignore}
// Load the `vec` module from `vec.rs`
mod vec;

mod thread {
    // Load the `local_data` module from `thread/local_data.rs`
    mod local_data;
}
```

The directories and files used for loading external file modules can be
influenced with the `path` attribute.

```{.ignore}
#[path = "thread_files"]
mod thread {
    // Load the `local_data` module from `thread_files/tls.rs`
    #[path = "tls.rs"]
    mod local_data;
}
```

##### Extern crate declarations

```{.ebnf .gram}
extern_crate_decl : "extern" "crate" crate_name
crate_name: ident | ( string_lit "as" ident )
```

An _`extern crate` declaration_ specifies a dependency on an external crate.
The external crate is then bound into the declaring scope as the `ident`
provided in the `extern_crate_decl`.

The external crate is resolved to a specific `soname` at compile time, and a
runtime linkage requirement to that `soname` is passed to the linker for
loading at runtime. The `soname` is resolved at compile time by scanning the
compiler's library path and matching the optional `crateid` provided as a
string literal against the `crateid` attributes that were declared on the
external crate when it was compiled. If no `crateid` is provided, a default
`name` attribute is assumed, equal to the `ident` given in the
`extern_crate_decl`.

Three examples of `extern crate` declarations:

```{.ignore}
extern crate pcre;

extern crate std; // equivalent to: extern crate std as std;

extern crate "std" as ruststd; // linking to 'std' under another name
```

##### Use declarations

```{.ebnf .gram}
use_decl : "pub" ? "use" [ path "as" ident
                          | path_glob ] ;

path_glob : ident [ "::" [ path_glob
                          | '*' ] ] ?
          | '{' path_item [ ',' path_item ] * '}' ;

path_item : ident | "self" ;
```

A _use declaration_ creates one or more local name bindings synonymous with
some other [path](#paths). Usually a `use` declaration is used to shorten the
path required to refer to a module item. These declarations may appear at the
top of [modules](#modules) and [blocks](#blocks).

> **Note**: Unlike in many languages,
> `use` declarations in Rust do *not* declare linkage dependency with external crates.
> Rather, [`extern crate` declarations](#extern-crate-declarations) declare linkage dependencies.

Use declarations support a number of convenient shortcuts:

* Rebinding the target name as a new local name, using the syntax `use p::q::r as x;`
* Simultaneously binding a list of paths differing only in their final element,
  using the glob-like brace syntax `use a::b::{c,d,e,f};`
* Binding all paths matching a given prefix, using the asterisk wildcard syntax
  `use a::b::*;`
* Simultaneously binding a list of paths differing only in their final element
  and their immediate parent module, using the `self` keyword, such as
  `use a::b::{self, c, d};`

An example of `use` declarations:

```
use std::iter::range_step;
use std::option::Option::{Some, None};
use std::collections::hash_map::{self, HashMap};

fn foo<T>(_: T){}
fn bar(map1: HashMap<String, usize>, map2: hash_map::HashMap<String, usize>){}

fn main() {
    // Equivalent to 'std::iter::range_step(0, 10, 2);'
    range_step(0, 10, 2);

    // Equivalent to 'foo(vec![std::option::Option::Some(1.0f64),
    // std::option::Option::None]);'
    foo(vec![Some(1.0f64), None]);

    // Both `hash_map` and `HashMap` are in scope.
    let map1 = HashMap::new();
    let map2 = hash_map::HashMap::new();
    bar(map1, map2);
}
```

Like items, `use` declarations are private to the containing module, by
default. Also like items, a `use` declaration can be public, if qualified by
the `pub` keyword. Such a `use` declaration serves to _re-export_ a name. A
public `use` declaration can therefore _redirect_ some public name to a
different target definition: even a definition with a private canonical path,
inside a different module. If a sequence of such redirections form a cycle or
cannot be resolved unambiguously, they represent a compile-time error.

An example of re-exporting:

```
# fn main() { }
mod quux {
    pub use quux::foo::{bar, baz};

    pub mod foo {
        pub fn bar() { }
        pub fn baz() { }
    }
}
```

In this example, the module `quux` re-exports two public names defined in
`foo`.

Also note that the paths contained in `use` items are relative to the crate
root. So, in the previous example, the `use` refers to `quux::foo::{bar,
baz}`, and not simply to `foo::{bar, baz}`. This also means that top-level
module declarations should be at the crate root if direct usage of the declared
modules within `use` items is desired. It is also possible to use `self` and
`super` at the beginning of a `use` item to refer to the current and direct
parent modules respectively. All rules regarding accessing declared modules in
`use` declarations applies to both module declarations and `extern crate`
declarations.

An example of what will and will not work for `use` items:

```
# #![allow(unused_imports)]
use foo::core::iter;  // good: foo is at the root of the crate
use foo::baz::foobaz;    // good: foo is at the root of the crate

mod foo {
    extern crate core;

    use foo::core::iter; // good: foo is at crate root
//  use core::iter;      // bad:  core is not at the crate root
    use self::baz::foobaz;  // good: self refers to module 'foo'
    use foo::bar::foobar;   // good: foo is at crate root

    pub mod bar {
        pub fn foobar() { }
    }

    pub mod baz {
        use super::bar::foobar; // good: super refers to module 'foo'
        pub fn foobaz() { }
    }
}

fn main() {}
```

### Functions

A _function item_ defines a sequence of [statements](#statements) and an
optional final [expression](#expressions), along with a name and a set of
parameters. Functions are declared with the keyword `fn`. Functions declare a
set of *input* [*slots*](#memory-slots) as parameters, through which the caller
passes arguments into the function, and an *output* [*slot*](#memory-slots)
through which the function passes results back to the caller.

A function may also be copied into a first-class *value*, in which case the
value has the corresponding [*function type*](#function-types), and can be used
otherwise exactly as a function item (with a minor additional cost of calling
the function indirectly).

Every control path in a function logically ends with a `return` expression or a
diverging expression. If the outermost block of a function has a
value-producing expression in its final-expression position, that expression is
interpreted as an implicit `return` expression applied to the final-expression.

An example of a function:

```
fn add(x: i32, y: i32) -> i32 {
    return x + y;
}
```

As with `let` bindings, function arguments are irrefutable patterns, so any
pattern that is valid in a let binding is also valid as an argument.

```
fn first((value, _): (i32, i32)) -> i32 { value }
```


#### Generic functions

A _generic function_ allows one or more _parameterized types_ to appear in its
signature. Each type parameter must be explicitly declared, in an
angle-bracket-enclosed, comma-separated list following the function name.

```{.ignore}
fn iter<T>(seq: &[T], f: |T|) {
    for elt in seq.iter() { f(elt); }
}
fn map<T, U>(seq: &[T], f: |T| -> U) -> Vec<U> {
    let mut acc = vec![];
    for elt in seq.iter() { acc.push(f(elt)); }
    acc
}
```

Inside the function signature and body, the name of the type parameter can be
used as a type name.

When a generic function is referenced, its type is instantiated based on the
context of the reference. For example, calling the `iter` function defined
above on `[1, 2]` will instantiate type parameter `T` with `i32`, and require
the closure parameter to have type `fn(i32)`.

The type parameters can also be explicitly supplied in a trailing
[path](#paths) component after the function name. This might be necessary if
there is not sufficient context to determine the type parameters. For example,
`mem::size_of::<u32>() == 4`.

Since a parameter type is opaque to the generic function, the set of operations
that can be performed on it is limited. Values of parameter type can only be
moved, not copied.

```
fn id<T>(x: T) -> T { x }
```

Similarly, [trait](#traits) bounds can be specified for type parameters to
allow methods with that trait to be called on values of that type.

#### Unsafety

Unsafe operations are those that potentially violate the memory-safety
guarantees of Rust's static semantics.

The following language level features cannot be used in the safe subset of
Rust:

- Dereferencing a [raw pointer](#pointer-types).
- Reading or writing a [mutable static variable](#mutable-statics).
- Calling an unsafe function (including an intrinsic or foreign function).

##### Unsafe functions

Unsafe functions are functions that are not safe in all contexts and/or for all
possible inputs. Such a function must be prefixed with the keyword `unsafe` and
can only be called from an `unsafe` block or another `unsafe` function.

##### Unsafe blocks

A block of code can be prefixed with the `unsafe` keyword, to permit calling
`unsafe` functions or dereferencing raw pointers within a safe function.

When a programmer has sufficient conviction that a sequence of potentially
unsafe operations is actually safe, they can encapsulate that sequence (taken
as a whole) within an `unsafe` block. The compiler will consider uses of such
code safe, in the surrounding context.

Unsafe blocks are used to wrap foreign libraries, make direct use of hardware
or implement features not directly present in the language. For example, Rust
provides the language features necessary to implement memory-safe concurrency
in the language but the implementation of threads and message passing is in the
standard library.

Rust's type system is a conservative approximation of the dynamic safety
requirements, so in some cases there is a performance cost to using safe code.
For example, a doubly-linked list is not a tree structure and can only be
represented with reference-counted pointers in safe code. By using `unsafe`
blocks to represent the reverse links as raw pointers, it can be implemented
with only boxes.

##### Behavior considered undefined

The following is a list of behavior which is forbidden in all Rust code,
including within `unsafe` blocks and `unsafe` functions. Type checking provides
the guarantee that these issues are never caused by safe code.

* Data races
* Dereferencing a null/dangling raw pointer
* Mutating an immutable value/reference without `UnsafeCell`
* Reads of [undef](http://llvm.org/docs/LangRef.html#undefined-values)
  (uninitialized) memory
* Breaking the [pointer aliasing
  rules](http://llvm.org/docs/LangRef.html#pointer-aliasing-rules)
  with raw pointers (a subset of the rules used by C)
* Invoking undefined behavior via compiler intrinsics:
  * Indexing outside of the bounds of an object with `std::ptr::offset`
    (`offset` intrinsic), with
    the exception of one byte past the end which is permitted.
  * Using `std::ptr::copy_nonoverlapping_memory` (`memcpy32`/`memcpy64`
    intrinsics) on overlapping buffers
* Invalid values in primitive types, even in private fields/locals:
  * Dangling/null references or boxes
  * A value other than `false` (0) or `true` (1) in a `bool`
  * A discriminant in an `enum` not included in the type definition
  * A value in a `char` which is a surrogate or above `char::MAX`
  * Non-UTF-8 byte sequences in a `str`
* Unwinding into Rust from foreign code or unwinding from Rust into foreign
  code. Rust's failure system is not compatible with exception handling in
  other languages. Unwinding must be caught and handled at FFI boundaries.

##### Behaviour not considered unsafe

This is a list of behaviour not considered *unsafe* in Rust terms, but that may
be undesired.

* Deadlocks
* Reading data from private fields (`std::repr`)
* Leaks due to reference count cycles, even in the global heap
* Exiting without calling destructors
* Sending signals
* Accessing/modifying the file system
* Unsigned integer overflow (well-defined as wrapping)
* Signed integer overflow (well-defined as two's complement representation
  wrapping)

#### Diverging functions

A special kind of function can be declared with a `!` character where the
output slot type would normally be. For example:

```
fn my_err(s: &str) -> ! {
    println!("{}", s);
    panic!();
}
```

We call such functions "diverging" because they never return a value to the
caller. Every control path in a diverging function must end with a `panic!()` or
a call to another diverging function on every control path. The `!` annotation
does *not* denote a type.

It might be necessary to declare a diverging function because as mentioned
previously, the typechecker checks that every control path in a function ends
with a [`return`](#return-expressions) or diverging expression. So, if `my_err`
were declared without the `!` annotation, the following code would not
typecheck:

```
# fn my_err(s: &str) -> ! { panic!() }

fn f(i: i32) -> i32 {
   if i == 42 {
     return 42;
   }
   else {
     my_err("Bad number!");
   }
}
```

This will not compile without the `!` annotation on `my_err`, since the `else`
branch of the conditional in `f` does not return an `i32`, as required by the
signature of `f`. Adding the `!` annotation to `my_err` informs the
typechecker that, should control ever enter `my_err`, no further type judgments
about `f` need to hold, since control will never resume in any context that
relies on those judgments. Thus the return type on `f` only needs to reflect
the `if` branch of the conditional.

#### Extern functions

Extern functions are part of Rust's foreign function interface, providing the
opposite functionality to [external blocks](#external-blocks). Whereas
external blocks allow Rust code to call foreign code, extern functions with
bodies defined in Rust code _can be called by foreign code_. They are defined
in the same way as any other Rust function, except that they have the `extern`
modifier.

```
// Declares an extern fn, the ABI defaults to "C"
extern fn new_i32() -> i32 { 0 }

// Declares an extern fn with "stdcall" ABI
extern "stdcall" fn new_i32_stdcall() -> i32 { 0 }
```

Unlike normal functions, extern fns have an `extern "ABI" fn()`. This is the
same type as the functions declared in an extern block.

```
# extern fn new_i32() -> i32 { 0 }
let fptr: extern "C" fn() -> i32 = new_i32;
```

Extern functions may be called directly from Rust code as Rust uses large,
contiguous stack segments like C.

### Type aliases

A _type alias_ defines a new name for an existing [type](#types). Type
aliases are declared with the keyword `type`. Every value has a single,
specific type; the type-specified aspects of a value include:

* Whether the value is composed of sub-values or is indivisible.
* Whether the value represents textual or numerical information.
* Whether the value represents integral or floating-point information.
* The sequence of memory operations required to access the value.
* The [kind](#type-kinds) of the type.

For example, the type `(u8, u8)` defines the set of immutable values that are
composite pairs, each containing two unsigned 8-bit integers accessed by
pattern-matching and laid out in memory with the `x` component preceding the
`y` component:

```
type Point = (u8, u8);
let p: Point = (41, 68);
```

### Structures

A _structure_ is a nominal [structure type](#structure-types) defined with the
keyword `struct`.

An example of a `struct` item and its use:

```
struct Point {x: i32, y: i32}
let p = Point {x: 10, y: 11};
let px: i32 = p.x;
```

A _tuple structure_ is a nominal [tuple type](#tuple-types), also defined with
the keyword `struct`. For example:

```
struct Point(i32, i32);
let p = Point(10, 11);
let px: i32 = match p { Point(x, _) => x };
```

A _unit-like struct_ is a structure without any fields, defined by leaving off
the list of fields entirely. Such types will have a single value, just like
the [unit value `()`](#unit-and-boolean-literals) of the unit type. For
example:

```
struct Cookie;
let c = [Cookie, Cookie, Cookie, Cookie];
```

The precise memory layout of a structure is not specified. One can specify a
particular layout using the [`repr` attribute](#ffi-attributes).

### Enumerations

An _enumeration_ is a simultaneous definition of a nominal [enumerated
type](#enumerated-types) as well as a set of *constructors*, that can be used
to create or pattern-match values of the corresponding enumerated type.

Enumerations are declared with the keyword `enum`.

An example of an `enum` item and its use:

```
enum Animal {
  Dog,
  Cat
}

let mut a: Animal = Animal::Dog;
a = Animal::Cat;
```

Enumeration constructors can have either named or unnamed fields:

```
# #![feature(struct_variant)]
# fn main() {
enum Animal {
    Dog (String, f64),
    Cat { name: String, weight: f64 }
}

let mut a: Animal = Animal::Dog("Cocoa".to_string(), 37.2);
a = Animal::Cat { name: "Spotty".to_string(), weight: 2.7 };
# }
```

In this example, `Cat` is a _struct-like enum variant_,
whereas `Dog` is simply called an enum variant.

Enums have a discriminant. You can assign them explicitly:

```
enum Foo {
    Bar = 123,
}
```

If a discriminant isn't assigned, they start at zero, and add one for each
variant, in order.

You can cast an enum to get this value:

```
# enum Foo { Bar = 123 }
let x = Foo::Bar as u32; // x is now 123u32
```

This only works as long as none of the variants have data attached. If
it were `Bar(i32)`, this is disallowed.

### Constant items

```{.ebnf .gram}
const_item : "const" ident ':' type '=' expr ';' ;
```

A *constant item* is a named _constant value_ which is not associated with a
specific memory location in the program. Constants are essentially inlined
wherever they are used, meaning that they are copied directly into the relevant
context when used. References to the same constant are not necessarily
guaranteed to refer to the same memory address.

Constant values must not have destructors, and otherwise permit most forms of
data. Constants may refer to the address of other constants, in which case the
address will have the `static` lifetime. The compiler is, however, still at
liberty to translate the constant many times, so the address referred to may not
be stable.

Constants must be explicitly typed. The type may be `bool`, `char`, a number, or
a type derived from those primitive types. The derived types are references with
the `static` lifetime, fixed-size arrays, tuples, enum variants, and structs.

```
const BIT1: u32 = 1 << 0;
const BIT2: u32 = 1 << 1;

const BITS: [u32; 2] = [BIT1, BIT2];
const STRING: &'static str = "bitstring";

struct BitsNStrings<'a> {
    mybits: [u32; 2],
    mystring: &'a str
}

const BITS_N_STRINGS: BitsNStrings<'static> = BitsNStrings {
    mybits: BITS,
    mystring: STRING
};
```

### Static items

```{.ebnf .gram}
static_item : "static" ident ':' type '=' expr ';' ;
```

A *static item* is similar to a *constant*, except that it represents a precise
memory location in the program. A static is never "inlined" at the usage site,
and all references to it refer to the same memory location. Static items have
the `static` lifetime, which outlives all other lifetimes in a Rust program.
Static items may be placed in read-only memory if they do not contain any
interior mutability.

Statics may contain interior mutability through the `UnsafeCell` language item.
All access to a static is safe, but there are a number of restrictions on
statics:

* Statics may not contain any destructors.
* The types of static values must ascribe to `Sync` to allow threadsafe access.
* Statics may not refer to other statics by value, only by reference.
* Constants cannot refer to statics.

Constants should in general be preferred over statics, unless large amounts of
data are being stored, or single-address and mutability properties are required.

```
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

// Note that ATOMIC_USIZE_INIT is a *const*, but it may be used to initialize a
// static. This static can be modified, so it is not placed in read-only memory.
static COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

// This table is a candidate to be placed in read-only memory.
static TABLE: &'static [usize] = &[1, 2, 3, /* ... */];

for slot in TABLE.iter() {
    println!("{}", slot);
}
COUNTER.fetch_add(1, Ordering::SeqCst);
```

#### Mutable statics

If a static item is declared with the `mut` keyword, then it is allowed to
be modified by the program. One of Rust's goals is to make concurrency bugs
hard to run into, and this is obviously a very large source of race conditions
or other bugs. For this reason, an `unsafe` block is required when either
reading or writing a mutable static variable. Care should be taken to ensure
that modifications to a mutable static are safe with respect to other threads
running in the same process.

Mutable statics are still very useful, however. They can be used with C
libraries and can also be bound from C libraries (in an `extern` block).

```
# fn atomic_add(_: &mut u32, _: u32) -> u32 { 2 }

static mut LEVELS: u32 = 0;

// This violates the idea of no shared state, and this doesn't internally
// protect against races, so this function is `unsafe`
unsafe fn bump_levels_unsafe1() -> u32 {
    let ret = LEVELS;
    LEVELS += 1;
    return ret;
}

// Assuming that we have an atomic_add function which returns the old value,
// this function is "safe" but the meaning of the return value may not be what
// callers expect, so it's still marked as `unsafe`
unsafe fn bump_levels_unsafe2() -> u32 {
    return atomic_add(&mut LEVELS, 1);
}
```

Mutable statics have the same restrictions as normal statics, except that the
type of the value is not required to ascribe to `Sync`.

### Traits

A _trait_ describes a set of method types.

Traits can include default implementations of methods, written in terms of some
unknown [`self` type](#self-types); the `self` type may either be completely
unspecified, or constrained by some other trait.

Traits are implemented for specific types through separate
[implementations](#implementations).

```
# type Surface = i32;
# type BoundingBox = i32;
trait Shape {
    fn draw(&self, Surface);
    fn bounding_box(&self) -> BoundingBox;
}
```

This defines a trait with two methods. All values that have
[implementations](#implementations) of this trait in scope can have their
`draw` and `bounding_box` methods called, using `value.bounding_box()`
[syntax](#method-call-expressions).

Type parameters can be specified for a trait to make it generic. These appear
after the trait name, using the same syntax used in [generic
functions](#generic-functions).

```
trait Seq<T> {
   fn len(&self) -> u32;
   fn elt_at(&self, n: u32) -> T;
   fn iter<F>(&self, F) where F: Fn(T);
}
```

Generic functions may use traits as _bounds_ on their type parameters. This
will have two effects: only types that have the trait may instantiate the
parameter, and within the generic function, the methods of the trait can be
called on values that have the parameter's type. For example:

```
# type Surface = i32;
# trait Shape { fn draw(&self, Surface); }
fn draw_twice<T: Shape>(surface: Surface, sh: T) {
    sh.draw(surface);
    sh.draw(surface);
}
```

Traits also define an [object type](#object-types) with the same name as the
trait. Values of this type are created by [casting](#type-cast-expressions)
pointer values (pointing to a type for which an implementation of the given
trait is in scope) to pointers to the trait name, used as a type.

```
# trait Shape { fn dummy(&self) { } }
# impl Shape for i32 { }
# let mycircle = 0i32;
let myshape: Box<Shape> = Box::new(mycircle) as Box<Shape>;
```

The resulting value is a box containing the value that was cast, along with
information that identifies the methods of the implementation that was used.
Values with a trait type can have [methods called](#method-call-expressions) on
them, for any method in the trait, and can be used to instantiate type
parameters that are bounded by the trait.

Trait methods may be static, which means that they lack a `self` argument.
This means that they can only be called with function call syntax (`f(x)`) and
not method call syntax (`obj.f()`). The way to refer to the name of a static
method is to qualify it with the trait name, treating the trait name like a
module. For example:

```
trait Num {
    fn from_i32(n: i32) -> Self;
}
impl Num for f64 {
    fn from_i32(n: i32) -> f64 { n as f64 }
}
let x: f64 = Num::from_i32(42);
```

Traits may inherit from other traits. For example, in

```
trait Shape { fn area(&self) -> f64; }
trait Circle : Shape { fn radius(&self) -> f64; }
```

the syntax `Circle : Shape` means that types that implement `Circle` must also
have an implementation for `Shape`. Multiple supertraits are separated by `+`,
`trait Circle : Shape + PartialEq { }`. In an implementation of `Circle` for a
given type `T`, methods can refer to `Shape` methods, since the typechecker
checks that any type with an implementation of `Circle` also has an
implementation of `Shape`.

In type-parameterized functions, methods of the supertrait may be called on
values of subtrait-bound type parameters. Referring to the previous example of
`trait Circle : Shape`:

```
# trait Shape { fn area(&self) -> f64; }
# trait Circle : Shape { fn radius(&self) -> f64; }
fn radius_times_area<T: Circle>(c: T) -> f64 {
    // `c` is both a Circle and a Shape
    c.radius() * c.area()
}
```

Likewise, supertrait methods may also be called on trait objects.

```{.ignore}
# trait Shape { fn area(&self) -> f64; }
# trait Circle : Shape { fn radius(&self) -> f64; }
# impl Shape for i32 { fn area(&self) -> f64 { 0.0 } }
# impl Circle for i32 { fn radius(&self) -> f64 { 0.0 } }
# let mycircle = 0i32;
let mycircle = Box::new(mycircle) as Box<Circle>;
let nonsense = mycircle.radius() * mycircle.area();
```

### Implementations

An _implementation_ is an item that implements a [trait](#traits) for a
specific type.

Implementations are defined with the keyword `impl`.

```
# #[derive(Copy)]
# struct Point {x: f64, y: f64};
# type Surface = i32;
# struct BoundingBox {x: f64, y: f64, width: f64, height: f64};
# trait Shape { fn draw(&self, Surface); fn bounding_box(&self) -> BoundingBox; }
# fn do_draw_circle(s: Surface, c: Circle) { }
struct Circle {
    radius: f64,
    center: Point,
}

impl Copy for Circle {}

impl Shape for Circle {
    fn draw(&self, s: Surface) { do_draw_circle(s, *self); }
    fn bounding_box(&self) -> BoundingBox {
        let r = self.radius;
        BoundingBox{x: self.center.x - r, y: self.center.y - r,
         width: 2.0 * r, height: 2.0 * r}
    }
}
```

It is possible to define an implementation without referring to a trait. The
methods in such an implementation can only be used as direct calls on the
values of the type that the implementation targets. In such an implementation,
the trait type and `for` after `impl` are omitted. Such implementations are
limited to nominal types (enums, structs), and the implementation must appear
in the same module or a sub-module as the `self` type:

```
struct Point {x: i32, y: i32}

impl Point {
    fn log(&self) {
        println!("Point is at ({}, {})", self.x, self.y);
    }
}

let my_point = Point {x: 10, y:11};
my_point.log();
```

When a trait _is_ specified in an `impl`, all methods declared as part of the
trait must be implemented, with matching types and type parameter counts.

An implementation can take type parameters, which can be different from the
type parameters taken by the trait it implements. Implementation parameters
are written after the `impl` keyword.

```
# trait Seq<T> { fn dummy(&self, _: T) { } }
impl<T> Seq<T> for Vec<T> {
   /* ... */
}
impl Seq<bool> for u32 {
   /* Treat the integer as a sequence of bits */
}
```

### External blocks

```{.ebnf .gram}
extern_block_item : "extern" '{' extern_block '}' ;
extern_block : [ foreign_fn ] * ;
```

External blocks form the basis for Rust's foreign function interface.
Declarations in an external block describe symbols in external, non-Rust
libraries.

Functions within external blocks are declared in the same way as other Rust
functions, with the exception that they may not have a body and are instead
terminated by a semicolon.

```
extern crate libc;
use libc::{c_char, FILE};

extern {
    fn fopen(filename: *const c_char, mode: *const c_char) -> *mut FILE;
}
# fn main() {}
```

Functions within external blocks may be called by Rust code, just like
functions defined in Rust. The Rust compiler automatically translates between
the Rust ABI and the foreign ABI.

A number of [attributes](#attributes) control the behavior of external blocks.

By default external blocks assume that the library they are calling uses the
standard C "cdecl" ABI. Other ABIs may be specified using an `abi` string, as
shown here:

```{.ignore}
// Interface to the Windows API
extern "stdcall" { }
```

The `link` attribute allows the name of the library to be specified. When
specified the compiler will attempt to link against the native library of the
specified name.

```{.ignore}
#[link(name = "crypto")]
extern { }
```

The type of a function declared in an extern block is `extern "abi" fn(A1, ...,
An) -> R`, where `A1...An` are the declared types of its arguments and `R` is
the declared return type.

## Visibility and Privacy

These two terms are often used interchangeably, and what they are attempting to
convey is the answer to the question "Can this item be used at this location?"

Rust's name resolution operates on a global hierarchy of namespaces. Each level
in the hierarchy can be thought of as some item. The items are one of those
mentioned above, but also include external crates. Declaring or defining a new
module can be thought of as inserting a new tree into the hierarchy at the
location of the definition.

To control whether interfaces can be used across modules, Rust checks each use
of an item to see whether it should be allowed or not. This is where privacy
warnings are generated, or otherwise "you used a private item of another module
and weren't allowed to."

By default, everything in Rust is *private*, with one exception. Enum variants
in a `pub` enum are also public by default. You are allowed to alter this
default visibility with the `priv` keyword. When an item is declared as `pub`,
it can be thought of as being accessible to the outside world. For example:

```
# fn main() {}
// Declare a private struct
struct Foo;

// Declare a public struct with a private field
pub struct Bar {
    field: i32
}

// Declare a public enum with two public variants
pub enum State {
    PubliclyAccessibleState,
    PubliclyAccessibleState2,
}
```

With the notion of an item being either public or private, Rust allows item
accesses in two cases:

1. If an item is public, then it can be used externally through any of its
   public ancestors.
2. If an item is private, it may be accessed by the current module and its
   descendants.

These two cases are surprisingly powerful for creating module hierarchies
exposing public APIs while hiding internal implementation details. To help
explain, here's a few use cases and what they would entail:

* A library developer needs to expose functionality to crates which link
  against their library. As a consequence of the first case, this means that
  anything which is usable externally must be `pub` from the root down to the
  destination item. Any private item in the chain will disallow external
  accesses.

* A crate needs a global available "helper module" to itself, but it doesn't
  want to expose the helper module as a public API. To accomplish this, the
  root of the crate's hierarchy would have a private module which then
  internally has a "public api". Because the entire crate is a descendant of
  the root, then the entire local crate can access this private module through
  the second case.

* When writing unit tests for a module, it's often a common idiom to have an
  immediate child of the module to-be-tested named `mod test`. This module
  could access any items of the parent module through the second case, meaning
  that internal implementation details could also be seamlessly tested from the
  child module.

In the second case, it mentions that a private item "can be accessed" by the
current module and its descendants, but the exact meaning of accessing an item
depends on what the item is. Accessing a module, for example, would mean
looking inside of it (to import more items). On the other hand, accessing a
function would mean that it is invoked. Additionally, path expressions and
import statements are considered to access an item in the sense that the
import/expression is only valid if the destination is in the current visibility
scope.

Here's an example of a program which exemplifies the three cases outlined
above:

```
// This module is private, meaning that no external crate can access this
// module. Because it is private at the root of this current crate, however, any
// module in the crate may access any publicly visible item in this module.
mod crate_helper_module {

    // This function can be used by anything in the current crate
    pub fn crate_helper() {}

    // This function *cannot* be used by anything else in the crate. It is not
    // publicly visible outside of the `crate_helper_module`, so only this
    // current module and its descendants may access it.
    fn implementation_detail() {}
}

// This function is "public to the root" meaning that it's available to external
// crates linking against this one.
pub fn public_api() {}

// Similarly to 'public_api', this module is public so external crates may look
// inside of it.
pub mod submodule {
    use crate_helper_module;

    pub fn my_method() {
        // Any item in the local crate may invoke the helper module's public
        // interface through a combination of the two rules above.
        crate_helper_module::crate_helper();
    }

    // This function is hidden to any module which is not a descendant of
    // `submodule`
    fn my_implementation() {}

    #[cfg(test)]
    mod test {

        #[test]
        fn test_my_implementation() {
            // Because this module is a descendant of `submodule`, it's allowed
            // to access private items inside of `submodule` without a privacy
            // violation.
            super::my_implementation();
        }
    }
}

# fn main() {}
```

For a rust program to pass the privacy checking pass, all paths must be valid
accesses given the two rules above. This includes all use statements,
expressions, types, etc.

### Re-exporting and Visibility

Rust allows publicly re-exporting items through a `pub use` directive. Because
this is a public directive, this allows the item to be used in the current
module through the rules above. It essentially allows public access into the
re-exported item. For example, this program is valid:

```
pub use self::implementation as api;

mod implementation {
    pub fn f() {}
}

# fn main() {}
```

This means that any external crate referencing `implementation::f` would
receive a privacy violation, while the path `api::f` would be allowed.

When re-exporting a private item, it can be thought of as allowing the "privacy
chain" being short-circuited through the reexport instead of passing through
the namespace hierarchy as it normally would.

## Attributes

```{.ebnf .gram}
attribute : "#!" ? '[' meta_item ']' ;
meta_item : ident [ '=' literal
                  | '(' meta_seq ')' ] ? ;
meta_seq : meta_item [ ',' meta_seq ] ? ;
```

Any item declaration may have an _attribute_ applied to it. Attributes in Rust
are modeled on Attributes in ECMA-335, with the syntax coming from ECMA-334
(C#). An attribute is a general, free-form metadatum that is interpreted
according to name, convention, and language and compiler version. Attributes
may appear as any of:

* A single identifier, the attribute name
* An identifier followed by the equals sign '=' and a literal, providing a
  key/value pair
* An identifier followed by a parenthesized list of sub-attribute arguments

Attributes with a bang ("!") after the hash ("#") apply to the item that the
attribute is declared within. Attributes that do not have a bang after the hash
apply to the item that follows the attribute.

An example of attributes:

```{.rust}
// General metadata applied to the enclosing module or crate.
#![crate_type = "lib"]

// A function marked as a unit test
#[test]
fn test_foo() {
  /* ... */
}

// A conditionally-compiled module
#[cfg(target_os="linux")]
mod bar {
  /* ... */
}

// A lint attribute used to suppress a warning/error
#[allow(non_camel_case_types)]
type int8_t = i8;
```

> **Note:** At some point in the future, the compiler will distinguish between
> language-reserved and user-available attributes. Until then, there is
> effectively no difference between an attribute handled by a loadable syntax
> extension and the compiler.

### Crate-only attributes

- `crate_name` - specify the this crate's crate name.
- `crate_type` - see [linkage](#linkage).
- `feature` - see [compiler features](#compiler-features).
- `no_builtins` - disable optimizing certain code patterns to invocations of
                  library functions that are assumed to exist
- `no_main` - disable emitting the `main` symbol. Useful when some other
   object being linked to defines `main`.
- `no_start` - disable linking to the `native` crate, which specifies the
  "start" language item.
- `no_std` - disable linking to the `std` crate.
- `plugin` — load a list of named crates as compiler plugins, e.g.
             `#![plugin(foo, bar)]`. Optional arguments for each plugin,
             i.e. `#![plugin(foo(... args ...))]`, are provided to the plugin's
             registrar function.  The `plugin` feature gate is required to use
             this attribute.

### Module-only attributes

- `no_implicit_prelude` - disable injecting `use std::prelude::*` in this
  module.
- `path` - specifies the file to load the module from. `#[path="foo.rs"] mod
  bar;` is equivalent to `mod bar { /* contents of foo.rs */ }`. The path is
  taken relative to the directory that the current module is in.

### Function-only attributes

- `main` - indicates that this function should be passed to the entry point,
  rather than the function in the crate root named `main`.
- `plugin_registrar` - mark this function as the registration point for
  [compiler plugins][plugin], such as loadable syntax extensions.
- `start` - indicates that this function should be used as the entry point,
  overriding the "start" language item. See the "start" [language
  item](#language-items) for more details.
- `test` - indicates that this function is a test function, to only be compiled
  in case of `--test`.
- `should_fail` - indicates that this test function should panic, inverting the success condition.
- `cold` - The function is unlikely to be executed, so optimize it (and calls
  to it) differently.

### Static-only attributes

- `thread_local` - on a `static mut`, this signals that the value of this
  static may change depending on the current thread. The exact consequences of
  this are implementation-defined.

### FFI attributes

On an `extern` block, the following attributes are interpreted:

- `link_args` - specify arguments to the linker, rather than just the library
  name and type. This is feature gated and the exact behavior is
  implementation-defined (due to variety of linker invocation syntax).
- `link` - indicate that a native library should be linked to for the
  declarations in this block to be linked correctly. `link` supports an optional `kind`
  key with three possible values: `dylib`, `static`, and `framework`. See [external blocks](#external-blocks) for more about external blocks. Two
  examples: `#[link(name = "readline")]` and
  `#[link(name = "CoreFoundation", kind = "framework")]`.

On declarations inside an `extern` block, the following attributes are
interpreted:

- `link_name` - the name of the symbol that this function or static should be
  imported as.
- `linkage` - on a static, this specifies the [linkage
  type](http://llvm.org/docs/LangRef.html#linkage-types).

On `enum`s:

- `repr` - on C-like enums, this sets the underlying type used for
  representation. Takes one argument, which is the primitive
  type this enum should be represented for, or `C`, which specifies that it
  should be the default `enum` size of the C ABI for that platform. Note that
  enum representation in C is undefined, and this may be incorrect when the C
  code is compiled with certain flags.

On `struct`s:

- `repr` - specifies the representation to use for this struct. Takes a list
  of options. The currently accepted ones are `C` and `packed`, which may be
  combined. `C` will use a C ABI compatible struct layout, and `packed` will
  remove any padding between fields (note that this is very fragile and may
  break platforms which require aligned access).

### Macro-related attributes

- `macro_use` on a `mod` — macros defined in this module will be visible in the
  module's parent, after this module has been included.

- `macro_use` on an `extern crate` — load macros from this crate.  An optional
  list of names `#[macro_use(foo, bar)]` restricts the import to just those
  macros named.  The `extern crate` must appear at the crate root, not inside
  `mod`, which ensures proper function of the [`$crate` macro
  variable](book/macros.html#the-variable-$crate).

- `macro_reexport` on an `extern crate` — re-export the named macros.

- `macro_export` - export a macro for cross-crate usage.

- `no_link` on an `extern crate` — even if we load this crate for macros, don't
  link it into the output.

See the [macros section of the
book](book/macros.html#scoping-and-macro-import/export) for more information on
macro scope.


### Miscellaneous attributes

- `export_name` - on statics and functions, this determines the name of the
  exported symbol.
- `link_section` - on statics and functions, this specifies the section of the
  object file that this item's contents will be placed into.
- `no_mangle` - on any item, do not apply the standard name mangling. Set the
  symbol for this item to its identifier.
- `packed` - on structs or enums, eliminate any padding that would be used to
  align fields.
- `simd` - on certain tuple structs, derive the arithmetic operators, which
  lower to the target's SIMD instructions, if any; the `simd` feature gate
  is necessary to use this attribute.
- `static_assert` - on statics whose type is `bool`, terminates compilation
  with an error if it is not initialized to `true`.
- `unsafe_destructor` - allow implementations of the "drop" language item
  where the type it is implemented for does not implement the "send" language
  item; the `unsafe_destructor` feature gate is needed to use this attribute
- `unsafe_no_drop_flag` - on structs, remove the flag that prevents
  destructors from being run twice. Destructors might be run multiple times on
  the same object with this attribute.
- `doc` - Doc comments such as `/// foo` are equivalent to `#[doc = "foo"]`.
- `rustc_on_unimplemented` - Write a custom note to be shown along with the error
   when the trait is found to be unimplemented on a type.
   You may use format arguments like `{T}`, `{A}` to correspond to the
   types at the point of use corresponding to the type parameters of the
   trait of the same name. `{Self}` will be replaced with the type that is supposed
   to implement the trait but doesn't. To use this, the `on_unimplemented` feature gate
   must be enabled.

### Conditional compilation

Sometimes one wants to have different compiler outputs from the same code,
depending on build target, such as targeted operating system, or to enable
release builds.

There are two kinds of configuration options, one that is either defined or not
(`#[cfg(foo)]`), and the other that contains a string that can be checked
against (`#[cfg(bar = "baz")]` (currently only compiler-defined configuration
options can have the latter form).

```
// The function is only included in the build when compiling for OSX
#[cfg(target_os = "macos")]
fn macos_only() {
  // ...
}

// This function is only included when either foo or bar is defined
#[cfg(any(foo, bar))]
fn needs_foo_or_bar() {
  // ...
}

// This function is only included when compiling for a unixish OS with a 32-bit
// architecture
#[cfg(all(unix, target_word_size = "32"))]
fn on_32bit_unix() {
  // ...
}

// This function is only included when foo is not defined
#[cfg(not(foo))]
fn needs_not_foo() {
  // ...
}
```

This illustrates some conditional compilation can be achieved using the
`#[cfg(...)]` attribute. `any`, `all` and `not` can be used to assemble
arbitrarily complex configurations through nesting.

The following configurations must be defined by the implementation:

* `target_arch = "..."`. Target CPU architecture, such as `"x86"`, `"x86_64"`
  `"mips"`, `"powerpc"`, `"arm"`, or `"aarch64"`.
* `target_endian = "..."`. Endianness of the target CPU, either `"little"` or
  `"big"`.
* `target_family = "..."`. Operating system family of the target, e. g.
  `"unix"` or `"windows"`. The value of this configuration option is defined
  as a configuration itself, like `unix` or `windows`.
* `target_os = "..."`. Operating system of the target, examples include
  `"win32"`, `"macos"`, `"linux"`, `"android"`, `"freebsd"`, `"dragonfly"` or
  `"openbsd"`.
* `target_word_size = "..."`. Target word size in bits. This is set to `"32"`
  for targets with 32-bit pointers, and likewise set to `"64"` for 64-bit
  pointers.
* `unix`. See `target_family`.
* `windows`. See `target_family`.

### Lint check attributes

A lint check names a potentially undesirable coding pattern, such as
unreachable code or omitted documentation, for the static entity to which the
attribute applies.

For any lint check `C`:

* `allow(C)` overrides the check for `C` so that violations will go
   unreported,
* `deny(C)` signals an error after encountering a violation of `C`,
* `forbid(C)` is the same as `deny(C)`, but also forbids changing the lint
   level afterwards,
* `warn(C)` warns about violations of `C` but continues compilation.

The lint checks supported by the compiler can be found via `rustc -W help`,
along with their default settings.  [Compiler
plugins](book/plugins.html#lint-plugins) can provide additional lint checks.

```{.ignore}
mod m1 {
    // Missing documentation is ignored here
    #[allow(missing_docs)]
    pub fn undocumented_one() -> i32 { 1 }

    // Missing documentation signals a warning here
    #[warn(missing_docs)]
    pub fn undocumented_too() -> i32 { 2 }

    // Missing documentation signals an error here
    #[deny(missing_docs)]
    pub fn undocumented_end() -> i32 { 3 }
}
```

This example shows how one can use `allow` and `warn` to toggle a particular
check on and off:

```{.ignore}
#[warn(missing_docs)]
mod m2{
    #[allow(missing_docs)]
    mod nested {
        // Missing documentation is ignored here
        pub fn undocumented_one() -> i32 { 1 }

        // Missing documentation signals a warning here,
        // despite the allow above.
        #[warn(missing_docs)]
        pub fn undocumented_two() -> i32 { 2 }
    }

    // Missing documentation signals a warning here
    pub fn undocumented_too() -> i32 { 3 }
}
```

This example shows how one can use `forbid` to disallow uses of `allow` for
that lint check:

```{.ignore}
#[forbid(missing_docs)]
mod m3 {
    // Attempting to toggle warning signals an error here
    #[allow(missing_docs)]
    /// Returns 2.
    pub fn undocumented_too() -> i32 { 2 }
}
```

### Language items

Some primitive Rust operations are defined in Rust code, rather than being
implemented directly in C or assembly language. The definitions of these
operations have to be easy for the compiler to find. The `lang` attribute
makes it possible to declare these operations. For example, the `str` module
in the Rust standard library defines the string equality function:

```{.ignore}
#[lang="str_eq"]
pub fn eq_slice(a: &str, b: &str) -> bool {
    // details elided
}
```

The name `str_eq` has a special meaning to the Rust compiler, and the presence
of this definition means that it will use this definition when generating calls
to the string equality function.

A complete list of the built-in language items will be added in the future.

### Inline attributes

The inline attribute is used to suggest to the compiler to perform an inline
expansion and place a copy of the function or static in the caller rather than
generating code to call the function or access the static where it is defined.

The compiler automatically inlines functions based on internal heuristics.
Incorrectly inlining functions can actually making the program slower, so it
should be used with care.

Immutable statics are always considered inlineable unless marked with
`#[inline(never)]`. It is undefined whether two different inlineable statics
have the same memory address. In other words, the compiler is free to collapse
duplicate inlineable statics together.

`#[inline]` and `#[inline(always)]` always causes the function to be serialized
into crate metadata to allow cross-crate inlining.

There are three different types of inline attributes:

* `#[inline]` hints the compiler to perform an inline expansion.
* `#[inline(always)]` asks the compiler to always perform an inline expansion.
* `#[inline(never)]` asks the compiler to never perform an inline expansion.

### `derive`

The `derive` attribute allows certain traits to be automatically implemented
for data structures. For example, the following will create an `impl` for the
`PartialEq` and `Clone` traits for `Foo`, the type parameter `T` will be given
the `PartialEq` or `Clone` constraints for the appropriate `impl`:

```
#[derive(PartialEq, Clone)]
struct Foo<T> {
    a: i32,
    b: T
}
```

The generated `impl` for `PartialEq` is equivalent to

```
# struct Foo<T> { a: i32, b: T }
impl<T: PartialEq> PartialEq for Foo<T> {
    fn eq(&self, other: &Foo<T>) -> bool {
        self.a == other.a && self.b == other.b
    }

    fn ne(&self, other: &Foo<T>) -> bool {
        self.a != other.a || self.b != other.b
    }
}
```

Supported traits for `derive` are:

* Comparison traits: `PartialEq`, `Eq`, `PartialOrd`, `Ord`.
* Serialization: `Encodable`, `Decodable`. These require `serialize`.
* `Clone`, to create `T` from `&T` via a copy.
* `Default`, to create an empty instance of a data type.
* `FromPrimitive`, to create an instance from a numeric primitive.
* `Hash`, to iterate over the bytes in a data type.
* `Rand`, to create a random instance of a data type.
* `Debug`, to format a value using the `{:?}` formatter.
* `Copy`, for "Plain Old Data" types which can be copied by simply moving bits.

### Compiler Features

Certain aspects of Rust may be implemented in the compiler, but they're not
necessarily ready for every-day use. These features are often of "prototype
quality" or "almost production ready", but may not be stable enough to be
considered a full-fledged language feature.

For this reason, Rust recognizes a special crate-level attribute of the form:

```{.ignore}
#![feature(feature1, feature2, feature3)]
```

This directive informs the compiler that the feature list: `feature1`,
`feature2`, and `feature3` should all be enabled. This is only recognized at a
crate-level, not at a module-level. Without this directive, all features are
considered off, and using the features will result in a compiler error.

The currently implemented features of the reference compiler are:

* `advanced_slice_patterns` - see the [match expressions](#match-expressions)
                              section for discussion; the exact semantics of
                              slice patterns are subject to change.

* `asm` - The `asm!` macro provides a means for inline assembly. This is often
          useful, but the exact syntax for this feature along with its
          semantics are likely to change, so this macro usage must be opted
          into.

* `associated_types` - Allows type aliases in traits. Experimental.

* `box_patterns` - Allows `box` patterns, the exact semantics of which
                   is subject to change.

* `box_syntax` - Allows use of `box` expressions, the exact semantics of which
                 is subject to change.

* `concat_idents` - Allows use of the `concat_idents` macro, which is in many
                    ways insufficient for concatenating identifiers, and may be
                    removed entirely for something more wholesome.

* `custom_attribute` - Allows the usage of attributes unknown to the compiler
                       so that new attributes can be added in a bacwards compatible
                       manner (RFC 572).

* `intrinsics` - Allows use of the "rust-intrinsics" ABI. Compiler intrinsics
                 are inherently unstable and no promise about them is made.

* `int_uint` - Allows the use of the `int` and `uint` types, which are deprecated.
               Use `isize` and `usize` instead.

* `lang_items` - Allows use of the `#[lang]` attribute. Like `intrinsics`,
                 lang items are inherently unstable and no promise about them
                 is made.

* `link_args` - This attribute is used to specify custom flags to the linker,
                but usage is strongly discouraged. The compiler's usage of the
                system linker is not guaranteed to continue in the future, and
                if the system linker is not used then specifying custom flags
                doesn't have much meaning.

* `link_llvm_intrinsics` – Allows linking to LLVM intrinsics via
                           `#[link_name="llvm.*"]`.

* `linkage` - Allows use of the `linkage` attribute, which is not portable.

* `log_syntax` - Allows use of the `log_syntax` macro attribute, which is a
                 nasty hack that will certainly be removed.

* `main` - Allows use of the `#[main]` attribute, which changes the entry point
           into a Rust program. This capabiilty is subject to change.

* `macro_reexport` - Allows macros to be re-exported from one crate after being imported
                     from another. This feature was originally designed with the sole
                     use case of the Rust standard library in mind, and is subject to
                     change.

* `non_ascii_idents` - The compiler supports the use of non-ascii identifiers,
                       but the implementation is a little rough around the
                       edges, so this can be seen as an experimental feature
                       for now until the specification of identifiers is fully
                       fleshed out.

* `no_std` - Allows the `#![no_std]` crate attribute, which disables the implicit
             `extern crate std`. This typically requires use of the unstable APIs
             behind the libstd "facade", such as libcore and libcollections. It
             may also cause problems when using syntax extensions, including
             `#[derive]`.

* `on_unimplemented` - Allows the `#[rustc_on_unimplemented]` attribute, which allows
                       trait definitions to add specialized notes to error messages
                       when an implementation was expected but not found.

* `optin_builtin_traits` - Allows the definition of default and negative trait
                           implementations. Experimental.

* `plugin` - Usage of [compiler plugins][plugin] for custom lints or syntax extensions.
             These depend on compiler internals and are subject to change.

* `plugin_registrar` - Indicates that a crate provides [compiler plugins][plugin].

* `quote` - Allows use of the `quote_*!` family of macros, which are
            implemented very poorly and will likely change significantly
            with a proper implementation.

* `rustc_attrs` - Gates internal `#[rustc_*]` attributes which may be
                  for internal use only or have meaning added to them in the future.

* `rustc_diagnostic_macros`- A mysterious feature, used in the implementation
                             of rustc, not meant for mortals.

* `simd` - Allows use of the `#[simd]` attribute, which is overly simple and
           not the SIMD interface we want to expose in the long term.

* `simd_ffi` - Allows use of SIMD vectors in signatures for foreign functions.
               The SIMD interface is subject to change.

* `staged_api` - Allows usage of stability markers and `#![staged_api]` in a crate

* `start` - Allows use of the `#[start]` attribute, which changes the entry point
            into a Rust program. This capabiilty, especially the signature for the
            annotated function, is subject to change.

* `struct_inherit` - Allows using struct inheritance, which is barely
                     implemented and will probably be removed. Don't use this.

* `struct_variant` - Structural enum variants (those with named fields). It is
                     currently unknown whether this style of enum variant is as
                     fully supported as the tuple-forms, and it's not certain
                     that this style of variant should remain in the language.
                     For now this style of variant is hidden behind a feature
                     flag.

* `thread_local` - The usage of the `#[thread_local]` attribute is experimental
                   and should be seen as unstable. This attribute is used to
                   declare a `static` as being unique per-thread leveraging
                   LLVM's implementation which works in concert with the kernel
                   loader and dynamic linker. This is not necessarily available
                   on all platforms, and usage of it is discouraged.

* `trace_macros` - Allows use of the `trace_macros` macro, which is a nasty
                   hack that will certainly be removed.

* `unboxed_closures` - Rust's new closure design, which is currently a work in
                       progress feature with many known bugs.

* `unsafe_destructor` - Allows use of the `#[unsafe_destructor]` attribute,
                        which is considered wildly unsafe and will be
                        obsoleted by language improvements.

* `unsafe_no_drop_flag` - Allows use of the `#[unsafe_no_drop_flag]` attribute,
                          which removes hidden flag added to a type that
                          implements the `Drop` trait. The design for the
                          `Drop` flag is subject to change, and this feature
                          may be removed in the future.

* `unmarked_api` - Allows use of items within a `#![staged_api]` crate
                   which have not been marked with a stability marker.
                   Such items should not be allowed by the compiler to exist,
                   so if you need this there probably is a compiler bug.

* `visible_private_types` - Allows public APIs to expose otherwise private
                            types, e.g. as the return type of a public function.
                            This capability may be removed in the future.

If a feature is promoted to a language feature, then all existing programs will
start to receive compilation warnings about #[feature] directives which enabled
the new feature (because the directive is no longer necessary). However, if a
feature is decided to be removed from the language, errors will be issued (if
there isn't a parser error first). The directive in this case is no longer
necessary, and it's likely that existing code will break if the feature isn't
removed.

If an unknown feature is found in a directive, it results in a compiler error.
An unknown feature is one which has never been recognized by the compiler.

# Statements and expressions

Rust is _primarily_ an expression language. This means that most forms of
value-producing or effect-causing evaluation are directed by the uniform syntax
category of _expressions_. Each kind of expression can typically _nest_ within
each other kind of expression, and rules for evaluation of expressions involve
specifying both the value produced by the expression and the order in which its
sub-expressions are themselves evaluated.

In contrast, statements in Rust serve _mostly_ to contain and explicitly
sequence expression evaluation.

## Statements

A _statement_ is a component of a block, which is in turn a component of an
outer [expression](#expressions) or [function](#functions).

Rust has two kinds of statement: [declaration
statements](#declaration-statements) and [expression
statements](#expression-statements).

### Declaration statements

A _declaration statement_ is one that introduces one or more *names* into the
enclosing statement block. The declared names may denote new slots or new
items.

#### Item declarations

An _item declaration statement_ has a syntactic form identical to an
[item](#items) declaration within a module. Declaring an item &mdash; a
function, enumeration, structure, type, static, trait, implementation or module
&mdash; locally within a statement block is simply a way of restricting its
scope to a narrow region containing all of its uses; it is otherwise identical
in meaning to declaring the item outside the statement block.

> **Note**: there is no implicit capture of the function's dynamic environment when
> declaring a function-local item.

#### Slot declarations

```{.ebnf .gram}
let_decl : "let" pat [':' type ] ? [ init ] ? ';' ;
init : [ '=' ] expr ;
```

A _slot declaration_ introduces a new set of slots, given by a pattern. The
pattern may be followed by a type annotation, and/or an initializer expression.
When no type annotation is given, the compiler will infer the type, or signal
an error if insufficient type information is available for definite inference.
Any slots introduced by a slot declaration are visible from the point of
declaration until the end of the enclosing block scope.

### Expression statements

An _expression statement_ is one that evaluates an [expression](#expressions)
and ignores its result. The type of an expression statement `e;` is always
`()`, regardless of the type of `e`. As a rule, an expression statement's
purpose is to trigger the effects of evaluating its expression.

## Expressions

An expression may have two roles: it always produces a *value*, and it may have
*effects* (otherwise known as "side effects"). An expression *evaluates to* a
value, and has effects during *evaluation*. Many expressions contain
sub-expressions (operands). The meaning of each kind of expression dictates
several things:

* Whether or not to evaluate the sub-expressions when evaluating the expression
* The order in which to evaluate the sub-expressions
* How to combine the sub-expressions' values to obtain the value of the expression

In this way, the structure of expressions dictates the structure of execution.
Blocks are just another kind of expression, so blocks, statements, expressions,
and blocks again can recursively nest inside each other to an arbitrary depth.

#### Lvalues, rvalues and temporaries

Expressions are divided into two main categories: _lvalues_ and _rvalues_.
Likewise within each expression, sub-expressions may occur in _lvalue context_
or _rvalue context_. The evaluation of an expression depends both on its own
category and the context it occurs within.

An lvalue is an expression that represents a memory location. These expressions
are [paths](#path-expressions) (which refer to local variables, function and
method arguments, or static variables), dereferences (`*expr`), [indexing
expressions](#index-expressions) (`expr[expr]`), and [field
references](#field-expressions) (`expr.f`). All other expressions are rvalues.

The left operand of an [assignment](#assignment-expressions) or
[compound-assignment](#compound-assignment-expressions) expression is an lvalue
context, as is the single operand of a unary
[borrow](#unary-operator-expressions). All other expression contexts are
rvalue contexts.

When an lvalue is evaluated in an _lvalue context_, it denotes a memory
location; when evaluated in an _rvalue context_, it denotes the value held _in_
that memory location.

When an rvalue is used in an lvalue context, a temporary un-named lvalue is
created and used instead. A temporary's lifetime equals the largest lifetime
of any reference that points to it.

#### Moved and copied types

When a [local variable](#memory-slots) is used as an
[rvalue](#lvalues,-rvalues-and-temporaries) the variable will either be moved
or copied, depending on its type. All values whose type implements `Copy` are
copied, all others are moved.

### Literal expressions

A _literal expression_ consists of one of the [literal](#literals) forms
described earlier. It directly describes a number, character, string, boolean
value, or the unit value.

```{.literals}
();        // unit type
"hello";   // string type
'5';       // character type
5;         // integer type
```

### Path expressions

A [path](#paths) used as an expression context denotes either a local variable
or an item. Path expressions are [lvalues](#lvalues,-rvalues-and-temporaries).

### Tuple expressions

Tuples are written by enclosing zero or more comma-separated expressions in
parentheses. They are used to create [tuple-typed](#tuple-types) values.

```{.tuple}
(0,);
(0.0, 4.5);
("a", 4us, true);
```

### Unit expressions

The expression `()` denotes the _unit value_, the only value of the type with
the same name.

### Structure expressions

```{.ebnf .gram}
struct_expr : expr_path '{' ident ':' expr
                      [ ',' ident ':' expr ] *
                      [ ".." expr ] '}' |
              expr_path '(' expr
                      [ ',' expr ] * ')' |
              expr_path ;
```

There are several forms of structure expressions. A _structure expression_
consists of the [path](#paths) of a [structure item](#structures), followed by
a brace-enclosed list of one or more comma-separated name-value pairs,
providing the field values of a new instance of the structure. A field name
can be any identifier, and is separated from its value expression by a colon.
The location denoted by a structure field is mutable if and only if the
enclosing structure is mutable.

A _tuple structure expression_ consists of the [path](#paths) of a [structure
item](#structures), followed by a parenthesized list of one or more
comma-separated expressions (in other words, the path of a structure item
followed by a tuple expression). The structure item must be a tuple structure
item.

A _unit-like structure expression_ consists only of the [path](#paths) of a
[structure item](#structures).

The following are examples of structure expressions:

```
# struct Point { x: f64, y: f64 }
# struct TuplePoint(f64, f64);
# mod game { pub struct User<'a> { pub name: &'a str, pub age: u32, pub score: uint } }
# struct Cookie; fn some_fn<T>(t: T) {}
Point {x: 10.0, y: 20.0};
TuplePoint(10.0, 20.0);
let u = game::User {name: "Joe", age: 35, score: 100_000};
some_fn::<Cookie>(Cookie);
```

A structure expression forms a new value of the named structure type. Note
that for a given *unit-like* structure type, this will always be the same
value.

A structure expression can terminate with the syntax `..` followed by an
expression to denote a functional update. The expression following `..` (the
base) must have the same structure type as the new structure type being formed.
The entire expression denotes the result of constructing a new structure (with
the same type as the base expression) with the given values for the fields that
were explicitly specified and the values in the base expression for all other
fields.

```
# struct Point3d { x: i32, y: i32, z: i32 }
let base = Point3d {x: 1, y: 2, z: 3};
Point3d {y: 0, z: 10, .. base};
```

### Block expressions

```{.ebnf .gram}
block_expr : '{' [ stmt ';' | item ] *
                 [ expr ] '}' ;
```

A _block expression_ is similar to a module in terms of the declarations that
are possible. Each block conceptually introduces a new namespace scope. Use
items can bring new names into scopes and declared items are in scope for only
the block itself.

A block will execute each statement sequentially, and then execute the
expression (if given). If the block ends in a statement, its value is `()`:

```
let x: () = { println!("Hello."); };
```

If it ends in an expression, its value and type are that of the expression:

```
let x: i32 = { println!("Hello."); 5 };

assert_eq!(5, x);
```

### Method-call expressions

```{.ebnf .gram}
method_call_expr : expr '.' ident paren_expr_list ;
```

A _method call_ consists of an expression followed by a single dot, an
identifier, and a parenthesized expression-list. Method calls are resolved to
methods on specific traits, either statically dispatching to a method if the
exact `self`-type of the left-hand-side is known, or dynamically dispatching if
the left-hand-side expression is an indirect [object type](#object-types).

### Field expressions

```{.ebnf .gram}
field_expr : expr '.' ident ;
```

A _field expression_ consists of an expression followed by a single dot and an
identifier, when not immediately followed by a parenthesized expression-list
(the latter is a [method call expression](#method-call-expressions)). A field
expression denotes a field of a [structure](#structure-types).

```{.ignore .field}
mystruct.myfield;
foo().x;
(Struct {a: 10, b: 20}).a;
```

A field access is an [lvalue](#lvalues,-rvalues-and-temporaries) referring to
the value of that field. When the type providing the field inherits mutability,
it can be [assigned](#assignment-expressions) to.

Also, if the type of the expression to the left of the dot is a pointer, it is
automatically dereferenced to make the field access possible.

### Array expressions

```{.ebnf .gram}
array_expr : '[' "mut" ? vec_elems? ']' ;

array_elems : [expr [',' expr]*] | [expr ';' expr] ;
```

An [array](#array,-and-slice-types) _expression_ is written by enclosing zero
or more comma-separated expressions of uniform type in square brackets.

In the `[expr ';' expr]` form, the expression after the `';'` must be a
constant expression that can be evaluated at compile time, such as a
[literal](#literals) or a [static item](#static-items).

```
[1, 2, 3, 4];
["a", "b", "c", "d"];
[0; 128];              // array with 128 zeros
[0u8, 0u8, 0u8, 0u8];
```

### Index expressions

```{.ebnf .gram}
idx_expr : expr '[' expr ']' ;
```

[Array](#array,-and-slice-types)-typed expressions can be indexed by
writing a square-bracket-enclosed expression (the index) after them. When the
array is mutable, the resulting [lvalue](#lvalues,-rvalues-and-temporaries) can
be assigned to.

Indices are zero-based, and may be of any integral type. Vector access is
bounds-checked at run-time. When the check fails, it will put the thread in a
_panicked state_.

```{should-fail}
([1, 2, 3, 4])[0];
(["a", "b"])[10]; // panics
```

### Unary operator expressions

Rust defines three unary operators. They are all written as prefix operators,
before the expression they apply to.

* `-`
  : Negation. May only be applied to numeric types.
* `*`
  : Dereference. When applied to a [pointer](#pointer-types) it denotes the
    pointed-to location. For pointers to mutable locations, the resulting
    [lvalue](#lvalues,-rvalues-and-temporaries) can be assigned to.
    On non-pointer types, it calls the `deref` method of the `std::ops::Deref`
    trait, or the `deref_mut` method of the `std::ops::DerefMut` trait (if
    implemented by the type and required for an outer expression that will or
    could mutate the dereference), and produces the result of dereferencing the
    `&` or `&mut` borrowed pointer returned from the overload method.

* `!`
  : Logical negation. On the boolean type, this flips between `true` and
    `false`. On integer types, this inverts the individual bits in the
    two's complement representation of the value.

### Binary operator expressions

```{.ebnf .gram}
binop_expr : expr binop expr ;
```

Binary operators expressions are given in terms of [operator
precedence](#operator-precedence).

#### Arithmetic operators

Binary arithmetic expressions are syntactic sugar for calls to built-in traits,
defined in the `std::ops` module of the `std` library. This means that
arithmetic operators can be overridden for user-defined types. The default
meaning of the operators on standard types is given here.

* `+`
  : Addition and array/string concatenation.
    Calls the `add` method on the `std::ops::Add` trait.
* `-`
  : Subtraction.
    Calls the `sub` method on the `std::ops::Sub` trait.
* `*`
  : Multiplication.
    Calls the `mul` method on the `std::ops::Mul` trait.
* `/`
  : Quotient.
    Calls the `div` method on the `std::ops::Div` trait.
* `%`
  : Remainder.
    Calls the `rem` method on the `std::ops::Rem` trait.

#### Bitwise operators

Like the [arithmetic operators](#arithmetic-operators), bitwise operators are
syntactic sugar for calls to methods of built-in traits. This means that
bitwise operators can be overridden for user-defined types. The default
meaning of the operators on standard types is given here.

* `&`
  : And.
    Calls the `bitand` method of the `std::ops::BitAnd` trait.
* `|`
  : Inclusive or.
    Calls the `bitor` method of the `std::ops::BitOr` trait.
* `^`
  : Exclusive or.
    Calls the `bitxor` method of the `std::ops::BitXor` trait.
* `<<`
  : Logical left shift.
    Calls the `shl` method of the `std::ops::Shl` trait.
* `>>`
  : Logical right shift.
    Calls the `shr` method of the `std::ops::Shr` trait.

#### Lazy boolean operators

The operators `||` and `&&` may be applied to operands of boolean type. The
`||` operator denotes logical 'or', and the `&&` operator denotes logical
'and'. They differ from `|` and `&` in that the right-hand operand is only
evaluated when the left-hand operand does not already determine the result of
the expression. That is, `||` only evaluates its right-hand operand when the
left-hand operand evaluates to `false`, and `&&` only when it evaluates to
`true`.

#### Comparison operators

Comparison operators are, like the [arithmetic
operators](#arithmetic-operators), and [bitwise operators](#bitwise-operators),
syntactic sugar for calls to built-in traits. This means that comparison
operators can be overridden for user-defined types. The default meaning of the
operators on standard types is given here.

* `==`
  : Equal to.
    Calls the `eq` method on the `std::cmp::PartialEq` trait.
* `!=`
  : Unequal to.
    Calls the `ne` method on the `std::cmp::PartialEq` trait.
* `<`
  : Less than.
    Calls the `lt` method on the `std::cmp::PartialOrd` trait.
* `>`
  : Greater than.
    Calls the `gt` method on the `std::cmp::PartialOrd` trait.
* `<=`
  : Less than or equal.
    Calls the `le` method on the `std::cmp::PartialOrd` trait.
* `>=`
  : Greater than or equal.
    Calls the `ge` method on the `std::cmp::PartialOrd` trait.

#### Type cast expressions

A type cast expression is denoted with the binary operator `as`.

Executing an `as` expression casts the value on the left-hand side to the type
on the right-hand side.

A numeric value can be cast to any numeric type. A raw pointer value can be
cast to or from any integral type or raw pointer type. Any other cast is
unsupported and will fail to compile.

An example of an `as` expression:

```
# fn sum(v: &[f64]) -> f64 { 0.0 }
# fn len(v: &[f64]) -> i32 { 0 }

fn avg(v: &[f64]) -> f64 {
  let sum: f64 = sum(v);
  let sz: f64 = len(v) as f64;
  return sum / sz;
}
```

#### Assignment expressions

An _assignment expression_ consists of an
[lvalue](#lvalues,-rvalues-and-temporaries) expression followed by an equals
sign (`=`) and an [rvalue](#lvalues,-rvalues-and-temporaries) expression.

Evaluating an assignment expression [either copies or
moves](#moved-and-copied-types) its right-hand operand to its left-hand
operand.

```
# let mut x = 0;
# let y = 0;

x = y;
```

#### Compound assignment expressions

The `+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `<<`, and `>>` operators may be
composed with the `=` operator. The expression `lval OP= val` is equivalent to
`lval = lval OP val`. For example, `x = x + 1` may be written as `x += 1`.

Any such expression always has the [`unit`](#primitive-types) type.

#### Operator precedence

The precedence of Rust binary operators is ordered as follows, going from
strong to weak:

```{.text .precedence}
as
* / %
+ -
<< >>
&
^
|
== != < > <= >=
&&
||
= ..
```

Operators at the same precedence level are evaluated left-to-right. [Unary
operators](#unary-operator-expressions) have the same precedence level and are
stronger than any of the binary operators.

### Grouped expressions

An expression enclosed in parentheses evaluates to the result of the enclosed
expression. Parentheses can be used to explicitly specify evaluation order
within an expression.

```{.ebnf .gram}
paren_expr : '(' expr ')' ;
```

An example of a parenthesized expression:

```
let x: i32 = (2 + 3) * 4;
```


### Call expressions

```{.ebnf .gram}
expr_list : [ expr [ ',' expr ]* ] ? ;
paren_expr_list : '(' expr_list ')' ;
call_expr : expr paren_expr_list ;
```

A _call expression_ invokes a function, providing zero or more input slots and
an optional reference slot to serve as the function's output, bound to the
`lval` on the right hand side of the call. If the function eventually returns,
then the expression completes.

Some examples of call expressions:

```
# fn add(x: i32, y: i32) -> i32 { 0 }

let x: i32 = add(1i32, 2i32);
let pi: Result<f32, _> = "3.14".parse();
```

### Lambda expressions

```{.ebnf .gram}
ident_list : [ ident [ ',' ident ]* ] ? ;
lambda_expr : '|' ident_list '|' expr ;
```

A _lambda expression_ (sometimes called an "anonymous function expression")
defines a function and denotes it as a value, in a single expression. A lambda
expression is a pipe-symbol-delimited (`|`) list of identifiers followed by an
expression.

A lambda expression denotes a function that maps a list of parameters
(`ident_list`) onto the expression that follows the `ident_list`. The
identifiers in the `ident_list` are the parameters to the function. These
parameters' types need not be specified, as the compiler infers them from
context.

Lambda expressions are most useful when passing functions as arguments to other
functions, as an abbreviation for defining and capturing a separate function.

Significantly, lambda expressions _capture their environment_, which regular
[function definitions](#functions) do not. The exact type of capture depends
on the [function type](#function-types) inferred for the lambda expression. In
the simplest and least-expensive form (analogous to a ```|| { }``` expression),
the lambda expression captures its environment by reference, effectively
borrowing pointers to all outer variables mentioned inside the function.
Alternately, the compiler may infer that a lambda expression should copy or
move values (depending on their type) from the environment into the lambda
expression's captured environment.

In this example, we define a function `ten_times` that takes a higher-order
function argument, and call it with a lambda expression as an argument:

```
fn ten_times<F>(f: F) where F: Fn(i32) {
    let mut i = 0i32;
    while i < 10 {
        f(i);
        i += 1;
    }
}

ten_times(|j| println!("hello, {}", j));
```

### While loops

```{.ebnf .gram}
while_expr : "while" no_struct_literal_expr '{' block '}' ;
```

A `while` loop begins by evaluating the boolean loop conditional expression.
If the loop conditional expression evaluates to `true`, the loop body block
executes and control returns to the loop conditional expression. If the loop
conditional expression evaluates to `false`, the `while` expression completes.

An example:

```
let mut i = 0;

while i < 10 {
    println!("hello");
    i = i + 1;
}
```

### Infinite loops

A `loop` expression denotes an infinite loop.

```{.ebnf .gram}
loop_expr : [ lifetime ':' ] "loop" '{' block '}';
```

A `loop` expression may optionally have a _label_. If a label is present, then
labeled `break` and `continue` expressions nested within this loop may exit out
of this loop or return control to its head. See [Break
expressions](#break-expressions) and [Continue
expressions](#continue-expressions).

### Break expressions

```{.ebnf .gram}
break_expr : "break" [ lifetime ];
```

A `break` expression has an optional _label_. If the label is absent, then
executing a `break` expression immediately terminates the innermost loop
enclosing it. It is only permitted in the body of a loop. If the label is
present, then `break foo` terminates the loop with label `foo`, which need not
be the innermost label enclosing the `break` expression, but must enclose it.

### Continue expressions

```{.ebnf .gram}
continue_expr : "continue" [ lifetime ];
```

A `continue` expression has an optional _label_. If the label is absent, then
executing a `continue` expression immediately terminates the current iteration
of the innermost loop enclosing it, returning control to the loop *head*. In
the case of a `while` loop, the head is the conditional expression controlling
the loop. In the case of a `for` loop, the head is the call-expression
controlling the loop. If the label is present, then `continue foo` returns
control to the head of the loop with label `foo`, which need not be the
innermost label enclosing the `break` expression, but must enclose it.

A `continue` expression is only permitted in the body of a loop.

### For expressions

```{.ebnf .gram}
for_expr : "for" pat "in" no_struct_literal_expr '{' block '}' ;
```

A `for` expression is a syntactic construct for looping over elements provided
by an implementation of `std::iter::Iterator`.

An example of a for loop over the contents of an array:

```
# type Foo = i32;
# fn bar(f: Foo) { }
# let a = 0;
# let b = 0;
# let c = 0;

let v: &[Foo] = &[a, b, c];

for e in v.iter() {
    bar(*e);
}
```

An example of a for loop over a series of integers:

```
# fn bar(b:usize) { }
for i in 0..256 {
    bar(i);
}
```

### If expressions

```{.ebnf .gram}
if_expr : "if" no_struct_literal_expr '{' block '}'
          else_tail ? ;

else_tail : "else" [ if_expr | if_let_expr
                   | '{' block '}' ] ;
```

An `if` expression is a conditional branch in program control. The form of an
`if` expression is a condition expression, followed by a consequent block, any
number of `else if` conditions and blocks, and an optional trailing `else`
block. The condition expressions must have type `bool`. If a condition
expression evaluates to `true`, the consequent block is executed and any
subsequent `else if` or `else` block is skipped. If a condition expression
evaluates to `false`, the consequent block is skipped and any subsequent `else
if` condition is evaluated. If all `if` and `else if` conditions evaluate to
`false` then any `else` block is executed.

### Match expressions

```{.ebnf .gram}
match_expr : "match" no_struct_literal_expr '{' match_arm * '}' ;

match_arm : attribute * match_pat "=>" [ expr "," | '{' block '}' ] ;

match_pat : pat [ '|' pat ] * [ "if" expr ] ? ;
```

A `match` expression branches on a *pattern*. The exact form of matching that
occurs depends on the pattern. Patterns consist of some combination of
literals, destructured arrays or enum constructors, structures and tuples,
variable binding specifications, wildcards (`..`), and placeholders (`_`). A
`match` expression has a *head expression*, which is the value to compare to
the patterns. The type of the patterns must equal the type of the head
expression.

In a pattern whose head expression has an `enum` type, a placeholder (`_`)
stands for a *single* data field, whereas a wildcard `..` stands for *all* the
fields of a particular variant. For example:

```
#![feature(box_patterns)]
#![feature(box_syntax)]
enum List<X> { Nil, Cons(X, Box<List<X>>) }

fn main() {
    let x: List<i32> = List::Cons(10, box List::Cons(11, box List::Nil));

    match x {
        List::Cons(_, box List::Nil) => panic!("singleton list"),
        List::Cons(..)               => return,
        List::Nil                    => panic!("empty list")
    }
}
```

The first pattern matches lists constructed by applying `Cons` to any head
value, and a tail value of `box Nil`. The second pattern matches _any_ list
constructed with `Cons`, ignoring the values of its arguments. The difference
between `_` and `..` is that the pattern `C(_)` is only type-correct if `C` has
exactly one argument, while the pattern `C(..)` is type-correct for any enum
variant `C`, regardless of how many arguments `C` has.

Used inside an array pattern, `..` stands for any number of elements, when the
`advanced_slice_patterns` feature gate is turned on. This wildcard can be used
at most once for a given array, which implies that it cannot be used to
specifically match elements that are at an unknown distance from both ends of a
array, like `[.., 42, ..]`. If preceded by a variable name, it will bind the
corresponding slice to the variable. Example:

```
# #![feature(advanced_slice_patterns)]
fn is_symmetric(list: &[u32]) -> bool {
    match list {
        [] | [_]                   => true,
        [x, inside.., y] if x == y => is_symmetric(inside),
        _                          => false
    }
}

fn main() {
    let sym     = &[0, 1, 4, 2, 4, 1, 0];
    let not_sym = &[0, 1, 7, 2, 4, 1, 0];
    assert!(is_symmetric(sym));
    assert!(!is_symmetric(not_sym));
}
```

A `match` behaves differently depending on whether or not the head expression
is an [lvalue or an rvalue](#lvalues,-rvalues-and-temporaries). If the head
expression is an rvalue, it is first evaluated into a temporary location, and
the resulting value is sequentially compared to the patterns in the arms until
a match is found. The first arm with a matching pattern is chosen as the branch
target of the `match`, any variables bound by the pattern are assigned to local
variables in the arm's block, and control enters the block.

When the head expression is an lvalue, the match does not allocate a temporary
location (however, a by-value binding may copy or move from the lvalue). When
possible, it is preferable to match on lvalues, as the lifetime of these
matches inherits the lifetime of the lvalue, rather than being restricted to
the inside of the match.

An example of a `match` expression:

```
#![feature(box_patterns)]
#![feature(box_syntax)]
# fn process_pair(a: i32, b: i32) { }
# fn process_ten() { }

enum List<X> { Nil, Cons(X, Box<List<X>>) }

fn main() {
    let x: List<i32> = List::Cons(10, box List::Cons(11, box List::Nil));

    match x {
        List::Cons(a, box List::Cons(b, _)) => {
            process_pair(a, b);
        }
        List::Cons(10, _) => {
            process_ten();
        }
        List::Nil => {
            return;
        }
        _ => {
            panic!();
        }
    }
}
```

Patterns that bind variables default to binding to a copy or move of the
matched value (depending on the matched value's type). This can be changed to
bind to a reference by using the `ref` keyword, or to a mutable reference using
`ref mut`.

Subpatterns can also be bound to variables by the use of the syntax `variable @
subpattern`. For example:

```
#![feature(box_patterns)]
#![feature(box_syntax)]

enum List { Nil, Cons(uint, Box<List>) }

fn is_sorted(list: &List) -> bool {
    match *list {
        List::Nil | List::Cons(_, box List::Nil) => true,
        List::Cons(x, ref r @ box List::Cons(_, _)) => {
            match *r {
                box List::Cons(y, _) => (x <= y) && is_sorted(&**r),
                _ => panic!()
            }
        }
    }
}

fn main() {
    let a = List::Cons(6, box List::Cons(7, box List::Cons(42, box List::Nil)));
    assert!(is_sorted(&a));
}

```

Patterns can also dereference pointers by using the `&`, `&mut` and `box`
symbols, as appropriate. For example, these two matches on `x: &i32` are
equivalent:

```
# let x = &3;
let y = match *x { 0 => "zero", _ => "some" };
let z = match x { &0 => "zero", _ => "some" };

assert_eq!(y, z);
```

A pattern that's just an identifier, like `Nil` in the previous example, could
either refer to an enum variant that's in scope, or bind a new variable. The
compiler resolves this ambiguity by forbidding variable bindings that occur in
`match` patterns from shadowing names of variants that are in scope. For
example, wherever `List` is in scope, a `match` pattern would not be able to
bind `Nil` as a new name. The compiler interprets a variable pattern `x` as a
binding _only_ if there is no variant named `x` in scope. A convention you can
use to avoid conflicts is simply to name variants with upper-case letters, and
local variables with lower-case letters.

Multiple match patterns may be joined with the `|` operator. A range of values
may be specified with `...`. For example:

```
# let x = 2;

let message = match x {
  0 | 1  => "not many",
  2 ... 9 => "a few",
  _      => "lots"
};
```

Range patterns only work on scalar types (like integers and characters; not
like arrays and structs, which have sub-components). A range pattern may not
be a sub-range of another range pattern inside the same `match`.

Finally, match patterns can accept *pattern guards* to further refine the
criteria for matching a case. Pattern guards appear after the pattern and
consist of a bool-typed expression following the `if` keyword. A pattern guard
may refer to the variables bound within the pattern they follow.

```
# let maybe_digit = Some(0);
# fn process_digit(i: i32) { }
# fn process_other(i: i32) { }

let message = match maybe_digit {
  Some(x) if x < 10 => process_digit(x),
  Some(x) => process_other(x),
  None => panic!()
};
```

### If let expressions

```{.ebnf .gram}
if_let_expr : "if" "let" pat '=' expr '{' block '}'
               else_tail ? ;
else_tail : "else" [ if_expr | if_let_expr | '{' block '}' ] ;
```

An `if let` expression is semantically identical to an `if` expression but in place
of a condition expression it expects a refutable let statement. If the value of the
expression on the right hand side of the let statement matches the pattern, the corresponding
block will execute, otherwise flow proceeds to the first `else` block that follows.

### While let loops

```{.ebnf .gram}
while_let_expr : "while" "let" pat '=' expr '{' block '}' ;
```

A `while let` loop is semantically identical to a `while` loop but in place of a
condition expression it expects a refutable let statement. If the value of the
expression on the right hand side of the let statement matches the pattern, the
loop body block executes and control returns to the pattern matching statement.
Otherwise, the while expression completes.

### Return expressions

```{.ebnf .gram}
return_expr : "return" expr ? ;
```

Return expressions are denoted with the keyword `return`. Evaluating a `return`
expression moves its argument into the output slot of the current function,
destroys the current function activation frame, and transfers control to the
caller frame.

An example of a `return` expression:

```
fn max(a: i32, b: i32) -> i32 {
   if a > b {
      return a;
   }
   return b;
}
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

The `usize` type is an unsigned integer type with the same number of bits as the
platform's pointer type. It can represent every memory address in the process.

The `isize` type is a signed integer type with the same number of bits as the
platform's pointer type. The theoretical upper bound on object and array size
is the maximum `isize` value. This ensures that `isize` can be used to calculate
differences between pointers into an object or array and can address every byte
within an object along with one byte past the end.

### Textual types

The types `char` and `str` hold textual data.

A value of type `char` is a [Unicode scalar value](
http://www.unicode.org/glossary/#unicode_scalar_value) (i.e. a code point that
is not a surrogate), represented as a 32-bit unsigned word in the 0x0000 to
0xD7FF or 0xE000 to 0x10FFFF range. A `[char]` array is effectively an UCS-4 /
UTF-32 string.

A value of type `str` is a Unicode string, represented as an array of 8-bit
unsigned bytes holding a sequence of UTF-8 codepoints. Since `str` is of
unknown size, it is not a _first-class_ type, but can only be instantiated
through a pointer type, such as `&str` or `String`.

### Tuple types

A tuple *type* is a heterogeneous product of other types, called the *elements*
of the tuple. It has no nominal name and is instead structurally typed.

Tuple types and values are denoted by listing the types or values of their
elements, respectively, in a parenthesized, comma-separated list.

Because tuple elements don't have a name, they can only be accessed by
pattern-matching.

An example of a tuple type and its use:

```
type Pair<'a> = (i32, &'a str);
let p: Pair<'static> = (10, "hello");
let (a, b) = p;
assert!(b != "world");
```

### Array, and Slice types

Rust has two different types for a list of items:

* `[T; N]`, an 'array'.
* `&[T]`, a 'slice'.

An array has a fixed size, and can be allocated on either the stack or the
heap.

A slice is a 'view' into an array. It doesn't own the data it points
to, it borrows it.

An example of each kind:

```{rust}
let vec: Vec<i32> = vec![1, 2, 3];
let arr: [i32; 3] = [1, 2, 3];
let s: &[i32] = &vec[..];
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

let a: List<i32> = List::Cons(7, Box::new(List::Cons(13, Box::new(List::Nil))));
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
fn add(x: i32, y: i32) -> i32 {
  return x + y;
}

let mut x = add(5,7);

type Binop = fn(i32, i32) -> i32;
let bo: Binop = add;
x = bo(5,7);
```

### Closure types

```{.ebnf .notation}
closure_type := [ 'unsafe' ] [ '<' lifetime-list '>' ] '|' arg-list '|'
                [ ':' bound-list ] [ '->' type ]
lifetime-list := lifetime | lifetime ',' lifetime-list
arg-list := ident ':' type | ident ':' type ',' arg-list
bound-list := bound | bound '+' bound-list
bound := path | lifetime
```

The type of a closure mapping an input of type `A` to an output of type `B` is
`|A| -> B`. A closure with no arguments or return values has type `||`.

An example of creating and calling a closure:

```rust
let captured_var = 10;

let closure_no_args = |&:| println!("captured_var={}", captured_var);

let closure_args = |&: arg: i32| -> i32 {
  println!("captured_var={}, arg={}", captured_var, arg);
  arg // Note lack of semicolon after 'arg'
};

fn call_closure<F: Fn(), G: Fn(i32) -> i32>(c1: F, c2: G) {
  c1();
  c2(2);
}

call_closure(closure_no_args, closure_args);

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

impl Printable for i32 {
  fn stringify(&self) -> String { self.to_string() }
}

fn print(a: Box<Printable>) {
   println!("{}", a.stringify());
}

fn main() {
   print(Box::new(10) as Box<Printable>);
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
  : Types of this kind can be safely sent between threads.
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

Rust has a memory model centered around concurrently-executing _threads_. Thus
its memory model and its concurrency model are best discussed simultaneously,
as parts of each only make sense when considered from the perspective of the
other.

When reading about the memory model, keep in mind that it is partitioned in
order to support threads; and when reading about threads, keep in mind that their
isolation and communication mechanisms are only possible due to the ownership
and lifetime semantics of the memory model.

## Memory model

A Rust program's memory consists of a static set of *items*, a set of
[threads](#threads) each with its own *stack*, and a *heap*. Immutable portions of
the heap may be shared between threads, mutable portions may not.

Allocations in the stack consist of *slots*, and allocations in the heap
consist of *boxes*.

### Memory allocation and lifetime

The _items_ of a program are those functions, modules and types that have their
value calculated at compile-time and stored uniquely in the memory image of the
rust process. Items are neither dynamically allocated nor freed.

A thread's _stack_ consists of activation frames automatically allocated on entry
to each function as the thread executes. A stack allocation is reclaimed when
control leaves the frame containing it.

The _heap_ is a general term that describes boxes.  The lifetime of an
allocation in the heap depends on the lifetime of the box values pointing to
it. Since box values may themselves be passed in and out of frames, or stored
in the heap, heap allocations may outlive the frame they are allocated within.

### Memory ownership

A thread owns all memory it can *safely* reach through local variables, as well
as boxes and references.

When a thread sends a value that has the `Send` trait to another thread, it loses
ownership of the value sent and can no longer refer to it. This is statically
guaranteed by the combined use of "move semantics", and the compiler-checked
_meaning_ of the `Send` trait: it is only instantiated for (transitively)
sendable kinds of data constructor and pointers, never including references.

When a stack frame is exited, its local allocations are all released, and its
references to boxes are dropped.

When a thread finishes, its stack is necessarily empty and it therefore has no
references to any boxes; the remainder of its heap is immediately freed.

### Memory slots

A thread's stack contains slots.

A _slot_ is a component of a stack frame, either a function parameter, a
[temporary](#lvalues,-rvalues-and-temporaries), or a local variable.

A _local variable_ (or *stack-local* allocation) holds a value directly,
allocated within the stack's memory. The value is a part of the stack frame.

Local variables are immutable unless declared otherwise like: `let mut x = ...`.

Function parameters are immutable unless declared with `mut`. The `mut` keyword
applies only to the following parameter (so `|mut x, y|` and `fn f(mut x:
Box<i32>, y: Box<i32>)` declare one mutable variable `x` and one immutable
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
let x: Box<i32> = Box::new(10);
```

Box values exist in 1:1 correspondence with their heap allocation, copying a
box value makes a shallow copy of the pointer. Rust will consider a shallow
copy of a box to move ownership of the value. After a value has been moved,
the source location cannot be used unless it is reinitialized.

```
let x: Box<i32> = Box::new(10);
let y = x;
// attempting to use `x` will result in an error here
```

## Threads

Rust's primary concurrency mechanism is called a **thread**.

### Communication between threads

Rust threads are isolated and generally unable to interfere with one another's
memory directly, except through [`unsafe` code](#unsafe-functions).  All
contact between threads is mediated by safe forms of ownership transfer, and data
races on memory are prohibited by the type system.

When you wish to send data between threads, the values are restricted to the
[`Send` type-kind](#type-kinds). Restricting communication interfaces to this
kind ensures that no references move between threads. Thus access to an entire
data structure can be mediated through its owning "root" value; no further
locking or copying is required to avoid data races within the substructure of
such a value.

### Thread

The _lifecycle_ of a threads consists of a finite set of states and events that
cause transitions between the states. The lifecycle states of a thread are:

* running
* blocked
* panicked
* dead

A thread begins its lifecycle &mdash; once it has been spawned &mdash; in the
*running* state. In this state it executes the statements of its entry
function, and any functions called by the entry function.

A thread may transition from the *running* state to the *blocked* state any time
it makes a blocking communication call. When the call can be completed &mdash;
when a message arrives at a sender, or a buffer opens to receive a message
&mdash; then the blocked thread will unblock and transition back to *running*.

A thread may transition to the *panicked* state at any time, due being killed by
some external event or internally, from the evaluation of a `panic!()` macro.
Once *panicking*, a thread unwinds its stack and transitions to the *dead* state.
Unwinding the stack of a thread is done by the thread itself, on its own control
stack. If a value with a destructor is freed during unwinding, the code for the
destructor is run, also on the thread's control stack. Running the destructor
code causes a temporary transition to a *running* state, and allows the
destructor code to cause any subsequent state transitions. The original thread
of unwinding and panicking thereby may suspend temporarily, and may involve
(recursive) unwinding of the stack of a failed destructor. Nonetheless, the
outermost unwinding activity will continue until the stack is unwound and the
thread transitions to the *dead* state. There is no way to "recover" from thread
panics. Once a thread has temporarily suspended its unwinding in the *panicking*
state, a panic occurring from within this destructor results in *hard* panic.
A hard panic currently results in the process aborting.

A thread in the *dead* state cannot transition to other states; it exists only to
have its termination status inspected by other threads, and/or to await
reclamation when the last reference to it drops.

# Runtime services, linkage and debugging

The Rust _runtime_ is a relatively compact collection of Rust code that
provides fundamental services and datatypes to all Rust threads at run-time. It
is smaller and simpler than many modern language runtimes. It is tightly
integrated into the language's execution model of memory, threads, communication
and logging.

### Memory allocation

The runtime memory-management system is based on a _service-provider
interface_, through which the runtime requests blocks of memory from its
environment and releases them back to its environment when they are no longer
needed. The default implementation of the service-provider interface consists
of the C runtime functions `malloc` and `free`.

The runtime memory-management system, in turn, supplies Rust threads with
facilities for allocating releasing stacks, as well as allocating and freeing
heap data.

### Built in types

The runtime provides C and Rust code to assist with various built-in types,
such as arrays, strings, and the low level communication system (ports,
channels, threads).

Support for other built-in types such as simple types, tuples and enums is
open-coded by the Rust compiler.

### Thread scheduling and communication

The runtime provides code to manage inter-thread communication. This includes
the system of thread-lifecycle state transitions depending on the contents of
queues, as well as code to copy values between queues and their recipients and
to serialize values for transmission over operating-system inter-process
communication facilities.

### Linkage

The Rust compiler supports various methods to link crates together both
statically and dynamically. This section will explore the various methods to
link Rust crates together, and more information about native libraries can be
found in the [ffi section of the book][ffi].

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

# Appendix: Influences

Rust is not a particularly original language, with design elements coming from
a wide range of sources. Some of these are listed below (including elements
that have since been removed):

* SML, OCaml: algebraic datatypes, pattern matching, type inference,
  semicolon statement separation
* C++: references, RAII, smart pointers, move semantics, monomorphisation,
  memory model
* ML Kit, Cyclone: region based memory management
* Haskell (GHC): typeclasses, type families
* Newsqueak, Alef, Limbo: channels, concurrency
* Erlang: message passing, task failure, ~~linked task failure~~,
  ~~lightweight concurrency~~
* Swift: optional bindings
* Scheme: hygienic macros
* C#: attributes
* Ruby: ~~block syntax~~
* NIL, Hermes: ~~typestate~~
* [Unicode Annex #31](http://www.unicode.org/reports/tr31/): identifier and
  pattern syntax

[ffi]: book/ffi.html
[plugin]: book/plugins.html
