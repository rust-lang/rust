# Tokens

Tokens are primitive productions in the grammar defined by regular
(non-recursive) languages. "Simple" tokens are given in [string table
production] form, and occur in the rest of the
grammar as double-quoted strings. Other tokens have exact rules given.

[string table production]: string-table-productions.html

## Literals

A literal is an expression consisting of a single token, rather than a sequence
of tokens, that immediately and directly denotes the value it evaluates to,
rather than referring to it by name or some other evaluation rule. A literal is
a form of constant expression, so is evaluated (primarily) at compile time.

### Examples

#### Characters and strings

|                                              | Example         | `#` sets   | Characters  | Escapes             |
|----------------------------------------------|-----------------|------------|-------------|---------------------|
| [Character](#character-literals)             | `'H'`           | `N/A`      | All Unicode | [Quote](#quote-escapes) & [Byte](#byte-escapes) & [Unicode](#unicode-escapes) |
| [String](#string-literals)                   | `"hello"`       | `N/A`      | All Unicode | [Quote](#quote-escapes) & [Byte](#byte-escapes) & [Unicode](#unicode-escapes) |
| [Raw](#raw-string-literals)                  | `r#"hello"#`    | `0...`     | All Unicode | `N/A`                                                      |
| [Byte](#byte-literals)                       | `b'H'`          | `N/A`      | All ASCII   | [Quote](#quote-escapes) & [Byte](#byte-escapes)                               |
| [Byte string](#byte-string-literals)         | `b"hello"`      | `N/A`      | All ASCII   | [Quote](#quote-escapes) & [Byte](#byte-escapes)                               |
| [Raw byte string](#raw-byte-string-literals) | `br#"hello"#`   | `0...`     | All ASCII   | `N/A`                                                      |

#### Byte escapes

|   | Name |
|---|------|
| `\x7F` | 8-bit character code (exactly 2 digits) |
| `\n` | Newline |
| `\r` | Carriage return |
| `\t` | Tab |
| `\\` | Backslash |
| `\0` | Null |

#### Unicode escapes

|   | Name |
|---|------|
| `\u{7FFF}` | 24-bit Unicode character code (up to 6 digits) |

#### Quote escapes

|   | Name |
|---|------|
| `\'` | Single quote |
| `\"` | Double quote |

#### Numbers

| [Number literals](#number-literals)`*` | Example | Exponentiation | Suffixes |
|----------------------------------------|---------|----------------|----------|
| Decimal integer | `98_222` | `N/A` | Integer suffixes |
| Hex integer | `0xff` | `N/A` | Integer suffixes |
| Octal integer | `0o77` | `N/A` | Integer suffixes |
| Binary integer | `0b1111_0000` | `N/A` | Integer suffixes |
| Floating-point | `123.0E+77` | `Optional` | Floating-point suffixes |

`*` All number literals allow `_` as a visual separator: `1_234.0E+18f64`

#### Suffixes

| Integer | Floating-point |
|---------|----------------|
| `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `isize`, `usize` | `f32`, `f64` |

### Character and string literals

#### Character literals

A _character literal_ is a single Unicode character enclosed within two
`U+0027` (single-quote) characters, with the exception of `U+0027` itself,
which must be _escaped_ by a preceding `U+005C` character (`\`).

#### String literals

A _string literal_ is a sequence of any Unicode characters enclosed within two
`U+0022` (double-quote) characters, with the exception of `U+0022` itself,
which must be _escaped_ by a preceding `U+005C` character (`\`).

Line-break characters are allowed in string literals. Normally they represent
themselves (i.e. no translation), but as a special exception, when an unescaped
`U+005C` character (`\`) occurs immediately before the newline (`U+000A`), the
`U+005C` character, the newline, and all whitespace at the beginning of the
next line are ignored. Thus `a` and `b` are equal:

```rust
let a = "foobar";
let b = "foo\
         bar";

assert_eq!(a,b);
```

#### Character escapes

Some additional _escapes_ are available in either character or non-raw string
literals. An escape starts with a `U+005C` (`\`) and continues with one of the
following forms:

* An _8-bit code point escape_ starts with `U+0078` (`x`) and is
  followed by exactly two _hex digits_. It denotes the Unicode code point
  equal to the provided hex value.
* A _24-bit code point escape_ starts with `U+0075` (`u`) and is followed
  by up to six _hex digits_ surrounded by braces `U+007B` (`{`) and `U+007D`
  (`}`). It denotes the Unicode code point equal to the provided hex value.
* A _whitespace escape_ is one of the characters `U+006E` (`n`), `U+0072`
  (`r`), or `U+0074` (`t`), denoting the Unicode values `U+000A` (LF),
  `U+000D` (CR) or `U+0009` (HT) respectively.
* The _null escape_ is the character `U+0030` (`0`) and denotes the Unicode
  value `U+0000` (NUL).
* The _backslash escape_ is the character `U+005C` (`\`) which must be
  escaped in order to denote *itself*.

#### Raw string literals

Raw string literals do not process any escapes. They start with the character
`U+0072` (`r`), followed by zero or more of the character `U+0023` (`#`) and a
`U+0022` (double-quote) character. The _raw string body_ can contain any sequence
of Unicode characters and is terminated only by another `U+0022` (double-quote)
character, followed by the same number of `U+0023` (`#`) characters that preceded
the opening `U+0022` (double-quote) character.

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

### Byte and byte string literals

#### Byte literals

A _byte literal_ is a single ASCII character (in the `U+0000` to `U+007F`
range) or a single _escape_ preceded by the characters `U+0062` (`b`) and
`U+0027` (single-quote), and followed by the character `U+0027`. If the character
`U+0027` is present within the literal, it must be _escaped_ by a preceding
`U+005C` (`\`) character. It is equivalent to a `u8` unsigned 8-bit integer
_number literal_.

#### Byte string literals

A non-raw _byte string literal_ is a sequence of ASCII characters and _escapes_,
preceded by the characters `U+0062` (`b`) and `U+0022` (double-quote), and
followed by the character `U+0022`. If the character `U+0022` is present within
the literal, it must be _escaped_ by a preceding `U+005C` (`\`) character.
Alternatively, a byte string literal can be a _raw byte string literal_, defined
below. A byte string literal of length `n` is equivalent to a `&'static [u8; n]` borrowed fixed-sized array
of unsigned 8-bit integers.

Some additional _escapes_ are available in either byte or non-raw byte string
literals. An escape starts with a `U+005C` (`\`) and continues with one of the
following forms:

* A _byte escape_ escape starts with `U+0078` (`x`) and is
  followed by exactly two _hex digits_. It denotes the byte
  equal to the provided hex value.
* A _whitespace escape_ is one of the characters `U+006E` (`n`), `U+0072`
  (`r`), or `U+0074` (`t`), denoting the bytes values `0x0A` (ASCII LF),
  `0x0D` (ASCII CR) or `0x09` (ASCII HT) respectively.
* The _null escape_ is the character `U+0030` (`0`) and denotes the byte
  value `0x00` (ASCII NUL).
* The _backslash escape_ is the character `U+005C` (`\`) which must be
  escaped in order to denote its ASCII encoding `0x5C`.

#### Raw byte string literals

Raw byte string literals do not process any escapes. They start with the
character `U+0062` (`b`), followed by `U+0072` (`r`), followed by zero or more
of the character `U+0023` (`#`), and a `U+0022` (double-quote) character. The
_raw string body_ can contain any sequence of ASCII characters and is terminated
only by another `U+0022` (double-quote) character, followed by the same number of
`U+0023` (`#`) characters that preceded the opening `U+0022` (double-quote)
character. A raw byte string literal can not contain any non-ASCII byte.

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

### Number literals

A _number literal_ is either an _integer literal_ or a _floating-point
literal_. The grammar for recognizing the two kinds of literals is mixed.

#### Integer literals

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

The type of an _unsuffixed_ integer literal is determined by type inference:

* If an integer type can be _uniquely_ determined from the surrounding
  program context, the unsuffixed integer literal has that type.

* If the program context under-constrains the type, it defaults to the
  signed 32-bit integer `i32`.

* If the program context over-constrains the type, it is considered a
  static type error.

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

Note that the Rust syntax considers `-1i8` as an application of the [unary minus
operator] to an integer literal `1i8`, rather than
a single integer literal.

[unary minus operator]: expressions.html#unary-operator-expressions

#### Floating-point literals

A _floating-point literal_ has one of two forms:

* A _decimal literal_ followed by a period character `U+002E` (`.`). This is
  optionally followed by another decimal literal, with an optional _exponent_.
* A single _decimal literal_ followed by an _exponent_.

Like integer literals, a floating-point literal may be followed by a
suffix, so long as the pre-suffix part does not end with `U+002E` (`.`).
The suffix forcibly sets the type of the literal. There are two valid
_floating-point suffixes_, `f32` and `f64` (the 32-bit and 64-bit floating point
types), which explicitly determine the type of the literal.

The type of an _unsuffixed_ floating-point literal is determined by
type inference:

* If a floating-point type can be _uniquely_ determined from the
  surrounding program context, the unsuffixed floating-point literal
  has that type.

* If the program context under-constrains the type, it defaults to `f64`.

* If the program context over-constrains the type, it is considered a
  static type error.

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

The representation semantics of floating-point numbers are described in
["Machine Types"].

["Machine Types"]: types.html#machine-types

### Boolean literals

The two values of the boolean type are written `true` and `false`.

## Symbols

Symbols are a general class of printable [tokens] that play structural
roles in a variety of grammar productions. They are a
set of remaining miscellaneous printable tokens that do not
otherwise appear as [unary operators], [binary
operators], or [keywords].
They are catalogued in [the Symbols section][symbols] of the Grammar document.

[unary operators]: expressions.html#unary-operator-expressions
[binary operators]: expressions.html#binary-operator-expressions
[tokens]: #tokens
[symbols]: ../grammar.html#symbols
[keywords]: ../grammar.html#keywords