- Start Date: 2014-05-05
- RFC PR: [rust-lang/rfcs#69](https://github.com/rust-lang/rfcs/pull/69)
- Rust Issue: [rust-lang/rust#14646](https://github.com/rust-lang/rust/issues/14646)

# Summary

Add ASCII byte literals and ASCII byte string literals to the language,
similar to the existing (Unicode) character and string literals.
Before the RFC process was in place, 
this was discussed in [#4334](https://github.com/mozilla/rust/issues/4334).


# Motivation

Programs dealing with text usually should use Unicode,
represented in Rust by the `str` and `char` types.
In some cases however,
a program may be dealing with bytes that can not be interpreted as Unicode as a whole,
but still contain ASCII compatible bits.

For example, the HTTP protocol was originally defined as Latin-1,
but in practice different pieces of the same request or response
can use different encodings.
The PDF file format is mostly ASCII,
but can contain UTF-16 strings and raw binary data.

There is a precedent at least in Python, which has both Unicode and byte strings.


# Drawbacks

The language becomes slightly more complex,
although that complexity should be limited to the parser.


# Detailed design

Using terminology from [the Reference Manual](http://static.rust-lang.org/doc/master/rust.html#character-and-string-literals):

Extend the syntax of expressions and patterns to add
byte literals of type `u8` and
byte string literals of type `&'static [u8]` (or `[u8]`, post-DST).
They are identical to the existing character and string literals, except that:

* They are prefixed with a `b` (for "binary"), to distinguish them.
  This is similar to the `r` prefix for raw strings.
* Unescaped code points in the body must be in the ASCII range: U+0000 to U+007F.
* `'\x5c' 'u' hex_digit 4` and `'\x5c' 'U' hex_digit 8` escapes are not allowed.
* `'\x5c' 'x' hex_digit 2` escapes represent a single byte rather than a code point.
  (They are the only way to express a non-ASCII byte.)

Examples: `b'A' == 65u8`, `b'\t' == 9u8`, `b'\xFF' == 0xFFu8`,
`b"A\t\xFF" == [65u8, 9, 0xFF]`

Assuming `buffer` of type `&[u8]`
```rust
match buffer[i] {
    b'a' .. b'z' => { /* ... */ }
    c => { /* ... */ }
}
```


# Alternatives

Status quo: patterns must use numeric literals for ASCII values,
or (for a single byte, not a byte string) cast to char

```rust
match buffer[i] {
    c @ 0x61 .. 0x7A => { /* ... */ }
    c => { /* ... */ }
}
match buffer[i] as char {
    // `c` is of the wrong type!
    c @ 'a' .. 'z' => { /* ... */ }
    c => { /* ... */ }
}
```

Another option is to change the syntax so that macros such as
[`bytes!()`](http://static.rust-lang.org/doc/master/std/macros/builtin/macro.bytes.html)
can be used in patterns, and add a `byte!()` macro:

```rust
match buffer[i] {
    c @ byte!('a') .. byte!('z') => { /* ... */ }
    c => { /* ... */ }
}q
```

This RFC was written to align the syntax with Python,
but there could be many variations such as using a different prefix (maybe `a` for ASCII),
or using a suffix instead (maybe `u8`, as in integer literals).

The code points from syntax could be encoded as UTF-8
rather than being mapped to bytes of the same value,
but assuming UTF-8 is not always appropriate when working with bytes.

See also previous discussion in [#4334](https://github.com/mozilla/rust/issues/4334).


# Unresolved questions

Should there be "raw byte string" literals?
E.g. `pdf_file.write(rb"<< /Title (FizzBuzz \(Part one\)) >>")`

Should control characters (U+0000 to U+001F) be disallowed in syntax?
This should be consistent across all kinds of literals.

Should the `bytes!()` macro be removed in favor of this?
