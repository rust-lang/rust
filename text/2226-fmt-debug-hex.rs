- Feature Name: fmt-debug-hex
- Start Date: 2017-11-24
- RFC PR: https://github.com/rust-lang/rfcs/pull/2226
- Rust Issue: https://github.com/rust-lang/rust/issues/48584

# Summary
[summary]: #summary

Add support for formatting integers as hexadecimal with the `fmt::Debug` trait,
including when they occur within larger types.

```rust
println!("{:02X?}", b"AZaz\0")
```
```
[41, 5A, 61, 7A, 00]
```

# Motivation
[motivation]: #motivation

Sometimes the bits that make up an integer are more meaningful than its purely numerical value.
For example, an RGBA color encoded in `u32` with 8 bits per channel is easier to understand
when shown as `00CC44FF` than `13387007`.

The `std::fmt::UpperHex` and `std::fmt::LowerHex` traits provide hexadecimal formatting
through `{:X}` and `{:x}` in formatting strings,
but they’re only implemented for plain integer types
and not other types like slices that might contain integers.

The `std::fmt::Debug` trait (used with `{:?}`) however is intended for
formatting “in a programmer-facing, debugging context”.
It can be derived, and doing so is recommended for most types.

This RFC proposes adding the missing combination of:

* Output intended primarily for end-users (`Display`) v.s. for programmers (`Debug`)
* Numbers shown in decimal v.s. hexadecimal

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

In formatting strings like in the `format!` and `println!` macros,
the formatting parameters `x` or `X` − to select lower-case or upper-case hexadecimal −
can now be combined with `?` which select the `Debug` trait.

For example, `format!("{:X?}", [65280].first())` returns `Some(FF00)`.

This can also be combined with other formatting parameters.
For example, `format!("{:02X?}", b"AZaz\0")` zero-pads each byte to two hexadecimal digits
and return `[41, 5A, 61, 7A, 00]`.

An API returning `Vec<u32>` might be tested like this:

```rust
let return_value = foo(bar);
let expected = &[ /* ... */ ][..];
assert!(return_value == expected, "{:08X?} != {:08X?}", return_value, expected);
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Formatting strings

The syntax of formatting strings
is [specified with a grammar](https://doc.rust-lang.org/std/fmt/#syntax)
which at the moment is as follows:

```
format_string := <text> [ maybe-format <text> ] *
maybe-format := '{' '{' | '}' '}' | <format>
format := '{' [ argument ] [ ':' format_spec ] '}'
argument := integer | identifier

format_spec := [[fill]align][sign]['#']['0'][width]['.' precision][type]
fill := character
align := '<' | '^' | '>'
sign := '+' | '-'
width := count
precision := count | '*'
type := identifier | ''
count := parameter | integer
parameter := argument '$'
```

This RFC adds an optional *radix* immediately before *type*:

```
format_spec := [[fill]align][sign]['#']['0'][width]['.' precision][radix][type]
radix: 'x' | 'X'
```

## `Formatter` API

Note that `x` and `X` are already valid *types*.
They are only interpreted as a radix when the type is `?`,
since combining them with other types doesn’t make sense.

This radix is exposed indirectly in two additional methods of `std::fmt::Formatter`:

```rust
impl<'a> Formatter<'a> {
    // ...

    /// Based on the radix and type: 16, 10, 8, or 2.
    ///
    /// This is mostly useful in `Debug` impls,
    /// where the trait itself doesn’t imply a radix.
    fn number_radix(&self) -> u32

    /// true for `X` or `E`
    ///
    /// This is mostly useful in `Debug` impls,
    /// where the trait itself doesn’t imply a case.
    fn number_uppercase(&self) -> bool
}
```

Although the radix and type are separate in the formatting string grammar,
they are intentionally conflated in this new API.

## `Debug` impls

The `Debug` implementation for primitive integer types `{u,i}{8,16,32,64,128,size}`
is modified to defer to `LowerHex` or `UpperHex` instead of `Display`,
based on `formatter.number_radix()` and `formatter.number_uppercase()`.
The *alternate* `#` flag is ignored, since it already has a separate meaning for `Debug`:
the `0x` prefix is *not* included.

As of Rust 1.22, impls using the `Formatter::debug_*` methods do not forward
formatting parameters such as *width* when formatting keys/values/items.
Doing so is important for this RFC to be useful.
This is fixed by [PR #46233](https://github.com/rust-lang/rust/pull/46233).

# Drawbacks
[drawbacks]: #drawbacks

The hexadecimal flag in the the `Debug` trait is superficially redundant
with the `LowerHex` and `UpperHex` traits.
If these traits were not stable yet, we could have considered a more unified design.

# Rationale and alternatives
[alternatives]: #alternatives

Implementing `LowerHex` and `UpperHex` was proposed and rejected
in [PR #44751](https://github.com/rust-lang/rust/pull/44751).

The status quo is that debugging or testing code that could be a one-liner
requires manual `Debug` impls and/or concatenating the results of separate
string formatting operations.

# Unresolved questions
[unresolved]: #unresolved-questions

* Should this be extended to octal and binary (as `{:o?}` and `{:b?}`)?
  Other formatting types/traits too?
* Details of the new `Formatter` API
