- Start Date: 2014-11-27
- RFC PR: https://github.com/rust-lang/rfcs/pull/486
- Rust Issue: https://github.com/rust-lang/rust/issues/19908

# Summary

Move the `std::ascii::Ascii` type and related traits to a new Cargo package on crates.io,
and instead expose its functionality for `u8`, `[u8]`, `char`, and `str` types.

# Motivation

The `std::ascii::Ascii` type is a `u8` wrapper that enforces
(unless `unsafe` code is used)
that the value is in the ASCII range,
similar to `char` with `u32` in the range of Unicode scalar values,
and `String` with `Vec<u8>` containing well-formed UTF-8 data.
`[Ascii]` and `Vec<Ascii>` are naturally strings of text entirely in the ASCII range.

Using the type system like this to enforce data invariants is interesting,
but in practice `Ascii` is not that useful.
Data (such as from the network) is rarely guaranteed to be ASCII only,
nor is it desirable to remove or replace non-ASCII bytes,
even if ASCII-range-only operations are used.
(For example, *ASCII case-insensitive matching* is common in HTML and CSS.)

Every single use of the `Ascii` type in the Rust distribution
is only to use the `to_lowercase` or `to_uppercase` method,
then immediately convert back to `u8` or `char`.

# Detailed design

The `Ascii` type
as well as the `AsciiCast`, `OwnedAsciiCast`, `AsciiStr`, and `IntoBytes` traits
should be copied into a new `ascii` Cargo package on crates.io.
The `std::ascii` copy should be deprecated and removed at some point before Rust 1.0.

Currently, the `AsciiExt` trait is:

```rust
pub trait AsciiExt<T> {
    fn to_ascii_upper(&self) -> T;
    fn to_ascii_lower(&self) -> T;
    fn eq_ignore_ascii_case(&self, other: &Self) -> bool;
}

impl AsciiExt<String> for str { ... }
impl AsciiExt<Vec<u8>> for [u8] { ... }
```

It should gain new methods for the functionality that is being removed with `Ascii`,
be implemented for `u8` and `char`,
and (if this is stable enough yet) use an associated type instead of the `T` parameter:

```rust
pub trait AsciiExt {
    type Owned = Self;
    fn to_ascii_upper(&self) -> Owned;
    fn to_ascii_lower(&self) -> Owned;
    fn eq_ignore_ascii_case(&self, other: &Self) -> bool;
    fn is_ascii(&self) -> bool;

    // Maybe? See unresolved questions
    fn is_ascii_lowercase(&self) -> bool;
    fn is_ascii_uppercase(&self) -> bool;
    ...
}

impl AsciiExt for str { type Owned = String; ... }
impl AsciiExt for [u8] { type Owned = Vec<u8>; ... }
impl AsciiExt char { ... }
impl AsciiExt u8 { ... }
```

The `OwnedAsciiExt` trait should stay as it is:

```rust
pub trait OwnedAsciiExt {
    fn into_ascii_upper(self) -> Self;
    fn into_ascii_lower(self) -> Self;
}

impl OwnedAsciiExt for String { ... }
impl OwnedAsciiExt for Vec<u8> { ... }
```

The `std::ascii::escape_default` function has little to do with ASCII.
I *think* it’s relevant to `b'x'` and `b"foo"` byte literals,
which have types `u8` and `&'static [u8]`.
I suggest moving it into `std::u8`.


I (@SimonSapin) can help with the implementation work.


# Drawbacks

Code using `Ascii` (not only for e.g. `to_lowercase`)
would need to install a Cargo package to get it.
This is strictly more work than having it in `std`,
but should still be easy.

# Alternatives

* The `Ascii` type could stay in `std::ascii`
* Some variations per *Unresolved questions* below.

# Unresolved questions

* What to do with `std::ascii::escape_default`?
* Rename the `AsciiExt` and `OwnedAsciiExt` traits?
* Should they be in the prelude? The `Ascii` type and the related traits currently are.
* Are associated type stable enough yet?
  If not, `AsciiExt` should temporarily keep its type parameter.
* Which of all the `Ascii::is_*` methods should `AsciiExt` include? Those included should have `ascii` added in their name.
  * *Maybe* `is_lowercase`, `is_uppercase`, `is_alphabetic`, or `is_alphanumeric` could be useful,
    but I’d be fine with dropping them and reconsider if someone asks for them.
    The same result can be achieved
    with `.is_ascii() &&` and the corresponding `UnicodeChar` method,
    which in most cases has an ASCII fast path.
    And in some cases it’s an easy range check like `'a' <= c && c <= 'z'`.
  * `is_digit` and `is_hex` are identical to `Char::is_digit(10)` and `Char::is_digit(16)`.
  * `is_blank`, `is_control`, `is_graph`, `is_print`, and `is_punctuation` are never used
    in the Rust distribution or Servo.
