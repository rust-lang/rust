- Feature Name: byte_concat
- Start Date: 2018-07-31
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow the use of `concat!()` to join byte sequences onto an `u8` array,
beyond the current support for `str` literals.

# Motivation
[motivation]: #motivation

`concat!()` is convenient and useful to create compile time `str` literals
from `str`, `bool`, numeric and `char` literals in the code. This RFC would
expand this capability to produce `[u8]` instead of `str` when any of its
arguments is a byte `str` or a byte `char`.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Whenever any of the arguments to `concat!()` is a byte literal, its output
will be a byte literal, and the other arguments will be evaluated on their
byte contents.

- `str`s and `char`s are evaluated in the same way as `String::as_bytes`,
- `bool`s are not accepted, use a numeric literal instead,
- numeric literals passed to `concat!()` must fit in `u8`, any number
  larger than `std::u8::MAX` causes a compile time error, like the
  following:  
```
error: cannot concatenate a non-`u8` literal in a byte string literal
  --> $FILE:XX:YY
   |
XX |     concat!(256, b"val");
   |             ^^^ this value is larger than `255`
```
- numeric array literals that can be coerced to `[u8]` are accepted, if the
literals are outside of `u8` range, it will cause a compile time error:
```
error: cannot concatenate a non-`u8` literal in a byte string literal
  --> $FILE:XX:YY
   |
XX |     concat!([300, 1, 2, 256], b"val");
   |              ^^^        ^^^ this value is larger than `255`
   |              |
   |              this value is larger than `255`
```

For example, `concat!(42, b"va", b'l', [1, 2])` evaluates to
`[42, 118, 97, 108, 1, 2]`.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

[PR #52838](https://github.com/rust-lang/rust/pull/52838) lays the
foundation for the implementation of the full RFC.

This new feature could be surprising when editting existing code, if
`concat!("foo", `b`, `a`, `r`, 3)` were changed to
`concat!("foo", `b`, b`a`, `r`, 3)`, as the macro call would change from
being evaluated as a `str` literal "foobar3" to `[u8]`
`[102, 111, 111, 98, 97, 114, 3]`.

# Drawbacks
[drawbacks]: #drawbacks

As mentioned in the previous section, this causes `concat!()`'s output to be
dependant on its input.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

A new macro `bconcat!()` could be introduced instead. People in the wild
have already intended to use `concat!()` for byte literals. A new macro
could be explained to users through diagnostics, but using the existing
macro adds support for something that a user could expect to work.

# Prior art
[prior-art]: #prior-art

[PR #52838](https://github.com/rust-lang/rust/pull/52838) lays the
foundation for the implementation of the full RFC, trying to enable a real
use seen in the wild.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
