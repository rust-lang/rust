- Start Date: 2014-04-18
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

Asserts are too expensive for release builds and mess up inlining. There must be a way to turn them off. I propose macros `assert!` and `enforce!`. For test cases, `enforce!` should be used.

# Motivation

Asserts are too expensive in release builds.

# Detailed design

There should be two macros, `assert!(EXPR)` and `enforce!(EXPR)`. In debug builds (without `--cfg ndebug`), `assert!()` is the same as `enforce!()`. In release builds (with `--cfg ndebug`), `assert!()` compiles away to nothing. The definition of `enforce!()` is `if (!EXPR) { fail!("assertion failed ({}, {}): {}", file!(), line!(), stringify!(expr) }`

# Alternatives

Other designs that have been considered are using `assert!` in test cases and not providing `enforce!`, but this doesn't work with separate compilation.

There has been an issue raised that `enforce!` is unintuitive for test cases, but I think all workarounds for this are worse because they add complexity.

The impact of not doing this is that `assert!` will become expensive, prompting people will write their own local `assert!` macros, duplicating functionality that should have been in the standard library.

# Unresolved questions

None.
