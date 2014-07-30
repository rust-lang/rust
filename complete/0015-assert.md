- Start Date: 2014-04-18
- RFC PR: [rust-lang/rfcs#50](https://github.com/rust-lang/rfcs/pull/50)
- Rust Issue: [rust-lang/rust#13789](https://github.com/rust-lang/rust/issues/13789)

# Summary

Asserts are too expensive for release builds and mess up inlining. There must be a way to turn them off. I propose macros `debug_assert!` and `assert!`. For test cases, `assert!` should be used.

# Motivation

Asserts are too expensive in release builds.

# Detailed design

There should be two macros, `debug_assert!(EXPR)` and `assert!(EXPR)`. In debug builds (without `--cfg ndebug`), `debug_assert!()` is the same as `assert!()`. In release builds (with `--cfg ndebug`), `debug_assert!()` compiles away to nothing. The definition of `assert!()` is `if (!EXPR) { fail!("assertion failed ({}, {}): {}", file!(), line!(), stringify!(expr) }`

# Alternatives

Other designs that have been considered are using `debug_assert!` in test cases and not providing `assert!`, but this doesn't work with separate compilation.

The impact of not doing this is that `assert!` will be expensive, prompting people will write their own local `debug_assert!` macros, duplicating functionality that should have been in the standard library.

# Unresolved questions

None.
