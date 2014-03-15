- Start Date: 2014-03-14
- RFC PR #: (leave this empty)
- Rust Issue #: (leave this empty)

# Summary

They way our intrinsics work forces them to be wrapped in order to
behave like normal functions. As a result, rustc is forced to inline a
great number of tiny intrinsic wrappers, which is bad for both
compile-time performance and runtime performance without
optimizations. This proposal changes the way intrinsics are surfaced
in the language so that they behave the same as normal Rust functions
by removing the "rust-intrinsic" foreign ABI and reusing the "Rust"
ABI.

# Motivation

A number of commonly-used intrinsics, including `transmute`, `forget`,
`init`, `uninit`, and `move_val_init`, are accessed through wrappers
whose only purpose is to present the intrinsics as normal functions.
As a result, rustc is forced to inline a great number of tiny
intrinsic wrappers, which is bad for both compile-time performance and
runtime performance without optimizations.

Intrinsics have a differently-named ABI from Rust functions
("rust-intrinsic" vs. "Rust") though the actual ABI implementation is
identical.  As a result one can't take the value of an intrinsic as a
function:

```
// error: the type of transmute is `extern "rust-intrinsic" fn ...`
let transmute: fn(int) -> uint = intrinsics::transmute;
```

This incongruity means that we can't just expose the intrinsics
directly as part of the public API.

# Detailed design

`extern "Rust" fn` is already equivalent to `fn`, so if intrinsics
have the "Rust" ABI then the problem is solved.

Under this scheme intrinsics will be declared as `extern "Rust"` functions
and identified as intrinsics with the `#[intrinsic]` attribute:

```
extern "Rust" {
    #[intrinsic]
    fn transmute<T, U>(T) -> U;
}
```

The compiler will type check and translate intrinsics the same as today.
Additionally, when trans sees a "Rust" extern tagged as an intrinsic
it will not emit a function declaration to LLVM bitcode.

# Alternatives

1. Instead of the new `#[intrinsic]` attribute we could make intrinsics
lang items. This would require either forcing them to be 'singletons'
or create a new type of lang item that can be multiply-declared.

2. We could also make "rust-intrinsic" coerce or otherwise be the same
as "Rust" externs and normal Rust functions.

# Unresolved questions

None.