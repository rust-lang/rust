- Start Date: 2014-03-14
- RFC PR: [rust-lang/rfcs#8](https://github.com/rust-lang/rfcs/pull/8)
- Rust Issue: 

** Note: this RFC was never implemented and has been retired. The
design may still be useful in the future, but before implementing we
would prefer to revisit it so as to be sure it is up to date. **

# Summary

The way our intrinsics work forces them to be wrapped in order to
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
and identified as intrinsics with the `#[lang = "..."]` attribute:

```
extern "Rust" {
    #[lang = "transmute"]
    fn transmute<T, U>(T) -> U;
}
```

The compiler will type check and translate intrinsics the same as today.
Additionally, when trans sees a "Rust" extern tagged as an intrinsic
it will not emit a function declaration to LLVM bitcode.

Because intrinsics will be lang items, they can no longer be redeclared
arbitrary number of times. This will require a small amount of existing
library code to be refactored, and all intrinsics to be exposed through public
abstractions.

Currently, "Rust" foreign functions may not be generic; this change
will require a special case that allows intrinsics to be generic.

# Alternatives

1. Instead of making intrinsics lang items we could create a slightly
different mechanism, like an `#[intrinsic]` attribute, that would
continue letting intrinsics to be redeclared.

2. While using lang items to identify intrinsics, intrinsic lang items
*could* be allowed to be redeclared.

3. We could also make "rust-intrinsic" coerce or otherwise be the same
as "Rust" externs and normal Rust functions.

# Unresolved questions

None.
