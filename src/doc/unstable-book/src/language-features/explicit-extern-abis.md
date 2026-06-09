# `explicit_extern_abis`

The tracking issue for this feature is: [#134986]

------

Disallow `extern` without an explicit ABI. We should write `extern "C"`
(or another ABI) instead of just `extern`.

By making the ABI explicit, it becomes much clearer that "C" is just one of the
possible choices, rather than the "standard" way for external functions.
Removing the default makes it easier to add a new ABI on equal footing as "C".

```rust,editionfuture,compile_fail
#![feature(explicit_extern_abis)]

extern fn function1() {}  // ERROR `extern` declarations without an explicit ABI
                          // are disallowed

extern "C" fn function2() {} // compiles

extern "aapcs" fn function3() {} // compiles
```

[#134986]: https://github.com/rust-lang/rust/issues/134986
