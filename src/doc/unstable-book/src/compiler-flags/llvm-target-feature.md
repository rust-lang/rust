# `llvm-target-feature`

The tracking issue for this feature is: [#157753](https://github.com/rust-lang/rust/issues/157753).

------------------------

The `-Zllvm-target-feature` flag passes target feature strings directly to the LLVM
backend, bypassing Rust's known-feature validation. It accepts a comma-separated list of
features in the form `+feat` or `-feat`, for example:

```sh
rustc -Zllvm-target-feature=+prefer-256-bit main.rs
```

Each feature string is forwarded verbatim to LLVM. This allows using LLVM target features
that are not (or not yet) part of Rust's target feature system, for example to match the
ABI of foreign libraries built with features Rust does not recognize.

Because such features can change the ABI, this flag is registered as a *target modifier*:
linking together crates compiled with different values of this flag is an error by
default. If you are sure the mismatch is sound, it can be overridden with
`-Cunsafe-allow-abi-mismatch=llvm-target-feature`.

## Scope and interactions

This flag:

- does **not** affect `cfg(target_feature)`: conditional compilation will not observe
  features passed this way;
- does **not** affect runtime feature detection macros such as `is_x86_feature_detected!`;
- is ignored by non-LLVM codegen backends (like `-Cllvm-args`);
- performs no validation: invalid or unsupported feature names are reported (or silently
  ignored) by LLVM itself during codegen.

## Relationship to `-Ctarget-feature`

`-Ctarget-feature` is the supported way to toggle target features that Rust recognizes.
Passing feature names unknown to Rust through `-Ctarget-feature` is deprecated and will
eventually become a hard error; use `-Zllvm-target-feature` for LLVM-only features
instead.
