# How to contribute

## compiler-builtins

1. From the [pending list](compiler-builtins/README.md#progress), pick one or
   more intrinsics.
2. Port the version from [`compiler-rt`] and, if applicable, their
   [tests][rt-tests]. Note that this crate has generic implementations for a lot
   of routines, which may be usable without porting the entire implementation.
3. Add a test to `builtins-test`, comparing the behavior of the ported
   intrinsic(s) with their implementation on the testing host.
4. Add the intrinsic to `builtins-test-intrinsics/src/main.rs` to verify it can
   be linked on all targets.
5. Send a Pull Request (PR) :tada:.

[`compiler-rt`]: https://github.com/llvm/llvm-project/tree/b6820c35c59a4da3e59c11f657093ffbd79ae1db/compiler-rt/lib/builtins
[rt-tests]: https://github.com/llvm/llvm-project/tree/b6820c35c59a4da3e59c11f657093ffbd79ae1db/compiler-rt/test/builtins

## Porting Reminders

1. [Rust][prec-rust] and [C][prec-c] have slightly different operator
   precedence. C evaluates comparisons (`== !=`) before bitwise operations
   (`& | ^`), while Rust evaluates the other way.
2. C assumes wrapping operations everywhere. Rust panics on overflow when in
   debug mode. Consider using the [Wrapping][wrap-ty] type or the explicit
   [wrapping_*][wrap-fn] functions where applicable.
3. Note [C implicit casts][casts], especially integer promotion. Rust is much
   more explicit about casting, so be sure that any cast which affects the
   output is ported to the Rust implementation.
4. Rust has [many functions][i32] for integer or floating point manipulation in
   the standard library. Consider using one of these functions rather than
   porting a new one.

[prec-rust]: https://doc.rust-lang.org/reference/expressions.html#expression-precedence
[prec-c]: http://en.cppreference.com/w/c/language/operator_precedence
[wrap-ty]: https://doc.rust-lang.org/core/num/struct.Wrapping.html
[wrap-fn]: https://doc.rust-lang.org/std/primitive.i32.html#method.wrapping_add
[casts]: http://en.cppreference.com/w/cpp/language/implicit_conversion
[i32]: https://doc.rust-lang.org/std/primitive.i32.html

## Tips and tricks

- _IMPORTANT_ The code in this crate will end up being used in the `core` crate
  so it can **not** have any external dependencies (other than a subset of
  `core` itself).
- Only use relative imports within the `math` directory / module, e.g.
  `use self::fabs::fabs` or `use super::k_cos`. Absolute imports from core are
  OK, e.g. `use core::u64`.
- To reinterpret a float as an integer use the `to_bits` method. The MUSL code
  uses the `GET_FLOAT_WORD` macro, or a union, to do this operation.
- To reinterpret an integer as a float use the `f32::from_bits` constructor. The
  MUSL code uses the `SET_FLOAT_WORD` macro, or a union, to do this operation.
- You may use other methods from core like `f64::is_nan`, etc. as appropriate.
- Rust does not have hex float literals. This crate provides two `hf16!`,
  `hf32!`, `hf64!`, and `hf128!` which convert string literals to floats at
  compile time.

  ```rust
  assert_eq!(hf32!("0x1.ffep+8").to_bits(), 0x43fff000);
  assert_eq!(hf64!("0x1.ffep+8").to_bits(), 0x407ffe0000000000);
  ```

- Rust code panics on arithmetic overflows when not optimized. You may need to
  use the [`Wrapping`] newtype to avoid this problem, or individual methods like
  [`wrapping_add`].

[`Wrapping`]: https://doc.rust-lang.org/std/num/struct.Wrapping.html
[`wrapping_add`]: https://doc.rust-lang.org/std/primitive.u32.html#method.wrapping_add

## Testing

Testing for these crates can be somewhat complex, so feel free to rely on CI.

The easiest way replicate CI testing is using Docker. This can be done by
running `./ci/run-docker.sh [target]`. If no target is specified, all targets
will be run.

Tests can also be run without Docker:

```sh
# Run basic tests
#
# --no-default-features always needs to be passed, an unfortunate limitation
# since the `#![compiler_builtins]` feature is enabled by default.
cargo test --workspace --no-default-features

# Test with all interesting features
cargo test --workspace --no-default-features \
    --features arch,unstable-float,unstable-intrinsics,mem

# Run with more detailed tests for libm
cargo test --workspace --no-default-features \
    --features arch,unstable-float,unstable-intrinsics,mem \
    --features build-mpfr,build-musl \
    --profile release-checked
```

The multiprecision tests use the [`rug`] crate for bindings to MPFR. MPFR can be
difficult to build on non-Unix systems, refer to [`gmp_mpfr_sys`] for help.

`build-musl` does not build with MSVC, Wasm, or Thumb.

[`rug`]: https://docs.rs/rug/latest/rug/
[`gmp_mpfr_sys`]: https://docs.rs/gmp-mpfr-sys/1.6.4/gmp_mpfr_sys/

In order to run all tests, some dependencies may be required:

```sh
# Allow testing compiler-builtins
./ci/download-compiler-rt.sh

# Optional, initialize musl for `--features build-musl`
git submodule init
git submodule update

# `--release` ables more test cases
cargo test --release
```

### Extensive tests

Libm also has tests that are exhaustive (for single-argument `f32` and 1- or 2-
argument `f16`) or extensive (for all other float and argument combinations).
These take quite a long time to run, but are launched in CI when relevant files
are changed.

Exhaustive tests can be selected by passing an environment variable:

```sh
LIBM_EXTENSIVE_TESTS=sqrt,sqrtf cargo test --features build-mpfr \
    --test z_extensive \
    --profile release-checked

# Run all tests for one type
LIBM_EXTENSIVE_TESTS=all_f16 cargo test ...

# Ensure `f64` tests can run exhaustively. Estimated completion test for a
# single test is 57306 years on my machine so this may be worth skipping.
LIBM_EXTENSIVE_TESTS=all LIBM_EXTENSIVE_ITERATIONS=18446744073709551615 cargo test ...
```

## Benchmarking

Regular walltime benchmarks can be run with `cargo bench`:

```sh
cargo bench --no-default-features \
    --features arch,unstable-float,unstable-intrinsics,mem \
    --features benchmarking-reports
```

There are also benchmarks that check instruction count behind the `icount`
feature. These require [`iai-callgrind-runner`] (via Cargo) and [Valgrind]
to be installed, which means these only run on limited platforms.

Instruction count benchmarks are run as part of CI to flag performance
regresions.

```sh
cargo bench --no-default-features \
    --features arch,unstable-float,unstable-intrinsics,mem \
    --features icount \
    --bench icount --bench mem_icount
```

[`iai-callgrind-runner`]: https://crates.io/crates/iai-callgrind-runner
[Valgrind]: https://valgrind.org/
