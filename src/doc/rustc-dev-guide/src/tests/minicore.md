# `minicore` test auxiliary: using `core` stubs

<!-- date-check Oct 2024 -->

[`tests/auxiliary/minicore.rs`][`minicore`] is a test auxiliary for
ui/codegen/assembly test suites. It provides `core` stubs for tests that need to
build for cross-compiled targets but do not need/want to run.

<div class="warning">

Please note that [`minicore`] is only intended for `core` items, and explicitly
**not** `std` or `alloc` items because `core` items are applicable to a wider
range of tests.

</div>

A test can use [`minicore`] by specifying the `//@ add-core-stubs` directive.
Then, mark the test with `#![feature(no_core)]` + `#![no_std]` + `#![no_core]`.
Due to Edition 2015 extern prelude rules, you will probably need to declare
`minicore` as an extern crate.

## Implied compiler flags

Due to the `no_std` + `no_core` nature of these tests, `//@ add-core-stubs`
implies and requires that the test will be built with `-C panic=abort`.
**Unwinding panics are not supported.**

Tests will also be built with `-C force-unwind-tables=yes` to preserve CFI
directives in assembly tests.

TL;DR: `//@ add-core-stubs` implies two compiler flags:

1. `-C panic=abort`
2. `-C force-unwind-tables=yes`

## Adding more `core` stubs

If you find a `core` item to be missing from the [`minicore`] stub, consider
adding it to the test auxiliary if it's likely to be used or is already needed
by more than one test.

## Staying in sync with `core`

The `minicore` items must be kept up to date with `core`. For consistent
diagnostic output between using `core` and `minicore`, any `diagnostic`
attributes (e.g. `on_unimplemented`) should be replicated exactly in `minicore`.

## Example codegen test that uses `minicore`

```rust,no_run
//@ add-core-stubs
//@ revisions: meow bark
//@[meow] compile-flags: --target=x86_64-unknown-linux-gnu
//@[meow] needs-llvm-components: x86
//@[bark] compile-flags: --target=wasm32-unknown-unknown
//@[bark] needs-llvm-components: webassembly

#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Meow;
impl Copy for Meow {} // `Copy` here is provided by `minicore`

// CHECK-LABEL: meow
#[unsafe(no_mangle)]
fn meow() {}
```

[`minicore`]: https://github.com/rust-lang/rust/tree/master/tests/auxiliary/minicore.rs
