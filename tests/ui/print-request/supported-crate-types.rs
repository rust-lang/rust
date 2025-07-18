//! Basic smoke test for `--print=supported-crate-types`, which should print a newline delimited
//! list of crate types supported by the given target. This test cherry-picks a few well-known
//! targets as examples.
//!
//! Tracking issue: <https://github.com/rust-lang/rust/issues/138640>

// ignore-tidy-linelength

//@ check-pass

// FIXME: musl targets are currently statically linked, but running on a musl host
// requires dynamic linkage, which in turn changes the supported crate types for
// x86_64-unknown-linux-musl.
//@ ignore-musl

//@ revisions: wasm musl linux

//@[wasm] compile-flags: --target=wasm32-unknown-unknown --print=supported-crate-types -Zunstable-options
//@[wasm] needs-llvm-components: webassembly

//@[musl] compile-flags: --target=x86_64-unknown-linux-musl --print=supported-crate-types -Zunstable-options
//@[musl] needs-llvm-components: x86

//@[linux] compile-flags: --target=x86_64-unknown-linux-gnu --print=supported-crate-types -Zunstable-options
//@[linux] needs-llvm-components: x86
