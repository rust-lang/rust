//@ build-fail
//@ failure-status: 101
//@ known-bug: #109681
//@ ignore-wasm32 this appears to SIGABRT on wasm, not fail cleanly

// This test verifies that we continue to hit the LLVM error for common linkage with non-zero
// initializers, since it generates invalid LLVM IR.
// Linkages are internal features marked as perma-unstable, so we don't need to fix the issue
// for now.
#![crate_type="lib"]
#![feature(linkage)]

#[linkage = "common"]
#[no_mangle]
pub static TEST: bool = true;
