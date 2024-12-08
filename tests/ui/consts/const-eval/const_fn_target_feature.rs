//@ only-x86_64
// Set the base cpu explicitly, in case the default has been changed.
//@ compile-flags: -C target-cpu=x86-64 -C target-feature=+ssse3
//@ check-pass

#![crate_type = "lib"]

// ok (ssse3 enabled at compile time)
const A: () = unsafe { ssse3_fn() };

// error (avx2 not enabled at compile time)
const B: () = unsafe { avx2_fn() };
// FIXME: currently we do not detect this UB, since we don't want the result of const-eval
// to depend on `tcx.sess` which can differ between crates in a crate graph.

#[target_feature(enable = "ssse3")]
const unsafe fn ssse3_fn() {}

#[target_feature(enable = "avx2")]
const unsafe fn avx2_fn() {}
