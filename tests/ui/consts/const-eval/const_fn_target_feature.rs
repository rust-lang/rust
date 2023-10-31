// only-x86_64
// compile-flags:-C target-feature=+ssse3

#![crate_type = "lib"]

// ok (ssse3 enabled at compile time)
const A: () = unsafe { ssse3_fn() };

// error (avx2 not enabled at compile time)
const B: () = unsafe { avx2_fn() };
//~^ ERROR evaluation of constant value failed

#[target_feature(enable = "ssse3")]
const unsafe fn ssse3_fn() {}

#[target_feature(enable = "avx2")]
const unsafe fn avx2_fn() {}
