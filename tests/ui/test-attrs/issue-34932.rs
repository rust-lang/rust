// run-pass
// compile-flags:--test
#![cfg(any())] // This test should be configured away
#![feature(rustc_attrs)] // Test that this is allowed on stable/beta
#![feature(iter_arith_traits)] // Test that this is not unused
#![deny(unused_features)]

#[test]
fn dummy() {
    let () = "this should not reach type-checking";
}
