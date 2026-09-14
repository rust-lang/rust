//! Regression test for https://github.com/rust-lang/rust/issues/113899.
//! When indexing into an array of an enum type with spare niches, the compiler
//! used to emit a superfluous branch checking whether the loaded value was
//! a niche value. Every element in the array is a valid variant, so this check
//! is unnecessary and should be optimised away.

//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[derive(Clone, Copy)]
pub enum Outer {
    A([u8; 8]),
    B([u8; 8]),
}

pub struct Error(u8);

// CHECK-LABEL: @test
#[no_mangle]
pub fn test(x: usize) -> Result<Outer, Error> {
    // There should be exactly one comparison: the bounds check on `x`.
    // There must be no second comparison checking the discriminant
    // against the niche value used by `Option<Outer>` (from `get()`).
    // CHECK: icmp ult
    // CHECK-NOT: icmp
    // CHECK: ret void
    [Outer::A([10; 8]), Outer::B([20; 8])].get(x).copied().ok_or(Error(5))
}
