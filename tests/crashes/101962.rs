//@ known-bug: #101962
//@ compile-flags: -Cdebuginfo=2

#![feature(core_intrinsics)]

pub fn wrapping<T: Copy>(a: T, b: T) {
    let _z = core::intrinsics::wrapping_mul(a, b);
}

fn main() {
    wrapping(1,2);
}
