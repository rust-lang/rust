// check-pass
// compile-flags: -Z unpretty=expanded

#![feature(core_intrinsics, generic_assert, generic_assert_internals)]

fn main() {
    let elem = 1i32;
    assert!(elem == 1);
}
