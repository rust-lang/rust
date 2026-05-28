// Tests that the `unwrap` branch is optimized out from the `pop` since the
// length has already been validated.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

pub enum Foo {
    First(usize),
    Second(usize),
}

// CHECK-LABEL: @check_only_one_panic
#[no_mangle]
pub fn check_only_one_panic(v: &mut Vec<Foo>) -> Foo {
    // CHECK-COUNT-1: call{{.+}}panic
    assert!(v.len() == 1);
    v.pop().unwrap()
}
