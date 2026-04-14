//@ compile-flags: -Copt-level=3
// Regression test for #155263: cold_path must propagate through
// FnOnce::call_once boundaries after LLVM inlining.
#![crate_type = "lib"]

use std::hint::cold_path;

fn dispatch<F: FnOnce(&mut u64) -> Option<u8>>(x: &mut u64, f: F) -> Option<u8> {
    if *x == 0 {
        cold_path();
        return None;
    }
    *x -= 1;

    let result = f(x);
    if result.is_none() {
        cold_path();
        return None;
    }
    result
}

fn dec(x: &mut u64) -> Option<u8> {
    if *x == 0 {
        None
    } else {
        *x -= 1;
        Some(1)
    }
}

// CHECK-LABEL: @test_cold_path_through_fnonce(
// CHECK: asm sideeffect
// CHECK: asm sideeffect
#[no_mangle]
pub fn test_cold_path_through_fnonce(x: &mut u64, y: &mut u64) -> Option<u8> {
    dispatch(x, |x| dec(y))
}
