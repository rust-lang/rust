// run-pass
#![feature(core_intrinsics)]

use std::intrinsics::assume;

unsafe fn f(x: i32) -> i32 {
    assume(x == 34);
    match x {
        34 => 42,
        _  => 30
    }
}

fn main() {
    let x = unsafe { f(34) };
    assert_eq!(x, 42);
}
