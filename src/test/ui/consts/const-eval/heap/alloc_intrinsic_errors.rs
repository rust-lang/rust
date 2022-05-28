#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(const_mut_refs)]
use std::intrinsics;

const FOO: i32 = foo();
const fn foo() -> i32 {
    unsafe {
        let _ = intrinsics::const_allocate(4, 3) as *mut i32;
        //~^ error: evaluation of constant value failed
    }
    1
}

fn main() {}
