#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const FOO: &i32 = foo();
//~^ error: encountered `const_allocate` pointer in final value that was not made global
const FOO_RAW: *const i32 = foo();
//~^ error: encountered `const_allocate` pointer in final value that was not made global

const fn foo() -> &'static i32 {
    let t = unsafe {
        let i = intrinsics::const_allocate(4, 4) as *mut i32;
        *i = 20;
        i
    };
    unsafe { &*t }
}

fn main() {}
