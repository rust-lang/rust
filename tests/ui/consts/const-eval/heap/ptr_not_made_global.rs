// Ensure that we reject interning `const_allocate`d allocations in the final value of constants
// if they have not been made global through `const_make_global`. The pointers are made *immutable*
// to focus the test on the missing `make_global`; `ptr_not_made_global_mut.rs` covers the case
// where the pointer remains mutable.

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
