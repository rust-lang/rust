//@ run-pass
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const FOO: &i32 = foo();
const FOO_RAW: *const i32 = foo();

const fn foo() -> &'static i32 {
    let t = unsafe {
        let i = intrinsics::const_allocate(4, 4) as *mut i32;
        *i = 20;
        i
    };
    unsafe { &*(intrinsics::const_make_global(t as *mut u8) as *const i32) }
}
fn main() {
    assert_eq!(*FOO, 20);
    assert_eq!(unsafe { *FOO_RAW }, 20);
}
