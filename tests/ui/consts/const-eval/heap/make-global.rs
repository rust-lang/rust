//@ run-pass
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const FOO: &i32 = foo();
const FOO_RAW: *const i32 = foo();

const fn foo() -> &'static i32 {
    unsafe {
        let ptr = intrinsics::const_allocate(4, 4);
        let t = ptr as *mut i32;
        *t = 20;
        intrinsics::const_make_global(ptr);
        &*t
    }
}
fn main() {
    assert_eq!(*FOO, 20);
    assert_eq!(unsafe { *FOO_RAW }, 20);
}
