//@ run-pass
#![feature(core_intrinsics)]
#![feature(const_heap)]
use std::intrinsics;

const FOO: i32 = foo();

const fn foo() -> i32 {
    let t = unsafe {
        let i = intrinsics::const_allocate(4, 4) as * mut i32;
        *i = 20;
        i
    };
    unsafe { *t }
}
fn main() {
    assert_eq!(FOO, 20);
}
