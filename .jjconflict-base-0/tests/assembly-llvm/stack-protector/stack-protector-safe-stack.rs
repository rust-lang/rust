//@ add-minicore
//@ revisions: all strong none safestack safestack_strong safestack_all
//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [none] compile-flags: -Z stack-protector=none
//@ [safestack] compile-flags: -Z stack-protector=none -Z sanitizer=safestack
//@ [safestack_strong] compile-flags: -Z stack-protector=strong -Z sanitizer=safestack
//@ [safestack_all] compile-flags: -Z stack-protector=all -Z sanitizer=safestack
//@ compile-flags: -C opt-level=2 -Z merge-functions=disabled --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![feature(unsized_fn_params)]
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]
#![no_std]

extern crate minicore;
use minicore::*;

extern "C" {
    fn test4spss(p: *mut u8);
}

// CHECK-LABEL: test1{{:|\[}}
#[no_mangle]
pub unsafe fn test1(x: *mut u8) -> u8 {
    let mut buf: [u8; 64] = [0; 64];

    let p = &mut buf as *mut [u8; 64] as *mut u8;

    *p = 42;

    test4spss(p);

    *x = *p;

    *p

    // none-NOT: __stack_chk_fail
    // strong: __stack_chk_fail
    // all: __stack_chk_fail

    // safestack: __safestack_unsafe_stack_ptr
    // safestack-NOT: __stack_chk_fail

    // safestack_strong: __safestack_unsafe_stack_ptr
    // safestack_strong: __stack_chk_fail

    // safestack_all: __safestack_unsafe_stack_ptr
    // safestack_all: __stack_chk_fail
}
