//@ add-minicore
//@ revisions: all strong none safestack safestack_strong safestack_all
//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu
//@ [all] compile-flags: -Zstack-protector=all
//@ [strong] compile-flags: -Zstack-protector=strong
//@ [none] compile-flags: -Zstack-protector=none
//@ [safestack] compile-flags: -Zstack-protector=none
//@ [safestack] compile-flags: -Cunsafe-allow-abi-mismatch=sanitize
//@ [safestack] compile-flags: -Zunstable-options -Csanitize=safestack
//@ [safestack_strong] compile-flags: -Zstack-protector=strong
//@ [safestack_strong] compile-flags: -Cunsafe-allow-abi-mismatch=sanitize
//@ [safestack_strong] compile-flags: -Zunstable-options -Csanitize=safestack
//@ [safestack_all] compile-flags: -Zstack-protector=all
//@ [safestack_all] compile-flags: -Cunsafe-allow-abi-mismatch=sanitize
//@ [safestack_all] compile-flags -Zunstable-options -Csanitize=safestack
//@ compile-flags: -Copt-level=2 -Zmerge-functions=disabled --target x86_64-unknown-linux-gnu
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

    // safestack_all: __stack_chk_fail
}
