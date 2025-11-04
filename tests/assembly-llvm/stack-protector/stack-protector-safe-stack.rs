//@ revisions: all strong none safestack safestack_strong safestack_all
//@ assembly-output: emit-asm
//@ ignore-msvc safestack sanitizer not supported
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-bare
//@ ignore-aarch64
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [none] compile-flags: -Z stack-protector=none
//@ [safestack] compile-flags: -Z stack-protector=none -Z sanitizer=safestack
//@ [safestack_strong] compile-flags: -Z stack-protector=strong -Z sanitizer=safestack
//@ [safestack_all] compile-flags: -Z stack-protector=all -Z sanitizer=safestack
//@ compile-flags: -C opt-level=2 -Z merge-functions=disabled

#![crate_type = "lib"]
#![allow(internal_features)]
#![feature(unsized_fn_params)]

// Check the coexistence of stack-protector and safe-stack.
// CHECK-LABEL: test1{{:|\[}}
#[no_mangle]
pub unsafe fn test1(src: *const u8, len: usize) -> u8 {
    let mut buf = [0u8; 64];
    std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), len.min(buf.len()));
    buf[0]

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
