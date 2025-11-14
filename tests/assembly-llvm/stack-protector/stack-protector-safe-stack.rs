//@ revisions: all strong none safestack safestack_strong safestack_all
//@ assembly-output: emit-asm
//@ ignore-msvc safestack sanitizer not supported
//@ ignore-musl safestack sanitizer not supported
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32 safestack sanitizer not supported
//@ ignore-aarch64
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [none] compile-flags: -Z stack-protector=none
//@ [safestack] compile-flags: -Z stack-protector=none -Z sanitizer=safestack
//@ [safestack_strong] compile-flags: -Z stack-protector=strong -Z sanitizer=safestack
//@ [safestack_all] compile-flags: -Z stack-protector=all -Z sanitizer=safestack
//@ compile-flags: -C opt-level=2 -Z merge-functions=disabled

#![no_std]
#![feature(link_llvm_intrinsics)]
#![feature(unsized_fn_params)]
#![crate_type = "lib"]

extern "C" {
    #[link_name = "llvm.memcpy.p0.p0.i64"]
    fn memcpy_intrinsic(dst: *mut u8, src: *const u8, size: u64, align: u32, is_volatile: bool);
}

// CHECK-LABEL: test1{{:|\[}}
#[no_mangle]
pub unsafe fn test1(src: *const u8, len: usize) -> u8 {
    let mut buf: [u8; 64] = [0; 64];

    let copy_len = if len < 64 { len } else { 64 };
    memcpy_intrinsic(buf.as_mut_ptr(), src, copy_len as u64, 1, false);

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
