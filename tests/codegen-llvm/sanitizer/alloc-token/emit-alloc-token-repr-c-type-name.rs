// Verifies that repr(C) user-defined type is encoded as its plain, unscoped name for cross-language
// LLVM AllocToken and heap partitioning support.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]
#![feature(rustc_attrs)]

use std::mem::{align_of, size_of};

unsafe extern "Rust" {
    #[rustc_allocator]
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

#[rustc_alloc_token_hint]
unsafe fn alloc<T>() -> *mut u8 {
    unsafe { __rust_alloc(size_of::<T>(), align_of::<T>()) }
}

#[repr(C)]
pub struct Buffer {
    len: usize,
    data: *mut u8,
}

pub unsafe fn foo() -> *mut u8 {
    unsafe { alloc::<Buffer>() }
}

// CHECK: call{{.*}}__rust_alloc{{.*}}!alloc_token ![[BUFFER:[0-9]+]]
// CHECK: ![[BUFFER]] = !{!"Buffer", i1 true}
