// Verifies that user-defined allocation token hints (i.e., contains-pointer classification and type
// name encoding) are emitted.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]
#![feature(alloc_token_hint, rustc_attrs)]

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

#[alloc_token_hint(contains_pointers = false)]
pub struct Counters([usize; 8]);

#[alloc_token_hint(type_name = "Foo", contains_pointers = true)]
#[repr(C)]
pub struct Foo {
    next: *mut Foo,
}

pub unsafe fn foo() -> (*mut u8, *mut u8) {
    (unsafe { alloc::<Counters>() }, unsafe { alloc::<Foo>() })
}

// CHECK-DAG: !{{[0-9]+}} = !{!"{{.*}}Counters{{.*}}", i1 false}
// CHECK-DAG: !{{[0-9]+}} = !{!"Foo", i1 true}
