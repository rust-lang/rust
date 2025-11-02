// LLVM does not support some atomic RMW operations on pointers, so inside codegen we lower those
// to integer atomics, followed by an inttoptr cast.
// This test ensures that we do the round-trip correctly for AtomicPtr::fetch_byte_add, and also
// ensures that we do not have such a round-trip for AtomicPtr::swap, because LLVM supports pointer
// arguments to `atomicrmw xchg`.

//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes
#![crate_type = "lib"]

use std::ptr::without_provenance_mut;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering::Relaxed;

// Portability hack so that we can say [[USIZE]] instead of i64/i32/i16 for usize.
// CHECK: @helper([[USIZE:i[0-9]+]] noundef %_1)
#[no_mangle]
pub fn helper(_: usize) {}

// CHECK-LABEL: @atomicptr_fetch_byte_add
#[no_mangle]
pub fn atomicptr_fetch_byte_add(a: &AtomicPtr<u8>, v: usize) -> *mut u8 {
    // CHECK: llvm.lifetime.start
    // CHECK-NEXT: %[[RET:.*]] = atomicrmw add ptr %{{.*}}, [[USIZE]] %v
    // CHECK-NEXT: inttoptr [[USIZE]] %[[RET]] to ptr
    a.fetch_byte_add(v, Relaxed)
}

// CHECK-LABEL: @atomicptr_swap
#[no_mangle]
pub fn atomicptr_swap(a: &AtomicPtr<u8>, ptr: *mut u8) -> *mut u8 {
    // CHECK-NOT: ptrtoint
    // CHECK: atomicrmw xchg ptr %{{.*}}, ptr %{{.*}} monotonic
    // CHECK-NOT: inttoptr
    a.swap(ptr, Relaxed)
}
