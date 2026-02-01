//@ add-minicore
//@ assembly-output: ptx-linker
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib -Ctarget-cpu=sm_30
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc

#![feature(abi_ptx, no_core, intrinsics)]
#![no_main]
#![no_core]

extern crate minicore;
use minicore::*;

//@ aux-build: non-inline-dependency.rs
extern crate non_inline_dependency as dep;

#[rustc_intrinsic]
pub const unsafe fn unchecked_add<T: Copy>(x: T, y: T) -> T;

// Make sure declarations are there.
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) overflowing_external_fn

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK:      call.uni (retval0),
    // CHECK:      wrapping_external_fn
    // CHECK:      ld.param.{{[ubs]}}32 %[[LHS:r[0-9]+]], [retval0];
    let lhs = dep::wrapping_external_fn(*a);

    // CHECK:      call.uni (retval0),
    // CHECK:      overflowing_external_fn
    // CHECK:      ld.param.{{[ubs]}}32 %[[RHS:r[0-9]+]], [retval0];
    let rhs = dep::overflowing_external_fn(*a);

    // CHECK: add.{{[us]}}32 %[[RES:r[0-9]+]], %[[RHS]], %[[LHS]];
    // CHECK: st.global.{{[ubs]}}32 [%{{rd[0-9]+}}], %[[RES]];
    *b = unchecked_add(lhs, rhs);
}

// Verify that external function bodies are available.
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) overflowing_external_fn
