//@ add-minicore
//@ assembly-output: linker-asm
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib
//@ needs-llvm-components: nvptx

#![feature(abi_ptx, no_core, intrinsics)]
#![no_main]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
pub const fn wrapping_add<T: Copy>(a: T, b: T) -> T;

//@ aux-build: non-inline-dependency.rs
extern crate non_inline_dependency as dep;

// Make sure declarations are there.
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) overflowing_external_fn

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK:      call.uni (retval0),
    // CHECK:      wrapping_external_fn
    // CHECK:      ld.param.{{[ubs]}}32 %[[LHS:r[0-9]+]], [{{retval0(\+0)?}}];
    let lhs = dep::wrapping_external_fn(*a);

    // CHECK:      call.uni (retval0),
    // CHECK:      overflowing_external_fn
    // CHECK:      ld.param.{{[ubs]}}32 %[[RHS:r[0-9]+]], [{{retval0(\+0)?}}];
    let rhs = dep::overflowing_external_fn(*a);

    // CHECK: add.{{[ubs]}}32 %[[RES:r[0-9]+]], %[[RHS]], %[[LHS]];
    // CHECK: st.global.{{[ubs]}}32 [%{{rd[0-9]+}}], %[[RES]];
    *b = wrapping_add(lhs, rhs);
}

// Verify that external function bodies are available.
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .{{[ubs]}}32 func_retval0) overflowing_external_fn
