// assembly-output: ptx-linker
// compile-flags: --crate-type cdylib
// only-nvptx64
// ignore-nvptx64

#![feature(abi_ptx)]
#![no_std]

// aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// aux-build: non-inline-dependency.rs
extern crate non_inline_dependency as dep;

// Make sure declarations are there.
// CHECK: .func (.param .b32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .b32 func_retval0) panicking_external_fn

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK:      call.uni (retval0),
    // CHECK-NEXT: wrapping_external_fn
    // CHECK:      ld.param.b32 %[[LHS:r[0-9]+]], [retval0+0];
    let lhs = dep::wrapping_external_fn(*a);

    // CHECK:      call.uni (retval0),
    // CHECK-NEXT: panicking_external_fn
    // CHECK:      ld.param.b32 %[[RHS:r[0-9]+]], [retval0+0];
    let rhs = dep::panicking_external_fn(*a);

    // CHECK: add.s32 %[[RES:r[0-9]+]], %[[RHS]], %[[LHS]];
    // CHECK: st.global.u32 [%{{rd[0-9]+}}], %[[RES]];
    *b = lhs + rhs;
}

// Verify that external function bodies are available.
// CHECK: .func (.param .b32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .b32 func_retval0) panicking_external_fn
