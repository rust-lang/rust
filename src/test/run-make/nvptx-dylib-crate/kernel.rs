#![no_std]
#![deny(warnings)]
#![feature(abi_ptx, core_intrinsics)]

extern crate dep;

// Verify the default CUDA arch.
// CHECK: .target sm_30
// CHECK: .address_size 64

// Make sure declarations are there.
// CHECK: .func (.param .b32 func_retval0) wrapping_external_fn
// CHECK: .func (.param .b32 func_retval0) panicking_external_fn
// CHECK: .func [[PANIC_HANDLER:_ZN4core9panicking5panic[a-zA-Z0-9]+]]

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
// CHECK-LABEL: .func (.param .b32 func_retval0) wrapping_external_fn
// CHECK: {
// CHECK:   st.param.b32 [func_retval0+0], %{{r[0-9]+}};
// CHECK: }

// Also verify panic behavior.
// CHECK-LABEL: .func (.param .b32 func_retval0) panicking_external_fn
// CHECK: {
// CHECK:   %{{p[0-9]+}} bra [[PANIC_LABEL:[a-zA-Z0-9_]+]];
// CHECK: [[PANIC_LABEL]]:
// CHECK:   call.uni
// CHECK:   [[PANIC_HANDLER]]
// CHECK: }

// Verify whether out dummy panic formatter has a correct body.
// CHECK: .func [[PANIC_FMT:_ZN4core9panicking9panic_fmt[a-zA-Z0-9]+]]()
// CHECK: {
// CHECK:   trap;
// CHECK: }

#[panic_handler]
unsafe fn breakpoint_panic_handler(_: &::core::panic::PanicInfo) -> ! {
    core::intrinsics::breakpoint();
    core::hint::unreachable_unchecked();
}
