#![no_std]
#![no_main]
#![deny(warnings)]
#![feature(abi_ptx, core_intrinsics)]

// Check the overriden CUDA arch.
// CHECK: .target sm_60
// CHECK: .address_size 64

// Verify that no extra function declarations are present.
// CHECK-NOT: .func

// CHECK-LABEL: .visible .entry top_kernel(
#[no_mangle]
pub unsafe extern "ptx-kernel" fn top_kernel(a: *const u32, b: *mut u32) {
    // CHECK: add.s32 %{{r[0-9]+}}, %{{r[0-9]+}}, 5;
    *b = *a + 5;
}

// Verify that no extra function definitions are there.
// CHECK-NOT: .func
// CHECK-NOT: .entry

#[panic_handler]
unsafe fn breakpoint_panic_handler(_: &::core::panic::PanicInfo) -> ! {
    core::intrinsics::breakpoint();
    core::hint::unreachable_unchecked();
}
