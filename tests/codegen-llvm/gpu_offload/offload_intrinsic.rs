//@ compile-flags: -Zoffload=Enable -Zunstable-options -C opt-level=0  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// This test is verifying that we generate __tgt_target_data_*_mapper before and after a call to the
// kernel_1. Better documentation to what each global or variable means is available in the gpu
// offlaod code, or the LLVM offload documentation. This code does not launch any GPU kernels yet,
// and will be rewritten once a proper offload frontend has landed.
//
// We currently only handle memory transfer for specific calls to functions named `kernel_{num}`,
// when inside of a function called main. This, too, is a temporary workaround for not having a
// frontend.

// CHECK: ;
#![feature(core_intrinsics)]
#![no_main]

#[unsafe(no_mangle)]
fn main() {
    let mut x = [3.0; 256];
    kernel(&mut x);
    core::hint::black_box(&x);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn kernel(x: &mut [f32; 256]) {
    core::intrinsics::offload(_kernel)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn _kernel(x: &mut [f32; 256]) {
    for i in 0..256 {
        x[i] = 21.0;
    }
}
