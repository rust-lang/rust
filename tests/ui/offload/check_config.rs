//@ revisions: pass fail
//@ no-prefer-dynamic
//@ needs-enzyme
//@[pass] build-pass
//@[fail] build-fail
//@[pass] compile-flags: -Zunstable-options -Zoffload=Test -Clto=fat --emit=metadata
//@[fail] compile-flags: -Clto=thin

//[fail]~? ERROR: using the offload feature requires -Z offload=<Device or Host=/absolute/path/to/device.bin>
//[fail]~? ERROR: using the offload feature requires -C lto=fat

#![feature(core_intrinsics)]

fn main() {
    let mut x = [3.0; 256];
    kernel_1(&mut x);
}

fn kernel_1(x: &mut [f32; 256]) {
    core::intrinsics::offload::<_, _, ()>(_kernel_1, [1, 1, 1], [1, 1, 1], 0, (x,))
}

fn _kernel_1(x: &mut [f32; 256]) {}
