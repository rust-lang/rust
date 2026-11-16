// Checks that `aarch64-unknown-uefi`, which is lowered through LLVM's dedicated
// UEFI target, keeps the Windows ARM64 conventions the UEFI specification
// mandates.

//@ assembly-output: emit-asm
//@ add-minicore
//@ compile-flags: --target aarch64-unknown-uefi -Copt-level=3
//@ needs-llvm-components: aarch64

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    fn variadic(n: i32, ...);
}

// Variadic arguments are passed in general purpose registers, even the floating
// point ones; the AAPCS64 of other aarch64 targets would leave `x` in d0.
// CHECK-LABEL: call_variadic:
// CHECK: fmov x1, d0
// CHECK: b{{l?}} variadic
#[no_mangle]
pub unsafe extern "C" fn call_variadic(x: f64) {
    variadic(1, x)
}

// Large stack frames are probed through `__chkstk`, which compiler-builtins
// provides for this target.
// CHECK-LABEL: big_frame:
// CHECK: bl __chkstk
#[no_mangle]
pub fn big_frame() {
    let mut buf = [0u8; 65536];
    hint::black_box(&mut buf);
}
