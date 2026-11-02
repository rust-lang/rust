//@ add-minicore
//@ revisions: powerpc64 aarch64_be
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//
//@[aarch64_be] compile-flags: --target aarch64_be-unknown-linux-gnu
//@[aarch64_be] needs-llvm-components: aarch64
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

// On big-endian targets, the singleton HFA and the unit are not ABI-compatible, so it matters
// which is used. Test that:
//
// - on powerpc64 SingletonStruct<f32> is passed like f32
// - on aarch64_be SingletonStruct<f32> is passed like [f32; 1]
//
// The test start with 13 float arguments, to exhaust float registers and force the last argument
// onto the stack (technically we only need 8 float arguments on aarch64_be but a couple more
// does not matter).

extern crate minicore;
use minicore::*;

#[repr(C)]
struct SingletonStruct<T>(T);

#[no_mangle]
extern "C" fn singleton_struct_float(
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    x: SingletonStruct<f32>,
) -> f32 {
    // powerpc64: lfs 1, 156(1)
    // aarch64_be: ldr s0, [sp, #40]
    x.0
}

// CHECK-LABEL: singleton_array_float:
#[no_mangle]
extern "C" fn singleton_array_float(
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    x: [f32; 1],
) -> f32 {
    // powerpc64: lfs 1, 156(1)
    // aarch64_be: ldr s0, [sp, #40]
    match x {
        [a] => a,
    }
}

// CHECK-LABEL: float:
#[no_mangle]
extern "C" fn float(
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    _: f64,
    x: f32,
) -> f32 {
    // powerpc64: lfs 1, 156(1)
    // aarch64_be: ldr s0, [sp, #44]
    x
}
