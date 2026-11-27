//@ add-minicore
//@ revisions: el eb
//@ assembly-output: emit-asm
//@ [el] compile-flags: --target bpfel-unknown-none -Copt-level=0
//@ [eb] compile-flags: --target bpfeb-unknown-none -Copt-level=0
//@ needs-llvm-components: bpf
//@ min-llvm-version: 23
#![feature(no_core, f16)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

// CHECK-LABEL: f16_to_f32:
// CHECK: call __extendhfsf2
#[no_mangle]
pub fn f16_to_f32(x: f16) -> f32 {
    x as f32
}
