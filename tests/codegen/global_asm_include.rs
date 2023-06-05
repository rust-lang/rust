// ignore-aarch64
// ignore-arm
// ignore-avr
// ignore-bpf
// ignore-bpf
// ignore-hexagon
// ignore-mips
// ignore-mips64
// ignore-msp430
// ignore-powerpc64
// ignore-powerpc
// ignore-sparc
// ignore-sparc64
// ignore-s390x
// ignore-thumb
// ignore-nvptx64
// ignore-spirv
// ignore-wasm32
// ignore-wasm64
// ignore-emscripten
// ignore-loongarch64
// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

use std::arch::global_asm;

// CHECK-LABEL: foo
// CHECK: module asm
// CHECK: module asm "{{[[:space:]]+}}jmp baz"
global_asm!(include_str!("foo.s"));

extern "C" {
    fn foo();
}

// CHECK-LABEL: @baz
#[no_mangle]
pub unsafe extern "C" fn baz() {}
