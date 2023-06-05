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
#![no_std]

use core::arch::global_asm;

// CHECK-LABEL: foo
// CHECK: module asm
// CHECK: module asm "{{[[:space:]]+}}jmp baz"
// any other global_asm will be appended to this first block, so:
// CHECK-LABEL: bar
// CHECK: module asm "{{[[:space:]]+}}jmp quux"
global_asm!(
    r#"
    .global foo
foo:
    jmp baz
"#
);

extern "C" {
    fn foo();
}

// CHECK-LABEL: @baz
#[no_mangle]
pub unsafe extern "C" fn baz() {}

// no checks here; this has been appended to the first occurrence
global_asm!(
    r#"
    .global bar
bar:
    jmp quux
"#
);

extern "C" {
    fn bar();
}

#[no_mangle]
pub unsafe extern "C" fn quux() {}
