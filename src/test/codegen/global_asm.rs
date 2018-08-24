// ignore-aarch64
// ignore-aarch64_be
// ignore-arm
// ignore-armeb
// ignore-avr
// ignore-bpfel
// ignore-bpfeb
// ignore-hexagon
// ignore-mips
// ignore-mips64
// ignore-msp430
// ignore-powerpc64
// ignore-powerpc64le
// ignore-powerpc
// ignore-r600
// ignore-amdgcn
// ignore-sparc
// ignore-sparcv9
// ignore-sparcel
// ignore-s390x
// ignore-tce
// ignore-thumb
// ignore-thumbeb
// ignore-xcore
// ignore-nvptx
// ignore-nvptx64
// ignore-le32
// ignore-le64
// ignore-amdil
// ignore-amdil64
// ignore-hsail
// ignore-hsail64
// ignore-spir
// ignore-spir64
// ignore-kalimba
// ignore-shave
// ignore-wasm32
// ignore-wasm64
// ignore-emscripten
// compile-flags: -C no-prepopulate-passes

#![feature(global_asm)]
#![crate_type = "lib"]

// CHECK-LABEL: foo
// CHECK: module asm
// this regex will capture the correct unconditional branch inst.
// CHECK: module asm "{{[[:space:]]+}}jmp baz"
global_asm!(r#"
    .global foo
foo:
    jmp baz
"#);

extern "C" {
    fn foo();
}

// CHECK-LABEL: @baz
#[no_mangle]
pub unsafe extern "C" fn baz() {}
