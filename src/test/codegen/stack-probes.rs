// ignore-arm
// ignore-aarch64
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-riscv64
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// ignore-wasm
// ignore-emscripten
// ignore-windows
// compile-flags: -C no-prepopulate-passes
// min-llvm-version: 11.0.1

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
// CHECK: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}"probe-stack"="inline-asm"{{.*}} }
}
