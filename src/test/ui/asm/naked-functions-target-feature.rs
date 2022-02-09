// ignore-arm
// ignore-aarch64
// ignore-wasm
// ignore-emscripten
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-riscv64
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// needs-asm-support

#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::asm;

#[target_feature(enable = "sse2")]
//~^ ERROR cannot use additional code generation attributes with `#[naked]`
#[naked]
pub unsafe extern "C" fn naked_target_feature() {
    asm!("", options(noreturn));
}
