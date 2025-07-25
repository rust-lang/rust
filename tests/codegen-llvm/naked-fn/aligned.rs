//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ needs-asm-support
//@ ignore-arm no "ret" mnemonic
//@ ignore-wasm32 aligning functions is not currently supported on wasm (#143368)

#![crate_type = "lib"]
// FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
#![feature(rustc_attrs)]
#![feature(fn_align)]

use std::arch::naked_asm;

// CHECK: .balign 16
// CHECK-LABEL: naked_empty:
#[rustc_align(16)]
#[no_mangle]
#[unsafe(naked)]
pub extern "C" fn naked_empty() {
    // CHECK: ret
    naked_asm!("ret")
}
