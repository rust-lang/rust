//@ add-minicore
//@ assembly-output: emit-asm
//@ needs-llvm-components: aarch64
//@ compile-flags: -Copt-level=3 --target=aarch64-unknown-linux-gnu

// Previously, Rust used to assume omnipresent zero-extension on bool in FFI returns
// which resulted in LLVM assuming that a register did not require explicit masking (e.g. `and`)
// and could be correctly handled via `cmp w0, #0`.
// `cmp` looks at the full register, but only the 8 bits that contain the bool are specified,
// and this is particularly glaring in the event of a branch or csel based on this.
// Thus we are looking for explicit handling like `tst w0, #0x1` or `and w0, #0xFF`

// NOTE: simplifying this further is risky as `tbnz x0, #0, ...` only examines 1 bit,
// so LLVM will optimize it all away if there's an immediate jump to another function

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;

#[repr(C)]
struct Bools {
    a: bool,
    b: bool,
}

#[link(name = "rust_test_helpers")]
unsafe extern "C" {
    safe fn bools_get_first_bool(bools: Bools) -> bool;
}

// CHECK-LABEL: broken
pub fn broken() -> i32 {
    let bools = Bools { a: false, b: true };
    // CHECK: bl bools_get_first_bool
    // CHECK-NOT: cmp
    // CHECK: tst w0, #0x1
    // CHECK-NOT: cmp
    if bools_get_first_bool(bools) { 123 } else { 321 }
}
