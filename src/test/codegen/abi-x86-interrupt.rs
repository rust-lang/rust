// Checks if the correct annotation for the x86-interrupt ABI is passed to
// llvm. Also checks that the abi_x86_interrupt feature gate allows usage
// of the x86-interrupt abi.

// ignore-arm
// ignore-aarch64

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(abi_x86_interrupt)]

// CHECK: define x86_intrcc i64 @has_x86_interrupt_abi
#[no_mangle]
pub extern "x86-interrupt" fn has_x86_interrupt_abi(a: i64) -> i64 {
    a * 2
}
