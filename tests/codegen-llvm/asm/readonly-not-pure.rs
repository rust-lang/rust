//@ add-core-stubs
//@ compile-flags: -Copt-level=3 --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![crate_type = "rlib"]
#![feature(no_core)]
#![no_core]

// Test that when an inline assembly block specifies `readonly` but not `pure`, a detailed
// `MemoryEffects` is provided to LLVM: this assembly block is not allowed to perform writes,
// but it may have side-effects.

extern crate minicore;
use minicore::*;

pub static mut VAR: i32 = 0;

// CHECK-LABEL: @no_options
// CHECK: call i32 asm
#[no_mangle]
pub unsafe fn no_options() -> i32 {
    VAR = 1;
    let _ignored: i32;
    asm!("mov {0}, 1", out(reg) _ignored);
    VAR
}

// CHECK-LABEL: @readonly_pure
// CHECK-NOT: call i32 asm
#[no_mangle]
pub unsafe fn readonly_pure() -> i32 {
    VAR = 1;
    let _ignored: i32;
    asm!("mov {0}, 1", out(reg) _ignored, options(pure, readonly));
    VAR
}

// CHECK-LABEL: @readonly_not_pure
// CHECK: call i32 asm {{.*}} #[[ATTR:[0-9]+]]
#[no_mangle]
pub unsafe fn readonly_not_pure() -> i32 {
    VAR = 1;
    let _ignored: i32;
    asm!("mov {0}, 1", out(reg) _ignored, options(readonly));
    VAR
}

// CHECK: attributes #[[ATTR]] = { nounwind memory(read, inaccessiblemem: readwrite) }
