//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "rlib"]

use std::arch::asm;

// CHECK-LABEL: @pure
// CHECK-NOT: asm
// CHECK: ret void
#[no_mangle]
pub unsafe fn pure(x: i32) {
    let y: i32;
    asm!("", out("ax") y, in("cx") x, options(pure, nomem));
}

// CHECK-LABEL: @noreturn
// CHECK: call void asm
// CHECK-NEXT: unreachable
#[no_mangle]
pub unsafe fn noreturn() {
    asm!("", options(noreturn));
}

pub static mut VAR: i32 = 0;
pub static mut DUMMY_OUTPUT: i32 = 0;

// CHECK-LABEL: @readonly
// CHECK: call i32 asm
// CHECK: ret i32 1
#[no_mangle]
pub unsafe fn readonly() -> i32 {
    VAR = 1;
    asm!("", out("ax") DUMMY_OUTPUT, options(pure, readonly));
    VAR
}

// CHECK-LABEL: @not_readonly
// CHECK: call i32 asm
// CHECK: ret i32 %
#[no_mangle]
pub unsafe fn not_readonly() -> i32 {
    VAR = 1;
    asm!("", out("ax") DUMMY_OUTPUT, options());
    VAR
}

// CHECK-LABEL: @nomem
// CHECK-NOT: store
// CHECK: call i32 asm
// CHECK: store
// CHECK: ret i32 2
#[no_mangle]
pub unsafe fn nomem() -> i32 {
    VAR = 1;
    asm!("", out("ax") DUMMY_OUTPUT, options(pure, nomem));
    VAR = 2;
    VAR
}

// CHECK-LABEL: @nomem_nopure
// CHECK-NOT: store
// CHECK: call i32 asm
// CHECK: store
// CHECK: ret i32 2
#[no_mangle]
pub unsafe fn nomem_nopure() -> i32 {
    VAR = 1;
    asm!("", out("ax") DUMMY_OUTPUT, options(nomem));
    VAR = 2;
    VAR
}

// CHECK-LABEL: @not_nomem
// CHECK: store
// CHECK: call i32 asm
// CHECK: store
// CHECK: ret i32 2
#[no_mangle]
pub unsafe fn not_nomem() -> i32 {
    VAR = 1;
    asm!("", out("ax") DUMMY_OUTPUT, options(pure, readonly));
    VAR = 2;
    VAR
}

// CHECK-LABEL: @dont_remove_nonpure
// CHECK: call void asm
// CHECK: call void asm
// CHECK: call void asm
// CHECK: ret void
#[no_mangle]
pub unsafe fn dont_remove_nonpure() {
    asm!("", options());
    asm!("", options(nomem));
    asm!("", options(readonly));
}

// CHECK-LABEL: @raw
// CHECK: call void asm sideeffect inteldialect "{} {}", ""()
#[no_mangle]
pub unsafe fn raw() {
    asm!("{} {}", options(nostack, nomem, preserves_flags, raw));
}
