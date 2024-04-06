// Make sure we do not request sanitizers for naked functions.
//@ add-core-stubs
//@ only-x86_64
//@ needs-llvm-components: x86
//@ needs-sanitizer-address
//@ compile-flags: --target x86_64-unknown-linux-gnu -Zunstable-options -Csanitize=address -Ctarget-feature=-crt-static

#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]
#![feature(abi_x86_interrupt)]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn caller() {
    unsafe { asm!("call {}", sym page_fault_handler) }
}

// CHECK: declare x86_intrcc void @page_fault_handler(){{.*}}#[[ATTRS:[0-9]+]]
#[unsafe(naked)]
#[no_mangle]
pub extern "x86-interrupt" fn page_fault_handler() {
    naked_asm!("ud2")
}

// CHECK: #[[ATTRS]] =
// CHECK-NOT: sanitize_address
// CHECK: !llvm.module.flags
