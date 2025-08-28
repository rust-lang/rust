//@ add-core-stubs
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu -Zsanitizer=address -Ctarget-feature=-crt-static

// Make sure we do not request sanitizers for naked functions.

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

// CHECK: declare x86_intrcc void @page_fault_handler(ptr {{.*}}, i64{{.*}}){{.*}}#[[ATTRS:[0-9]+]]
#[unsafe(naked)]
#[no_mangle]
pub extern "x86-interrupt" fn page_fault_handler(_: u64, _: u64) {
    naked_asm!("ud2")
}

// CHECK: #[[ATTRS]] =
// CHECK-NOT: sanitize_address
// CHECK: !llvm.module.flags
