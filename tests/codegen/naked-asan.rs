// Make sure we do not request sanitizers for naked functions.

//@ only-x86_64
//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Ctarget-feature=-crt-static

#![crate_type = "lib"]
#![no_std]
#![feature(abi_x86_interrupt)]

pub fn caller() {
    page_fault_handler(1, 2);
}

// CHECK: declare x86_intrcc void @page_fault_handler(ptr {{.*}}, i64{{.*}}){{.*}}#[[ATTRS:[0-9]+]]
#[unsafe(naked)]
#[no_mangle]
pub extern "x86-interrupt" fn page_fault_handler(_: u64, _: u64) {
    core::arch::naked_asm!("ud2")
}

// CHECK: #[[ATTRS]] =
// CHECK-NOT: sanitize_address
// CHECK: !llvm.module.flags
