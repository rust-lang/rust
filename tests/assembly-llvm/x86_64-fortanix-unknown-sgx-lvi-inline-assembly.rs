// Test LVI load hardening on SGX inline assembly code

//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target x86_64-fortanix-unknown-sgx
//@ needs-llvm-components: x86

#![feature(no_core, lang_items, f16)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub extern "C" fn get(ptr: *const u64) -> u64 {
    // CHECK-LABEL: get
    // CHECK: movq
    // CHECK-NEXT: lfence
    let value: u64;
    unsafe {
        asm!("mov {}, [{}]",
            out(reg) value,
            in(reg) ptr);
    }
    value
}

#[no_mangle]
pub extern "C" fn myret() {
    // CHECK-LABEL: myret
    // CHECK: shlq $0, (%rsp)
    // CHECK-NEXT: lfence
    // CHECK-NEXT: retq
    unsafe {
        asm!("ret");
    }
}
