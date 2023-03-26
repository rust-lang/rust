// Test LVI load hardening on SGX inline assembly code

// assembly-output: emit-asm
// compile-flags: --crate-type staticlib
// only-x86_64-fortanix-unknown-sgx

use std::arch::asm;

#[no_mangle]
pub extern "C" fn get(ptr: *const u64) -> u64 {
    let value: u64;
    unsafe {
        asm!("mov {}, [{}]",
            out(reg) value,
            in(reg) ptr);
    }
    value
}

// CHECK: get
// CHECK: movq
// CHECK-NEXT: lfence

#[no_mangle]
pub extern "C" fn myret() {
    unsafe {
        asm!("ret");
    }
}

// CHECK: myret
// CHECK: shlq $0, (%rsp)
// CHECK-NEXT: lfence
// CHECK-NEXT: retq
