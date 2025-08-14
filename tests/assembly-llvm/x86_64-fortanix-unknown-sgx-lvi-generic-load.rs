// Test LVI load hardening on SGX enclave code, specifically that `ret` is rewritten.

//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target x86_64-fortanix-unknown-sgx -Copt-level=0
//@ needs-llvm-components: x86

#![feature(no_core, lang_items, f16)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub extern "C" fn dereference(a: &mut u64) -> u64 {
    // CHECK-LABEL: dereference
    // CHECK: lfence
    // CHECK: mov
    // CHECK: popq [[REGISTER:%[a-z]+]]
    // CHECK-NEXT: lfence
    // CHECK-NEXT: jmpq *[[REGISTER]]
    *a
}
