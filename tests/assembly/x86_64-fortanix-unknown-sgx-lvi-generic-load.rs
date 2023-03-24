// Test LVI load hardening on SGX enclave code

// assembly-output: emit-asm
// compile-flags: --crate-type staticlib
// only-x86_64-fortanix-unknown-sgx

#[no_mangle]
pub extern fn plus_one(r: &mut u64) {
    *r = *r + 1;
}

// CHECK: plus_one
// CHECK: lfence
// CHECK-NEXT: incq
// CHECK: popq [[REGISTER:%[a-z]+]]
// CHECK-NEXT: lfence
// CHECK-NEXT: jmpq *[[REGISTER]]
