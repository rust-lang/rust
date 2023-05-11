// Test LVI ret hardening on generic rust code

// assembly-output: emit-asm
// compile-flags: --crate-type staticlib
// only-x86_64-fortanix-unknown-sgx

#[no_mangle]
pub extern fn myret() {}
// CHECK: myret:
// CHECK: popq [[REGISTER:%[a-z]+]]
// CHECK-NEXT: lfence
// CHECK-NEXT: jmpq *[[REGISTER]]
