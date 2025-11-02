// Test LVI ret hardening on generic rust code

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
pub extern "C" fn myret() {}
// CHECK-LABEL: myret:
// CHECK: popq [[REGISTER:%[a-z]+]]
// CHECK-NEXT: lfence
// CHECK-NEXT: jmpq *[[REGISTER]]
