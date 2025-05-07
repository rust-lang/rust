//@ add-core-stubs
//@ compile-flags: --target riscv64imac-unknown-none-elf -Zsanitizer=shadow-call-stack
//@ needs-llvm-components: riscv

#![allow(internal_features)]
#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: ; Function Attrs:{{.*}}shadowcallstack
// CHECK: define dso_local void @foo() unnamed_addr #0
#[no_mangle]
pub fn foo() {}

// CHECK: attributes #0 = {{.*}}shadowcallstack{{.*}}
