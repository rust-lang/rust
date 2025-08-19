// Makes sure that `-C dwarf-version=5` causes `rustc` to emit DWARF version 5.
//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -g --target x86_64-unknown-linux-gnu -C dwarf-version=5 -Copt-level=0
//@ needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn wibble() {}

// CHECK: .section .debug_info
// CHECK-NOT: .short 2
// CHECK-NOT: .short 4
// CHECK: .short 5
// CHECK: .section .debug_names
