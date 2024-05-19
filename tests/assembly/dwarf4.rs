// Makes sure that `-Z dwarf-version=4` causes `rustc` to emit DWARF version 4.
//@ assembly-output: emit-asm
//@ compile-flags: -g --target x86_64-unknown-linux-gnu -Z dwarf-version=4 -Copt-level=0
//@ needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![crate_type = "rlib"]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub fn wibble() {}

pub struct X;

// CHECK: .section .debug_info
// CHECK-NOT: .short 2
// CHECK-NOT: .short 5
// CHECK: .short 4
// CHECK-NOT: .section .debug_pubnames
// CHECK-NOT: .section .debug_pubtypes
