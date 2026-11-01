//@ add-minicore
//@ compile-flags: --target mips-unknown-linux-gnu
//@ needs-llvm-components: mips
//@ ignore-backends: gcc

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

unsafe fn main() {
    asm!("");
    //~^ ERROR inline assembly is not stable yet on this architecture
}
