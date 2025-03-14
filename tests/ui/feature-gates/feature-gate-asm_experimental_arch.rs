//@ add-core-stubs
//@ compile-flags: --target mips-unknown-linux-gnu
//@ needs-llvm-components: mips

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

unsafe fn main() {
    asm!("");
    //~^ ERROR inline assembly is not stable yet on this architecture
}
