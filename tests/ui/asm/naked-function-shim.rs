// The indirect call will generate a shim that then calls the actual function. Test that
// this is handled correctly. See also https://github.com/rust-lang/rust/issues/143266.

//@ build-pass
//@ add-core-stubs
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

trait MyTrait {
    #[unsafe(naked)]
    extern "C" fn foo(&self) {
        naked_asm!("ret")
    }
}

impl MyTrait for i32 {}

fn main() {
    let x: extern "C" fn(&_) = <dyn MyTrait as MyTrait>::foo;
    x(&1);
}
