//@ add-minicore
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn main() {
    let mut foo = 0;
    let mut bar = 0;
    unsafe {
        asm!("", a = in("x0") foo);
        //~^ ERROR explicit register arguments cannot have names
        asm!("{a}", in("x0") foo, a = const bar);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{a}", in("x0") foo, a = const bar);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{1}", in("x0") foo, const bar);
        //~^ ERROR positional arguments cannot follow named arguments or explicit register arguments
        //~^^ ERROR attempt to use a non-constant value in a constant
    }
}
