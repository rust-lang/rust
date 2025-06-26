//@ add-core-stubs
//@ needs-asm-support
#![no_core]
#![feature(no_core, lang_items)]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

#[unsafe(naked)]
unsafe extern "custom" fn f7() {
    //~^ ERROR "custom" ABI is experimental
    naked_asm!("")
}
trait Tr {
    extern "custom" fn m7();
    //~^ ERROR "custom" ABI is experimental
    //~| ERROR functions with the "custom" ABI must be unsafe
    #[unsafe(naked)]
    extern "custom" fn dm7() {
        //~^ ERROR "custom" ABI is experimental
        //~| ERROR functions with the "custom" ABI must be unsafe
        naked_asm!("")
    }
}

struct S;

// Methods in trait impl
impl Tr for S {
    #[unsafe(naked)]
    extern "custom" fn m7() {
        //~^ ERROR "custom" ABI is experimental
        //~| ERROR functions with the "custom" ABI must be unsafe
        naked_asm!("")
    }
}

// Methods in inherent impl
impl S {
    #[unsafe(naked)]
    extern "custom" fn im7() {
        //~^ ERROR "custom" ABI is experimental
        //~| ERROR functions with the "custom" ABI must be unsafe
        naked_asm!("")
    }
}

type A7 = extern "custom" fn(); //~ ERROR "custom" ABI is experimental

extern "custom" {} //~ ERROR "custom" ABI is experimental
