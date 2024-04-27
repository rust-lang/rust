//@ compile-flags: --target thumbv6m-none-eabi
//@ needs-llvm-components: arm
//@ needs-asm-support

#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]
#![crate_type = "rlib"]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[lang = "sized"]
trait Sized {}

pub fn foo() {
    unsafe {
        asm!("", in("r8") 0);
        //~^ cannot use register `r8`: high registers (r8+) can only be used as clobbers in Thumb-1 code
    }
}
