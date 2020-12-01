// compile-flags: --target sparc-unknown-linux-gnu
// needs-llvm-components: sparc

#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[lang = "sized"]
trait Sized {}

fn main() {
    unsafe {
        asm!("");
        //~^ ERROR asm! is unsupported on this target
    }
}
