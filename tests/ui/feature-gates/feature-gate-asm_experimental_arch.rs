// compile-flags: --target mips-unknown-linux-gnu
// needs-llvm-components: mips

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

unsafe fn main() {
    asm!("");
    //~^ ERROR inline assembly is not stable yet on this architecture
}
