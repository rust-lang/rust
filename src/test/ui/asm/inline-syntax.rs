// needs-llvm-components: arm
// revisions: x86_64 arm
//[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//[x86_64] check-pass
//[arm] compile-flags: --target armv7-unknown-linux-gnueabihf
//[arm] build-fail

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

#[lang = "sized"]
trait Sized {}

pub fn main() {
    unsafe {
        asm!(".intel_syntax noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.intel_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".intel_syntax aaa noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.intel_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".att_syntax noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.att_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".att_syntax bbb noprefix", "nop");
        //[x86_64]~^ WARN avoid using `.att_syntax`
        //[arm]~^^ ERROR unknown directive
        asm!(".intel_syntax noprefix; nop");
        //[x86_64]~^ WARN avoid using `.intel_syntax`
        //[arm]~^^ ERROR unknown directive

        asm!(
            r"
            .intel_syntax noprefix
            nop"
        );
        //[x86_64]~^^^ WARN avoid using `.intel_syntax`
        //[arm]~^^^^ ERROR unknown directive
    }
}
