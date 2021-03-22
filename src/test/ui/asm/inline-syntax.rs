// needs-llvm-components: arm
// revisions: x86_64 arm
//[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//[arm] compile-flags: --target armv7-unknown-linux-gnueabihf

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
        asm!(".intel_syntax noprefix", "nop");
        //[x86_64]~^ ERROR intel syntax is the default syntax on this target
        asm!(".intel_syntax aaa noprefix", "nop");
        //[x86_64]~^ ERROR intel syntax is the default syntax on this target
        asm!(".att_syntax noprefix", "nop");
        //[x86_64]~^ ERROR using the .att_syntax directive may cause issues
        //[arm]~^^ att syntax is the default syntax on this target
        asm!(".att_syntax bbb noprefix", "nop");
        //[x86_64]~^ ERROR using the .att_syntax directive may cause issues
        //[arm]~^^ att syntax is the default syntax on this target
        asm!(".intel_syntax noprefix; nop");
        //[x86_64]~^ ERROR intel syntax is the default syntax on this target

        asm!(
            r"
            .intel_syntax noprefix
            nop"
        );
        //[x86_64]~^^^ ERROR intel syntax is the default syntax on this target
    }
}
