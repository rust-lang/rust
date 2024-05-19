//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, lang_items, no_core)]
#![no_core]
#[lang="sized"]
trait Sized { }

extern "C-cmse-nonsecure-call" { //~ ERROR [E0781]
    fn test();
}
