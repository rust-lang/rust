//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

extern "x86-interrupt" fn f7() {} //~ ERROR x86-interrupt ABI is experimental
trait Tr {
    extern "x86-interrupt" fn m7(); //~ ERROR x86-interrupt ABI is experimental
    extern "x86-interrupt" fn dm7() {} //~ ERROR x86-interrupt ABI is experimental
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "x86-interrupt" fn m7() {} //~ ERROR x86-interrupt ABI is experimental
}

// Methods in inherent impl
impl S {
    extern "x86-interrupt" fn im7() {} //~ ERROR x86-interrupt ABI is experimental
}

type A7 = extern "x86-interrupt" fn(); //~ ERROR x86-interrupt ABI is experimental

extern "x86-interrupt" {} //~ ERROR x86-interrupt ABI is experimental
