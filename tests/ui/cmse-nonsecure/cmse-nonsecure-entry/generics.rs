//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(cmse_nonsecure_entry, c_variadic, no_core, lang_items)]
#![no_core]
#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
impl Copy for u32 {}

#[repr(C)]
struct Wrapper<T>(T);

impl<T: Copy> Wrapper<T> {
    extern "C-cmse-nonsecure-entry" fn ambient_generic(_: T, _: u32, _: u32, _: u32) -> u64 {
        //~^ ERROR [E0798]
        0
    }

    extern "C-cmse-nonsecure-entry" fn ambient_generic_nested(
        //~^ ERROR [E0798]
        _: Wrapper<T>,
        _: u32,
        _: u32,
        _: u32,
    ) -> u64 {
        0
    }
}

extern "C-cmse-nonsecure-entry" fn introduced_generic<U: Copy>(
    //~^ ERROR [E0798]
    _: U,
    _: u32,
    _: u32,
    _: u32,
) -> u64 {
    0
}

extern "C-cmse-nonsecure-entry" fn impl_trait(_: impl Copy, _: u32, _: u32, _: u32) -> u64 {
    //~^ ERROR [E0798]
    0
}

extern "C-cmse-nonsecure-entry" fn reference(x: &usize) -> usize {
    *x
}

trait Trait {}

extern "C-cmse-nonsecure-entry" fn trait_object(x: &dyn Trait) -> &dyn Trait {
    //~^ ERROR return value of `"C-cmse-nonsecure-entry"` function too large to pass via registers [E0798]
    x
}

extern "C-cmse-nonsecure-entry" fn static_trait_object(
    x: &'static dyn Trait,
) -> &'static dyn Trait {
    //~^ ERROR return value of `"C-cmse-nonsecure-entry"` function too large to pass via registers [E0798]
    x
}

#[repr(transparent)]
struct WrapperTransparent<'a>(&'a dyn Trait);

extern "C-cmse-nonsecure-entry" fn wrapped_trait_object(
    x: WrapperTransparent,
) -> WrapperTransparent {
    //~^ ERROR return value of `"C-cmse-nonsecure-entry"` function too large to pass via registers [E0798]
    x
}

extern "C-cmse-nonsecure-entry" fn c_variadic(_: u32, _: ...) {
    //~^ ERROR only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg
    //~| ERROR requires `va_list` lang_item
}
