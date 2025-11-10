//@ add-minicore
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct Wrapper<T>(T);

impl<T: Copy> Wrapper<T> {
    extern "cmse-nonsecure-entry" fn ambient_generic(_: T, _: u32, _: u32, _: u32) -> u64 {
        //~^ ERROR [E0798]
        0
    }

    extern "cmse-nonsecure-entry" fn ambient_generic_nested(
        _: Wrapper<T>,
        //~^ ERROR [E0798]
        _: u32,
        _: u32,
        _: u32,
    ) -> u64 {
        0
    }
}

extern "cmse-nonsecure-entry" fn introduced_generic<U: Copy>(
    _: U,
    //~^ ERROR [E0798]
    _: u32,
    _: u32,
    _: u32,
) -> u64 {
    0
}

extern "cmse-nonsecure-entry" fn reference(x: &usize) -> usize {
    *x
}

trait Trait {}

extern "cmse-nonsecure-entry" fn trait_object(x: &dyn Trait) -> &dyn Trait {
    //~^ ERROR return value of `"cmse-nonsecure-entry"` function too large to pass via registers [E0798]
    x
}

extern "cmse-nonsecure-entry" fn static_trait_object(x: &'static dyn Trait) -> &'static dyn Trait {
    //~^ ERROR return value of `"cmse-nonsecure-entry"` function too large to pass via registers [E0798]
    x
}

#[repr(transparent)]
struct WrapperTransparent<'a>(&'a dyn Trait);

extern "cmse-nonsecure-entry" fn wrapped_trait_object(x: WrapperTransparent) -> WrapperTransparent {
    //~^ ERROR return value of `"cmse-nonsecure-entry"` function too large to pass via registers [E0798]
    x
}

extern "cmse-nonsecure-entry" fn impl_trait(_: impl Copy, _: u32, _: u32, _: u32) -> u64 {
    //~^ ERROR [E0798]
    0
}

extern "cmse-nonsecure-entry" fn return_impl_trait() -> impl Copy {
    //~^ ERROR `impl Trait` is not allowed in `extern "cmse-nonsecure-entry"` signatures
    0u128
}

extern "cmse-nonsecure-entry" fn return_impl_trait_nested() -> (impl Copy, i32) {
    //~^ ERROR `impl Trait` is not allowed in `extern "cmse-nonsecure-entry"` signatures
    (0u128, 0i32)
}

extern "cmse-nonsecure-entry" fn identity_impl_trait(v: impl Copy) -> impl Copy {
    //~^ ERROR generics are not allowed in `extern "cmse-nonsecure-entry"` signatures
    //~| ERROR `impl Trait` is not allowed in `extern "cmse-nonsecure-entry"` signatures
    v
}

extern "cmse-nonsecure-entry" fn identity_impl_trait_nested(
    v: (impl Copy, i32),
    //~^ ERROR generics are not allowed in `extern "cmse-nonsecure-entry"` signatures
) -> (impl Copy, i32) {
    //~^ ERROR `impl Trait` is not allowed in `extern "cmse-nonsecure-entry"` signatures
    v
}
