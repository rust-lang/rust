//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct Wrapper<T>(T);

struct Test<T: Copy> {
    f1: extern "C-cmse-nonsecure-call" fn<U: Copy>(U, u32, u32, u32) -> u64,
    //~^ ERROR cannot find type `U` in this scope
    //~| ERROR function pointer types may not have generic parameters
    f2: extern "C-cmse-nonsecure-call" fn(impl Copy, u32, u32, u32) -> u64,
    //~^ ERROR `impl Trait` is not allowed in `fn` pointer parameters
    f3: extern "C-cmse-nonsecure-call" fn(T, u32, u32, u32) -> u64, //~ ERROR [E0798]
    f4: extern "C-cmse-nonsecure-call" fn(Wrapper<T>, u32, u32, u32) -> u64, //~ ERROR [E0798]
}

type WithReference = extern "C-cmse-nonsecure-call" fn(&usize);

trait Trait {}
type WithTraitObject = extern "C-cmse-nonsecure-call" fn(&dyn Trait) -> &dyn Trait;
//~^ ERROR return value of `"C-cmse-nonsecure-call"` function too large to pass via registers [E0798]

type WithStaticTraitObject =
    extern "C-cmse-nonsecure-call" fn(&'static dyn Trait) -> &'static dyn Trait;
//~^ ERROR return value of `"C-cmse-nonsecure-call"` function too large to pass via registers [E0798]

#[repr(transparent)]
struct WrapperTransparent<'a>(&'a dyn Trait);

type WithTransparentTraitObject =
    extern "C-cmse-nonsecure-call" fn(WrapperTransparent) -> WrapperTransparent;
//~^ ERROR return value of `"C-cmse-nonsecure-call"` function too large to pass via registers [E0798]

type WithVarArgs = extern "C-cmse-nonsecure-call" fn(u32, ...);
//~^ ERROR C-variadic functions with the "C-cmse-nonsecure-call" calling convention are not supported
