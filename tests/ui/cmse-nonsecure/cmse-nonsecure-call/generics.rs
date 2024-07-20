//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, no_core, lang_items)]
#![no_core]
#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
impl Copy for u32 {}

#[repr(C)]
struct Wrapper<T>(T);

struct Test<T: Copy> {
    f1: extern "C-cmse-nonsecure-call" fn<U: Copy>(U, u32, u32, u32) -> u64, //~ ERROR cannot find type `U` in this scope
    //~^ ERROR function pointer types may not have generic parameters
    f2: extern "C-cmse-nonsecure-call" fn(impl Copy, u32, u32, u32) -> u64,
    //~^ ERROR `impl Trait` is not allowed in `fn` pointer parameters
    f3: extern "C-cmse-nonsecure-call" fn(T, u32, u32, u32) -> u64, //~ ERROR [E0798]
    f4: extern "C-cmse-nonsecure-call" fn(Wrapper<T>, u32, u32, u32) -> u64, //~ ERROR [E0798]
}
