// compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
// only-thumbv8m.main-none-eabi
#![feature(abi_c_cmse_nonsecure_call)]
#![no_std]

pub extern "C-cmse-nonsecure-call" fn test() {} //~ ERROR [E0781]
