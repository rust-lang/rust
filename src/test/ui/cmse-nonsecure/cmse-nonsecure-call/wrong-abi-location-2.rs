// compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
// only-thumbv8m.main-none-eabi
#![feature(abi_c_cmse_nonsecure_call)]
#![no_std]

extern "C-cmse-nonsecure-call" { //~ ERROR [E0781]
    fn test();
}
