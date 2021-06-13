// build-pass
// compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
// needs-llvm-components: arm
#![feature(cmse_nonsecure_entry)]
#![no_std]

#[no_mangle]
#[cmse_nonsecure_entry]
pub extern "C" fn entry_function(a: u32, b: u32, c: u32, d: u32) -> u32 {
    a + b + c + d
}
