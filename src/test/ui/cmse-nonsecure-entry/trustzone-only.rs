// ignore-thumbv8m.main-none-eabi
#![feature(cmse_nonsecure_entry)]

#[no_mangle]
#[cmse_nonsecure_entry] //~ ERROR [E0775]
pub extern "C" fn entry_function(input: u32) -> u32 {
    input + 6
}

fn main() {}
