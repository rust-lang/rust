//@ ignore-thumbv8m.main-none-eabi
#![feature(cmse_nonsecure_entry)]

#[no_mangle]
pub extern "C-cmse-nonsecure-entry" fn entry_function(input: u32) -> u32 {
    //~^ ERROR [E0570]
    input + 6
}

fn main() {}
