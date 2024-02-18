#![no_std]
#![feature(lang_items)]
#![warn(clippy::transmute_int_to_char)]

use core::panic::PanicInfo;

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}

fn int_to_char() {
    let _: char = unsafe { core::mem::transmute(0_u32) };
    //~^ ERROR: transmute from a `u32` to a `char`
    //~| NOTE: `-D clippy::transmute-int-to-char` implied by `-D warnings`
    let _: char = unsafe { core::mem::transmute(0_i32) };
    //~^ ERROR: transmute from a `i32` to a `char`

    // These shouldn't warn
    const _: char = unsafe { core::mem::transmute(0_u32) };
    const _: char = unsafe { core::mem::transmute(0_i32) };
}

fn main() {}
