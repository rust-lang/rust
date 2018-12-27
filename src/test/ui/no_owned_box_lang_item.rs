// Test that we don't ICE when we are missing the owned_box lang item.

// error-pattern: requires `owned_box` lang_item

#![feature(lang_items, box_syntax)]
#![no_std]

use core::panic::PanicInfo;

fn main() {
    let x = box 1i32;
}

#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "eh_unwind_resume"] extern fn eh_unwind_resume() {}
#[lang = "panic_impl"] fn panic_impl(panic: &PanicInfo) -> ! { loop {} }
