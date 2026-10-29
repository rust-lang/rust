// Test that a missing lang item (in this case `sized`) does not cause an ICE,
// see #17392.

#![feature(lang_items, no_core)]
#![no_core]
#![no_main]

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 { //~ ERROR requires `sized` lang_item
    loop {}
}
