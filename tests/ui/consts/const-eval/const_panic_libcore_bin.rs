#![crate_type = "bin"]
#![feature(lang_items)]
#![feature(panic_unwind)]
#![no_main]
#![no_std]

extern crate unwind;

use core::panic::PanicInfo;

const Z: () = panic!("cheese");
//~^ ERROR evaluation panicked

const Y: () = unreachable!();
//~^ ERROR evaluation panicked

const X: () = unimplemented!();
//~^ ERROR evaluation panicked

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
